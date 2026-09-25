"""
Behavioural benchmark runner for the movie recommendation agent.

Drives the 50-case suite against a live agent over HTTP, applies deterministic
checks to what came back, and writes a tagged report. No LLM judge: every
pass/fail traces to a named assertion in checks.py.

Each case gets a fresh session (DELETE /sessions/{id}) so no case can be
helped or hurt by a previous one's history.

Prerequisites: backend (8000), MCP server (8001), agent (8002) all running.
For a comparable run, start the agent with AGENT_TEMPERATURE=0 -- the runner
reads /meta and records what it found, and warns loudly if it sees sampling.

Usage:
    python runner.py --tag baseline
    python runner.py --tag after-fix --category open_ended
    python runner.py --case open_01 --case arg_05 --verbose
    python runner.py --compare results/agent-baseline-*.json
"""

import argparse
import glob
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

from cases import DEFAULT_CASE_USER_ID, CATEGORIES, select
from checks import run_check

AGENT_URL_DEFAULT = "http://localhost:8002"


# --------------------------------------------------------------------------
# agent driving
# --------------------------------------------------------------------------

def fetch_meta(client: httpx.Client, base: str) -> dict:
    try:
        r = client.get(f"{base}/meta", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        # /meta is new; an older agent simply will not have it.
        return {"agent_tag": "unknown", "temperature": None, "tools": None}


def reset_session(client: httpx.Client, base: str, session_id: str) -> None:
    try:
        client.delete(f"{base}/sessions/{session_id}", timeout=10.0)
    except Exception:
        pass  # a session that never existed is already clean


def run_case(client: httpx.Client, base: str, case: dict, timeout: float) -> dict:
    """Send every turn in one session, then apply the case's checks."""
    session_id = f"eval-{case['id']}-{int(time.time() * 1000)}"
    reset_session(client, base, session_id)

    user_id = case.get("user_id", DEFAULT_CASE_USER_ID)
    turns = []
    error = None

    for message in case["turns"]:
        payload = {"message": message, "session_id": session_id, "user_id": user_id}
        try:
            r = client.post(f"{base}/chat", json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        turns.append({
            "message": message,
            "response": data.get("response", ""),
            "tool_calls": data.get("tool_calls", []),
            "latency_ms": data.get("latency_ms", 0.0),
        })

    reset_session(client, base, session_id)

    if error:
        return {
            "id": case["id"],
            "category": case["category"],
            "passed": False,
            "error": error,
            "turns": turns,
            "checks": [],
        }

    results = [run_check(spec, turns) for spec in case["checks"]]
    return {
        "id": case["id"],
        "category": case["category"],
        "passed": all(c["passed"] for c in results),
        "error": None,
        "turns": turns,
        "checks": results,
        "latency_ms": round(sum(t["latency_ms"] for t in turns), 1),
        "tools_used": [c.get("name") for t in turns for c in (t.get("tool_calls") or [])],
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def summarize(results: list[dict]) -> dict:
    by_cat = defaultdict(lambda: {"passed": 0, "total": 0})
    check_fails = defaultdict(int)

    for r in results:
        by_cat[r["category"]]["total"] += 1
        if r["passed"]:
            by_cat[r["category"]]["passed"] += 1
        for c in r.get("checks", []):
            if not c["passed"]:
                check_fails[c["check"]] += 1

    passed = sum(1 for r in results if r["passed"])
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "by_category": {
            k: {**v, "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] else 0.0}
            for k, v in sorted(by_cat.items())
        },
        "failures_by_check": dict(sorted(check_fails.items(), key=lambda x: -x[1])),
    }


def format_report(report: dict) -> str:
    s = report["summary"]
    meta = report["agent_meta"]
    lines = [
        f"# Agent behavioural benchmark — {report['tag']}",
        "",
        f"- timestamp: {report['timestamp']}",
        f"- agent tag: {meta.get('agent_tag')}",
        f"- temperature: {meta.get('temperature')}",
        f"- tools registered: {', '.join(meta.get('tools') or []) or 'unknown'}",
        f"- **{s['passed']}/{s['total']} passed ({s['pass_rate']:.1%})**",
        "",
        "## By category",
        "",
        "| category | passed | total | rate |",
        "|---|---|---|---|",
    ]
    for cat, v in s["by_category"].items():
        lines.append(f"| {cat} | {v['passed']} | {v['total']} | {v['pass_rate']:.1%} |")

    if s["failures_by_check"]:
        lines += ["", "## Failures by check type", "",
                  "| check | failures |", "|---|---|"]
        for name, n in s["failures_by_check"].items():
            lines.append(f"| {name} | {n} |")

    failed = [r for r in report["results"] if not r["passed"]]
    if failed:
        lines += ["", "## Failed cases", ""]
        for r in failed:
            lines.append(f"**{r['id']}** ({r['category']})")
            if r.get("error"):
                lines.append(f"  - transport error: {r['error']}")
            for c in r.get("checks", []):
                if not c["passed"]:
                    lines.append(f"  - `{c['check']}`: {c['detail']}")
            if r.get("tools_used"):
                lines.append(f"  - tools called: {r['tools_used']}")
            lines.append("")

    return "\n".join(lines)


def format_comparison(before: dict, after: dict) -> str:
    b, a = before["summary"], after["summary"]
    lines = [
        "# Agent benchmark: before vs after",
        "",
        f"- before: `{before['tag']}` ({before['agent_meta'].get('agent_tag')}), {before['timestamp']}",
        f"- after:  `{after['tag']}` ({after['agent_meta'].get('agent_tag')}), {after['timestamp']}",
        "",
        "| category | before | after | delta |",
        "|---|---|---|---|",
    ]
    cats = sorted(set(b["by_category"]) | set(a["by_category"]))
    for cat in cats:
        bv = b["by_category"].get(cat, {"pass_rate": 0.0, "passed": 0, "total": 0})
        av = a["by_category"].get(cat, {"pass_rate": 0.0, "passed": 0, "total": 0})
        delta = av["pass_rate"] - bv["pass_rate"]
        lines.append(
            f"| {cat} | {bv['passed']}/{bv['total']} | {av['passed']}/{av['total']} | {delta:+.1%} |"
        )
    lines.append(
        f"| **overall** | **{b['passed']}/{b['total']}** | **{a['passed']}/{a['total']}** | "
        f"**{a['pass_rate'] - b['pass_rate']:+.1%}** |"
    )

    # Case-level movement is the interesting part: a flat overall rate can
    # still hide fixes offset by regressions.
    bmap = {r["id"]: r["passed"] for r in before["results"]}
    amap = {r["id"]: r["passed"] for r in after["results"]}
    fixed = sorted(i for i in bmap if not bmap[i] and amap.get(i))
    broke = sorted(i for i in bmap if bmap[i] and amap.get(i) is False)
    if fixed:
        lines += ["", f"**Fixed ({len(fixed)}):** {', '.join(fixed)}"]
    if broke:
        lines += ["", f"**Regressed ({len(broke)}):** {', '.join(broke)}"]
    if not fixed and not broke:
        lines += ["", "No case-level changes."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    p.add_argument("--tag", default="baseline", help="label for the output files")
    p.add_argument("--category", action="append", choices=CATEGORIES,
                   help="restrict to a category (repeatable)")
    p.add_argument("--case", action="append", help="run specific case ids (repeatable)")
    p.add_argument("--timeout", type=float, default=90.0, help="per-turn timeout, seconds")
    p.add_argument("--verbose", action="store_true", help="print each case as it runs")
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                   help="compare two result JSON files and exit")
    args = p.parse_args()

    if args.compare:
        before_path = sorted(glob.glob(args.compare[0]))[-1]
        after_path = sorted(glob.glob(args.compare[1]))[-1]
        before = json.loads(Path(before_path).read_text())
        after = json.loads(Path(after_path).read_text())
        print(format_comparison(before, after))
        return 0

    cases = select(categories=args.category, ids=args.case)
    if not cases:
        raise SystemExit("No cases selected.")

    with httpx.Client() as client:
        try:
            client.get(f"{args.agent_url}/health", timeout=10.0).raise_for_status()
        except Exception as exc:
            raise SystemExit(
                f"Agent not reachable at {args.agent_url} ({exc}).\n"
                "Start backend (8000), MCP server (8001), and agent (8002) first."
            )

        meta = fetch_meta(client, args.agent_url)
        temp = meta.get("temperature")
        if temp not in (0, 0.0):
            print(
                f"WARNING: agent temperature is {temp}, not 0. Results will vary\n"
                f"         run to run and a before/after delta may be sampling noise.\n"
                f"         Restart the agent with AGENT_TEMPERATURE=0 for a comparable run.\n",
                file=sys.stderr,
            )

        print(f"Running {len(cases)} cases against {args.agent_url} "
              f"(agent_tag={meta.get('agent_tag')}, temp={temp}) ...\n", flush=True)

        results = []
        start = time.time()
        for i, case in enumerate(cases, 1):
            r = run_case(client, args.agent_url, case, args.timeout)
            results.append(r)
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"  [{i:>2}/{len(cases)}] {mark}  {r['id']:<10} {r['category']}", flush=True)
            if args.verbose and not r["passed"]:
                if r.get("error"):
                    print(f"          transport error: {r['error']}")
                for c in r.get("checks", []):
                    if not c["passed"]:
                        print(f"          {c['check']}: {c['detail']}")

    report = {
        "tag": args.tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_meta": meta,
        "elapsed_s": round(time.time() - start, 1),
        "summary": summarize(results),
        "results": results,
    }

    markdown = format_report(report)
    print()
    print(markdown)

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"agent-{args.tag}-{stamp}.md").write_text(markdown)
    (out / f"agent-{args.tag}-{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote results/agent-{args.tag}-{stamp}.md")

    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
