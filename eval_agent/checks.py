"""
Deterministic behavioural checks for the movie recommendation agent.

Every function here is pure: it takes the turns an agent produced and returns
(passed, detail). No network, no LLM, no model. That is the whole design
premise -- an LLM-as-a-judge introduces a second stochastic system into the
measurement, and the Valorant project already covers that ground. A check
either fires or it does not, and the reason is always inspectable.

A "turn" is one request/response pair, shaped like the agent's ChatResponse:

    {
      "message": "I loved Interstellar",     # what we sent
      "response": "Great choice! ...",       # what came back
      "tool_calls": [
        {"name": "search_movies", "args": {...}, "status": "success",
         "result": "..."},
      ],
      "latency_ms": 1240.0,
    }

Checks operate on the full list of turns for a case, so multi-turn assertions
(memory, recall) work the same way as single-turn ones.
"""

from __future__ import annotations

import re

CheckResult = tuple[bool, str]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def all_tool_calls(turns: list[dict]) -> list[dict]:
    """Every tool call across every turn, in order."""
    calls = []
    for turn in turns:
        calls.extend(turn.get("tool_calls") or [])
    return calls


def tool_names(turns: list[dict]) -> list[str]:
    return [c.get("name", "") for c in all_tool_calls(turns)]


def all_text(turns: list[dict]) -> str:
    return "\n".join(t.get("response", "") or "" for t in turns)


def _norm(s: str) -> str:
    """Lowercase and collapse whitespace, so phrasing differences don't fail."""
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


# --------------------------------------------------------------------------
# tool selection
# --------------------------------------------------------------------------

def tool_called(turns: list[dict], name: str) -> CheckResult:
    names = tool_names(turns)
    if name in names:
        return True, f"{name} called"
    return False, f"{name} never called; called {names or '[]'}"


def tool_not_called(turns: list[dict], name: str) -> CheckResult:
    names = tool_names(turns)
    if name not in names:
        return True, f"{name} correctly not called"
    return False, f"{name} should not have been called; called {names}"


def any_tool_called(turns: list[dict], names: list[str]) -> CheckResult:
    called = tool_names(turns)
    hit = [n for n in names if n in called]
    if hit:
        return True, f"called {hit[0]}"
    return False, f"none of {names} called; called {called or '[]'}"


def no_tools_called(turns: list[dict]) -> CheckResult:
    called = tool_names(turns)
    if not called:
        return True, "no tools called"
    return False, f"expected no tool calls, got {called}"


def tool_order(turns: list[dict], first: str, then: str) -> CheckResult:
    """
    `first` must appear before the first occurrence of `then`.

    This is the system prompt's central rule: confirm a title via
    search_movies before calling get_recommendations. A violation means the
    agent is inventing titles rather than verifying them.
    """
    names = tool_names(turns)
    if then not in names:
        return False, f"{then} never called; order not satisfied (called {names or '[]'})"
    if first not in names:
        return False, f"{first} never called before {then} (called {names})"
    if names.index(first) < names.index(then):
        return True, f"{first} preceded {then}"
    return False, f"{then} called before {first} (order: {names})"


def tool_call_count(turns: list[dict], name: str, maximum: int) -> CheckResult:
    """Guards against retry loops, which burn tokens and look broken."""
    n = tool_names(turns).count(name)
    if n <= maximum:
        return True, f"{name} called {n}x (max {maximum})"
    return False, f"{name} called {n}x, exceeds max {maximum}"


# --------------------------------------------------------------------------
# tool arguments
# --------------------------------------------------------------------------

def tool_arg_equals(turns: list[dict], name: str, arg: str, expected) -> CheckResult:
    """
    Catches the argument-propagation bugs: top_n ignored, user_id not threaded.
    Checks every call to `name`, so one bad call fails the case.
    """
    calls = [c for c in all_tool_calls(turns) if c.get("name") == name]
    if not calls:
        return False, f"{name} never called, cannot check {arg}"
    for call in calls:
        actual = (call.get("args") or {}).get(arg, "<missing>")
        if actual != expected:
            return False, f"{name}.{arg} = {actual!r}, expected {expected!r}"
    return True, f"{name}.{arg} = {expected!r} on all {len(calls)} call(s)"


def tool_arg_present(turns: list[dict], name: str, arg: str) -> CheckResult:
    calls = [c for c in all_tool_calls(turns) if c.get("name") == name]
    if not calls:
        return False, f"{name} never called, cannot check {arg}"
    missing = [i for i, c in enumerate(calls) if arg not in (c.get("args") or {})]
    if missing:
        return False, f"{name} missing {arg} on call(s) {missing}"
    return True, f"{name} passed {arg} on all {len(calls)} call(s)"


# --------------------------------------------------------------------------
# failure handling
# --------------------------------------------------------------------------

def no_tool_errors(turns: list[dict]) -> CheckResult:
    bad = [
        f"{c.get('name')}({c.get('status')})"
        for c in all_tool_calls(turns)
        if c.get("status") != "success"
    ]
    if not bad:
        return True, "all tool calls succeeded"
    return False, f"failed tool calls: {bad}"


def graceful_failure(turns: list[dict]) -> CheckResult:
    """
    The agent hit a dead end and must say so rather than inventing an answer.

    Passes when the reply acknowledges the problem and does not leak a raw
    error. Deliberately does not require an apology or specific wording --
    that would test phrasing, not behaviour.
    """
    text = _norm(all_text(turns))
    if not text:
        return False, "empty response"

    leaked = ["traceback", "httpstatuserror", "internal server error",
              "500 ", "connectionerror", "keyerror", "nonetype"]
    for token in leaked:
        if token in text:
            return False, f"leaked raw error text: {token!r}"

    acknowledges = [
        "couldn't find", "could not find", "can't find", "cannot find",
        "not found", "no results", "doesn't appear", "does not appear",
        "unable to find", "don't have", "do not have", "not in",
        "no movie", "didn't find", "did not find",
    ]
    if any(p in text for p in acknowledges):
        return True, "acknowledged the failure"
    return False, "did not acknowledge the failure"


def no_fabricated_titles(turns: list[dict], forbidden: list[str]) -> CheckResult:
    """
    After a failed lookup the agent must not present made-up titles as real
    recommendations. `forbidden` is seeded with the bogus title we sent.
    """
    text = _norm(all_text(turns))
    hits = [t for t in forbidden if _norm(t) in text]
    if not hits:
        return True, "no fabricated titles"
    return False, f"echoed fabricated title(s): {hits}"


# --------------------------------------------------------------------------
# response content and memory
# --------------------------------------------------------------------------

def response_contains(turns: list[dict], needle: str) -> CheckResult:
    if _norm(needle) in _norm(all_text(turns)):
        return True, f"response contains {needle!r}"
    return False, f"response missing {needle!r}"


def response_contains_any(turns: list[dict], needles: list[str]) -> CheckResult:
    text = _norm(all_text(turns))
    hits = [n for n in needles if _norm(n) in text]
    if hits:
        return True, f"response contains {hits[0]!r}"
    return False, f"response contains none of {needles}"


def response_not_contains(turns: list[dict], needle: str) -> CheckResult:
    if _norm(needle) not in _norm(all_text(turns)):
        return True, f"response correctly omits {needle!r}"
    return False, f"response unexpectedly contains {needle!r}"


def recall_prior_title(turns: list[dict], title: str, from_turn: int = 0) -> CheckResult:
    """
    Multi-turn memory: a title established in an earlier turn must resurface
    in a later one, without the user restating it.

    This is the check that catches session history being dropped -- the agent
    answers each turn in isolation and the conversation stops being one.
    """
    if len(turns) <= from_turn + 1:
        return False, f"case has {len(turns)} turn(s), need at least {from_turn + 2}"
    later = _norm("\n".join(t.get("response", "") or "" for t in turns[from_turn + 1:]))
    if _norm(title) in later:
        return True, f"recalled {title!r} in a later turn"
    return False, f"did not recall {title!r} after turn {from_turn}"


def recommends_at_least(turns: list[dict], count: int) -> CheckResult:
    """
    Counts recommendation-shaped lines in the final response. Deliberately
    loose: it verifies the agent actually presented a list rather than
    describing one, without pinning exact formatting.
    """
    text = turns[-1].get("response", "") if turns else ""
    lines = [
        ln for ln in text.splitlines()
        if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", ln)
    ]
    if len(lines) >= count:
        return True, f"presented {len(lines)} list items (need {count})"
    return False, f"presented {len(lines)} list items, need at least {count}"


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

CHECKS = {
    "tool_called": tool_called,
    "tool_not_called": tool_not_called,
    "any_tool_called": any_tool_called,
    "no_tools_called": no_tools_called,
    "tool_order": tool_order,
    "tool_call_count": tool_call_count,
    "tool_arg_equals": tool_arg_equals,
    "tool_arg_present": tool_arg_present,
    "no_tool_errors": no_tool_errors,
    "graceful_failure": graceful_failure,
    "no_fabricated_titles": no_fabricated_titles,
    "response_contains": response_contains,
    "response_contains_any": response_contains_any,
    "response_not_contains": response_not_contains,
    "recall_prior_title": recall_prior_title,
    "recommends_at_least": recommends_at_least,
}


def run_check(spec: dict, turns: list[dict]) -> dict:
    """
    spec is {"check": "tool_called", "name": "get_for_you"} -- the "check" key
    selects the function, every other key is passed as a keyword argument.
    """
    kind = spec.get("check")
    fn = CHECKS.get(kind)
    if fn is None:
        return {"check": kind, "passed": False, "detail": f"unknown check {kind!r}"}
    kwargs = {k: v for k, v in spec.items() if k != "check"}
    try:
        passed, detail = fn(turns, **kwargs)
    except TypeError as exc:
        return {"check": kind, "passed": False, "detail": f"bad check args: {exc}"}
    return {"check": kind, "passed": passed, "detail": detail, "args": kwargs}
