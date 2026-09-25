# LinkedIn — Projects section

Paste-ready. Primary version is ~1,750 characters, inside LinkedIn's ~2,000
limit. A shorter variant follows.

**Project name:** Movie Recommendation Agent — Conversational Recommender with MCP Tooling

**URL:** https://github.com/tpatil17/Movie_recommender

---

## Primary version

A conversational movie recommendation system built to production standards, with
the evaluation layer treated as a first-class component rather than an
afterthought.

**Architecture.** Four services: a FastAPI backend serving a hybrid recommender
(SVD collaborative filtering via Surprise + TF-IDF content similarity over
45,000 movies), an MCP server exposing four tools over SSE, a LangChain/LangGraph
agent on GPT-4o handling multi-turn conversation, and a React frontend. Deployed
on GCP Cloud Run and Firebase, instrumented with Prometheus and Grafana, with
GitHub Actions CI running 93 tests.

**The part I care about most.** The Precision@10 figure I originally reported
came from a notebook that couldn't be reproduced from the repository, so I
rebuilt offline evaluation properly: per-user held-out splits, SVD trained on
train data only, disjoint query and answer sets, and a popularity baseline
scored on the same users.

That baseline beat both of my personalised methods on precision@10 (0.1537 vs
0.0777). I reported it rather than omitting it, then diagnosed the cause:
the seed-anchored hybrid was retrieval-bound — 80.7% of users received a
candidate pool containing nothing they had rated highly, capping precision at
0.0263 while the ranker achieved 0.025. The ranker was at 95% of its achievable
ceiling, so the fix was candidate generation, not scoring. Rebuilding it as a
pure collaborative path tripled precision (0.026 → 0.0777) and shipped as its
own endpoint.

**Agent evaluation.** A 52-case behavioural benchmark with 119 deterministic
assertions covering tool selection, multi-turn memory, argument propagation, and
graceful failure — no LLM-as-judge, because grading one stochastic system with
another makes a before/after delta unattributable.

**Stack:** Python · FastAPI · LangChain · LangGraph · MCP · scikit-learn ·
Surprise · React · Prometheus · Grafana · Docker · GCP

---

## Shorter variant (~950 characters)

A conversational movie recommender built across four services: a FastAPI backend
(SVD collaborative filtering + TF-IDF content similarity over 45,000 movies), an
MCP server exposing tools over SSE, a LangChain/LangGraph agent on GPT-4o, and a
React frontend — deployed on GCP Cloud Run with Prometheus/Grafana observability
and CI running 93 tests.

The engineering I'm proudest of is the evaluation layer. My original Precision@10
came from an unreproducible notebook, so I rebuilt offline evaluation with
per-user held-out splits and a popularity baseline. That baseline beat my
personalised methods (0.1537 vs 0.0777) — I reported it rather than hiding it,
then diagnosed why: the hybrid was retrieval-bound, with 80.7% of users getting a
candidate pool containing nothing they liked. The ranker was already at 95% of
its ceiling, so I rebuilt candidate generation instead of tuning scores, tripling
precision.

Also built a 52-case deterministic behavioural benchmark for the agent, with no
LLM-as-judge.

**Stack:** Python · FastAPI · LangChain · LangGraph · MCP · scikit-learn ·
Surprise · React · Prometheus · Grafana · Docker · GCP

---

## Notes

- Every figure traces to `eval_offline/results/offline-three-way-*.json` and
  `diagnose-*.json` in the repo. Nothing here is rounded or estimated.
- 93 tests = 21 offline-eval metric tests + 52 behavioural check tests + 20
  backend model tests, all running in CI.
- The 52-case benchmark is described as *built*, not *run* — a baseline has not
  been captured yet. Once it is, add the pass rate; until then this phrasing is
  accurate.
- Keywords placed for recruiter search: LangChain, LangGraph, MCP, GPT-4o,
  FastAPI, collaborative filtering, Prometheus, Grafana, GCP, CI/CD.
