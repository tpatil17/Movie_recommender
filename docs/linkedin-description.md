# LinkedIn — Projects section

Paste-ready. Every figure traces to `eval_offline/results/offline-three-way-*.json`
and `diagnose-*.json`.

**Project name:** Movie Recommendation Agent — Conversational Recommender with MCP Tooling

**URL:** https://github.com/tpatil17/Movie_recommender

---

## Five-bullet version (preferred)

- Built a conversational movie recommender across four services: a FastAPI
  backend (SVD collaborative filtering and TF-IDF content similarity over 45,000
  movies), an MCP server exposing four tools over SSE, a LangChain/LangGraph
  agent on GPT-4o, and a React frontend, deployed on GCP Cloud Run and Firebase.
- Built an offline evaluation harness using holdout splits by user, with SVD
  trained only on train data and a popularity baseline scored on the same users.
  On open ended requests the baseline beat both personalised methods on
  precision@10 (0.1537 vs 0.0777), which is expected given that precision on
  logged ratings is biased toward popular items.
- Diagnosed the seed based path as limited by retrieval, measured inside its own
  candidate pool: 80.7% of users got a pool containing nothing they rated highly,
  capping precision at 0.0263 against 0.025 achieved. Built a separate
  collaborative endpoint for open ended requests, reaching 0.0777.
- Built a behavioural benchmark of 52 cases and 119 deterministic assertions
  covering tool selection, memory across turns, argument propagation, and
  graceful failure. No LLM judge, so every delta is attributable.
- Instrumented all services with Prometheus and Grafana, including tool selection
  distribution by tool and status, with GitHub Actions CI running 93 tests.

**Stack:** Python · FastAPI · LangChain · LangGraph · MCP · scikit-learn ·
Surprise · React · Prometheus · Grafana · Docker · GCP

---

## Prose version (~1,500 characters)

A conversational movie recommendation system built to production standards, with
the evaluation layer treated as a first-class component rather than an
afterthought.

**Architecture.** Four services: a FastAPI backend serving a hybrid recommender
(SVD collaborative filtering via Surprise + TF-IDF content similarity over 45,000
movies), an MCP server exposing four tools over SSE, a LangChain/LangGraph agent
on GPT-4o handling multi-turn conversation, and a React frontend. Deployed on GCP
Cloud Run and Firebase, instrumented with Prometheus and Grafana, with GitHub
Actions CI running 93 tests.

**Evaluation.** Offline evaluation uses per-user held-out splits with SVD trained
on train data only, disjoint query and answer sets, and a popularity baseline
scored on the same users. That baseline beat both personalised methods on
precision@10 (0.1537 vs 0.0777) — reported rather than omitted, then diagnosed:
the seed-anchored hybrid was retrieval-bound, with 80.7% of users receiving a
candidate pool containing nothing they had rated highly. That capped precision at
0.0263 while the ranker reached 0.025, already 95% of its achievable ceiling. The
fix was candidate generation, not scoring — rebuilding it as a pure collaborative
path tripled precision to 0.0777 and shipped as its own endpoint.

**Agent evaluation.** A 52-case behavioural benchmark with 119 deterministic
assertions covering tool selection, multi-turn memory, argument propagation and
graceful failure — no LLM-as-judge, because grading one stochastic system with
another makes a before/after delta unattributable.

**Stack:** Python · FastAPI · LangChain · LangGraph · MCP · scikit-learn ·
Surprise · React · Prometheus · Grafana · Docker · GCP

---

## Notes

- 93 tests = 21 offline-eval metric tests + 52 behavioural check tests + 20
  backend model tests, all running in CI.
- The 52-case benchmark is described as *built*, not *run* — no baseline has been
  captured yet. Once one is, add the pass rate; until then this phrasing is
  accurate.
- Keywords placed for recruiter search: LangChain, LangGraph, MCP, GPT-4o,
  FastAPI, collaborative filtering, Prometheus, Grafana, GCP, CI/CD.
