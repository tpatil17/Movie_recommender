# 🎬 Movie Recommendation Agent

A conversational movie recommendation system: a hybrid recommender behind an
MCP tool server, driven by a LangChain agent, with offline and behavioural
evaluation harnesses and Prometheus/Grafana observability.

**[Live Demo](https://movie-recommender-1726.web.app)** · **[GitHub](https://github.com/tpatil17/Movie_recommender)**

---

## Architecture

```
React (Firebase)  →  Agent :8002  →  MCP server :8001  →  Backend :8000
                     LangChain        FastMCP / SSE        FastAPI
                     gpt-4o           4 tools              SVD + TF-IDF
                          ↓                                     ↓
                     Prometheus :9090  →  Grafana :3001
```

| Service | Role |
|---|---|
| `services/backend` | Recommendation models and REST API. No LLM — testable and evaluable on its own. |
| `services/mcp-server` | Exposes four tools over MCP/SSE. Tool docstrings are the descriptions the LLM reads. |
| `services/agent` | LangChain/LangGraph agent over `gpt-4o`, multi-turn conversation, tool-call tracing. |
| `frontend` | React + Vite + Tailwind UI. |
| `eval_offline` | Offline recommender evaluation: held-out split, popularity baseline. |
| `eval_agent` | 52-case deterministic behavioural benchmark for the agent. |

---

## Recommendation approaches

Two distinct paths, because they answer different questions.

**Seed-anchored ("movies like X")** — `POST /api/recommendations`

```
title → content-based retrieval (TF-IDF + cosine, top 25)
      → collaborative re-ranking (SVD predicted rating)
      → top N
```

**Pure collaborative ("recommend for you")** — `GET /api/recommendations/for-you`

```
user_id → rank the full candidate pool (2,848 movies, support ≥ 5)
        → exclude already-rated titles
        → top N
```

The second exists because offline evaluation showed the first is
*retrieval-bound* — see below. Users with no rating history get an explicit
`cold_start: true` flag and a popularity fallback rather than a fabricated
personalised list.

---

## Evaluation

All numbers below are reproducible from this repository. Run
`eval_offline/eval_offline.py`; results are written to `eval_offline/results/`
as paired markdown and JSON.

### Offline results

300 users, K = 10, relevance = rated ≥ 4.0, per-user 70/30 split, SVD trained
on the train split only.

Figures below are copied verbatim from
`eval_offline/results/offline-three-way-20260804T234956Z.json`.

| metric@10 | pure CF | seed-anchored hybrid | popularity baseline |
|---|---|---|---|
| precision | 0.0777 | 0.0260 | **0.1537** |
| recall | 0.0378 | 0.0110 | **0.0912** |
| NDCG | 0.0878 | 0.0386 | **0.1948** |
| catalog coverage | 0.0051 | 0.0191 | 0.0009 |

**A popularity baseline beats both personalised methods on precision@10** — on
the full catalogue and on the long tail with the 200 most-popular titles removed
(0.0313 vs 0.0204 for CF, see `offline-tail-*.json`). This is reported rather
than omitted, with three caveats that matter:

1. On MovieLens, what users rate is overwhelmingly what is popular, so a
   non-personalised baseline is genuinely hard to beat on precision@10.
2. Random selection from the 42,277-title catalogue scores ≈ 0.0006. CF at
   0.0777 is roughly 130× random — the signal is real, it just loses to
   popularity on this metric.
3. Coverage inverts the ranking. Popularity recommends the same ~10 titles to
   every user (0.0009). The personalised methods actually differentiate.

### Why the hybrid underperforms

Diagnostics isolated the cause. The seed-anchored hybrid retrieves 25
content-neighbours of one movie and re-ranks them with SVD:

| measure | value |
|---|---|
| mean candidate pool | 9.81 (nominal 25) |
| users whose pool contained nothing they rated highly | 80.7% |
| precision@10 ceiling given that pool | 0.0263 |
| precision@10 achieved | 0.025 |

The ranker reached 95% of the maximum the candidate pool allowed. **Ranking was
never the bottleneck** — no amount of score tuning could have helped, because
the relevant movies were not in the pool. Two causes: `get_similar_movies`
applied its `vote_count` filter *after* slicing to the top 25, collapsing the
pool; and single-seed retrieval structurally cannot represent a user's whole
taste profile. The fix was candidate generation, which is what the pure-CF path
is.

### Behavioural benchmark

`eval_agent/` contains 52 cases and 119 deterministic assertions across six
categories — `seed_movie`, `open_ended`, `similarity`, `graceful_failure`,
`memory`, `arg_propagation`.

No LLM-as-judge. An LLM judge places two stochastic systems in series, so when
a score moves you cannot tell whether the agent changed or the judge did. Every
assertion here is a pure function over the agent's tool calls and response text,
so each pass or fail names its own cause. The tradeoff: these checks verify
contract adherence, not recommendation quality — quality is `eval_offline`'s
job.

Benchmark runs require all three services plus an OpenAI key. Set
`AGENT_TEMPERATURE=0` for any run intended for comparison; the runner records
the temperature it observed and warns if it sees sampling.

> **Status:** the harness is complete and unit-tested, but a baseline run has
> not yet been captured. No behavioural numbers are claimed.

---

## API

```
GET  /health                                    service health
GET  /api/movies/search?q={title}               autocomplete search
GET  /api/movies/similar?title={title}          content-only similarity
GET  /api/movies/{tmdb_id}                      movie detail
POST /api/recommendations                       seed-anchored hybrid
GET  /api/recommendations/for-you?user_id={id}  pure collaborative
```

Agent service: `POST /chat`, `DELETE /sessions/{id}`, `GET /meta`, `GET /health`.

Interactive docs at `/docs` (Swagger UI).

**Example**

```bash
curl 'http://localhost:8000/api/recommendations/for-you?user_id=1&top_n=5'
```

```json
{
  "user_id": 1,
  "cold_start": false,
  "results": [
    {
      "title": "The Conversation",
      "predicted_rating": 4.41,
      "genres": ["crime", "drama", "mystery"],
      "reason": "Users with similar taste rated this highly"
    }
  ]
}
```

---

## Observability

Prometheus scrapes `/metrics` on the backend and agent every 15s; Grafana
dashboards sit on top.

| Service | Metrics |
|---|---|
| Backend | `recommendation_requests_total{status}`, `for_you_requests_total{status}`, `search_requests_total`, `similar_requests_total`, latency histograms, `content_model_cache_hits/misses_total`, `models_loaded` |
| Agent | `agent_chat_requests_total{status}`, `agent_chat_latency_seconds`, `agent_tool_calls_total{tool_name,status}`, `agent_active_sessions` |

`agent_tool_calls_total{tool_name, status}` is the useful one: it shows the
agent's tool-selection distribution in production, the operational counterpart
to what the behavioural benchmark measures offline.

MCP tools return `{"error": ...}` payloads rather than raising, so LangChain
reports them as successful. The agent parses tool results to derive real status
before incrementing the counter — otherwise the success rate would read 100%
while tools were failing.

---

## Performance

| Optimization | Details |
|---|---|
| Similarity cache | Per-title cosine results cached in memory; repeat queries skip the pass entirely |
| Startup pre-warming | Top 50 titles by vote count pre-computed at boot |
| Bisect search index | `/movies/search` uses a sorted title list + `bisect` for O(log n) prefix lookup |
| Frontend debounce | Search input debounced 300ms |
| Connection warm-up | `GET /health` on page load establishes the TCP connection early |

Content similarity is O(n) per uncached query across ~42k rows. The cache makes
this acceptable at this scale; an ANN index (FAISS, HNSW) would be the next
step.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite, Tailwind CSS 4 |
| Backend | FastAPI, Python 3.11 |
| Agent | LangChain, LangGraph, OpenAI `gpt-4o` |
| Tool protocol | MCP via FastMCP (SSE) |
| ML | scikit-learn (TF-IDF), scikit-surprise (SVD) |
| Data | pandas, numpy |
| Observability | Prometheus, Grafana |
| Hosting | GCP Cloud Run (backend), Firebase Hosting (frontend), GCS (data) |
| CI | GitHub Actions |

---

## Project structure

```
movie_recommender/
├── services/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── data/loader.py           data loading + feature engineering
│   │   │   ├── models/
│   │   │   │   ├── content_based.py     TF-IDF + cosine similarity
│   │   │   │   ├── collaborative.py     SVD; recommend_for_user
│   │   │   │   └── hybrid.py            seed-anchored retrieve-then-rank
│   │   │   ├── routes/recommendations.py
│   │   │   ├── main.py                  FastAPI app + model lifecycle
│   │   │   ├── metrics.py               Prometheus metrics
│   │   │   └── schemas.py
│   │   └── tests/                       20 model tests
│   ├── mcp-server/
│   │   ├── tools/                       search, recommendations, similar, for_you
│   │   └── main.py                      FastMCP server (SSE)
│   └── agent/
│       ├── agent/chain.py               agent construction, temperature/tag config
│       ├── agent/prompts.py             system prompt
│       └── main.py                      /chat, tool-call extraction, /meta
├── eval_offline/                        recommender evaluation (21 tests)
├── eval_agent/                          behavioural benchmark (52 cases, 52 tests)
├── frontend/src/App.jsx
├── docs/
│   ├── technical-walkthrough.md         architecture and decision rationale
│   └── ui-plan.md
├── prometheus/prometheus.yml
├── .github/workflows/test.yml
└── docker-compose.yml
```

---

## Running locally

**Prerequisites:** Python 3.11, Node 18+, an OpenAI API key for the agent.

```bash
git clone https://github.com/tpatil17/Movie_recommender
cd Movie_recommender

python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install numpy==1.26.4 Cython
pip install git+https://github.com/NicolasHug/Surprise.git
pip install -r services/backend/requirements.txt
```

**Dataset** — download [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
and place these in `services/backend/data/raw/`:
`movies_metadata.csv`, `ratings_small.csv`, `credits.csv`, `links_small.csv`.

```bash
# backend
cd services/backend && uvicorn app.main:app --port 8000

# MCP server
cd services/mcp-server && pip install -r requirements.txt && python main.py

# agent (needs OPENAI_API_KEY)
cd services/agent && pip install -r requirements.txt && python main.py

# frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`. The backend alone is enough for the REST API;
the MCP server and agent are only needed for the chat surface.

> **Known issue:** `docker-compose up` does not currently work. The backend
> Dockerfile serves port 8080 while compose maps 8000, and compose sets
> `BACKEND_URL` without the `/api` prefix the MCP tools expect. Run the
> services directly as above until this is fixed.

---

## Tests

```bash
pytest eval_offline/test_eval_offline.py -v          # 21 — metric correctness
pytest eval_agent/test_checks.py -v                  # 52 — behavioural checks
PYTHONPATH=services/backend pytest services/backend/tests/ -v   # 20 — models
```

All three run in CI on every push and pull request to `main`. The two
evaluation suites are pure functions over synthetic data — no CSVs, no models,
no API key — so they finish in well under a second. A broken metric silently
invalidates every number reported above, which is why they are tested at all.

---

## Dataset

[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
— 45,000+ movies with ~100,000 MovieLens ratings (`ratings_small.csv`). Full
MovieLens is 26M ratings; results here may not hold at that scale.

---

## Known limitations

| Limitation | Detail |
|---|---|
| Popularity beats personalisation | On precision@10, full catalogue and long tail. Coverage tells the other half of the story. |
| `DEFAULT_USER_ID = 1` hardcoded | In the MCP recommendations tool — personalisation is inert on the agent path. Encoded as failing benchmark cases. |
| Conversation state in-process | A dict capped at 20 messages; does not survive restart or scale past one instance. |
| Content feature weights hand-tuned | Director repeated 3×, genres 2× in the TF-IDF "soup" — never validated against the harness. |
| Behavioural baseline not captured | Harness complete, no run recorded. |
| Docker Compose broken | See the port/prefix issue above. |

---

## Author

**Tanishq Patil** — MS Computer Science, San Diego State University

[LinkedIn](https://linkedin.com/in/tanishq-patil) · [GitHub](https://github.com/tpatil17)
