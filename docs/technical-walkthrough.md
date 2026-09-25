# Movie Recommendation Agent — technical walkthrough

A study guide, not a README. Read Parts 1–5 to rebuild the reasoning, Part 6
for the decision forks, Part 7 to drill. Part 8 is the material you should
raise yourself before an interviewer finds it.

The goal is not memorised answers. It is being able to *derive* the answer live
when the follow-up goes somewhere this doc did not.

---

## Part 0 — The 60-second pitch

> A conversational movie recommendation agent. Three services: a FastAPI
> backend with a hybrid recommender, an MCP server exposing four tools over
> SSE, and a LangChain agent that talks to them. What I care about most is the
> evaluation layer — I found the precision number on my own resume wasn't
> reproducible, rebuilt the offline evaluation honestly, and discovered my
> hybrid loses to a popularity baseline. I kept that result and diagnosed why
> rather than hiding it. There's also a 52-case deterministic behavioural
> benchmark for the agent, with no LLM judge.

That last sentence is the hook. Most portfolio recommenders assert a metric.
Yours reports a result that makes the model look worse, which is a far stronger
signal about how you work.

**Do not lead with the model quality.** Lead with the measurement discipline.

---

## Part 1 — Architecture: why three services

```
React (Firebase)  →  Agent :8002  →  MCP server :8001  →  Backend :8000
                     LangChain        FastMCP/SSE          FastAPI
                     gpt-4o           4 tools              SVD + TF-IDF
                          ↓                                     ↓
                     Prometheus :9090  →  Grafana :3001
```

**Why the backend is separate from the agent.** The recommender is a
deterministic service that trains models at startup and answers HTTP. It has no
notion of conversation. Keeping it separate means it can be tested, evaluated,
and benchmarked without an LLM anywhere in the loop — which is exactly what
`eval_offline/` does. If the model logic lived inside the agent, every offline
evaluation run would need an OpenAI key.

**Why MCP is its own service rather than LangChain tools defined inline.**
This is the question you will get, and "because MCP is the standard" is not an
answer. The real reasons:

- The tool layer has its own lifecycle. You can restart the agent without
  restarting tools, and vice versa.
- Tool definitions live next to the code that implements them, not inside the
  agent's prompt assembly.
- Any MCP-speaking client can consume them — Claude Desktop, another agent
  framework, a different LLM — with no change to the tool code.
- It forces the tool contract to be explicit and serialisable, rather than an
  in-process Python function that happens to be passed around.

**The honest cost:** an extra network hop and an extra process to run. For a
single agent with four tools, inline LangChain tools would be simpler. You
chose MCP because the interop boundary is the point of the exercise, and that
is a legitimate answer — just do not pretend it is free.

---

## Part 2 — The recommender internals

### Content-based: TF-IDF over a "soup"

For each movie, `loader.py` builds a text blob:

```python
f"{cast} {director*3} {genres*2} {overview}"
```

Top 3 cast members, director repeated three times, genres twice, plus the
overview. Names are lowercased and de-spaced so "Tom Hanks" becomes one token
`tomhanks` rather than two common words.

Then `TfidfVectorizer(stop_words='english', min_df=2)` and cosine similarity of
one movie's vector against the entire matrix.

**What repetition is doing.** Repeating the director three times triples its
term frequency, so it dominates the vector. It is crude manual feature
weighting. A cleaner approach would be separate vectorisers per field with
explicit weights, or learned weights. If asked "why 3 and 2?" the honest answer
is: hand-tuned, not validated. That is a real weakness — those constants were
never swept against the evaluation harness, and now that the harness exists,
they could be.

**Why TF-IDF rather than raw counts.** IDF downweights terms appearing in many
documents. "drama" appears in thousands of movies and carries almost no
discriminative signal; a specific director appears in a handful and carries a
lot. Raw counts would let common genres dominate.

**The cost.** `cosine_similarity(movie_vec, self.count_matrix)` compares against
every movie on every query — O(n) per lookup over ~42k movies. That is why
there is an in-memory cache and a warm-up of the 50 most-voted titles at
startup, with `cache_hits`/`cache_misses` counters. At real scale you would
reach for an ANN index (FAISS, HNSW) instead.

### Collaborative: SVD

Matrix factorisation from the `surprise` library. Users and items each get a
latent factor vector; a predicted rating is:

```
r̂(u,i) = μ + b_u + b_i + qᵢ · pᵤ
```

Global mean, plus a user bias, plus an item bias, plus the dot product of the
item and user factor vectors. Trained by SGD over **observed ratings only** —
it never treats missing entries as zeros.

**Two consequences that matter, and both show up in your results:**

*Cold start.* A user the model never saw has no `pᵤ`. The prediction collapses
to roughly `μ + b_i` — identical ordering for every unknown user. This is why
`/recommendations/for-you` returns an explicit `cold_start: true` and falls
back to popularity rather than pretending to personalise. Rendering that as
personalised would be the same dishonesty the evaluation work was meant to stop.

*Support.* An item with three ratings has a barely-trained `qᵢ`, so its
prediction is dominated by biases and is nearly noise. That is why the CF
candidate pool filters to `support >= 5`, and why `CF_MIN_SUPPORT` in the
backend deliberately matches `--cf-min-support` in the harness — otherwise the
product ranks a different pool than the one you measured.

### The hybrid, and why it failed

```python
similar = content.get_similar_movies(title, top_n=25)   # retrieve
for movie in similar:                                    # rank
    score = collab.predict_rating(user_id, movielens_id)
```

This is a standard two-stage **retrieve-then-rank** design: a cheap model
narrows the catalogue, an expensive/personalised model orders the shortlist.
The architecture is sound. The instance was not.

Diagnostics found:

| measure | value |
|---|---|
| mean candidate pool | **9.81** (nominal 25) |
| users with pool < 10 | 51.3% |
| mean relevant items in pool | 0.263 |
| users whose pool contained nothing they liked | **80.7%** |
| precision@10 ceiling | 0.0263 |
| precision@10 achieved | 0.025 |

**Read that carefully, because it is the most interesting thing in the
project.** The ranker achieved 95% of the maximum score the candidate pool
permitted. Ranking was never the bottleneck. No amount of blending, tuning, or
re-weighting could have moved the number, because the relevant movies were not
in the pool to be ranked.

Two causes:

1. **A filter-ordering bug.** `get_similar_movies` slices to the top 25 *then*
   drops everything with `vote_count < 100`. Obscure content-neighbours are
   exactly what that filter kills, so the pool collapsed to ~9.8 before ranking
   ever ran.
2. **Single-seed retrieval, structurally.** The hybrid asks "what resembles
   this one film," while the metric asks "what will this user rate ≥ 4." Even
   with a full 25 candidates the ceiling lands near 0.05.

The fix was therefore candidate generation, not scoring — which is why
`recommend_for_user` ranks a broad 2,848-movie pool with no seed at all, and
scores 0.078 versus the hybrid's 0.026.

**There was also a genuine bug**: `hybrid.py` scored every candidate twice with
two different formulas and kept whichever was higher, so a movie's score
depended on which formula happened to be more generous. Worth mentioning, but
be precise — fixing it barely moved precision, because ranking was already at
ceiling. Claiming that fix drove an improvement would be exactly the kind of
overclaiming this project is meant to avoid.

---

## Part 3 — Agent and MCP

### What MCP actually does here

`services/mcp-server/main.py` runs a FastMCP server over SSE on port 8001 with
four tools. Each tool is a plain async Python function that calls the backend
over HTTP.

**The critical detail most people miss: the docstring is the prompt.** When
`MultiServerMCPClient` pulls tools, the function signature becomes the tool's
JSON schema and the docstring becomes the description the LLM reads when
deciding what to call. That is why `search.py`'s docstring says what the tool
is for *and* what it is not for:

> Do not use this tool for genre or mood requests such as 'a comedy'...

That negative instruction is prompt engineering living inside a tool
definition. When a tool gets mis-selected, the docstring is the first place to
look — not the system prompt.

### The four tools

| tool | purpose | backend |
|---|---|---|
| `search_movies` | resolve partial/misspelled titles | `GET /api/movies/search` |
| `get_recommendations` | seed-anchored, personalised | `POST /api/recommendations` |
| `get_similar` | content-only, no personalisation | `GET /api/movies/similar` |
| `get_for_you` | pure CF, no seed | `GET /api/recommendations/for-you` |

`get_for_you` exists because evaluation showed the seed-anchored path was
retrieval-bound. The measurement produced a product change — that is the
strongest narrative thread in the project. Say it that way.

### The agent loop

`create_agent` builds a LangGraph ReAct-style agent over `gpt-4o`. One turn:

1. LLM receives system prompt + tool schemas + conversation history
2. It either emits an `AIMessage` with `tool_calls`, or a final answer
3. Each tool call executes; results come back as `ToolMessage`
4. Loop until the LLM produces a message with no tool calls

`_extract_tool_calls` walks the messages produced this turn and pairs each
requested call with its result **by `tool_call_id`** — args live on the
`AIMessage`, results on the `ToolMessage`, so you need both.

**One subtlety worth knowing cold.** Your MCP tools return `{"error": "..."}`
dicts instead of raising. LangChain sees a successful tool response, so
`ToolMessage.status` stays `"success"` even when the call failed. `_tool_status`
parses the payload to catch this. Without it, `agent_tool_calls_total{status}`
would report a 100% success rate while tools were failing — a metric that looks
healthy precisely when it should not.

### Conversation state

An in-process dict keyed by `session_id`, capped at 20 messages. It is not
persistent and does not survive a restart or scale past one instance. Know this
and say it before you are asked — the honest framing is that it is a
deliberate scope decision, and the fix is Redis or Postgres.

---

## Part 4 — Evaluation

This is your strongest material. Know it in the most depth.

### The origin

The Precision@10 of 0.74 on your resume came from a gitignored notebook. It
could not be reproduced from the repository, and 0.74 is implausibly high for
top-N recommendation on MovieLens. Almost certainly it measured something
easier — likely rating prediction on held-out ratings, or ranking with the
answer visible.

**The lesson to state out loud:** a metric you cannot regenerate from a clean
checkout is not a result, it is a claim.

### Offline protocol (`eval_offline/`)

Four properties, each blocking a specific way the number could inflate:

| choice | inflation it prevents |
|---|---|
| Per-user split; SVD fits train only | Model scoring ratings it trained on |
| Seed from train, relevant set from test | Seed-to-target leakage |
| Relevance = rated ≥ 4.0, not merely rated | Counting "watched" as "liked" |
| Popularity baseline on the same users | Reporting a number with nothing to compare it to |
| A title credits only on first appearance | Recall exceeding 1.0 via duplicates |

### The metrics — be able to define these cold

**Precision@10** = hits in top 10 / **10**. Note the denominator is `k`, not
the number of items returned. Return two items and get one right and precision
is 0.1, not 0.5 — otherwise a model that returns one lucky item scores
perfectly.

**Recall@10** = hits in top 10 / size of the user's relevant set.

**NDCG@10** = DCG / IDCG, where DCG sums `1/log2(rank+1)` over hits. A hit at
rank 1 contributes 1.0, at rank 2 contributes 0.63. IDCG places
`min(|relevant|, k)` hits at the top. **The cap matters**: with 50 relevant
items and k=5, without capping, five perfect hits would score 0.1 instead of
1.0 — you would be penalising the model for the list being short.

**Catalog coverage** = distinct titles recommended across all users / catalog
size. This is your counter-argument, so keep it handy.

### The result

| metric@10 | cf | hybrid | popularity |
|---|---|---|---|
| precision | 0.078 | 0.026 | **0.154** |
| recall | 0.038 | 0.011 | **0.091** |
| ndcg | 0.088 | 0.039 | **0.195** |
| coverage | 0.005 | 0.019 | **0.001** |

Popularity wins on precision — on the full catalogue *and* on the long tail
with the 200 most-popular titles removed. That second run matters: it kills the
easy excuse that popularity only wins because blockbusters dominate.

**The three things to say about this, in order:**

1. On MovieLens, what people rate is overwhelmingly what is popular. A
   non-personalised baseline is genuinely hard to beat on precision@10. This is
   a well-known result, not a personal failure.
2. Random selection from 42k titles scores ~0.0006. CF at 0.078 is roughly 130×
   random. The signal is real; it just loses to popularity on this metric.
3. Coverage inverts the ranking. Popularity recommends the same ~10 titles to
   everyone (0.001). CF and the hybrid differentiate between users. Precision@10
   on held-out ratings structurally rewards recommending what is already
   popular — which is the metric's blind spot, not the model's.

### Behavioural benchmark (`eval_agent/`)

52 cases, 119 assertions, six categories: `seed_movie`, `open_ended`,
`similarity`, `graceful_failure`, `memory`, `arg_propagation`.

**Why no LLM judge.** An LLM judge puts two stochastic systems in series: when
the number moves, you cannot tell whether the agent changed, the judge changed,
or neither. For a benchmark whose entire purpose is a trustworthy before/after
delta, that is disqualifying. `tool_order(search_movies, get_recommendations)`
either held or it did not.

**The tradeoff — state it before they do.** Deterministic checks cannot assess
whether a recommendation was *good*. They assess whether the agent honoured its
contract. Quality is `eval_offline`'s job. The two harnesses are complementary,
and neither is sufficient alone.

**Why temperature 0 for benchmark runs.** The product runs at 0.7 so
conversation is not canned. A behavioural benchmark at 0.7 measures sampling
noise as much as behaviour, so `AGENT_TEMPERATURE` overrides it to 0 for
measurement. The runner reads `/meta`, records the temperature in every report,
and warns if it sees sampling — you cannot accidentally compare two
incomparable runs.

**Cases designed to fail.** `arg_propagation` asserts `user_id` reaches the
tools. It does not — `DEFAULT_USER_ID = 1` is hardcoded, so personalisation is
inert on the agent path. Those failures encode the bug so the fix has something
to prove itself against.

---

## Part 5 — Infrastructure and observability

**Metrics.** Prometheus scrapes `/metrics` on backend and agent every 15s.

Backend: `recommendation_requests_total{status}`, `search_requests_total`,
`similar_requests_total`, `for_you_requests_total{status}`, latency histograms,
`content_model_cache_hits/misses_total`, `models_loaded` gauge.

Agent: `agent_chat_requests_total{status}`, `agent_chat_latency_seconds`,
`agent_tool_calls_total{tool_name,status}`, `agent_active_sessions`.

**Counter vs Histogram vs Gauge** — a standard interview question:

- *Counter* only increases; you query its rate. Request counts, error counts.
- *Histogram* buckets observations so you can compute quantiles server-side.
  Latency.
- *Gauge* goes up and down. Active sessions, models-loaded.

**The most interesting metric is `agent_tool_calls_total{tool_name, status}`.**
Labelled by tool, it shows the agent's tool-selection distribution in
production — the operational counterpart to what the behavioural benchmark
measures offline. If `get_for_you` never fires in production, routing is broken
regardless of what the benchmark says.

**CI.** GitHub Actions on push/PR to main: installs numpy and Cython before
building `surprise` from source (it needs numpy headers at build time), then
runs three suites — offline eval metric tests, agent behavioural check tests,
backend tests. The two evaluation suites are pure functions over synthetic data,
so they need no CSVs, no models, and no API key, and finish in under a tenth of
a second.

**A dependency conflict worth understanding.** CI went red on
`starlette==0.37.2` against `fastapi==0.137.1`, which requires `>=0.46.0`. Fixing
it exposed a second conflict — `pydantic==2.7.1` versus a `>=2.9.0` requirement.
Root cause: the file is a `pip freeze` where fastapi was upgraded without
regenerating the rest, so the local environment was already inconsistent and
only a clean resolve surfaced it. The general lesson: a lockfile that only ever
gets partially updated is a lockfile that has stopped locking.

**Deployment.** Backend on Cloud Run, frontend on Firebase Hosting, data from
GCS when `GCS_BUCKET` is set (see `loader.py`), local disk otherwise.

---

## Part 6 — The decision log

For each: the fork, the choice, and the cost. Interviewers probe the cost.

**1. Recompute the baseline against buggy code before fixing.**
Alternative: fix first, measure after. Chosen because a before/after delta is
only meaningful if "before" was measured, not remembered. Cost: an extra
evaluation run, and living with a bad number in the interim.

**2. Include a popularity baseline.**
Alternative: report hybrid precision alone. Chosen because a ranking metric
without a baseline is uninterpretable. Cost: it produced a result where the
model loses. Kept anyway — that is the point.

**3. Deterministic checks over LLM-as-judge.**
Alternative: LLM judge, which scales to subjective quality. Chosen for
measurement stability and because a separate project already demonstrates
judge-based evaluation. Cost: cannot assess recommendation quality, only
contract adherence.

**4. Expose CF as a new endpoint rather than replacing the hybrid.**
Alternative: swap the hybrid's internals. Chosen because they answer different
questions — "movies like X" and "movies for you" are both legitimate, and the
UI surfaces them separately. Cost: two code paths, two things to maintain.

**5. Explicit `cold_start` flag instead of silent popularity fallback.**
Alternative: return popular titles and say nothing. Chosen because unlabelled
fallback is the product-layer version of an unreproducible metric. Cost: the
frontend must handle a state most demos would hide.

**6. Keep the raw CF score, drop the `vote_average` blend.**
When removing the double-scoring bug, one loop blended in TMDB `vote_average`.
Dropped it: `vote_average` is a popularity prior, and blending it in while
benchmarking against a popularity baseline makes "we beat popularity" partly
circular. Cost: lost a tie-breaking signal for sparse users, where SVD
predictions bunch near the mean. Now testable with the harness rather than
guessed at.

**7. `AGENT_TEMPERATURE` env override rather than pinning to 0.**
Alternative: temperature 0 everywhere. Chosen to keep the product
conversational while making measurement reproducible. Cost: the benchmark
tests a slightly different configuration than production runs — a real caveat,
and you should name it before an interviewer does.

---

## Part 7 — Drill questions

Answer out loud before reading on. If you cannot answer in your own words,
reread the relevant part.

### Warm-up

**Q: Walk me through what happens when a user types "what should I watch?"**
Frontend POSTs to the agent. The agent sends the system prompt, tool schemas
and history to gpt-4o. No title was named, so the model should select
`get_for_you`, which calls the MCP server, which calls
`GET /api/recommendations/for-you`. The backend ranks the ~2,848-movie CF pool
by that user's predicted rating, excluding movies they have already rated,
returns the top 10. The agent phrases the result. `tool_calls` comes back to
the frontend so the trace can be rendered.

**Q: What does `predicted_rating: 4.3` mean?**
SVD's estimated rating that specific user would give that movie:
`μ + b_u + b_i + qᵢ·pᵤ`, clipped to the 0.5–5 scale. Not a probability, not a
confidence. For an unknown user it degenerates toward the global mean, which is
why cold start is flagged rather than served silently.

**Q: Why does the content model need a cache?**
Cosine similarity runs against all ~42k rows per query — O(n) each time. The
cache makes repeat lookups free, and the 50 most-voted titles are warmed at
startup so the first real query is not cold. `cache_hits`/`cache_misses` track
whether it is actually working.

### Core

**Q: Your precision is 0.078 and a popularity baseline gets 0.154. Why is this
project worth anything?**
Two answers. First, the finding itself is the value — most people never build
the baseline and so never learn this. Second, precision@10 on held-out
MovieLens ratings structurally favours popular items, because what people rate
is what is popular. Coverage inverts the ranking entirely: popularity shows the
same ten titles to everyone at 0.001, CF differentiates. Which metric matters
depends on whether the product's job is being right or being useful.

**Q: Then why not just ship the popularity baseline?**
For precision@10 alone, you would. But a recommender that shows everyone the
same ten movies has no reason to exist as a product, cannot support a
conversational agent, and cannot improve with user data. The honest position:
popularity is the right *baseline* and the wrong *product*, and I can now
quantify the gap instead of assuming it.

**Q: You said the hybrid was "retrieval-bound." Prove it.**
The candidate pool held 0.263 relevant items on average, and 80.7% of users got
a pool with none at all. That caps precision@10 at 0.0263 by construction. The
hybrid achieved 0.025 — 95% of the ceiling. The ranker was doing nearly
everything possible with what it was given. That is why the fix was candidate
generation, not scoring.

**Q: Why not just raise `top_n` from 25 to 200?**
It would help — the filter-ordering bug alone costs 60% of the pool. But
estimated gains put the ceiling near 0.05, still under popularity's 0.154. The
limitation is structural: one seed movie cannot represent a user's whole taste
profile. Widening a narrow funnel does not change that it is anchored to a
single point.

**Q: Why MCP instead of defining LangChain tools directly?**
Process isolation and interop. The tool server has an independent lifecycle,
tool definitions live with their implementations, and any MCP client can
consume them. The cost is a network hop and another process — for four tools
and one agent, inline tools would be simpler. I chose it because the
interoperability boundary is the part I wanted to build.

**Q: How does the LLM know when to call `get_for_you` versus
`get_recommendations`?**
Tool descriptions come from the Python docstrings, plus the system prompt.
`get_for_you`'s docstring specifies open-ended requests with no title named;
`get_recommendations` requires a confirmed title. `search_movies`' docstring
explicitly says not to use it for genre or mood requests. Routing is prompt
engineering distributed across tool definitions — which is why the behavioural
benchmark tests routing directly rather than trusting it.

**Q: Your benchmark has 52 cases. How do you know they are the right ones?**
I do not, fully. They were derived from bugs found in code review — hardcoded
`user_id`, unreachable `get_for_you`, `top_n` not propagating — plus categories
for known agent failure modes: memory, graceful failure, injection. The suite
encodes hypotheses about how this agent breaks. Cases should be added whenever
a new failure is found in production, which is what
`agent_tool_calls_total{tool_name}` is for.

### Hostile

**Q: Deterministic checks cannot tell whether a recommendation is good. Isn't
your benchmark measuring the easy thing?**
Yes, and deliberately. It measures contract adherence, not quality — quality is
what `eval_offline` measures, with actual relevance judgments from held-out
ratings. Splitting them is the design: one harness answers "did the agent do
what it was told," the other "were the results any good." A single
LLM-judged score would blur both into one unstable number.

**Q: You benchmark at temperature 0 but ship at 0.7. You are testing a
different system.**
Correct, and it is a real caveat. The alternative is worse: at 0.7 a
before/after delta is partly sampling noise, so I could not attribute changes
to my fixes. The reproducible option would be pinning production to 0 as well,
at the cost of more rigid conversation. Running each case n times and reporting
pass *rates* would capture stochastic behaviour properly — that is the upgrade
path, and it costs n× the API spend.

**Q: You repeat the director three times in the soup. Where did 3 come from?**
Hand-tuned, never validated. It is crude term weighting standing in for proper
per-field feature weights. Before the evaluation harness existed I had no way
to test it; now I do, and it is a legitimate thing to sweep. I would rather say
that than invent a justification.

**Q: Your resume said 0.74. It is now 0.078. What happened?**
The 0.74 came from a notebook that was gitignored and could not be reproduced
from the repository. It almost certainly measured an easier task — likely
rating prediction rather than top-N ranking. I rebuilt the evaluation with a
proper held-out split and a popularity baseline, and 0.078 is what the model
actually does. I updated the claim rather than the code.

*This is the single most valuable question you will get. Do not get defensive.
Volunteering it is stronger than being caught by it.*

**Q: What would you do differently starting over?**
Build the evaluation harness first. Every significant problem — the
unreproducible metric, the retrieval-bound hybrid, the double-scoring bug —
existed for months and was invisible because there was nothing measuring them.
The order should have been: baseline, then build, then measure again. I would
also not have hand-tuned soup weights before having a way to test them.

---

## Part 8 — Known weaknesses, own them first

Raise these yourself. Being the person who already knows their system's flaws
reads as far stronger than being the person who discovers them mid-interview.

| weakness | honest framing |
|---|---|
| Model loses to popularity | Kept and diagnosed rather than hidden; coverage tells the other half |
| `DEFAULT_USER_ID = 1` hardcoded | Personalisation is inert on the agent path; benchmark cases encode it; fix is identity threading |
| Conversation state in-process | Does not survive restart or scale past one instance; needs Redis |
| Soup weights hand-tuned | Never validated; now testable with the harness |
| Content similarity is O(n) per query | Cached and warmed; needs an ANN index at real scale |
| `ratings_small.csv` (~100k ratings) | Full MovieLens is 26M; results may not hold at scale |
| Compose port mismatch | Dockerfile serves 8080, compose maps 8000; `BACKEND_URL` missing `/api` |
| Behavioural benchmark not yet run | Harness exists, baseline not captured — do this before touching the agent |

---

## Study sequence

1. Read Parts 1–5 once, slowly.
2. Answer Part 7 out loud without looking. Note where you stall.
3. Reread only the sections behind the stalls.
4. Open the code and trace one full request end to end: frontend → agent →
   MCP → backend → model → back. Reading the path beats memorising it.
5. Re-answer the hostile questions. Those are the ones that decide the
   conversation.
