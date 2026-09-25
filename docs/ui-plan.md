# UI plan: making the agent legible

## Premise

This UI is a portfolio artifact first. Its job is not to be the best movie
recommender — the evaluation already shows a popularity baseline beats the
model. Its job is to make a reviewer conclude, within about three minutes:

> This person built an agent, instrumented it, measured it honestly, and can
> explain exactly what is wrong with it.

That reframes every design decision. Movie posters do not serve that goal.
Visible tool calls, a real identity model, and an evaluation page that admits
the model loses do.

Three things are currently invisible in the product, and all three are the
strongest parts of the engineering:

| Asset | Where it lives | Visible in UI today |
|---|---|---|
| Agent tool selection | `ChatResponse.tool_calls`, already returned | No — UI renders only `response` |
| Pure-CF recommendations | `GET /api/recommendations/for-you` | No — no surface at all |
| Honest offline evaluation | `eval_offline/results/*.json` | No |

---

## Sequencing warning: read before starting

Phase 0 changes the MCP tool signatures (adding `user_id`) and the agent
system prompt. Both directly affect which tools the agent picks.

The five-module plan calls for a **50-case behavioral baseline against the
current agent** (Module 3) with before/after measurement. If Phase 0 lands
first, that baseline is gone — there is nothing left to measure "before"
against, and the same reproducibility problem that produced the 0.74 repeats
itself in the agent layer.

**Capture the 50-case agent baseline before Phase 0.** If the behavioral
harness does not exist yet, that is the real next task, not this plan.

---

## Information architecture

Keep the existing tab shell. Four surfaces, plus one persistent global control.

```
┌─────────────────────────────────────────────────────────────┐
│  Movie Recommender          [Profile: User 42 ▾]            │
│  Hybrid + collaborative     Chat | For You | Discover | Eval │
└─────────────────────────────────────────────────────────────┘
```

**Profile switcher (global).** Replaces the bare `user_id` number input. A
dropdown of ~6 demo MovieLens users, each labelled by taste derived from
their top-rated genres — "User 42 · sci-fi, thriller" — plus a "New user"
option that triggers the cold-start path. Selecting a profile sets `user_id`
for chat, For You, and Discover simultaneously.

This is the important move: it converts the `DEFAULT_USER_ID` fix from
invisible plumbing into the demo itself. A reviewer switches profiles, sees
recommendations change, and personalization becomes a claim they verified
rather than one they read.

**Chat.** Existing conversation, plus an inline agent trace (below).

**For You.** Pure CF, no seed. Card grid with predicted rating and reason.
Explicit cold-start banner when `cold_start: true` — never dress up popular
titles as personalized.

**Discover.** The existing seed-anchored search. Renamed from "Quick Search"
because it now sits beside a second recommendation mode and the distinction
matters: "movies like X" versus "movies for you".

**Evaluation.** The differentiator. Details below.

---

## Surface detail

### Agent trace

Under each assistant message, a compact row of tool chips in call order:

```
  search_movies ✓   get_for_you ✓                        1.24s
```

Clicking a chip expands args, status, and truncated result. Failed calls
render red — `_tool_status` already catches the `{"error": ...}` payloads that
LangChain reports as successes, so failure states are real and worth showing.

Everything here comes from data the agent already returns. No backend work.

Design notes:
- Collapsed by default. The conversation stays readable; the trace is opt-in.
- Show `latency_ms` per turn. It is already in `ChatResponse`.
- A turn with zero tool calls should render nothing, not an empty container.

### Evaluation page

Static render of the committed results JSON. Three parts:

1. **The comparison**, as a bar chart plus table:

   | metric@10 | cf | hybrid | popularity |
   |---|---|---|---|
   | precision | 0.078 | 0.026 | **0.154** |
   | coverage | 0.005 | 0.019 | 0.001 |

   Popularity is bolded as the winner. Hiding that would defeat the purpose.

2. **The diagnosis**, in two sentences: the hybrid is retrieval-bound — mean
   candidate pool 9.81 against a nominal 25, and 80.7% of users get a pool
   containing nothing they rated highly, capping precision at 0.026 while the
   ranker achieved 0.025. Ranking was never the bottleneck.

3. **The honest read**, stated plainly: random selection scores ~0.0006, so CF
   carries real signal at 130× random; it just does not beat popularity on
   precision@10, and coverage is where personalization actually shows up
   (0.005 and 0.019 versus popularity's 0.001 — the baseline shows the same
   ten titles to everyone).

Source the numbers from a committed JSON file, not a live endpoint. These are
research artifacts, not application state.

---

## Phase 0 — backend prerequisites (blocking)

Nothing in the frontend can be built honestly until identity is real.

| Change | File | Note |
|---|---|---|
| `ChatRequest.user_id: int` | `services/agent/main.py` | Thread into agent invocation |
| Pass `user_id` to tools | `services/agent/agent/chain.py` | Inject into tool call context |
| `get_recommendations(user_id, ...)` | `services/mcp-server/tools/recommendations.py` | Delete `DEFAULT_USER_ID` |
| `GET /api/users/demo` | `services/backend/app/routes/` | Returns sample users + top-rated titles + dominant genres |
| Copy latest eval JSON | `frontend/src/data/eval-results.json` | Committed artifact |

`/api/users/demo` needs a real design decision: pick users with enough ratings
to be genuinely personalized (say 100+) and visibly different tastes, so the
profile switcher demonstrates something. Choosing six random user IDs would
produce six nearly identical result sets and quietly undercut the whole point.

**Verify Phase 0:** same chat message with two different `user_id` values
returns different recommendations. If it does not, identity is not threaded
and no amount of frontend work will fix it.

---

## Phase 1 — frontend restructure (no behavior change)

Split the 318-line `App.jsx`. A single-file React app is itself a code-review
smell for a portfolio, independent of what it renders.

```
src/
  api/client.js               fetch wrappers, base URLs, error normalization
  context/UserContext.jsx     selected user_id, shared by every surface
  data/eval-results.json      committed eval artifact
  components/
    layout/       Header, TabNav, ProfileSwitcher
    chat/         ChatTab, MessageBubble, ToolTrace, ToolCallChip
    forYou/       ForYouTab, MovieCard, ColdStartBanner
    discover/     DiscoverTab, MovieSearch
    evaluation/   EvaluationTab, MetricTable, MetricBar
```

`api/client.js` matters more than it looks: the fetch calls are currently
inline with ad-hoc error handling, and three surfaces are about to need the
same base-URL and failure logic.

**Verify Phase 1:** app behaves identically to today. Pure refactor, committed
separately, so the diff for later phases stays reviewable.

## Phase 2 — profile switcher

`UserContext` + header dropdown, consuming `/api/users/demo`. Wire into
Discover (replacing the number input) and into the chat request body.

**Verify:** switching profiles changes Discover results for the same seed.

## Phase 3 — agent trace

`ToolTrace` and `ToolCallChip`. Frontend-only.

**Verify:** ask "what should I watch" and confirm `get_for_you` appears in the
trace with the selected `user_id` in its args. This is also the fastest way to
confirm Phase 0 threading actually works end to end.

## Phase 4 — For You

`ForYouTab` against `/api/recommendations/for-you`, with cold-start banner.

**Verify:** the "New user" profile shows the cold-start state rather than
silently rendering popular titles as personalized.

## Phase 5 — evaluation page

`EvaluationTab` from the committed JSON.

**Verify:** rendered numbers match `offline-three-way-*.json` exactly. Any
drift here is the same reproducibility failure as the original 0.74.

## Phase 6 — deploy

Firebase build, CORS check for the agent origin (currently only ports 5173 and
3000 plus the two Firebase domains are allowed — the deployed agent origin may
need adding), README screenshots.

---

## Risks

**Agent baseline.** Covered above. The most likely way this plan damages the
project is by landing Phase 0 before Module 3's measurement.

**Cold start is a real product state, not an error.** If the frontend renders
`cold_start: true` results without saying so, the UI is lying in exactly the
way the evaluation work was meant to stop.

**Scope.** Phases 2–5 are each independently shippable. If time runs short,
Phase 3 (agent trace) and Phase 5 (evaluation) carry the most portfolio weight
per hour; Phase 4 is the smallest.

**Compose bugs.** `docker-compose.yml` maps `8000:8000` while the backend
Dockerfile serves `8080`, and sets `BACKEND_URL` without the `/api` prefix.
Anyone who tries to run this stack from compose — including a reviewer — hits
both immediately. Worth fixing before the UI invites people to run it.
