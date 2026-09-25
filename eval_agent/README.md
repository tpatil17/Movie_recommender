# Agent behavioural benchmark

52 cases that test what the agent **does**, not what it says. Companion to
`eval_offline/`, which tests the models. This tests the agent on top of them.

## Why no LLM judge

Every check here is a deterministic assertion over the agent's tool calls and
response text. No second model grades the first.

Scoring an agent with an LLM judge puts two stochastic systems in series: when
the number moves you cannot tell whether the agent changed, the judge changed,
or neither did. For a benchmark whose entire purpose is a trustworthy
before/after delta, that is disqualifying. `tool_order(search_movies,
get_recommendations)` either held or it did not, and the failure message names
the exact violation.

The tradeoff is real and worth stating: deterministic checks cannot assess
whether a recommendation was *good*. They assess whether the agent followed
its own contract. Recommendation quality is `eval_offline`'s job.

## Categories

| category | cases | what it catches |
|---|---|---|
| `seed_movie` | 12 | Recommending without verifying the title first |
| `open_ended` | 10 | Reaching for a seed-anchored tool when no film was named |
| `similarity` | 6 | Confusing "exactly like X" with personalized recommendation |
| `graceful_failure` | 10 | Inventing titles, leaking stack traces, prompt injection |
| `memory` | 6 | Session history dropped between turns |
| `arg_propagation` | 8 | `top_n` ignored, `user_id` not threaded |

`arg_propagation` and `open_ended` are expected to fail on the current agent.
That is deliberate — they encode the known `DEFAULT_USER_ID` and tool-routing
bugs so the fix has something to prove itself against.

## Running

Needs all three services up, plus an OpenAI key. One full run is ~52 cases with
some multi-turn, so budget roughly 70 agent turns of API spend.

```bash
# terminal 1
cd services/backend && uvicorn app.main:app --port 8000

# terminal 2
cd services/mcp-server && python main.py

# terminal 3 -- temperature 0 matters, see below
cd services/agent && AGENT_TEMPERATURE=0 AGENT_TAG=v2-for-you python main.py

# terminal 4
cd eval_agent
python runner.py --tag baseline
```

Useful flags:

```bash
python runner.py --category open_ended --verbose   # one category, show failures
python runner.py --case arg_05 --case arg_06       # specific cases
python runner.py --compare 'results/agent-baseline-*.json' 'results/agent-after-*.json'
```

Results land in `results/` as paired `.md` and `.json`. The runner exits
non-zero if any case fails, so it can gate a pipeline.

### Temperature

The agent defaults to `temperature=0.7` so conversation does not feel canned.
A behavioural benchmark at 0.7 measures sampling noise as much as behaviour, so
set `AGENT_TEMPERATURE=0` for any run you intend to compare. The runner reads
`/meta`, records what it found in the report, and warns on stderr if it sees
sampling.

`AGENT_TAG` is recorded the same way. Bump it whenever a change is expected to
alter behaviour, so a report can never be silently attributed to the wrong
prompt or tool set.

## The before/after sequence

The point of this harness is a defensible delta, which requires capturing the
baseline **before** touching the agent:

```bash
python runner.py --tag baseline           # current agent, bugs included
# ... apply user_id threading, prompt fixes, bump AGENT_TAG ...
python runner.py --tag after-fix
python runner.py --compare 'results/agent-baseline-*.json' 'results/agent-after-fix-*.json'
```

The comparison reports per-category deltas plus case-level movement — which
cases were fixed and which regressed. A flat overall pass rate can hide fixes
offset by regressions, and that distinction is the whole point.

## Tests

The check functions are unit-tested on synthetic payloads — no agent, no
network, no API key:

```bash
python -m pytest test_checks.py -v
```

These run in CI. A broken check silently invalidates every benchmark number,
the same way a broken metric invalidated the original precision figure.

## Files

| file | role |
|---|---|
| `checks.py` | Pure assertion functions. All logic lives here. |
| `cases.py` | The 52 cases, as data. |
| `runner.py` | Drives the agent over HTTP, applies checks, writes reports. |
| `test_checks.py` | Unit tests for `checks.py`. |
| `results/` | Tagged, timestamped run artifacts. |

Adding a case means appending to `cases.py`. Adding a new kind of assertion
means a function in `checks.py`, an entry in `CHECKS`, and a test.
