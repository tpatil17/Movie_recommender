# Offline recommender evaluation

Measures ranking quality of the hybrid recommender on held-out ratings. This
tests the **model**, not the agent.

## What it does

Given a movie a user liked, does the recommender surface other movies that same
user rated highly? Scored with Precision@10, Recall@10, NDCG@10, and catalog
coverage, against a popularity baseline on the same users.

## Why the protocol resists inflation

- SVD is trained on the train split only; test ratings are never seen in fit.
- The seed movie comes from the user's train ratings, the relevant set from
  their test ratings. Query and answer key are disjoint.
- Relevance means rated >= 4.0, not merely rated.
- A title only counts the first time it appears in a result list, so a model
  cannot inflate recall by returning the same relevant movie twice.
- The popularity baseline is the honesty check. If the hybrid cannot beat
  "recommend the most-rated titles to everyone," that is the real result.

## Running

Requires the backend's Kaggle CSVs at `services/backend/data/raw/`. No agent, no
MCP server, no OpenAI key needed. Needs a venv with `surprise`, `sklearn`, and
`pandas`, since it imports `app.models.*` and `app.data.loader`.

```bash
cd eval_offline
source ../.venv/bin/activate
python eval_offline.py --tag hybrid-fixed
```

Sampled at 300 users by default for runtime. `--all-users` for the full set.
Results are written to `eval_offline/results/` as paired `.md` and `.json`.

### Getting the before/after

The double-scoring bug in `hybrid.py` is already fixed, so a plain run measures
the fixed model. To produce the honest baseline for comparison, stash the fix:

```bash
git stash push ../services/backend/app/models/hybrid.py
python eval_offline.py --tag baseline        # buggy double-scoring
git stash pop
python eval_offline.py --tag hybrid-fixed    # single scoring loop
```

## Tests

The scoring math is unit-tested on synthetic data — no CSVs, no models, no
backend import. Run these before trusting any number the harness prints:

```bash
cd eval_offline
python -m pytest test_eval_offline.py -v
```

## Reading the result

The number to record is precision@10 for the hybrid, and its delta over the
popularity baseline. A hybrid that only ties popularity is not adding value from
the collaborative signal, which given the hardcoded `DEFAULT_USER_ID` on the
agent path is a plausible outcome worth knowing.
