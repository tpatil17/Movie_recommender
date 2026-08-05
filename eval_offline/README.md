# Offline recommender evaluation

Measures ranking quality on held-out ratings. This tests the **models**, not
the agent.

Three methods are scored on the same users and the same relevant sets:

| method | question it answers |
|---|---|
| `hybrid` | "movies like X" — content neighbours of one seed movie, CF-ranked |
| `cf` | "recommend for you" — every candidate ranked by the user's predicted rating, no seed |
| `popularity` | recommend the most-rated titles to everyone — the baseline any personalized method must beat |

## Protocol

- Per-user train/test split, fixed seed. SVD fits on train only, so no test
  rating is ever seen during training.
- The hybrid seed comes from the user's train high-ratings; the relevant set is
  their test high-ratings. Query and answer key are disjoint.
- Relevance means rated >= 4.0, not merely rated.
- All three methods exclude movies the user already rated in train, so none
  gets credit for recommending something already seen.
- A title only counts the first time it appears in a result list, so no method
  can inflate recall by returning the same movie twice.

## Running

Requires the backend's Kaggle CSVs at `services/backend/data/raw/`. No agent,
no MCP server, no OpenAI key. Needs a venv with `surprise`, `sklearn`, and
`pandas`, since it imports `app.models.*` and `app.data.loader`.

```bash
cd eval_offline
source ../.venv/bin/activate

python eval_offline.py --tag three-way      # all three methods
python eval_offline.py --all-users          # full evaluable set
python eval_offline.py --cf-min-support 10  # stricter CF candidate pool
python eval_offline.py --head-n 200         # also report long-tail results
```

Results land in `results/` as paired `.md` and `.json`, tagged and timestamped.

`diagnose_cf_coverage.py` reports how much of the catalog the CF pool actually
covers at a given support threshold.

## Tests

The scoring math is unit-tested on synthetic data — no CSVs, no models, no
backend import. Run these before trusting any number the harness prints:

```bash
python -m pytest test_eval_offline.py -v
```

## Findings so far

Full catalog, 300 users, K = 10:

| metric@10 | cf | hybrid | popularity |
|---|---|---|---|
| precision | 0.078 | 0.026 | 0.154 |
| recall | 0.038 | 0.011 | 0.091 |
| ndcg | 0.088 | 0.039 | 0.195 |
| coverage | 0.005 | 0.019 | 0.001 |

Two things this establishes:

**The seed-anchored hybrid is retrieval-bound.** Diagnostics showed a mean
candidate pool of 9.81 movies against a nominal 25, and 80.7% of users got a
pool containing nothing they had rated highly. Precision@10 was capped at
0.0263 by the pool itself while the hybrid achieved 0.025 — the ranker was
already extracting 95% of what was available. The fix had to be candidate
generation, not scoring. Pure CF over a broad pool triples it.

**Popularity still wins.** On the full catalog and on the long tail with the
200 most-popular titles removed. This is the honest headline: on MovieLens,
what users rate is overwhelmingly what is popular, so a non-personalized
baseline is hard to beat on precision@10. For context, random selection from
the 42k catalog scores about 0.0006 — CF is roughly 130x random. The signal is
real; it just does not beat popularity on this metric.

Coverage is the counter-argument worth reporting alongside precision:
popularity recommends the same ~10 titles to everyone (coverage 0.001), while
CF and the hybrid actually differentiate between users.
