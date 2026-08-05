# Offline recommender evaluation

- timestamp: 2026-08-04T23:03:56.153935+00:00
- users evaluated: 300
- relevance threshold: rating >= 4.0
- mean relevant-set size: 24.74
- catalog size: 42277
- split: 70/30 per user, seed 42

| metric@10 | hybrid | popularity baseline |
|---|---|---|
| precision | 0.0253 | 0.1 |
| recall | 0.0106 | 0.0692 |
| ndcg | 0.0326 | 0.1234 |
| catalog coverage | 0.0192 | 0.0003 |

Hybrid LOSES TO the popularity baseline on precision@10 (0.0253 vs 0.1, delta -0.0747).