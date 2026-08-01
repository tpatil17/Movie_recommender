# Offline recommender evaluation

- timestamp: 2026-07-31T19:10:35.174212+00:00
- users evaluated: 300
- relevance threshold: rating >= 4.0
- mean relevant-set size: 24.74
- catalog size: 42277
- split: 70/30 per user, seed 42

| metric@10 | hybrid | popularity baseline |
|---|---|---|
| precision | 0.024 | 0.1 |
| recall | 0.0103 | 0.0692 |
| ndcg | 0.0326 | 0.1234 |
| catalog coverage | 0.0193 | 0.0003 |

Hybrid LOSES TO the popularity baseline on precision@10 (0.024 vs 0.1, delta -0.0760).