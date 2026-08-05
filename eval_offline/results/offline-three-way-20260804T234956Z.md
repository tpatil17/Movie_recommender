# Offline recommender evaluation (three-way)

- timestamp: 2026-08-04T23:49:55.881459+00:00
- users evaluated: 300
- relevance threshold: rating >= 4.0
- mean relevant-set size: 24.74
- CF candidate pool: 2848 movies (support >= 5)
- split: 70/30 per user, seed 42

| metric@10 | cf (recommend-for-you) | hybrid (movies-like-X) | popularity |
|---|---|---|---|
| precision | 0.0777 | 0.026 | 0.1537 |
| recall | 0.0378 | 0.011 | 0.0912 |
| ndcg | 0.0878 | 0.0386 | 0.1948 |
| coverage | 0.0051 | 0.0191 | 0.0009 |

CF loses to popularity on precision@10 (0.0777 vs 0.1537, delta -0.0760).