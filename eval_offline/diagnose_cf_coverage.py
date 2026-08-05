"""
Diagnostic: how often does the collaborative signal actually reach a candidate?

The offline eval showed the hybrid losing to a popularity baseline by ~4x.
The hypothesis is that most content candidates never receive a real
collaborative prediction, so the hybrid degenerates into ranking by TMDB
vote average. This measures that directly, on the same population and split
the evaluation used.

Three buckets per scored candidate:
  unmapped         - tmdb_id not in tmdb_to_movielens; hits the hardcoded
                     neutral floor, CF never consulted.
  mapped_fallback  - in the map, but SVD never saw this item in training, so
                     .predict returns the global mean with was_impossible=True.
  mapped_real      - in the map AND SVD made a genuine personalized prediction.

Only mapped_real candidates carry any personalization. If that bucket is
small, the collaborative half of the "hybrid" is mostly inert.

Run from eval_offline/ with the backend venv active:
    python diagnose_cf_coverage.py
    python diagnose_cf_coverage.py --sample-users 300 --seed 42
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

from eval_offline import per_user_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=str(Path(__file__).parent.parent / "services" / "backend"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--sample-users", type=int, default=300)
    parser.add_argument("--candidates", type=int, default=25, help="content candidates per seed, matching hybrid.py")
    parser.add_argument("--probe-seeds", type=int, default=5, help="how many popular seeds to run the personalization probe on")
    parser.add_argument("--probe-users", type=int, default=50)
    args = parser.parse_args()

    backend = Path(args.backend).resolve()
    if not (backend / "app").is_dir():
        raise SystemExit(f"--backend {backend} has no app/ directory")
    sys.path.insert(0, str(backend))

    from app.data.loader import load_clean_data
    from app.models.content_based import ContentBasedModel
    from app.models.collaborative import CollaborativeModel

    print("Loading data ...", flush=True)
    data, ratings, tmdb_to_movielens = load_clean_data()

    title_by_tmdb = dict(zip(data["id"].astype(int), data["title"]))
    movielens_to_title: dict[int, str] = {}
    for tmdb_id, ml_id in tmdb_to_movielens.items():
        t = title_by_tmdb.get(int(tmdb_id))
        if t is not None:
            movielens_to_title.setdefault(int(ml_id), t)

    user_to_items: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in ratings.itertuples(index=False):
        ml_id = int(row.movieId)
        if ml_id in movielens_to_title:
            user_to_items[int(row.userId)].append((ml_id, float(row.rating)))

    rng = random.Random(args.seed)
    train, test = per_user_split(user_to_items, args.test_frac, rng)

    import pandas as pd
    train_df = pd.DataFrame(
        [{"userId": u, "movieId": m, "rating": r}
         for u, items in train.items() for (m, r) in items]
    )
    print(f"Training SVD on {len(train_df)} held-in ratings ...", flush=True)
    collab = CollaborativeModel()
    collab.train(train_df)
    svd = collab.svd

    print("Building content model ...", flush=True)
    content = ContentBasedModel(data)

    threshold = args.rating_threshold
    # Same evaluable population as the eval, up to sampling.
    evaluable = []
    for user in test:
        relevant = {movielens_to_title[m] for (m, r) in test[user]
                    if r >= threshold and m in movielens_to_title}
        if not relevant:
            continue
        train_liked = sorted(
            [(m, r) for (m, r) in train.get(user, []) if r >= threshold and m in movielens_to_title],
            key=lambda x: (-x[1], x[0]),
        )
        if not train_liked:
            continue
        seed_title = movielens_to_title[train_liked[0][0]]
        relevant.discard(seed_title)
        if relevant:
            evaluable.append((user, seed_title))

    rng.shuffle(evaluable)
    evaluable = evaluable[: args.sample_users]
    print(f"Measuring CF coverage over {len(evaluable)} users x {args.candidates} candidates ...\n", flush=True)

    unmapped = mapped_fallback = mapped_real = total = 0
    pred_values = []

    for user, seed_title in evaluable:
        candidates = content.get_similar_movies(seed_title, top_n=args.candidates)
        for movie in candidates:
            total += 1
            tmdb_id = movie["tmdb_id"]
            if tmdb_id not in tmdb_to_movielens:
                unmapped += 1
                continue
            ml_id = tmdb_to_movielens[tmdb_id]
            pred = svd.predict(user, ml_id)
            if pred.details.get("was_impossible", False):
                mapped_fallback += 1
            else:
                mapped_real += 1
                pred_values.append(pred.est)

    def pct(n):
        return f"{100 * n / total:.1f}%" if total else "n/a"

    # Personalization probe: does the same seed produce different rankings
    # for different users? Uses the actual hybrid so the answer reflects
    # exactly what users receive.
    from app.models.hybrid import HybridRecommender
    hybrid = HybridRecommender(content, collab, tmdb_to_movielens)

    pop = defaultdict(int)
    for u, items in train.items():
        for (m, _r) in items:
            pop[m] += 1
    popular_seeds = [movielens_to_title[m] for m, _ in
                     sorted(pop.items(), key=lambda x: x[1], reverse=True)
                     if m in movielens_to_title][: args.probe_seeds]
    probe_user_ids = [u for u, _ in evaluable][: args.probe_users]

    probe_lines = []
    for seed_title in popular_seeds:
        rankings = set()
        for u in probe_user_ids:
            recs = hybrid.recommend(u, seed_title, top_n=10)
            rankings.add(tuple(r["title"] for r in recs))
        distinct = len(rankings)
        probe_lines.append(
            f"  seed {seed_title!r}: {distinct} distinct top-10 lists across {len(probe_user_ids)} users"
        )

    spread = ""
    if pred_values:
        pred_values.sort()
        lo, hi = pred_values[0], pred_values[-1]
        mean = sum(pred_values) / len(pred_values)
        spread = f"{mean:.2f} mean, range {lo:.2f}-{hi:.2f}"

    print("=" * 60)
    print("CF COVERAGE")
    print("=" * 60)
    print(f"total candidates scored : {total}")
    print(f"unmapped (3.0 floor)    : {unmapped:>7}  {pct(unmapped)}")
    print(f"mapped, fallback pred   : {mapped_fallback:>7}  {pct(mapped_fallback)}")
    print(f"mapped, REAL CF signal  : {mapped_real:>7}  {pct(mapped_real)}")
    if spread:
        print(f"real-pred distribution  : {spread}")
    print()
    print("PERSONALIZATION PROBE (same seed, many users)")
    print("\n".join(probe_lines))
    print()
    print("Reading: personalization only exists via the 'REAL CF signal' bucket.")
    print("If that bucket is small and the probe shows ~1 distinct list per seed,")
    print("the hybrid is effectively ranking everyone identically by quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())