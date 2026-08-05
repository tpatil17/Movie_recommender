"""
Offline evaluation for the recommender, comparing three approaches on the
same users and the same held-out relevant sets:

  hybrid       - seed-anchored "movies like X": content neighbours of one
                 movie the user liked, CF-ranked. Answers a narrow question.
  cf           - pure "recommend for you": every candidate movie ranked by
                 the user's SVD predicted rating, no seed. The standard
                 top-N task SVD is built for.
  popularity   - recommend the most-rated titles to everyone. The baseline
                 any personalized method must beat to justify itself.

Protocol (unchanged, chosen to resist inflation):
  * Per-user train/test split, fixed seed. SVD fits on train only.
  * The hybrid seed is drawn from the user's TRAIN high ratings; the relevant
    set is the user's TEST high ratings. Query and answer key are disjoint.
  * Relevance means rated >= threshold (default 4.0), not merely rated.
  * All three methods exclude movies the user already rated in train, so none
    gets credit for recommending something already seen.

Imports the real models from the backend, so it measures current code.

Usage:
    python eval_offline.py                      # sampled users, all three methods
    python eval_offline.py --all-users
    python eval_offline.py --cf-min-support 10  # candidate pool support filter
    python eval_offline.py --tag three-way
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# --------------------------------------------------------------------------
# Pure metric functions (unit-tested, no model or data dependency)
# --------------------------------------------------------------------------

def dcg_at_k(hits: list[bool], k: int) -> float:
    total = 0.0
    for i, hit in enumerate(hits[:k]):
        if hit:
            total += 1.0 / math.log2(i + 2)
    return total


def ndcg_at_k(hits: list[bool], num_relevant: int, k: int) -> float:
    ideal_hits = [True] * min(num_relevant, k)
    idcg = dcg_at_k(ideal_hits, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(hits, k) / idcg


def score_recommendations(recommended: list[str], relevant: set[str], k: int) -> dict:
    top = recommended[:k]
    # A title only counts the first time it appears. A method that returns the
    # same relevant movie twice has wasted a slot, not earned a second hit --
    # without this, recall can exceed 1.0 and precision rewards duplication.
    credited = set()
    hits = []
    for title in top:
        is_hit = title in relevant and title not in credited
        if is_hit:
            credited.add(title)
        hits.append(is_hit)
    hit_count = sum(hits)
    return {
        "precision": hit_count / k if k else 0.0,
        "recall": hit_count / len(relevant) if relevant else 0.0,
        "ndcg": ndcg_at_k(hits, len(relevant), k),
        "hits": hit_count,
    }


def per_user_split(user_to_items, test_frac, rng):
    train, test = {}, {}
    for user, items in user_to_items.items():
        if len(items) < 2:
            train[user] = list(items)
            test[user] = []
            continue
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_frac)))
        test[user] = shuffled[:n_test]
        train[user] = shuffled[n_test:]
    return train, test


def head_sets(popular_ids, movielens_to_title, head_n):
    """
    The 'head' is the head_n most-popular movies (popular_ids is already sorted
    by descending train support). Returns (head_id_set, head_title_set). The
    long-tail evaluation removes these from candidate pools and relevant sets so
    a popularity-by-count method cannot lean on blockbusters.
    """
    head_ids = set(popular_ids[:head_n])
    head_titles = {movielens_to_title[m] for m in head_ids if m in movielens_to_title}
    return head_ids, head_titles


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

@dataclass
class MetricAccumulator:
    label: str
    precision: list = field(default_factory=list)
    recall: list = field(default_factory=list)
    ndcg: list = field(default_factory=list)
    recommended_titles: set = field(default_factory=set)

    def add(self, scored, recommended):
        self.precision.append(scored["precision"])
        self.recall.append(scored["recall"])
        self.ndcg.append(scored["ndcg"])
        self.recommended_titles.update(recommended)

    def summary(self, catalog_size):
        n = len(self.precision) or 1
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        return {
            "label": self.label,
            "users_evaluated": len(self.precision),
            "precision_at_k": round(mean(self.precision), 4),
            "recall_at_k": round(mean(self.recall), 4),
            "ndcg_at_k": round(mean(self.ndcg), 4),
            "catalog_coverage": round(len(self.recommended_titles) / catalog_size, 4) if catalog_size else 0.0,
        }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_evaluation(args):
    backend = Path(args.backend).resolve()
    if not (backend / "app").is_dir():
        raise SystemExit(f"--backend {backend} has no app/ directory")
    sys.path.insert(0, str(backend))

    from app.data.loader import load_clean_data
    from app.models.content_based import ContentBasedModel
    from app.models.collaborative import CollaborativeModel
    from app.models.hybrid import HybridRecommender

    print("Loading data ...", flush=True)
    data, ratings, tmdb_to_movielens = load_clean_data()

    title_by_tmdb = dict(zip(data["id"].astype(int), data["title"]))
    movielens_to_title = {}
    for tmdb_id, ml_id in tmdb_to_movielens.items():
        t = title_by_tmdb.get(int(tmdb_id))
        if t is not None:
            movielens_to_title.setdefault(int(ml_id), t)

    reachable_titles = set(data["title"])
    catalog_size = len(reachable_titles)

    user_to_items = defaultdict(list)
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

    print("Building content model ...", flush=True)
    content = ContentBasedModel(data)
    hybrid = HybridRecommender(content, collab, tmdb_to_movielens)

    # CF candidate pool: movies with enough train support, mapped to a title.
    # Support computed from train only to avoid leakage.
    support = defaultdict(int)
    for u, items in train.items():
        for (m, _r) in items:
            support[m] += 1
    cf_candidates = [m for m, c in support.items()
                     if c >= args.cf_min_support and m in movielens_to_title]
    print(f"CF candidate pool: {len(cf_candidates)} movies (support >= {args.cf_min_support})", flush=True)

    # Popularity order (train-derived), as (movielens_id, title) so we can
    # apply the same per-user seen-exclusion the other methods get.
    popular_ids = [m for m, _ in sorted(support.items(), key=lambda x: x[1], reverse=True)
                   if m in movielens_to_title]

    # Long-tail split: remove the head_n most-popular movies from the CF pool,
    # the popularity list, and (per user) the relevant set.
    head_ids, head_titles = head_sets(popular_ids, movielens_to_title, args.head_n)
    cf_candidates_tail = [m for m in cf_candidates if m not in head_ids]
    popular_ids_tail = [m for m in popular_ids if m not in head_ids]
    print(f"Long tail: removing {len(head_titles)} head titles; "
          f"CF tail pool {len(cf_candidates_tail)} movies", flush=True)

    threshold = args.rating_threshold

    # Per-user seen set (all train-rated movie ids) as titles, for exclusion.
    train_seen_titles = {
        u: {movielens_to_title[m] for (m, _r) in items if m in movielens_to_title}
        for u, items in train.items()
    }
    train_seen_ids = {u: {m for (m, _r) in items} for u, items in train.items()}

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
            evaluable.append((user, seed_title, relevant))

    if not evaluable:
        raise SystemExit("No evaluable users after filtering.")

    if not args.all_users and len(evaluable) > args.sample_users:
        rng.shuffle(evaluable)
        evaluable = evaluable[: args.sample_users]

    print(f"Evaluating {len(evaluable)} users (threshold >= {threshold}, K = {args.top_n}) ...", flush=True)

    hy = MetricAccumulator("hybrid")
    cf = MetricAccumulator("cf")
    pop = MetricAccumulator("popularity")
    hy_t = MetricAccumulator("hybrid_tail")
    cf_t = MetricAccumulator("cf_tail")
    pop_t = MetricAccumulator("popularity_tail")
    rel_sizes = []
    tail_rel_sizes = []

    for i, (user, seed_title, relevant) in enumerate(evaluable):
        rel_sizes.append(len(relevant))
        seen_titles = train_seen_titles.get(user, set())
        seen_ids = train_seen_ids.get(user, set())
        k = args.top_n

        # Pull a generous ranked list from the hybrid once, then slice for both
        # the full and tail evaluations.
        hy_ranked = [r["title"] for r in
                     hybrid.recommend(user, seed_title, top_n=k + len(seen_titles) + args.head_n + 5)
                     if r["title"] not in seen_titles]

        # ---- full evaluation ----
        hy_titles = hy_ranked[:k]
        hy.add(score_recommendations(hy_titles, relevant, k), hy_titles)

        cf_pairs = collab.recommend_for_user(user, cf_candidates, exclude_ids=seen_ids, top_n=k)
        cf_titles = [movielens_to_title[m] for (m, _s) in cf_pairs]
        cf.add(score_recommendations(cf_titles, relevant, k), cf_titles)

        pop_titles = [movielens_to_title[m] for m in popular_ids if m not in seen_ids][:k]
        pop.add(score_recommendations(pop_titles, relevant, k), pop_titles)

        # ---- long-tail evaluation (head removed from pools and relevant set) ----
        relevant_tail = relevant - head_titles
        if not relevant_tail:
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(evaluable)}", flush=True)
            continue
        tail_rel_sizes.append(len(relevant_tail))

        hy_titles_tail = [t for t in hy_ranked if t not in head_titles][:k]
        hy_t.add(score_recommendations(hy_titles_tail, relevant_tail, k), hy_titles_tail)

        cf_pairs_tail = collab.recommend_for_user(user, cf_candidates_tail, exclude_ids=seen_ids, top_n=k)
        cf_titles_tail = [movielens_to_title[m] for (m, _s) in cf_pairs_tail]
        cf_t.add(score_recommendations(cf_titles_tail, relevant_tail, k), cf_titles_tail)

        pop_titles_tail = [movielens_to_title[m] for m in popular_ids_tail if m not in seen_ids][:k]
        pop_t.add(score_recommendations(pop_titles_tail, relevant_tail, k), pop_titles_tail)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(evaluable)}", flush=True)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": {
            "top_n": args.top_n,
            "rating_threshold": threshold,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "cf_min_support": args.cf_min_support,
            "cf_pool_size": len(cf_candidates),
            "head_n": args.head_n,
            "head_titles_removed": len(head_titles),
            "cf_tail_pool_size": len(cf_candidates_tail),
            "users_evaluated": len(evaluable),
            "tail_users_evaluated": len(tail_rel_sizes),
            "mean_relevant_set_size": round(sum(rel_sizes) / len(rel_sizes), 2),
            "mean_tail_relevant_set_size": round(sum(tail_rel_sizes) / len(tail_rel_sizes), 2) if tail_rel_sizes else 0.0,
            "catalog_size": catalog_size,
        },
        "full": {
            "cf": cf.summary(catalog_size),
            "hybrid": hy.summary(catalog_size),
            "popularity": pop.summary(catalog_size),
        },
        "tail": {
            "cf": cf_t.summary(catalog_size),
            "hybrid": hy_t.summary(catalog_size),
            "popularity": pop_t.summary(catalog_size),
        },
    }


def _table(block, k):
    c, h, b = block["cf"], block["hybrid"], block["popularity"]
    return [
        f"| metric@{k} | cf (recommend-for-you) | hybrid (movies-like-X) | popularity |",
        "|---|---|---|---|",
        f"| precision | {c['precision_at_k']} | {h['precision_at_k']} | {b['precision_at_k']} |",
        f"| recall | {c['recall_at_k']} | {h['recall_at_k']} | {b['recall_at_k']} |",
        f"| ndcg | {c['ndcg_at_k']} | {h['ndcg_at_k']} | {b['ndcg_at_k']} |",
        f"| coverage | {c['catalog_coverage']} | {h['catalog_coverage']} | {b['catalog_coverage']} |",
    ]


def _verdict(block, k, label):
    c, b = block["cf"], block["popularity"]
    d = c["precision_at_k"] - b["precision_at_k"]
    v = "beats" if d > 0 else ("ties" if d == 0 else "loses to")
    return f"{label}: CF {v} popularity on precision@{k} ({c['precision_at_k']} vs {b['precision_at_k']}, delta {d:+.4f})."


def format_report(report):
    p = report["params"]
    k = p["top_n"]
    lines = [
        "# Offline recommender evaluation (full catalog vs long tail)",
        "",
        f"- timestamp: {report['timestamp']}",
        f"- users evaluated: {p['users_evaluated']} (full), {p['tail_users_evaluated']} (tail)",
        f"- relevance threshold: rating >= {p['rating_threshold']}",
        f"- mean relevant-set size: {p['mean_relevant_set_size']} (full), {p['mean_tail_relevant_set_size']} (tail)",
        f"- CF pool: {p['cf_pool_size']} movies full, {p['cf_tail_pool_size']} tail (support >= {p['cf_min_support']})",
        f"- head removed for tail: {p['head_titles_removed']} most-popular titles (head_n = {p['head_n']})",
        f"- split: {int((1 - p['test_frac']) * 100)}/{int(p['test_frac'] * 100)} per user, seed {p['seed']}",
        "",
        "## Full catalog",
        "",
        *_table(report["full"], k),
        "",
        "## Long tail (blockbusters removed)",
        "",
        *_table(report["tail"], k),
        "",
        _verdict(report["full"], k, "Full"),
        _verdict(report["tail"], k, "Tail"),
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=str(Path(__file__).parent.parent / "services" / "backend"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-users", type=int, default=300)
    parser.add_argument("--all-users", action="store_true")
    parser.add_argument("--cf-min-support", type=int, default=5)
    parser.add_argument("--head-n", type=int, default=200,
                        help="number of most-popular titles removed for the long-tail evaluation")
    parser.add_argument("--tag", default="three-way")
    args = parser.parse_args()

    report = run_evaluation(args)
    markdown = format_report(report)
    print()
    print(markdown)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"offline-{args.tag}-{stamp}.md").write_text(markdown)
    (out_dir / f"offline-{args.tag}-{stamp}.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote results/offline-{args.tag}-{stamp}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())