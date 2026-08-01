"""
Offline evaluation for the hybrid recommender.

Measures ranking quality on held-out ratings, the way the recommender is
actually used: given a movie a user liked, does it surface other movies that
same user rated highly?

Protocol, chosen to resist the inflation that produces implausible numbers:

  * Per-user split. Each user's ratings are divided into train and test with
    a fixed seed. SVD is fit on train only, so no test rating is ever seen
    during training.
  * The seed movie (the query) is drawn from the user's TRAIN high ratings.
    The relevant set (the answer key) is the user's TEST high ratings. Query
    and answer key are disjoint, so there is no seed-to-target leakage.
  * Relevance means rated >= threshold (default 4.0), not merely rated. We
    measure whether recommendations are liked, not whether they were watched.
  * A popularity baseline is scored on the same users and same relevant sets.
    If the hybrid cannot beat "recommend the most-rated titles to everyone,"
    that is the headline finding.

This imports the real HybridRecommender from the backend, so it measures
whatever is currently in hybrid.py. Run it, note the number, fix the bug,
run it again.

Usage:
    python eval_offline.py                         # sampled users, default params
    python eval_offline.py --all-users             # every evaluable user
    python eval_offline.py --sample-users 500 --seed 7
    python eval_offline.py --rating-threshold 4.0 --top-n 10
    python eval_offline.py --backend ../services/backend
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
# Pure metric functions (unit-tested separately, no model or data dependency)
# --------------------------------------------------------------------------

def dcg_at_k(hits: list[bool], k: int) -> float:
    """Binary-relevance DCG. hits[i] True if the item at rank i is relevant."""
    total = 0.0
    for i, hit in enumerate(hits[:k]):
        if hit:
            total += 1.0 / math.log2(i + 2)  # rank i is position i+1
    return total


def ndcg_at_k(hits: list[bool], num_relevant: int, k: int) -> float:
    """NDCG@k with binary relevance. IDCG places min(num_relevant, k) hits up top."""
    ideal_hits = [True] * min(num_relevant, k)
    idcg = dcg_at_k(ideal_hits, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(hits, k) / idcg


def score_recommendations(recommended: list[str], relevant: set[str], k: int) -> dict:
    """
    recommended: ordered list of recommended titles (best first).
    relevant: set of titles the user rated highly in the held-out set.
    Returns precision@k, recall@k, ndcg@k, and the raw hit count.
    """
    top = recommended[:k]
    # A title only counts the first time it appears. A model that returns the
    # same relevant movie twice has wasted a slot, not earned a second hit —
    # without this, recall can exceed 1.0 and precision rewards duplication.
    credited: set[str] = set()
    hits = []
    for title in top:
        is_hit = title in relevant and title not in credited
        if is_hit:
            credited.add(title)
        hits.append(is_hit)
    hit_count = sum(hits)
    precision = hit_count / k if k else 0.0
    recall = hit_count / len(relevant) if relevant else 0.0
    ndcg = ndcg_at_k(hits, len(relevant), k)
    return {
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg,
        "hits": hit_count,
    }


def per_user_split(
    user_to_items: dict[int, list[tuple[int, float]]],
    test_frac: float,
    rng: random.Random,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[tuple[int, float]]]]:
    """
    Split each user's (movie_id, rating) list into train and test.
    Users with fewer than 2 ratings go entirely to train (nothing to hold out).
    """
    train: dict[int, list] = {}
    test: dict[int, list] = {}
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


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

@dataclass
class MetricAccumulator:
    label: str
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    ndcg: list[float] = field(default_factory=list)
    recommended_titles: set[str] = field(default_factory=set)

    def add(self, scored: dict, recommended: list[str]) -> None:
        self.precision.append(scored["precision"])
        self.recall.append(scored["recall"])
        self.ndcg.append(scored["ndcg"])
        self.recommended_titles.update(recommended)

    def summary(self, catalog_size: int) -> dict:
        n = len(self.precision)
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        return {
            "label": self.label,
            "users_evaluated": n,
            "precision_at_k": round(mean(self.precision), 4),
            "recall_at_k": round(mean(self.recall), 4),
            "ndcg_at_k": round(mean(self.ndcg), 4),
            "catalog_coverage": round(len(self.recommended_titles) / catalog_size, 4) if catalog_size else 0.0,
        }


# --------------------------------------------------------------------------
# Main evaluation (wires the real models)
# --------------------------------------------------------------------------

def prepare_evaluation(args) -> dict:
    """
    Loads data, splits it, trains the models, and assembles the evaluable user
    list. Shared by the scoring run in this file and by diagnose.py, so both
    measure the identical population under the identical split.

    Returns a context dict with the models, the evaluable users, and the
    lookups the callers need.
    """
    backend = Path(args.backend).resolve()
    if not (backend / "app").is_dir():
        raise SystemExit(f"--backend {backend} does not contain an app/ directory")
    sys.path.insert(0, str(backend))

    from app.data.loader import load_clean_data
    from app.models.content_based import ContentBasedModel
    from app.models.collaborative import CollaborativeModel
    from app.models.hybrid import HybridRecommender

    print("Loading data ...", flush=True)
    data, ratings, tmdb_to_movielens = load_clean_data()

    # movielens_id -> catalog title, built from the same frame the recommender
    # draws from, so recommended titles and relevant titles are directly comparable.
    movielens_to_title: dict[int, str] = {}
    title_by_tmdb = dict(zip(data["id"].astype(int), data["title"]))
    for tmdb_id, ml_id in tmdb_to_movielens.items():
        title = title_by_tmdb.get(int(tmdb_id))
        if title is not None:
            movielens_to_title.setdefault(int(ml_id), title)

    reachable_titles = set(data["title"])
    catalog_size = len(reachable_titles)

    # Group ratings by user, keeping only movies that map to a reachable title.
    user_to_items: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in ratings.itertuples(index=False):
        ml_id = int(row.movieId)
        if ml_id in movielens_to_title:
            user_to_items[int(row.userId)].append((ml_id, float(row.rating)))

    rng = random.Random(args.seed)
    train, test = per_user_split(user_to_items, args.test_frac, rng)

    # Train SVD on the train split only. CollaborativeModel.train wants a
    # DataFrame with userId, movieId, rating.
    import pandas as pd
    train_rows = [
        {"userId": u, "movieId": m, "rating": r}
        for u, items in train.items()
        for (m, r) in items
    ]
    train_df = pd.DataFrame(train_rows)
    print(f"Training SVD on {len(train_df)} held-in ratings ...", flush=True)
    collab = CollaborativeModel()
    collab.train(train_df)

    print("Building content model ...", flush=True)
    content = ContentBasedModel(data)
    hybrid = HybridRecommender(content, collab, tmdb_to_movielens)

    # Popularity baseline: most-rated titles overall, computed from train only.
    pop_counts: dict[str, int] = defaultdict(int)
    for u, items in train.items():
        for (m, _r) in items:
            pop_counts[movielens_to_title[m]] += 1
    popular_titles = [t for t, _ in sorted(pop_counts.items(), key=lambda x: x[1], reverse=True)]

    threshold = args.rating_threshold

    # Assemble the evaluable user list: needs a train-side seed and a
    # non-empty test-side relevant set, both restricted to reachable titles.
    evaluable = []
    for user in test:
        relevant = {
            movielens_to_title[m]
            for (m, r) in test[user]
            if r >= threshold and m in movielens_to_title
        }
        if not relevant:
            continue
        train_liked = sorted(
            [(m, r) for (m, r) in train.get(user, []) if r >= threshold and m in movielens_to_title],
            key=lambda x: (-x[1], x[0]),
        )
        if not train_liked:
            continue
        seed_title = movielens_to_title[train_liked[0][0]]
        # A seed whose title equals a relevant title would trivially help; the
        # content model already excludes the seed itself from its output, but
        # guard against the degenerate case where they coincide.
        relevant.discard(seed_title)
        if not relevant:
            continue
        evaluable.append((user, seed_title, relevant))

    if not evaluable:
        raise SystemExit("No evaluable users after filtering. Check thresholds and data.")

    if not args.all_users and len(evaluable) > args.sample_users:
        rng.shuffle(evaluable)
        evaluable = evaluable[: args.sample_users]

    return {
        "hybrid": hybrid,
        "content": content,
        "collab": collab,
        "evaluable": evaluable,
        "popular_titles": popular_titles,
        "catalog_size": catalog_size,
        "tmdb_to_movielens": tmdb_to_movielens,
        "threshold": threshold,
    }


def run_evaluation(args) -> dict:
    ctx = prepare_evaluation(args)
    hybrid = ctx["hybrid"]
    evaluable = ctx["evaluable"]
    popular_titles = ctx["popular_titles"]
    catalog_size = ctx["catalog_size"]
    threshold = ctx["threshold"]

    print(f"Evaluating {len(evaluable)} users (threshold >= {threshold}, K = {args.top_n}) ...", flush=True)

    hybrid_acc = MetricAccumulator("hybrid")
    pop_acc = MetricAccumulator("popularity")
    rel_sizes = []

    for i, (user, seed_title, relevant) in enumerate(evaluable):
        rel_sizes.append(len(relevant))

        recs = hybrid.recommend(user, seed_title, top_n=args.top_n)
        rec_titles = [r["title"] for r in recs]
        hybrid_acc.add(score_recommendations(rec_titles, relevant, args.top_n), rec_titles)

        # Popularity baseline: same K, exclude the seed so both methods are
        # judged on titles other than the query.
        pop_recs = [t for t in popular_titles if t != seed_title][: args.top_n]
        pop_acc.add(score_recommendations(pop_recs, relevant, args.top_n), pop_recs)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(evaluable)}", flush=True)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": {
            "top_n": args.top_n,
            "rating_threshold": threshold,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "users_evaluated": len(evaluable),
            "mean_relevant_set_size": round(sum(rel_sizes) / len(rel_sizes), 2),
            "catalog_size": catalog_size,
        },
        "hybrid": hybrid_acc.summary(catalog_size),
        "popularity_baseline": pop_acc.summary(catalog_size),
    }
    return report


def format_report(report: dict) -> str:
    p = report["params"]
    h = report["hybrid"]
    b = report["popularity_baseline"]
    k = p["top_n"]
    lines = [
        "# Offline recommender evaluation",
        "",
        f"- timestamp: {report['timestamp']}",
        f"- users evaluated: {p['users_evaluated']}",
        f"- relevance threshold: rating >= {p['rating_threshold']}",
        f"- mean relevant-set size: {p['mean_relevant_set_size']}",
        f"- catalog size: {p['catalog_size']}",
        f"- split: {int((1 - p['test_frac']) * 100)}/{int(p['test_frac'] * 100)} per user, seed {p['seed']}",
        "",
        f"| metric@{k} | hybrid | popularity baseline |",
        "|---|---|---|",
        f"| precision | {h['precision_at_k']} | {b['precision_at_k']} |",
        f"| recall | {h['recall_at_k']} | {b['recall_at_k']} |",
        f"| ndcg | {h['ndcg_at_k']} | {b['ndcg_at_k']} |",
        f"| catalog coverage | {h['catalog_coverage']} | {b['catalog_coverage']} |",
        "",
    ]
    delta = h["precision_at_k"] - b["precision_at_k"]
    verdict = "beats" if delta > 0 else ("ties" if delta == 0 else "LOSES TO")
    lines.append(f"Hybrid {verdict} the popularity baseline on precision@{k} "
                 f"({h['precision_at_k']} vs {b['precision_at_k']}, delta {delta:+.4f}).")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=str(Path(__file__).parent.parent / "services" / "backend"))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rating-threshold", type=float, default=4.0)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-users", type=int, default=300)
    parser.add_argument("--all-users", action="store_true")
    parser.add_argument("--tag", default="baseline", help="label for the output file, e.g. baseline or hybrid-fixed")
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