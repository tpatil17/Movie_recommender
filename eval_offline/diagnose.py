"""
Diagnostic run for the hybrid recommender.

eval_offline.py tells you the hybrid loses to a popularity baseline. It does
not tell you WHY, and the two candidate explanations call for opposite fixes:

  RETRIEVAL is at fault
      The 25 content-similar candidates rarely contain anything the user liked.
      Precision@10 is then capped no matter how well SVD re-ranks. The fix is
      candidate generation: widen the pool, use the whole user profile, or add
      a collaborative retrieval path.

  RANKING is at fault
      The candidate pool does contain relevant movies, but SVD's ordering
      pushes them out of the top 10. The fix is the scoring blend, not
      retrieval.

This script measures which one it is. It reuses prepare_evaluation() from
eval_offline.py, so it runs on the identical users under the identical split
and the numbers are directly comparable to the scoring run.

Usage:
    python diagnose.py                      # 300 users, same defaults
    python diagnose.py --sample-users 100   # faster
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval_offline import prepare_evaluation, score_recommendations


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def run_diagnostics(args) -> dict:
    ctx = prepare_evaluation(args)
    hybrid = ctx["hybrid"]
    content = ctx["content"]
    collab = ctx["collab"]
    evaluable = ctx["evaluable"]
    tmdb_to_movielens = ctx["tmdb_to_movielens"]
    k = args.top_n

    print(f"Diagnosing {len(evaluable)} users ...", flush=True)

    pool_sizes = []          # candidates surviving the vote_count filter
    recs_returned = []       # how many recommendations actually came back
    ceilings = []            # best achievable precision@k given the pool
    pool_hit_counts = []     # relevant items present anywhere in the pool
    rating_spreads = []      # max - min predicted rating within a pool
    rating_stdevs = []
    unmapped_fracs = []      # candidates with no movielens id -> flat 3.0
    content_order_prec = []  # precision if we never re-ranked at all
    hybrid_prec = []

    for i, (user, seed_title, relevant) in enumerate(evaluable):
        # The candidate pool the hybrid actually sees. Note get_similar_movies
        # applies vote_count >= 100 AFTER slicing to top_n, so this can return
        # far fewer than 25.
        candidates = content.get_similar_movies(seed_title, top_n=25)
        pool_sizes.append(len(candidates))

        cand_titles = [c["title"] for c in candidates]
        in_pool = sum(1 for t in set(cand_titles) if t in relevant)
        pool_hit_counts.append(in_pool)

        # Ceiling: even with perfect ranking you cannot place more than
        # min(hits_available, k) relevant items in k slots.
        ceilings.append(min(in_pool, k) / k if k else 0.0)

        # Content order, unranked — precision if SVD were removed entirely.
        content_order_prec.append(
            score_recommendations(cand_titles, relevant, k)["precision"]
        )

        # What SVD does to that pool.
        preds = []
        unmapped = 0
        for c in candidates:
            tmdb_id = c["tmdb_id"]
            if tmdb_id in tmdb_to_movielens:
                preds.append(collab.predict_rating(user, tmdb_to_movielens[tmdb_id]))
            else:
                unmapped += 1
                preds.append(3.0)
        if candidates:
            unmapped_fracs.append(unmapped / len(candidates))
        if len(preds) >= 2:
            rating_spreads.append(max(preds) - min(preds))
            rating_stdevs.append(statistics.pstdev(preds))

        recs = hybrid.recommend(user, seed_title, top_n=k)
        recs_returned.append(len(recs))
        hybrid_prec.append(
            score_recommendations([r["title"] for r in recs], relevant, k)["precision"]
        )

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(evaluable)}", flush=True)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "users": len(evaluable),
        "top_n": k,
        "pool": {
            "mean_size": round(mean(pool_sizes), 2),
            "min_size": min(pool_sizes) if pool_sizes else 0,
            "pct_below_k": round(
                100 * mean([1.0 if s < k else 0.0 for s in pool_sizes]), 1
            ),
            "mean_recs_returned": round(mean(recs_returned), 2),
            "pct_short_of_k": round(
                100 * mean([1.0 if r < k else 0.0 for r in recs_returned]), 1
            ),
        },
        "retrieval": {
            "mean_relevant_in_pool": round(mean(pool_hit_counts), 3),
            "pct_users_with_empty_pool_hits": round(
                100 * mean([1.0 if h == 0 else 0.0 for h in pool_hit_counts]), 1
            ),
            "precision_ceiling": round(mean(ceilings), 4),
        },
        "ranking": {
            "content_order_precision": round(mean(content_order_prec), 4),
            "hybrid_precision": round(mean(hybrid_prec), 4),
            "mean_predicted_rating_spread": round(mean(rating_spreads), 3),
            "mean_predicted_rating_stdev": round(mean(rating_stdevs), 3),
            "mean_unmapped_candidate_frac": round(mean(unmapped_fracs), 3),
        },
    }


def format_diagnostics(d: dict) -> str:
    p, r, rk = d["pool"], d["retrieval"], d["ranking"]
    k = d["top_n"]
    lines = [
        "# Hybrid recommender diagnostics",
        "",
        f"- timestamp: {d['timestamp']}",
        f"- users: {d['users']}, K = {k}",
        "",
        "## Candidate pool",
        "",
        f"| measure | value |",
        f"|---|---|",
        f"| mean pool size (after vote_count filter) | {p['mean_size']} |",
        f"| smallest pool seen | {p['min_size']} |",
        f"| % of users with pool < K | {p['pct_below_k']}% |",
        f"| mean recommendations returned | {p['mean_recs_returned']} |",
        f"| % of users given fewer than K | {p['pct_short_of_k']}% |",
        "",
        "## Retrieval quality",
        "",
        f"| measure | value |",
        f"|---|---|",
        f"| mean relevant items present in pool | {r['mean_relevant_in_pool']} |",
        f"| % of users whose pool contains none | {r['pct_users_with_empty_pool_hits']}% |",
        f"| precision@{k} ceiling (perfect ranking) | {r['precision_ceiling']} |",
        "",
        "## Ranking quality",
        "",
        f"| measure | value |",
        f"|---|---|",
        f"| precision@{k}, content order (no SVD) | {rk['content_order_precision']} |",
        f"| precision@{k}, hybrid (SVD re-ranked) | {rk['hybrid_precision']} |",
        f"| mean predicted-rating spread in pool | {rk['mean_predicted_rating_spread']} |",
        f"| mean predicted-rating stdev in pool | {rk['mean_predicted_rating_stdev']} |",
        f"| mean fraction of candidates unmapped (flat 3.0) | {rk['mean_unmapped_candidate_frac']} |",
        "",
        "## Verdict",
        "",
    ]

    ceiling = r["precision_ceiling"]
    achieved = rk["hybrid_precision"]
    content_prec = rk["content_order_precision"]
    spread = rk["mean_predicted_rating_spread"]

    if ceiling < 0.05:
        lines.append(
            f"RETRIEVAL-BOUND. Even a perfect ranker tops out at {ceiling} because the "
            f"candidate pool holds only {r['mean_relevant_in_pool']} relevant items on "
            f"average, and {r['pct_users_with_empty_pool_hits']}% of users get a pool "
            f"with none at all. Re-ranking cannot fix this. Change candidate generation."
        )
    elif achieved < ceiling * 0.5:
        lines.append(
            f"RANKING-BOUND. The pool supports precision up to {ceiling}, but the hybrid "
            f"only reaches {achieved}. SVD is pushing relevant candidates out of the top "
            f"{k}. Change the scoring blend."
        )
    else:
        lines.append(
            f"The hybrid reaches {achieved} against a ceiling of {ceiling}, so ranking is "
            f"extracting most of what the pool allows. Gains have to come from a better "
            f"pool."
        )

    if p["pct_short_of_k"] > 10:
        lines.append(
            f"\nAlso: {p['pct_short_of_k']}% of users receive fewer than {k} "
            f"recommendations (mean {p['mean_recs_returned']}). The vote_count >= 100 "
            f"filter in get_similar_movies runs AFTER the top-25 slice, so the pool is "
            f"cut before it is ever ranked. Filtering before slicing would recover those "
            f"slots."
        )

    if spread < 0.5:
        lines.append(
            f"\nAlso: predicted ratings vary by only {spread} across a pool, so SVD is "
            f"close to inert here and the output is near content order "
            f"({content_prec} unranked vs {achieved} ranked)."
        )

    if rk["mean_unmapped_candidate_frac"] > 0.2:
        lines.append(
            f"\nAlso: {rk['mean_unmapped_candidate_frac']:.0%} of candidates have no "
            f"MovieLens id and are scored a flat 3.0, so they are ranked by nothing at all."
        )

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
    parser.add_argument("--tag", default="diagnose")
    args = parser.parse_args()

    d = run_diagnostics(args)
    markdown = format_diagnostics(d)
    print()
    print(markdown)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"{args.tag}-{stamp}.md").write_text(markdown)
    (out_dir / f"{args.tag}-{stamp}.json").write_text(json.dumps(d, indent=2))
    print(f"\nWrote results/{args.tag}-{stamp}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
