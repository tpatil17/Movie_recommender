"""
Unit tests for the offline evaluation scoring math.

These run on synthetic data only — no CSVs, no models, no backend import.
The point is to prove the metrics are correct before we trust any number the
harness prints against real data. Every case here has a precision, recall or
NDCG value that can be checked by hand.

Run:
    python -m pytest test_eval_offline.py -v
"""

import math
import random

import pytest

from eval_offline import (
    dcg_at_k,
    ndcg_at_k,
    per_user_split,
    score_recommendations,
)


# --------------------------------------------------------------------------
# precision / recall
# --------------------------------------------------------------------------

def test_precision_all_hits():
    """5 of 5 recommendations relevant -> precision 1.0."""
    recs = ["a", "b", "c", "d", "e"]
    relevant = {"a", "b", "c", "d", "e"}
    assert score_recommendations(recs, relevant, 5)["precision"] == 1.0


def test_precision_no_hits():
    recs = ["x", "y", "z"]
    relevant = {"a", "b"}
    assert score_recommendations(recs, relevant, 3)["precision"] == 0.0


def test_precision_planted_case():
    """
    Hand-checkable: 3 hits in the top 10 -> precision 0.3.
    Hits are deliberately scattered so position cannot affect precision.
    """
    recs = ["a", "x", "b", "x", "x", "c", "x", "x", "x", "x"]
    relevant = {"a", "b", "c"}
    scored = score_recommendations(recs, relevant, 10)
    assert scored["hits"] == 3
    assert scored["precision"] == pytest.approx(0.3)


def test_precision_divides_by_k_not_list_length():
    """
    Only 2 recommendations returned but K = 10. Precision must be 1/10, not
    1/2 — otherwise a model that returns one lucky item scores 1.0.
    """
    scored = score_recommendations(["a", "x"], {"a"}, 10)
    assert scored["precision"] == pytest.approx(0.1)


def test_recall_is_over_relevant_set_size():
    """2 of 4 relevant items retrieved -> recall 0.5."""
    recs = ["a", "b", "q", "r"]
    relevant = {"a", "b", "c", "d"}
    assert score_recommendations(recs, relevant, 4)["recall"] == pytest.approx(0.5)


def test_recall_full():
    recs = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert score_recommendations(recs, relevant, 3)["recall"] == 1.0


def test_only_top_k_are_scored():
    """A hit at rank 11 must not count when K = 10."""
    recs = ["x"] * 10 + ["a"]
    scored = score_recommendations(recs, {"a"}, 10)
    assert scored["hits"] == 0
    assert scored["precision"] == 0.0


def test_duplicate_recommendation_does_not_double_count_recall():
    """
    Recall is bounded by 1.0 even if the same relevant title appears twice.
    This is the metric-side guard against exactly the hybrid.py dedup bug.
    """
    scored = score_recommendations(["a", "a"], {"a"}, 2)
    assert scored["recall"] <= 1.0


# --------------------------------------------------------------------------
# DCG / NDCG
# --------------------------------------------------------------------------

def test_dcg_single_hit_at_rank_one():
    """Rank 1 uses log2(2) = 1, so DCG is exactly 1.0."""
    assert dcg_at_k([True], 1) == pytest.approx(1.0)


def test_dcg_position_weighting():
    """Rank 2 is discounted by log2(3)."""
    assert dcg_at_k([False, True], 2) == pytest.approx(1.0 / math.log2(3))


def test_ndcg_perfect_ranking_is_one():
    """All relevant items at the top -> NDCG 1.0."""
    hits = [True, True, True] + [False] * 7
    assert ndcg_at_k(hits, num_relevant=3, k=10) == pytest.approx(1.0)


def test_ndcg_rewards_earlier_hits():
    """Same hit count, better positions -> strictly higher NDCG."""
    early = ndcg_at_k([True, True, False, False], 2, 4)
    late = ndcg_at_k([False, False, True, True], 2, 4)
    assert early > late


def test_ndcg_idcg_capped_at_k():
    """
    50 relevant items but K = 5. IDCG must assume only min(50, 5) = 5 hits
    fit, so 5 hits in the top 5 is a perfect 1.0 — not a fraction of 50.
    """
    hits = [True] * 5
    assert ndcg_at_k(hits, num_relevant=50, k=5) == pytest.approx(1.0)


def test_ndcg_zero_when_no_hits():
    assert ndcg_at_k([False, False, False], num_relevant=3, k=3) == 0.0


def test_ndcg_never_exceeds_one():
    for num_relevant in (1, 3, 7, 20):
        hits = [True] * min(num_relevant, 10)
        hits += [False] * (10 - len(hits))
        assert ndcg_at_k(hits, num_relevant, 10) <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# empty-set guards
# --------------------------------------------------------------------------

def test_empty_relevant_set_does_not_divide_by_zero():
    scored = score_recommendations(["a", "b"], set(), 10)
    assert scored["recall"] == 0.0
    assert scored["ndcg"] == 0.0


def test_empty_recommendation_list_scores_zero():
    scored = score_recommendations([], {"a", "b"}, 10)
    assert scored["precision"] == 0.0
    assert scored["recall"] == 0.0
    assert scored["ndcg"] == 0.0


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------

def test_split_is_disjoint_and_lossless():
    """No rating may appear in both halves, and none may be dropped."""
    user_to_items = {
        u: [(m, 4.0) for m in range(10)] for u in range(5)
    }
    train, test = per_user_split(user_to_items, 0.3, random.Random(42))
    for u in user_to_items:
        assert not set(train[u]) & set(test[u])
        assert sorted(train[u] + test[u]) == sorted(user_to_items[u])


def test_split_is_reproducible_under_same_seed():
    user_to_items = {u: [(m, 4.0) for m in range(10)] for u in range(5)}
    a = per_user_split(user_to_items, 0.3, random.Random(42))
    b = per_user_split(user_to_items, 0.3, random.Random(42))
    assert a == b


def test_split_holds_out_at_least_one_rating():
    """A user with 2 ratings must still contribute a test item."""
    train, test = per_user_split({1: [(10, 5.0), (11, 4.0)]}, 0.3, random.Random(0))
    assert len(test[1]) >= 1
    assert len(train[1]) >= 1


def test_single_rating_user_goes_entirely_to_train():
    """Nothing to hold out — the user is unevaluable, not half-evaluated."""
    train, test = per_user_split({1: [(10, 5.0)]}, 0.3, random.Random(0))
    assert train[1] == [(10, 5.0)]
    assert test[1] == []
