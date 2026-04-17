#test the hybrid models
from app.models.hybrid import HybridRecommender
import pandas as pd

# backend/tests/test_hybrid.py


class TestHybridRecommender:

    def test_returns_results_for_known_movie(self, hybrid_model):
        """Should return results for a known movie and valid user."""
        results = hybrid_model.recommend(user_id=2, title="The Godfather")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_respects_top_n(self, hybrid_model):
        """Should return exactly top_n results."""
        results = hybrid_model.recommend(user_id=2, title="The Godfather", top_n=5)
        assert len(results) == 5

    def test_unknown_movie_returns_empty_list(self, hybrid_model):
        """Should return empty list for movie not in dataset."""
        results = hybrid_model.recommend(user_id=2, title="This Movie Does Not Exist 99999")
        assert results == []

    def test_result_has_correct_keys(self, hybrid_model):
        """Each result should have title, genres, predicted_rating, reason."""
        results = hybrid_model.recommend(user_id=2, title="The Godfather", top_n=1)
        assert "title" in results[0]
        assert "genres" in results[0]
        assert "predicted_rating" in results[0]
        assert "reason" in results[0]

    def test_result_types_are_correct(self, hybrid_model):
        """Field types should match expected schema."""
        results = hybrid_model.recommend(user_id=2, title="The Godfather", top_n=1)
        assert isinstance(results[0]["title"], str)
        assert isinstance(results[0]["genres"], list)
        assert isinstance(results[0]["predicted_rating"], float)
        assert isinstance(results[0]["reason"], str)

    def test_results_sorted_by_rating_descending(self, hybrid_model):
        """Results should be sorted highest predicted rating first."""
        results = hybrid_model.recommend(user_id=2, title="The Godfather", top_n=10)
        ratings = [r["predicted_rating"] for r in results]
        assert ratings == sorted(ratings, reverse=True)

    def test_query_movie_not_in_results(self, hybrid_model):
        """The query movie itself should not appear in recommendations."""
        results = hybrid_model.recommend(user_id=2, title="The Godfather")
        titles = [r["title"] for r in results]
        assert "The Godfather" not in titles