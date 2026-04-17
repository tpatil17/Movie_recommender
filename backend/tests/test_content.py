# backend/tests/test_content.py
import pytest


class TestContentBasedModel:

    def test_known_movie_returns_results(self, content_model):
        """Should return a list of results for a known movie."""
        results = content_model.get_similar_movies("The Godfather")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_returns_correct_number_of_results(self, content_model):
        """Should respect top_n parameter."""
        results = content_model.get_similar_movies("The Godfather", top_n=10)
        assert len(results) == 10

    def test_unknown_movie_returns_empty_list(self, content_model):
        """Should return empty list for a movie not in the dataset."""
        results = content_model.get_similar_movies("This Movie Does Not Exist 99999")
        assert results == []

    def test_result_has_correct_keys(self, content_model):
        """Each result should have title, tmdb_id, and genres."""
        results = content_model.get_similar_movies("The Godfather", top_n=1)
        assert "title" in results[0]
        assert "tmdb_id" in results[0]
        assert "genres" in results[0]

    def test_result_types_are_correct(self, content_model):
        """title should be str, tmdb_id int, genres list."""
        results = content_model.get_similar_movies("The Godfather", top_n=1)
        assert isinstance(results[0]["title"], str)
        assert isinstance(results[0]["tmdb_id"], int)
        assert isinstance(results[0]["genres"], list)

    def test_does_not_return_query_movie(self, content_model):
        """The query movie itself should not appear in results."""
        results = content_model.get_similar_movies("The Godfather")
        titles = [r["title"] for r in results]
        assert "The Godfather" not in titles