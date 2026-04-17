# backend/tests/test_collaborative.py
from app.models.collaborative import CollaborativeModel
import pandas as pd


class TestCollaborativeModel:

    def test_prediction_returns_float(self, collab_model):
        """Should return a float for a known user and movie."""
        result = collab_model.predict_rating(1, 302)
        assert isinstance(result, float)

    def test_prediction_within_rating_scale(self, collab_model):
        """Predicted rating should be between 0.5 and 5.0."""
        result = collab_model.predict_rating(1, 302)
        assert 0.5 <= result <= 5.0

    def test_unknown_user_returns_float(self, collab_model):
        """Unknown user should still return a valid float, not crash."""
        result = collab_model.predict_rating(999999, 302)
        assert isinstance(result, float)
        assert 0.5 <= result <= 5.0

    def test_unknown_movie_returns_float(self, collab_model):
        """Unknown movie should still return a valid float, not crash."""
        result = collab_model.predict_rating(1, 999999)
        assert isinstance(result, float)

    def test_untrained_model_returns_neutral(self):
        """Untrained model should return neutral score of 3.0."""
        model = CollaborativeModel()
        result = model.predict_rating(1, 302)
        assert result == 3.0

    def test_trained_flag_set_after_training(self, loaded_data):
        """_trained flag should be True after calling train()."""
        _, ratings, _ = loaded_data
        model = CollaborativeModel()
        assert model._trained is False
        model.train(ratings)
        assert model._trained is True

    def test_untrained_model_returns_neutral(self):
        """This test runs in CI — no dataset needed."""
        from app.models.collaborative import CollaborativeModel
        model = CollaborativeModel()
        result = model.predict_rating(1, 302)
        assert result == 3.0