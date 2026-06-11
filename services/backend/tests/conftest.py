# backend/tests/conftest.py
import pytest
from app.data.loader import load_clean_data
from app.models.content_based import ContentBasedModel
from app.models.collaborative import CollaborativeModel
from app.models.hybrid import HybridRecommender

import os

# Skip tests that require local CSV files in CI
skip_in_ci = pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping — dataset not available in CI"
)


@pytest.fixture(scope="session")
def loaded_data():
    if os.getenv("CI") == "true":
        pytest.skip("No dataset in CI")
    from app.data.loader import load_clean_data
    data, ratings, tmdb_to_movielens = load_clean_data()
    return data, ratings, tmdb_to_movielens


@pytest.fixture(scope="session")
def content_model(loaded_data):
    from app.models.content_based import ContentBasedModel
    data, _, _ = loaded_data
    return ContentBasedModel(data)


@pytest.fixture(scope="session")
def collab_model(loaded_data):
    from app.models.collaborative import CollaborativeModel
    _, ratings, _ = loaded_data
    model = CollaborativeModel()
    model.train(ratings)
    return model


@pytest.fixture(scope="session")
def hybrid_model(content_model, collab_model, loaded_data):
    from app.models.hybrid import HybridRecommender
    _, _, tmdb_to_movielens = loaded_data
    return HybridRecommender(content_model, collab_model, tmdb_to_movielens)