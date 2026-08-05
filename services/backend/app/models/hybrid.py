import pandas as pd
from app.models.content_based import ContentBasedModel
from app.models.collaborative import CollaborativeModel


class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering.
    1. Content-based finds the top 25 similar movies to the given title.
    2. Each candidate is scored by blending the collaborative prediction
       (how much THIS user is predicted to like it) with the movie's TMDB
       quality signal.
    3. Returns the top_n candidates by blended score.
    """

    # Blend weights: collaborative prediction vs TMDB quality signal.
    CF_WEIGHT = 0.7
    QUALITY_WEIGHT = 0.3
    NEUTRAL_RATING = 3.0

    def __init__(
        self,
        content_model: ContentBasedModel,
        collab_model: CollaborativeModel,
        tmdb_to_movielens: dict
    ):
        self.content_model = content_model
        self.collab_model = collab_model
        self.tmdb_to_movielens = tmdb_to_movielens

    def recommend(self, user_id: int, title: str, top_n: int = 10) -> list[dict]:
        """
        Returns top_n recommendations for a user based on a movie they liked.
        Each result includes title, genres, predicted_rating, and reason.
        Returns an empty list if the title is not found or nothing maps.
        """
        # Step 1 — content-based: get similar movies to the seed title.
        similar_movies = self.content_model.get_similar_movies(title, top_n=25)
        if not similar_movies:
            return []

        # Step 2 — score each candidate once, blending the collaborative
        # prediction with the movie's quality signal.
        scored: dict[str, dict] = {}
        for movie in similar_movies:
            tmdb_id = movie['tmdb_id']
            vote_average = movie.get('vote_average', 5.0)

            if tmdb_id in self.tmdb_to_movielens:
                movielens_id = self.tmdb_to_movielens[tmdb_id]
                predicted_rating = self.collab_model.predict_rating(user_id, movielens_id)
            else:
                # Movie not in the ratings dataset — fall back to neutral.
                predicted_rating = self.NEUTRAL_RATING

            # vote_average is on a 0-10 scale; halve it to align with the
            # 0-5 predicted-rating scale before blending.
            blended_score = round(
                self.CF_WEIGHT * predicted_rating
                + self.QUALITY_WEIGHT * (vote_average / 2),
                2
            )

            title_key = movie['title']
            existing = scored.get(title_key)
            if existing is None or blended_score > existing['predicted_rating']:
                scored[title_key] = {
                    "title": title_key,
                    "genres": movie['genres'],
                    "predicted_rating": blended_score,
                    "reason": self._build_reason(movie['genres'])
                }

        ranked = sorted(
            scored.values(),
            key=lambda x: x['predicted_rating'],
            reverse=True
        )
        return ranked[:top_n]

    def _build_reason(self, genres: list) -> str:
        """
        Builds a human-readable explanation for why a movie was recommended.
        """
        if not genres:
            return "Similar cast & director"
        # Genres come in lowercased from the loader; capitalise for display.
        genre_names = [g.capitalize() for g in genres[:2]]
        return f"{' • '.join(genre_names)} match"