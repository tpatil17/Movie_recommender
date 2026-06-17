import pandas as pd
from app.models.content_based import ContentBasedModel
from app.models.collaborative import CollaborativeModel


class HybridRecommender:
    """
    Hybrid recommender combining content-based and collaborative filtering.
    1. Content-based finds top 25 similar movies to the given title
    2. Collaborative filtering ranks them by predicted user rating
    3. Returns top_n results sorted by predicted rating
    """

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
        Returns empty list if title not found or no mappable results.
        """
        # Step 1 — content-based: get 25 similar movies
        similar_movies = self.content_model.get_similar_movies(title, top_n=25)

        if not similar_movies:
            return []
        
        #---
        scored = []

        for movie in similar_movies:
            tmdb_id = movie['tmdb_id']
            vote_average = movie.get('vote_average', 5.0)

            if tmdb_id in self.tmdb_to_movielens:
                movielens_id = self.tmdb_to_movielens[tmdb_id]
                predicted_rating = self.collab_model.predict_rating(user_id, movielens_id)
            else:
                predicted_rating = 3.0

            # Blend collaborative score with TMDB quality signal
            # 70% collaborative, 30% vote_average (normalised to 0-5 scale)
            blended_score = round(
                0.7 * predicted_rating + 0.3 * (vote_average / 2),
                2
            )

            scored.append({
                "title": movie['title'],
                "genres": movie['genres'],
                "predicted_rating": blended_score,
                "reason": self._build_reason(movie['genres'])
            })
        #---

        # Step 2 — collaborative: score each candidate with user's preferences

        for movie in similar_movies:
            tmdb_id = movie['tmdb_id']

            if tmdb_id in self.tmdb_to_movielens:
                movielens_id = self.tmdb_to_movielens[tmdb_id]
                predicted_rating = self.collab_model.predict_rating(user_id, movielens_id)
            else:
                # Movie not in ratings dataset — use neutral score
                predicted_rating = 3.0

            scored.append({
                "title": movie['title'],
                "genres": movie['genres'],
                "predicted_rating": round(predicted_rating, 2),
                "reason": self._build_reason(movie['genres'])
            })
        seen: dict[str, dict] = {}
        for movie in scored:
            t = movie['title']
            if t not in seen or movie['predicted_rating'] > seen[t]['predicted_rating']:
                seen[t] = movie

        deduped = list(seen.values())
        deduped.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return deduped[:top_n]


    def _build_reason(self, genres: list) -> str:
        """
        Builds a human-readable explanation for why a movie was recommended.
        This is the explainability feature that makes the app look professional.
        """
        if not genres:
            return "Similar cast & director"
        # Genres come in lowercased from loader, capitalize for display
        genre_names = [g.capitalize() for g in genres[:2]]
        return f"{' • '.join(genre_names)} match"
    


    