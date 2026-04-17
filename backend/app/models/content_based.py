import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedModel:
    """
    Content-based recommender using cast, director, and genre similarity.
    Computes similarity on-demand per query to avoid storing an O(n^2) matrix.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data.reset_index(drop=True)

        count = CountVectorizer(stop_words='english')
        self.count_matrix = count.fit_transform(self.data['soup'])

        # Reverse map: movie title → DataFrame index
        self.indices = pd.Series(
            self.data.index, index=self.data['title']
        ).drop_duplicates()

    def get_similar_movies(self, title: str, top_n: int = 25) -> list[dict]:
        """
        Returns top_n movies most similar to the given title.
        Each result is a dict with title, tmdb_id, and genres.
        Returns empty list if title not found.
        """
        if title not in self.indices:
            return []

        idx = self.indices[title]

        # Guard against duplicate titles returning a Series instead of scalar
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        scores = self._compute_similarity(idx)
        sim_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        # Skip index 0 — that's the movie itself with score 1.0
        sim_scores = sim_scores[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]

        movies = self.data.iloc[movie_indices][['title', 'id', 'genres']].copy()

        return [
            {
                "title": row['title'],
                "tmdb_id": int(row['id']),
                "genres": row['genres'] if isinstance(row['genres'], list) else []
            }
            for _, row in movies.iterrows()
        ]

    def _compute_similarity(self, idx: int):
        """Computes cosine similarity between one movie and all others."""
        movie_vec = self.count_matrix[idx]
        return cosine_similarity(movie_vec, self.count_matrix).flatten()