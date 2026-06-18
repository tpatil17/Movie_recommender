import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.metrics import cache_hits, cache_misses

class ContentBasedModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data.reset_index(drop=True)

        # TF-IDF downweights common terms like popular genres
        # so distinctive signals like director name carry more weight
        tfidf = TfidfVectorizer(stop_words='english', min_df=2)
        self.count_matrix = tfidf.fit_transform(self.data['soup'])

        self.indices = pd.Series(
            self.data.index, index=self.data['title']
        ).drop_duplicates()

        self._cache: dict[str, list] = {}

    def get_similar_movies(self, title: str, top_n: int = 25) -> list[dict]:
        if title not in self.indices:
            return []

        if title in self._cache:
            cache_hits.inc()
            return self._cache[title]
        cache_misses.inc()
        idx = self.indices[title]

        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        scores = self._compute_similarity(idx)
        sim_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        movie_indices = [i[0] for i in sim_scores]

        cols = ['title', 'id', 'genres']
        if 'vote_count' in self.data.columns:
            cols.append('vote_count')
        if 'vote_average' in self.data.columns:
            cols.append('vote_average')

        movies = self.data.iloc[movie_indices][cols].copy()

        # Filter out low-quality movies before caching
        if 'vote_count' in movies.columns:
            movies['vote_count'] = pd.to_numeric(
                movies['vote_count'], errors='coerce'
            ).fillna(0)
            movies = movies[movies['vote_count'] >= 100]

        result = [
            {
                "title": row['title'],
                "tmdb_id": int(row['id']),
                "genres": row['genres'] if isinstance(row['genres'], list) else [],
                "vote_average": float(row['vote_average']) if 'vote_average' in row else 5.0
            }
            for _, row in movies.iterrows()
        ]
        # After building the result list, deduplicate by title keeping highest vote_average
        seen_titles = set()
        deduped_result = []
        for movie in result:
            if movie['title'] not in seen_titles:
                seen_titles.add(movie['title'])
                deduped_result.append(movie)

        self._cache[title] = deduped_result
        return deduped_result


    def warm_cache(self, titles: list[str]) -> None:
        for title in titles:
            if title not in self._cache:
                self.get_similar_movies(title)

    def _compute_similarity(self, idx: int):
        movie_vec = self.count_matrix[idx]
        return cosine_similarity(movie_vec, self.count_matrix).flatten()