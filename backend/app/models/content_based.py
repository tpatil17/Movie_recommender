from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ContentBasedModel:

    def __init__(self, data: pd.DataFrame):
        # fit CountVectorizer, compute cosine_sim, build indices
        #  Define the Vectorizer
        # stop_words='english' removes common words like "the", "and", etc.
        self.count = CountVectorizer(stop_words='english')

        # Fit and Transform the data
        # This creates a massive matrix where rows = movies, columns = all unique words in our soup
        self.count_matrix = self.count.fit_transform(data['soup'])

        # Create a reverse mapping of titles and indices
        # look up the index of a movie based on its title
        self.indices = pd.Series(data.index, index=data['title']).drop_duplicates()

        self.data = data

        

    def get_similar_movies(self, title: str, top_n=25) -> list:
        # your existing logic, returns list of (title, tmdb_id, score)
            # Get the index of the movie that matches the title
        if title not in self.indices:
            return "Movie not found"

        idx = self.indices[title]

        # Get the pairwise similarity scores of all movies with that movie
        # returns a list of (index, score) tuples
        sim_scores = list(enumerate(self.get_similar(idx)))

        # Sort the movies based on the similarity scores (highest first)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top_n most similar movies
        # ignore index 0; the movie itself (score = 1.0)
        sim_scores = sim_scores[1:top_n+1]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        movies = self.data.iloc[movie_indices][['title', 'id', 'genres']].copy()
        return [
    {"title": row['title'], "tmdb_id": row['id'], "genres": row['genres']}
    for _, row in movies.iterrows()
]
     
    
    def get_similar(self, idx):
        movie_vec = self.count_matrix[idx]
        scores = cosine_similarity(movie_vec, self.count_matrix).flatten()
        return scores
    
