import pandas as pd
from surprise import Reader, Dataset, SVD


class CollaborativeModel:
    """
    SVD-based collaborative filter trained on user-movie ratings.
    Predicts how much a specific user would enjoy a specific movie.
    """

    def __init__(self):
        self.svd = SVD()
        self._trained = False

    def train(self, ratings_df: pd.DataFrame):
        """
        Trains the SVD model on a ratings DataFrame.
        Expects columns: userId, movieId, rating
        """
        reader = Reader(rating_scale=(0.5, 5))
        data = Dataset.load_from_df(
            ratings_df[['userId', 'movieId', 'rating']], reader
        )
        trainset = data.build_full_trainset()
        self.svd.fit(trainset)
        self._trained = True
        print(f"CollaborativeModel trained on {trainset.n_ratings} ratings")

    def predict_rating(self, user_id: int, movielens_id: int) -> float:
        """
        Predicts a user's rating for a movie.
        Returns float between 0.5 and 5.0.
        Falls back to 3.0 (neutral) if model not trained or user unknown.
        """
        if not self._trained:
            return 3.0
        return self.svd.predict(user_id, movielens_id).est