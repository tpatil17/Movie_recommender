import pandas as pd
from surprise import Reader, Dataset, SVD


def rank_candidates_by_prediction(predict_fn, user_id, candidate_ids, exclude_ids, top_n):
    """
    Rank candidate item ids by predicted rating for a user, highest first.

    predict_fn(user_id, item_id) -> float. Injected so this is testable
    without a trained model and so the eval and the product path share one
    implementation.

    exclude_ids are removed before ranking (typically the movies the user has
    already rated, so we never recommend something they have seen).
    Returns a list of (item_id, predicted_score) of length <= top_n.
    """
    exclude = set(exclude_ids or [])
    scored = [
        (item_id, predict_fn(user_id, item_id))
        for item_id in candidate_ids
        if item_id not in exclude
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


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

    def recommend_for_user(
        self,
        user_id: int,
        candidate_ids: list[int],
        exclude_ids: list[int] | None = None,
        top_n: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Pure collaborative 'recommend for you': rank the candidate movies by
        this user's predicted rating, excluding movies they have already seen.

        This is the standard top-N task SVD is designed for, with no seed
        movie. Returns (movielens_id, predicted_score) pairs, best first.
        """
        if not self._trained:
            return []
        return rank_candidates_by_prediction(
            lambda u, m: self.svd.predict(u, m).est,
            user_id,
            candidate_ids,
            exclude_ids,
            top_n,
        )
