from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    user_id: int
    title: str
    top_n: int = 10


class MovieResult(BaseModel):
    title: str
    predicted_rating: float
    genres: list[str]
    reason: str


class RecommendationResponse(BaseModel):
    query_title: str
    results: list[MovieResult]


class ForYouResponse(BaseModel):
    """
    Pure-collaborative recommendations, with no seed movie.

    cold_start is True when the user has no rating history the model was
    trained on, in which case results fall back to popular titles and are not
    personalized. The frontend should say so rather than implying otherwise.
    """
    user_id: int
    cold_start: bool
    results: list[MovieResult]