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