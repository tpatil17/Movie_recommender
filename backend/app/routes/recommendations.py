from fastapi import APIRouter, HTTPException
from app.state import models
from app.schemas import RecommendationRequest, RecommendationResponse

router = APIRouter()


@router.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    hybrid = models.get("hybrid")
    if not hybrid:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    results = hybrid.recommend(
        user_id=request.user_id,
        title=request.title,
        top_n=request.top_n
    )

    if not results:
        raise HTTPException(status_code=404, detail=f"Movie '{request.title}' not found")

    return RecommendationResponse(query_title=request.title, results=results)


@router.get("/movies/search")
def search_movies(q: str):
    data = models.get("data")
    if data is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Case-insensitive search against movie titles
    mask = data['title'].str.contains(q, case=False, na=False)
    matches = data[mask]['title'].drop_duplicates().head(10).tolist()

    return {"results": matches}


@router.get("/movies/{tmdb_id}")
def get_movie(tmdb_id: int):
    data = models.get("data")
    if data is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    movie = data[data['id'] == tmdb_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")

    row = movie.iloc[0]
    return {
        "tmdb_id": tmdb_id,
        "title": row['title'],
        "genres": row['genres'] if isinstance(row['genres'], list) else [],
        "overview": row.get('overview', ''),
    }