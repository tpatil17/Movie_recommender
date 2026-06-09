import bisect

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
    titles_sorted = models.get("titles_sorted")
    if titles_sorted is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    q_lower = q.lower()

    # Fast O(log n) prefix scan using bisect, then linear walk for matches
    # Falls back to substring search so "Dark Knight" still matches "The Dark Knight"
    prefix_matches: list[str] = []
    substring_matches: list[str] = []

    lo = bisect.bisect_left(titles_sorted, q.capitalize())
    for title in titles_sorted[lo:lo + 500]:
        if title.lower().startswith(q_lower):
            prefix_matches.append(title)
            if len(prefix_matches) >= 10:
                break

    # If we have fewer than 10 prefix hits, fill with substring matches
    if len(prefix_matches) < 10:
        for title in titles_sorted:
            if q_lower in title.lower() and title not in prefix_matches:
                substring_matches.append(title)
                if len(prefix_matches) + len(substring_matches) >= 10:
                    break

    return {"results": (prefix_matches + substring_matches)[:10]}


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