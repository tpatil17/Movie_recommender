import bisect
import time
from fastapi import APIRouter, HTTPException
from app.state import models
from app.schemas import RecommendationRequest, RecommendationResponse, ForYouResponse
from app.metrics import (
    recommendation_requests,
    recommendation_latency,
    search_requests,
    search_latency,
    similar_requests,
    for_you_requests,
    for_you_latency,
)

router = APIRouter()


def _result_from_movielens(ml_id: int, score: float, reason: str) -> dict:
    meta = models["movielens_to_meta"].get(ml_id, {})
    genres = meta.get("genres", [])
    return {
        "title": meta.get("title", "Unknown"),
        "predicted_rating": round(float(score), 2),
        "genres": genres,
        "reason": reason,
    }


@router.get("/recommendations/for-you", response_model=ForYouResponse)
def get_for_you(user_id: int, top_n: int = 10):
    """
    Pure collaborative filtering: rank the whole candidate pool by this user's
    predicted rating. No seed movie.

    This exists because offline evaluation showed the seed-anchored hybrid is
    retrieval-bound -- it can only rank the 25 content-neighbours of one movie,
    and 80% of the time none of them are movies the user would like. Ranking
    the broad pool instead scores about 3x higher on precision@10.

    Movies the user has already rated are excluded, so nothing is recommended
    back to someone who has seen it.
    """
    collab = models.get("collab")
    if collab is None or "cf_candidates" not in models:
        for_you_requests.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    seen = models["user_seen"].get(user_id, set())

    # Cold start: SVD has no factors for an unknown user, so every prediction
    # would be the global mean and the "ranking" would be arbitrary. Fall back
    # to popularity and label it honestly rather than faking personalization.
    if not seen:
        popular = [m for m in models["popular_ids"]][:top_n]
        for_you_requests.labels(status="cold_start").inc()
        return ForYouResponse(
            user_id=user_id,
            cold_start=True,
            results=[
                _result_from_movielens(m, 0.0, "Popular right now")
                for m in popular
            ],
        )

    start = time.time()
    ranked = collab.recommend_for_user(
        user_id=user_id,
        candidate_ids=models["cf_candidates"],
        exclude_ids=seen,
        top_n=top_n,
    )
    for_you_latency.observe(time.time() - start)

    if not ranked:
        for_you_requests.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Collaborative model not trained")

    for_you_requests.labels(status="success").inc()
    return ForYouResponse(
        user_id=user_id,
        cold_start=False,
        results=[
            _result_from_movielens(m, score, "Users with similar taste rated this highly")
            for m, score in ranked
        ],
    )


@router.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    hybrid = models.get("hybrid")
    if not hybrid:
        recommendation_requests.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    start = time.time()
    results = hybrid.recommend(
        user_id=request.user_id,
        title=request.title,
        top_n=request.top_n
    )
    recommendation_latency.observe(time.time() - start)
    if not results:
        recommendation_requests.labels(status="not_found").inc()
        raise HTTPException(status_code=404, detail=f"Movie '{request.title}' not found")
    recommendation_requests.labels(status="success").inc()
    return RecommendationResponse(query_title=request.title, results=results)


@router.get("/movies/search")
def search_movies(q: str):
    titles_sorted = models.get("titles_sorted")
    if titles_sorted is None:
        search_requests.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    start = time.time()
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
    search_latency.observe(time.time() - start)
    search_requests.labels(status="success").inc()

    return {"results": (prefix_matches + substring_matches)[:10]}


@router.get("/movies/similar")
def get_similar_movies(title: str, top_n: int = 10):
    content_model = models.get("content")
    if content_model is None:
        similar_requests.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    
    results = content_model.get_similar_movies(title, top_n=top_n)
    if not results:

        similar_requests.labels(status="not_found").inc()
        raise HTTPException(
            status_code=404,
            
            detail=f"Movie '{title}' not found in content index"
        )
    similar_requests.labels(status="success").inc()

    return {"query_title": title, "results": results}


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
