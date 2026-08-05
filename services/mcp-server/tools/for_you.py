"""
Tool: get_for_you
Calls the backend's pure-collaborative endpoint to get personalized
recommendations for a user with no seed movie.
"""

import os

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api")


async def get_for_you(user_id: int, top_n: int = 10) -> dict:
    """
    Get personalized movie recommendations for a user based on their whole
    rating history, without needing a seed movie.

    Use this tool for open-ended requests like "what should I watch",
    "recommend me something", or "any good movies for me" — cases where the
    user has NOT named a specific film. If the user names a movie they liked
    and wants more like it, use get_recommendations instead.

    Requires a real user_id. If the response has cold_start set to true, the
    results are popular titles rather than personalized ones, and you should
    tell the user their recommendations will improve once they rate some
    movies.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(
                f"{BACKEND_URL}/recommendations/for-you",
                params={"user_id": user_id, "top_n": top_n},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                return {
                    "error": "Recommendation models are still loading. Please try again in a moment."
                }
            return {"error": f"Backend returned {e.response.status_code}"}
        except httpx.RequestError as e:
            return {"error": f"Could not reach backend: {str(e)}"}
