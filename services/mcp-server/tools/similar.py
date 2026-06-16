"""
Tool: get_similar
Calls the content-based similarity model directly and returns
movies most similar to a seed title by cosine similarity score.
"""

import httpx
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api")

async def get_similar(title: str, top_n: int = 10) -> dict:
    """
    Get movies most similar to a given title using content similarity.

    Use this tool when the user wants to explore movies that closely
    resemble a specific film in terms of cast, director, and genre —
    without personalisation. Results are ranked by content similarity
    score only, not by user preferences.

    Use get_recommendations instead when you want personalised results
    that factor in predicted ratings. Use this tool when the user asks
    something like 'what is most similar to X' or 'movies exactly like X'.

    Title must be an exact match. Use search_movies first to confirm
    the exact title if unsure.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(
                f"{BACKEND_URL}/movies/similar",
                params={"title": title, "top_n": top_n}
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {
                    "error": f"Movie '{title}' not found. "
                             f"Use search_movies to confirm the exact title."
                }
            if e.response.status_code == 503:
                return {"error": "Content model not loaded yet. Try again shortly."}
            return {"error": f"Backend returned {e.response.status_code}"}
        except httpx.RequestError as e:
            return {"error": f"Could not reach backend: {str(e)}"}