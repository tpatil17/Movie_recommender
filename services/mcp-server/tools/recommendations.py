
"""
Tool: get_recommendations
Calls the hybrid recommendation backend with a seed movie title
and returns a ranked list of similar movies.
"""

import httpx
import os
from typing import Optional

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api")

# Placeholder user_id until Phase 2 session memory is implemented.
# Phase 2 will derive a user profile from conversation history via
# ChromaDB and pass it through to the backend.
DEFAULT_USER_ID = 1

async def get_recommendations(
    title: str,
    top_n: int = 10
) -> dict:
    """
    Get movie recommendations based on a seed movie title.

    Use this tool when the user names a specific movie they enjoyed
    or wants something similar to. The title must be a real movie name.
    Returns a ranked list of similar movies with title and relevance score.

    Do not use this tool for vague requests like 'something funny' or
    'a good action movie' — use search_movies first to find a seed title,
    then call this tool with that title.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                f"{BACKEND_URL}/recommendations",
                json={
                    "user_id": DEFAULT_USER_ID,
                    "title": title,
                    "top_n": top_n
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {
                    "error": f"Movie '{title}' not found. Try search_movies to find the exact title."
                }
            if e.response.status_code == 503:
                return {
                    "error": "Recommendation models are still loading. Please try again in a moment."
                    }
            return {"error": f"Backend returned {e.response.status_code}"}
            
        except httpx.RequestError as e:
            return {"error": f"Could not reach backend: {str(e)}"}