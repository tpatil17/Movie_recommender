"""
Tool: search_movies
Searches the movie catalogue by title keyword and returns
up to 10 matching titles to use as seeds for get_recommendations.
"""

import httpx
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api")

async def search_movies(query: str) -> dict:
    """
    Search for movies by title keyword.

    Use this tool when the user mentions a movie title but you are not
    sure of the exact name, or when they reference a partial title such
    as 'that movie with Leonardo DiCaprio on a boat'. Returns up to 10
    matching titles ordered by relevance.

    Do not use this tool for genre or mood requests such as 'a comedy'
    or 'something relaxing'. For those, ask the user to name a specific
    movie they enjoyed in that genre or mood, then call this tool with
    that title to confirm the exact name before calling get_recommendations.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(
                f"{BACKEND_URL}/movies/search",
                params={"q": query}
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            if not results:
                return {
                    "results": [],
                    "message": f"No movies found matching '{query}'. "
                               f"Try a different spelling or a shorter keyword."
                }
            return {"results": results}

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                return {"error": "Search index not loaded yet. Please try again shortly."}
            return {"error": f"Backend returned {e.response.status_code}"}
        except httpx.RequestError as e:
            return {"error": f"Could not reach backend: {str(e)}"}