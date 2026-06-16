import asyncio
import os

os.environ["BACKEND_URL"] = "http://localhost:8000/api"

from tools.search import search_movies
from tools.recommendations import get_recommendations
from tools.similar import get_similar

async def main():
    print("--- Testing search_movies ---")
    result = await search_movies("inception")
    print(result)

    print("\n--- Testing get_recommendations ---")
    result = await get_recommendations("Inception")
    print(result)

    print("\n--- Testing get_similar ---")
    result = await get_similar("Inception", 10)
    print(result)

asyncio.run(main())