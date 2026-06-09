from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.data.loader import load_clean_data
from app.models.content_based import ContentBasedModel
from app.models.collaborative import CollaborativeModel
from app.models.hybrid import HybridRecommender
from app.state import models
from app.routes.recommendations import router as recommendations_router




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup, clean up on shutdown."""
    print("Loading data and training models...")

    data, ratings, tmdb_to_movielens = load_clean_data()

    content_model = ContentBasedModel(data)

    collab_model = CollaborativeModel()
    collab_model.train(ratings)

    models["hybrid"] = HybridRecommender(content_model, collab_model, tmdb_to_movielens)
    models["content"] = content_model
    models["data"] = data

    # Pre-warm similarity cache for the most popular movies so the first
    # real user query is served from cache rather than computed cold.
    print("Pre-warming similarity cache...")
    if 'vote_count' in data.columns:
        popular_titles = (
            data.dropna(subset=['vote_count'])
            .nlargest(50, 'vote_count')['title']
            .drop_duplicates()
            .tolist()
        )
        content_model.warm_cache(popular_titles)
        print(f"Cache warmed for {len(popular_titles)} popular titles.")

    # Build a sorted title list for fast prefix search (O(log n) vs O(n) scan)
    models["titles_sorted"] = sorted(data['title'].dropna().drop_duplicates().tolist())

    print("Models ready.")
    yield

    # Cleanup on shutdown
    models.clear()


app = FastAPI(
    title="Movie Recommender API",
    description="Hybrid content + collaborative filtering recommendation engine",
    version="1.0.0",
    lifespan=lifespan
)

# Allow React frontend to call this API locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000","https://movie-recommender-1726.web.app","https://movie-recommender-1726.firebaseapp.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models) > 0}