from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.data.loader import load_clean_data
from app.models.content_based import ContentBasedModel
from app.models.collaborative import CollaborativeModel
from app.models.hybrid import HybridRecommender
from app.state import models
from app.routes.recommendations import router as recommendations_router

# Global model state — loaded once at startup
models = {}


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
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations_router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models) > 0}