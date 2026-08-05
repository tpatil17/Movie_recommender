from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.data.loader import load_clean_data
from app.models.content_based import ContentBasedModel
from app.models.collaborative import CollaborativeModel
from app.models.hybrid import HybridRecommender
from app.state import models
from app.routes.recommendations import router as recommendations_router

from prometheus_client import make_asgi_app
from app.metrics import models_loaded

# Minimum ratings a movie needs before SVD has learned a meaningful factor for
# it. Below this, predictions regress to the global mean and the ranking is
# noise. Matches --cf-min-support in eval_offline so the product path and the
# measured path use the same pool.
CF_MIN_SUPPORT = 5


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
    models["collab"] = collab_model
    models["data"] = data

    # --- Collaborative "recommend for you" support ---------------------------
    # Offline evaluation showed the seed-anchored hybrid is retrieval-bound:
    # 80% of users get a candidate pool containing nothing they liked. Pure CF
    # over a broad pool scores roughly 3x better (precision@10 0.078 vs 0.026),
    # so it is exposed as its own endpoint rather than left in the eval harness.
    #
    # Pool = movies with enough ratings for SVD to have learned a real factor.
    # Below that support threshold predictions collapse toward the global mean.
    movielens_to_meta = {}
    meta_by_tmdb = {
        int(row.id): (row.title, row.genres if isinstance(row.genres, list) else [])
        for row in data.itertuples(index=False)
    }
    for tmdb_id, ml_id in tmdb_to_movielens.items():
        meta = meta_by_tmdb.get(int(tmdb_id))
        if meta is not None:
            movielens_to_meta.setdefault(int(ml_id), {"title": meta[0], "genres": meta[1]})

    support = ratings.groupby("movieId").size()
    cf_candidates = [
        int(m) for m, c in support.items()
        if c >= CF_MIN_SUPPORT and int(m) in movielens_to_meta
    ]

    models["movielens_to_meta"] = movielens_to_meta
    models["cf_candidates"] = cf_candidates
    models["user_seen"] = (
        ratings.groupby("userId")["movieId"].apply(lambda s: set(int(x) for x in s)).to_dict()
    )
    models["popular_ids"] = [
        int(m) for m in support.sort_values(ascending=False).index
        if int(m) in movielens_to_meta
    ]
    print(f"CF candidate pool: {len(cf_candidates)} movies (support >= {CF_MIN_SUPPORT})")

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
    models_loaded.set(1)  # Indicate that models are loaded and ready
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
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
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