# 🎬 Movie Recommender

A full-stack hybrid recommendation engine combining **content-based filtering** and **collaborative filtering** to deliver personalized movie recommendations.

**[Live Demo](https://movie-recommender-1726.web.app)** · **[GitHub](https://github.com/tpatil17/Movie_recommender)**

> ⚠️ First load may take 30–60 seconds due to model initialization on GCP Cloud Run.

---

## Demo

Search for a movie you like, enter a User ID, and get 10 personalized recommendations ranked by predicted rating — each with an explainability tag showing why it was recommended.

---

## How It Works

```
User inputs a movie title
        │
        ▼
Content-Based Filtering         ← finds 25 similar movies using
(CountVectorizer + Cosine Sim)    cast, director, and genre features
        │
        ▼
Collaborative Filtering         ← scores each candidate using
(SVD Matrix Factorization)        the user's rating history
        │
        ▼
Hybrid Ranking                  ← sorts by predicted rating,
                                  returns top 10 with reason tags
```

The hybrid approach outperforms either method alone — content-based filtering finds semantically similar movies while collaborative filtering personalizes results to the specific user's taste.

**Model Performance:**
- Precision@10: **0.74**
- Recall@10: **0.44**
- Significantly outperforms random baseline (Precision@10: 0.31)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, Tailwind CSS |
| Backend | FastAPI, Python 3.11 |
| ML Models | scikit-learn, scikit-surprise (SVD) |
| Data Processing | pandas, numpy |
| Containerization | Docker |
| Backend Hosting | GCP Cloud Run |
| Data Storage | GCP Cloud Storage |
| Frontend Hosting | Firebase Hosting |
| CI/CD | GitHub Actions |
| Testing | pytest (19 tests) |

---

## Project Structure

```
movie-recommender/
├── backend/
│   ├── app/
│   │   ├── data/
│   │   │   └── loader.py           ← data loading + feature engineering
│   │   ├── models/
│   │   │   ├── content_based.py    ← CountVectorizer + cosine similarity
│   │   │   ├── collaborative.py    ← SVD collaborative filtering
│   │   │   └── hybrid.py           ← hybrid ranking engine
│   │   ├── routes/
│   │   │   └── recommendations.py  ← REST API endpoints
│   │   ├── main.py                 ← FastAPI app + model lifecycle
│   │   ├── schemas.py              ← Pydantic request/response models
│   │   └── state.py                ← shared model state
│   ├── tests/                      ← 19 pytest tests
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── src/
│       └── App.jsx                 ← React UI with autocomplete
├── .github/
│   └── workflows/
│       └── test.yml                ← GitHub Actions CI
└── docker-compose.yml
```

---

## API Endpoints

```
GET  /health                        → service health check
GET  /api/movies/search?q={title}   → autocomplete movie search
POST /api/recommendations           → get hybrid recommendations
GET  /api/movies/{tmdb_id}          → movie detail
```

**Example request:**
```bash
curl -X POST https://movie-recommender-w2s64z6b6q-uc.a.run.app/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": 2, "title": "The Godfather", "top_n": 5}'
```

**Example response:**
```json
{
  "query_title": "The Godfather",
  "results": [
    {
      "title": "The Conversation",
      "predicted_rating": 4.41,
      "genres": ["crime", "drama", "mystery"],
      "reason": "Crime • Drama match"
    }
  ]
}
```

Interactive API docs available at `/docs` (Swagger UI).

---

## Running Locally

**Prerequisites:** Python 3.11, Node.js 18+, pyenv

```bash
# Clone the repo
git clone https://github.com/tpatil17/Movie_recommender
cd Movie_recommender

# Set up Python environment
pyenv local 3.11.9
python -m venv appvenv
source appvenv/bin/activate
pip install --upgrade pip setuptools wheel
pip install numpy==1.26.4 Cython
pip install git+https://github.com/NicolasHug/Surprise.git
pip install -r backend/requirements.txt
```

**Download the dataset** from [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and place these files in `backend/data/raw/`:
- `movies_metadata.csv`
- `ratings_small.csv`
- `credits.csv`
- `links_small.csv`

```bash
# Start the backend
PYTHONPATH=backend uvicorn app.main:app --reload

# In a new terminal, start the frontend
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## Running with Docker

```bash
docker-compose up
```

Open `http://localhost:8000/docs` for the API.

---

## Tests

```bash
PYTHONPATH=backend pytest backend/tests/ -v
```

19 tests covering content-based model, collaborative filtering, and hybrid recommender.

---

## Dataset

[The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) — 45,000+ movies, 100,000+ ratings from MovieLens.

---

## Author

**Tanishq Patil** — MS Computer Science, San Diego State University

[LinkedIn](https://linkedin.com/in/tanishq-patil) · [GitHub](https://github.com/tpatil17)