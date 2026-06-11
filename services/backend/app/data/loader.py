import os
import pandas as pd
import ast
import numpy as np
from io import BytesIO

# Resolves to backend/data/raw regardless of where Python is run from
_DEFAULT_DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),  # backend/app/data/
    "..",                        # backend/app/
    "..",                        # backend/
    "data",
    "raw"
))
# Detect if running in GCP Cloud Run
_GCS_BUCKET = os.getenv("GCS_BUCKET")  # set this env var in Cloud Run

def _read_csv(filename: str, data_dir: str, **kwargs) -> pd.DataFrame:
    """
    Reads a CSV either from GCS (in production) or local disk (in development).
    """
    if _GCS_BUCKET:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(_GCS_BUCKET)
        blob = bucket.blob(f"raw/{filename}")
        data = blob.download_as_bytes()
        return pd.read_csv(BytesIO(data), **kwargs)
    else:
        return pd.read_csv(os.path.join(data_dir, filename), **kwargs)

def load_clean_data(data_dir: str = _DEFAULT_DATA_DIR):
    meta = _read_csv('movies_metadata.csv', data_dir, low_memory=False)
    ratings = _read_csv('ratings_small.csv', data_dir)
    credits = _read_csv('credits.csv', data_dir)
    links = _read_csv('links_small.csv', data_dir)

    # Clean IDs — metadata has messy mixed-type ID column
    meta['id'] = pd.to_numeric(meta['id'], errors='coerce')
    meta = meta.dropna(subset=['id'])
    meta['id'] = meta['id'].astype(int)
    credits['id'] = credits['id'].astype(int)

    # Merge metadata with credits
    data = meta.merge(credits, on='id')

    # Build TMDB → MovieLens ID map from links file
    links = links.dropna(subset=['tmdbId', 'movieId'])
    tmdb_to_movielens = dict(
        zip(links['tmdbId'].astype(int), links['movieId'].astype(int))
    )

    # Run full feature engineering pipeline
    data = _prepare_data(data)

    return data, ratings, tmdb_to_movielens


# --- Internal pipeline --- #

def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Runs all feature engineering steps in order."""
    data = _parse_json_features(data)
    data = _extract_director(data)
    data = _extract_top_cast(data)
    data = _extract_genres(data)
    data = _clean_text_features(data)
    data = _build_soup(data)
    return data


def _parse_json_features(data: pd.DataFrame) -> pd.DataFrame:
    """Parses stringified JSON columns into Python lists."""
    for feature in ['cast', 'crew', 'genres']:
        data[feature] = data[feature].apply(_safe_parse)
    return data

def _safe_parse(x):
    try:
        return ast.literal_eval(x)
    except:
        return []


def _extract_director(data: pd.DataFrame) -> pd.DataFrame:
    data['director'] = data['crew'].apply(_get_director)
    return data

def _get_director(crew: list):
    for member in crew:
        if member.get('job') == 'Director':
            return member['name']
    return np.nan


def _extract_top_cast(data: pd.DataFrame) -> pd.DataFrame:
    data['top_cast'] = data['cast'].apply(_get_top_3)
    return data

def _get_top_3(cast: list) -> list:
    if isinstance(cast, list):
        return [member['name'] for member in cast[:3]]
    return []


def _extract_genres(data: pd.DataFrame) -> pd.DataFrame:
    data['genres'] = data['genres'].apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else []
    )
    return data


def _clean_text_features(data: pd.DataFrame) -> pd.DataFrame:
    """Lowercases and removes spaces so 'Tom Hanks' → 'tomhanks'."""
    def clean(x):
        if isinstance(x, list):
            return [s.lower().replace(" ", "") for s in x]
        if isinstance(x, str):
            return x.lower().replace(" ", "")
        return ''

    for feature in ['top_cast', 'genres', 'director']:
        data[feature] = data[feature].apply(clean)
    return data


def _build_soup(data: pd.DataFrame) -> pd.DataFrame:
    """Concatenates cast, director, genres into a single text blob."""
    def make_soup(row):
        cast = ' '.join(row['top_cast'])
        director = row['director'] if isinstance(row['director'], str) else ''
        genres = ' '.join(row['genres'])
        return f"{cast} {director} {genres}"

    data['soup'] = data.apply(make_soup, axis=1)
    return data

