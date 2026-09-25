FROM python:3.11-slim

WORKDIR /app

# Build tools are needed to compile scikit-surprise from source. They are
# installed and then the apt cache is dropped in the same layer to keep the
# image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so code changes do not invalidate the (slow) surprise build.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.26.4 && \
    pip install --no-cache-dir Cython && \
    pip install --no-cache-dir git+https://github.com/NicolasHug/Surprise.git && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT=8080

EXPOSE 8080

# Cloud Run injects PORT and ignores EXPOSE, so the server must read it rather
# than hardcode a value. The shell form is required for ${PORT} to expand.
# Single worker on purpose: models are trained into process memory at startup,
# so each extra worker would duplicate the SVD and TF-IDF matrices.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
