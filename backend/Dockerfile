FROM python:3.11-slim

WORKDIR /app

# Install build tools needed for scikit-surprise
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install numpy==1.26.4 && \
    pip install Cython && \
    pip install git+https://github.com/NicolasHug/Surprise.git && \
    pip install -r requirements.txt

# Copy application code
COPY app/ ./app/

# Set Python path so imports resolve correctly
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]