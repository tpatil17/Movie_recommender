FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py metrics.py ./
COPY agent/ ./agent/

ENV PYTHONUNBUFFERED=1 \
    PORT=8002

EXPOSE 8002

# OPENAI_API_KEY comes from Secret Manager on Cloud Run and from the environment
# locally — never baked into the image (see .dockerignore).
#
# Single worker: conversation history lives in an in-process dict, so a second
# worker would serve some turns from an empty history. Deployed with
# --max-instances=1 for the same reason.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
