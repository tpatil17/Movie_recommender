FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY tools/ ./tools/

ENV PYTHONUNBUFFERED=1 \
    PORT=8001

EXPOSE 8001

CMD exec python main.py
