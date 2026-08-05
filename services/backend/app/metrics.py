"""
Prometheus metrics for the Movie Recommender backend.
All metrics are defined here and imported where needed.
"""

from prometheus_client import Counter, Histogram, Gauge

# Request counts
recommendation_requests = Counter(
    "recommendation_requests_total",
    "Total number of recommendation requests",
    ["status"]  # success, not_found, error
)

search_requests = Counter(
    "search_requests_total",
    "Total number of movie search requests",
    ["status"]
)

similar_requests = Counter(
    "similar_requests_total",
    "Total number of similar movie requests",
    ["status"]
)

for_you_requests = Counter(
    "for_you_requests_total",
    "Total number of pure-collaborative 'recommend for you' requests",
    ["status"]  # success, cold_start, error
)

# Latency histograms
recommendation_latency = Histogram(
    "recommendation_latency_seconds",
    "Time taken to return recommendations",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

search_latency = Histogram(
    "search_latency_seconds",
    "Time taken to return search results",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5]
)

for_you_latency = Histogram(
    "for_you_latency_seconds",
    "Time taken to rank the collaborative candidate pool for a user",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

# Cache metrics
cache_hits = Counter(
    "content_model_cache_hits_total",
    "Number of similarity cache hits"
)

cache_misses = Counter(
    "content_model_cache_misses_total",
    "Number of similarity cache misses"
)

# Model health
models_loaded = Gauge(
    "models_loaded",
    "Whether all models are loaded and ready",
)