from prometheus_client import Gauge, Histogram, Counter

chat_requests = Counter(
    "agent_chat_requests_total",
    "Total chat requests received by the agent",
    ["status"]
)

chat_latency = Histogram(
    "agent_chat_latency_seconds",
    "End to end chat response latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

tool_calls = Counter(
    "agent_tool_calls_total",
    "Total MCP tool calls made by the agent",
    ["tool_name", "status"]
)

active_sessions = Gauge(
    "agent_active_sessions",
    "Number of active conversation sessions"
)