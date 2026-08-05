"""
Agent service — LangChain conversational agent that connects
to the MCP server and handles multi-turn movie recommendation chat.

Phase 3 change: the chat response now surfaces the tool calls the agent
made during the turn, so the evaluation harness can score tool selection
without scraping logs or Prometheus.
"""

import json
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from agent.chain import build_agent
from metrics import (
    active_sessions,
    chat_latency,
    chat_requests,
    tool_calls as tool_calls_metric,
)

agent_executor = None
conversation_history: dict[str, list] = {}

# Cap history so long sessions do not grow unbounded in memory or tokens.
MAX_HISTORY_MESSAGES = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    agent_executor = await build_agent()
    print("Agent ready.")
    yield
    agent_executor = None


app = FastAPI(title="Movie Recommender Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ToolCallRecord(BaseModel):
    name: str
    args: dict
    status: str
    result: str


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[ToolCallRecord] = []
    latency_ms: float = 0.0


def _tool_status(message: ToolMessage) -> str:
    """
    Our MCP tools return {"error": "..."} dicts rather than raising, so
    ToolMessage.status stays "success" even on a failed call. Parse the
    payload to catch those. Falls back to success on unparseable content.
    """
    if getattr(message, "status", None) == "error":
        return "error"
    content = message.content
    if not isinstance(content, str):
        return "success"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return "success"
    if isinstance(payload, dict) and "error" in payload:
        return "error"
    return "success"


def _extract_tool_calls(new_messages: list) -> list[ToolCallRecord]:
    """
    Walk the messages the agent produced this turn and pair each requested
    tool call with the ToolMessage carrying its result. Args come from the
    AIMessage because ToolMessage does not carry them.
    """
    requested: dict[str, dict] = {}
    records: list[ToolCallRecord] = []

    for message in new_messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                requested[call["id"]] = {
                    "name": call["name"],
                    "args": call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            meta = requested.get(
                message.tool_call_id,
                {"name": message.name or "unknown", "args": {}},
            )
            status = _tool_status(message)
            records.append(
                ToolCallRecord(
                    name=meta["name"],
                    args=meta["args"],
                    status=status,
                    result=str(message.content)[:2000],
                )
            )
            tool_calls_metric.labels(tool_name=meta["name"], status=status).inc()

    return records


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    start = time.time()

    if session_id not in conversation_history:
        conversation_history[session_id] = []
        active_sessions.inc()

    history = conversation_history[session_id]
    history.append(HumanMessage(content=request.message))
    sent_count = len(history)

    try:
        result = await agent_executor.ainvoke({"messages": history})
    except Exception as exc:
        chat_requests.labels(status="error").inc()
        chat_latency.observe(time.time() - start)
        return ChatResponse(
            response=f"The agent failed to complete this turn: {exc}",
            session_id=session_id,
            tool_calls=[],
            latency_ms=(time.time() - start) * 1000,
        )

    new_messages = result["messages"][sent_count:]
    tool_records = _extract_tool_calls(new_messages)

    last_message = result["messages"][-1]
    history.append(AIMessage(content=last_message.content))

    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]

    elapsed = time.time() - start
    chat_latency.observe(elapsed)
    chat_requests.labels(status="success").inc()

    return ChatResponse(
        response=last_message.content,
        session_id=session_id,
        tool_calls=tool_records,
        latency_ms=elapsed * 1000,
    )


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str):
    """Lets the eval harness guarantee a clean session per test case."""
    if session_id in conversation_history:
        del conversation_history[session_id]
        active_sessions.dec()
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(conversation_history)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)