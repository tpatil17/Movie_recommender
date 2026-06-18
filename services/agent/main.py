"""
Agent service — LangChain conversational agent that connects
to the MCP server and handles multi-turn movie recommendation chat.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.chain import build_agent
from langchain_core.messages import HumanMessage, AIMessage
import uvicorn
from metrics import chat_requests, chat_latency, active_sessions
from prometheus_client import make_asgi_app
import time

agent_executor = None
conversation_history: dict[str, list] ={}

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

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    start = time.time()

    if session_id not in conversation_history:
        conversation_history[session_id] = []
        active_sessions.inc()

    conversation_history[session_id].append(
        HumanMessage(content=request.message)
    )

    result = await agent_executor.ainvoke({
        "messages": conversation_history[session_id]
    })


    last_message = result["messages"][-1]

    conversation_history[session_id].append(
        AIMessage(content=last_message.content)
    )
    chat_latency.observe(time.time() - start)
    chat_requests.labels(status="success").inc()
    return ChatResponse(
        
        response=last_message.content,
        session_id=request.session_id
    )

@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(conversation_history)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)