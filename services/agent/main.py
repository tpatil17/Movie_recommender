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
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id

    if session_id not in conversation_history:
        conversation_history[session_id] = []

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
    return ChatResponse(
        
        response=last_message.content,
        session_id=request.session_id
    )

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)