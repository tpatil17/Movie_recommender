"""
Builds the LangChain agent connected to the MCP server via SSE.
"""

import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from agent.prompts import SYSTEM_PROMPT
from dotenv import load_dotenv




load_dotenv() # load environment variables from .env

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def build_agent():
    client = MultiServerMCPClient(
        {
            "movie-recommender": {
                "url": MCP_SERVER_URL,
                "transport": "sse"
            }
        }
    )

    tools = await client.get_tools()

    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        temperature=0.7
    )

    agent = create_agent(
        llm,
        tools,
        system_prompt=SYSTEM_PROMPT
    )

    return agent