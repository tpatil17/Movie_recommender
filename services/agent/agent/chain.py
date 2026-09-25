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

# Product default is 0.7 so conversation does not feel canned. The behavioral
# benchmark sets AGENT_TEMPERATURE=0, because a before/after tool-selection
# delta measured under sampling is mostly noise.
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))

# Bump when a change is expected to alter agent behaviour. The benchmark
# records this, so a report can never be silently attributed to the wrong
# prompt or tool set.
AGENT_TAG = os.getenv("AGENT_TAG", "v2-for-you")

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
        temperature=AGENT_TEMPERATURE
    )

    agent = create_agent(
        llm,
        tools,
        system_prompt=SYSTEM_PROMPT
    )

    return agent