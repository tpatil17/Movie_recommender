"""
Movie Recommender MCP Server
Exposes recommendation, search, and similarity tools for LLM agents
via the Model Context Protocol using FastMCP.
"""

from fastmcp import FastMCP
from tools.recommendations import get_recommendations
from tools.search import search_movies
from tools.similar import get_similar

mcp = FastMCP( name= "Movie Recommender", instructions=""" 
    
    You are a movie recommendation assistant with access to three tools.

    Use search_movies to find exact movie titles when the user gives a
    partial name or you are unsure of the spelling.

    Use get_recommendations when you have a confirmed title and want
    personalised similar movies ranked by predicted rating.

    Use get_similar when the user wants movies that closely resemble
    a specific film by content only, without personalisation.

    Always confirm a title exists via search_movies before calling
    get_recommendations or get_similar.
""")

mcp.tool()(get_recommendations)
mcp.tool()(search_movies)
mcp.tool()(get_similar)

if __name__ == "__main__":
    mcp.run(transport="sse", port= 8001)

