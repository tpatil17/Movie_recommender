"""
Movie Recommender MCP Server
Exposes recommendation, search, and similarity tools for LLM agents
via the Model Context Protocol using FastMCP.
"""

from fastmcp import FastMCP
from tools.recommendations import get_recommendations
from tools.search import search_movies
from tools.similar import get_similar
from tools.for_you import get_for_you

mcp = FastMCP( name= "Movie Recommender", instructions="""

    You are a movie recommendation assistant with access to four tools.

    Use search_movies to find exact movie titles when the user gives a
    partial name or you are unsure of the spelling.

    Use get_recommendations when you have a confirmed title and want
    personalised similar movies ranked by predicted rating.

    Use get_similar when the user wants movies that closely resemble
    a specific film by content only, without personalisation.

    Use get_for_you when the user asks for recommendations WITHOUT naming
    a movie, for example "what should I watch" or "recommend me something".
    It ranks the full catalogue by the user's own rating history and needs
    no seed title.

    Always confirm a title exists via search_movies before calling
    get_recommendations or get_similar. get_for_you needs no title.
""")

mcp.tool()(get_recommendations)
mcp.tool()(search_movies)
mcp.tool()(get_similar)
mcp.tool()(get_for_you)

if __name__ == "__main__":
    mcp.run(transport="sse", port= 8001)

