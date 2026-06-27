import logging
import os

from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_web(query: str, count: int = 3) -> dict:
    """Search the web via Tavily with advanced depth for higher-quality snippets.

    Args:
        query: The search query string.
        count: Maximum number of results to return.

    Returns:
        Tavily response dict with a 'results' key containing a list of docs.
    """
    return tavily_client.search(
        query=query,
        max_results=count,
        search_depth="fast"
    )


# def search_web_fallback(query: str, count: int = 3) -> list[dict]:
#     """Fallback web search using DuckDuckGo when Tavily is unavailable / rate-limited.

#     Requires: uv add duckduckgo-search

#     Returns:
#         List of dicts with keys: title, url, content.
#         Returns an empty list on any error so the caller can continue gracefully.
#     """
#     try:
#         from duckduckgo_search import DDGS

#         results: list[dict] = []
#         with DDGS() as ddgs:
#             for r in ddgs.text(query, max_results=count):
#                 results.append({
#                     "title": r.get("title", "No Title"),
#                     "url":   r.get("href", ""),
#                     "content": r.get("body", ""),
#                 })
#         logger.info("[Fallback] DuckDuckGo returned %d results for: '%s'", len(results), query)
#         return results

#     except ImportError:
#         logger.error(
#             "[Fallback] 'duckduckgo_search' not installed. "
#             "Run:  uv add duckduckgo-search"
#         )
#         return []
#     except Exception as exc:
#         logger.error("[Fallback] DuckDuckGo failed for '%s': %s", query, exc)
#         return []