"""
search.py
---------
Owns all interaction with the live web search tool (DuckDuckGo).
Responsible for pulling fresh, recent sports news that ChromaDB's static
offline dataset cannot know about (e.g., "who won last night's match").
"""

from duckduckgo_search import DDGS

DEFAULT_MAX_RESULTS = 3


def get_live_news_context(sport_name: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
    """
    Searches the live web for recent tournament results, winners, and news
    for the given sport. Returns a single text block of joined snippets.

    Falls back to a clear "unavailable" message on any failure (e.g., no
    network access, rate limiting) so the rest of the pipeline never crashes.
    """
    search_query = f"{sport_name} latest tournament results championship winners news 2026"
    retrieved_texts = []

    print(f"[search] Executing web search for: '{search_query}'")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=max_results)
            for index, r in enumerate(results, start=1):
                title = r.get("title", "No Title")
                snippet = r.get("body", "No snippet content available")
                retrieved_texts.append(f"Web Source {index}: {title}\nSnippet: {snippet}")
    except Exception as e:
        print(f"[search] Web search failed: {e}")
        return "No recent search engine updates available due to a connectivity issue."

    if not retrieved_texts:
        return "No recent web search results were found for this sport."

    return "\n\n".join(retrieved_texts)
