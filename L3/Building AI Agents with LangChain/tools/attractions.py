"""
tools/attractions.py – Tourist attractions search via DuckDuckGo.

Uses the free DuckDuckGo Search API (no key required) to find
top tourist attractions for a given city.
"""

from langchain_core.tools import tool

try:
    from ddgs import DDGS  # new package name (ddgs >= 9.x)
except ImportError:
    from duckduckgo_search import DDGS  # legacy fallback

from config import MAX_SEARCH_RESULTS


@tool
def get_attractions(city: str) -> str:
    """Search for top tourist attractions in a city.

    Returns a bullet-point list of popular places to visit.
    Use this tool when the user asks about sightseeing,
    tourist spots, or things to do in a destination.

    Args:
        city: The name of the city (e.g. "Rome", "New York").
    """
    query = f"top tourist attractions in {city}"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))

        if not results:
            return f"No attraction results found for '{city}'."

        lines = [f"📍 Top attractions in {city}:\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Unknown")
            body = r.get("body", "")
            link = r.get("href", "")
            lines.append(f"  {i}. **{title}**")
            if body:
                lines.append(f"     {body[:200]}")
            if link:
                lines.append(f"     🔗 {link}")
            lines.append("")

        return "\n".join(lines)

    except Exception as exc:  # noqa: BLE001
        return (
            f"❌ Error searching attractions for '{city}': {exc}\n"
            "Tip: Make sure you have internet connectivity."
        )
