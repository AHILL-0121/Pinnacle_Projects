"""
Tool: competitor_fetch_tool

Searches for nearby clothing-store competitors using the Places service,
returning a formatted Markdown table of results.
"""

from __future__ import annotations

import json
import logging
from langchain_core.tools import tool

import config
from models.schemas import Location
from services.places_service import geocode, search_nearby_competitors
from services import cache_get, cache_put, make_key
from tools._store import save_search

logger = logging.getLogger(__name__)


def _format_table(location_name: str, radius_km: float, competitors: list[dict]) -> str:
    """Build a human-readable Markdown table from competitor data."""
    total = len(competitors)
    lines = [
        f"### Clothing Stores near {location_name} ({total} found, {radius_km} km radius)",
        "",
        "| # | Store Name | Distance | Type | Address |",
        "|---|-----------|----------|------|--------|",
    ]
    for i, c in enumerate(competitors, 1):
        name = c.get("name", "Unknown")
        dist = c.get("distance_km", "N/A")
        btype = c.get("business_type", "clothes")
        addr = c.get("address", "N/A")
        lines.append(f"| {i} | {name} | {dist} km | {btype} | {addr} |")
    lines.append("")
    lines.append("*Data source: OpenStreetMap. Ratings/reviews not available (OSM data).*")
    return "\n".join(lines)


@tool
def competitor_fetch_tool(
    location_name: str,
    radius_km: float = 0.0,
) -> str:
    """Find nearby clothing stores for a given location.

    Just pass the location name (e.g. 'Avarampalayam, Coimbatore').
    The tool will geocode it and search automatically.

    Args:
        location_name: Full location name with city (e.g. 'RS Puram, Coimbatore').
        radius_km: Search radius in km (default: 10 km from config).

    Returns:
        A formatted Markdown table of clothing stores found nearby.
    """
    # Use config default if radius not supplied
    if radius_km <= 0:
        radius_km = config.DEFAULT_RADIUS_KM

    # Auto-geocode
    logger.info("Geocoding '%s'", location_name)
    loc = geocode(location_name)
    if loc.latitude == 0.0 and loc.longitude == 0.0:
        return (
            f"Could not resolve location '{location_name}'. "
            f"Please try a more specific name like 'Avarampalayam, Coimbatore'."
        )

    # Check cache
    cache_key = make_key("competitors", location_name, str(loc.latitude), str(loc.longitude), str(radius_km))
    cached_json = cache_get(cache_key)
    if cached_json:
        data = json.loads(cached_json)
    else:
        competitors = search_nearby_competitors(loc, radius_km)
        data = {
            "search_area": location_name,
            "radius_km": radius_km,
            "total_found": len(competitors),
            "competitors": [c.model_dump() for c in competitors],
        }
        cache_put(cache_key, json.dumps(data))

    # Save raw data for downstream tools (footfall, report)
    save_search(location_name, data)

    if not data["competitors"]:
        return f"No clothing stores found within {radius_km} km of {location_name}."

    return _format_table(location_name, radius_km, data["competitors"])
