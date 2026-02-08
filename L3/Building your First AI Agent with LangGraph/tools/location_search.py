"""
Tool: location_search_tool

Geocodes a user-provided locality into coordinates and validates it.
"""

from __future__ import annotations

import json
from langchain_core.tools import tool

from services.places_service import geocode
from services import cache_get, cache_put, make_key


@tool
def location_search_tool(location_name: str) -> str:
    """Geocode a locality/area name and return its coordinates.

    IMPORTANT: Always pass the FULL location with city, e.g.:
      - "Koramangala, Bangalore"  (not just "Koramangala")
      - "Gandhipuram, Coimbatore" (not just "Gandhipuram")
      - "Coimbatore"              (city-level search is fine)

    Args:
        location_name: Full location name including city (e.g. 'RS Puram, Coimbatore').

    Returns:
        JSON with name, latitude, longitude, formatted_address.
        If formatted_address contains "NOT_FOUND", geocoding failed – ask user for clarification.
    """
    cache_key = make_key("geo", location_name.lower().strip())
    cached = cache_get(cache_key)
    if cached:
        return cached

    loc = geocode(location_name)
    result = json.dumps(loc.model_dump(), indent=2)
    cache_put(cache_key, result)
    return result
