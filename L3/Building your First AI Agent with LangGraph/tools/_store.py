"""
Shared in-memory data store for tool results.

Allows downstream tools (footfall, report) to access competitor data
without the LLM having to pass raw JSON between tool calls.
"""

from __future__ import annotations
from typing import Any

# Keyed by location_name (lowered) → dict with search results
_data: dict[str, dict[str, Any]] = {}

# Track the most recent search location
_last_location: str = ""


def save_search(location_name: str, data: dict) -> None:
    """Store search results for a location."""
    global _last_location
    key = location_name.lower().strip()
    _data[key] = data
    _last_location = key


def get_search(location_name: str = "") -> dict | None:
    """Retrieve stored search results. If no location given, use last search."""
    key = (location_name or _last_location).lower().strip()
    return _data.get(key)


def get_last_location() -> str:
    """Return the most recently searched location name."""
    return _last_location
