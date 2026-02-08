"""
Location & competitor search service using **OpenStreetMap** (100 % free).

APIs used (no key / billing required):
  • Nominatim  – geocoding (locality name → lat/lng)
  • Overpass   – POI search (find clothing shops within a radius)

Falls back to demo data only when DEMO_MODE=true.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import httpx

import config
from models.schemas import Competitor, Location

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Polite User-Agent (required by Nominatim TOS)
_USER_AGENT = "CompetitorIQ/1.0 (educational-project; github.com/competitor-iq)"
_HEADERS = {"User-Agent": _USER_AGENT, "Accept-Language": "en"}

# Broader Overpass query – clothing, textiles, department stores, malls
_OVERPASS_QUERY = """
[out:json][timeout:30];
(
  node["shop"~"clothes|fashion|boutique|department_store|textiles|fabric|tailor"](around:{radius_m},{lat},{lng});
  way["shop"~"clothes|fashion|boutique|department_store|textiles|fabric|tailor"](around:{radius_m},{lat},{lng});
  node["shop"="mall"](around:{radius_m},{lat},{lng});
  way["shop"="mall"](around:{radius_m},{lat},{lng});
);
out center body;
"""


# ── Public API ──────────────────────────────────────────────

def geocode(location_name: str) -> Location:
    """Convert a locality name to coordinates via Nominatim.

    Tries multiple query variations to handle small localities:
      1. Exact query
      2. Query + ", India"
      3. Query + ", Tamil Nadu, India" (common state for south-Indian localities)

    Returns a Location with `formatted_address` containing the full
    resolved address so callers can verify the correct city was matched.
    If nothing matches, returns None-like Location with latitude=0.
    """
    if config.DEMO_MODE:
        return _demo_geocode(location_name)

    # Try multiple variations to handle small locality names
    variations = [
        location_name,
        f"{location_name}, India",
    ]

    for query in variations:
        result = _nominatim_search(query)
        if result:
            # Use user's original name but with resolved coordinates
            result.name = location_name
            return result

    # Nothing found – return a clear "not found" Location
    logger.warning("Nominatim: could not resolve '%s' after %d attempts", location_name, len(variations))
    return Location(
        name=location_name,
        latitude=0.0,
        longitude=0.0,
        formatted_address=f"NOT_FOUND: {location_name}",
    )


def _nominatim_search(query: str) -> Location | None:
    """Single Nominatim search attempt. Returns Location or None."""
    try:
        time.sleep(1.2)  # Nominatim requires ≥1 s between requests
        params = {
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 1,
        }
        resp = httpx.get(
            _NOMINATIM_URL, params=params, headers=_HEADERS, timeout=10,
            follow_redirects=True,
        )
        resp.raise_for_status()
        results = resp.json()

        if not results:
            return None

        hit = results[0]
        return Location(
            name=query,
            latitude=float(hit["lat"]),
            longitude=float(hit["lon"]),
            formatted_address=hit.get("display_name", query),
        )
    except Exception as e:
        logger.warning("Nominatim error for '%s': %s", query, e)
        return None


def search_nearby_competitors(
    location: Location,
    radius_km: float = 10.0,
    business_type: str = "clothing_store",
) -> list[Competitor]:
    """Find nearby clothing stores via the Overpass API (OpenStreetMap).

    Returns an empty list (not demo data) if the location has lat=0/lng=0
    (i.e. geocoding failed), so the agent can detect the failure and ask
    the user for clarification.
    """
    if config.DEMO_MODE:
        return _demo_competitors(location, radius_km)

    # Guard: don't search if geocoding failed
    if location.latitude == 0.0 and location.longitude == 0.0:
        logger.warning("Skipping Overpass search – location not resolved: %s", location.name)
        return []

    try:
        time.sleep(1)  # courtesy pause

        radius_m = int(radius_km * 1000)
        query = _OVERPASS_QUERY.format(
            radius_m=radius_m,
            lat=location.latitude,
            lng=location.longitude,
        )
        resp = httpx.post(
            _OVERPASS_URL,
            data={"data": query},
            headers=_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        elements = data.get("elements", [])
        if not elements:
            logger.warning("Overpass: no clothing shops near %s", location.name)
            return []

        competitors: list[Competitor] = []
        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name", tags.get("brand", ""))
            if not name:
                continue  # skip unnamed nodes

            # Nodes have lat/lon directly; ways have a "center" sub-object
            if el["type"] == "node":
                lat, lng = el["lat"], el["lon"]
            else:
                center = el.get("center", {})
                lat = center.get("lat", location.latitude)
                lng = center.get("lon", location.longitude)

            dist = _haversine(location.latitude, location.longitude, lat, lng)

            brand = tags.get("brand", "")
            shop_type = tags.get("shop", "clothes")
            addr = _build_address(tags, location.name)

            competitors.append(
                Competitor(
                    name=name,
                    address=addr,
                    rating=0.0,
                    review_count=0,
                    distance_km=round(dist, 2),
                    place_id=f"osm_{el['type']}_{el['id']}",
                    business_type=shop_type,
                    is_open_now=None,
                )
            )

        competitors.sort(key=lambda c: c.distance_km)
        total = len(competitors)
        logger.info("Overpass: found %d clothing stores near %s", total, location.name)
        return competitors[:25]  # cap at 25

    except Exception as e:
        logger.warning("Overpass error: %s", e)
        return []


def _build_address(tags: dict, fallback_area: str) -> str:
    """Construct a human-readable address from OSM tags."""
    parts = []
    for key in ("addr:full", "addr:street", "addr:housenumber", "addr:city", "addr:postcode"):
        val = tags.get(key)
        if val:
            parts.append(val)
    if parts:
        return ", ".join(parts)
    return f"Near {fallback_area}"


# ── Demo / Fallback Data ────────────────────────────────────

_DEMO_FILE = config.DEMO_DATA_DIR / "sample_competitors.json"


def _demo_geocode(name: str) -> Location:
    """Return a plausible Location for common demo localities."""
    presets: dict[str, tuple[float, float]] = {
        "koramangala": (12.9352, 77.6245),
        "indiranagar": (12.9784, 77.6408),
        "hsr layout": (12.9116, 77.6389),
        "jayanagar": (12.9250, 77.5938),
        "whitefield": (12.9698, 77.7500),
        "bangalore": (12.9716, 77.5946),
        "coimbatore": (11.0168, 76.9558),
        "chennai": (13.0827, 80.2707),
        "mumbai": (19.0760, 72.8777),
        "delhi": (28.6139, 77.2090),
        "hyderabad": (17.3850, 78.4867),
    }
    key = name.lower().strip()
    for k, (lat, lng) in presets.items():
        if k in key:
            return Location(name=name, latitude=lat, longitude=lng, formatted_address=name)
    # Default to Bangalore center
    return Location(name=name, latitude=12.9716, longitude=77.5946, formatted_address=name)


def _demo_competitors(location: Location, radius_km: float) -> list[Competitor]:
    """Load demo competitors from JSON, or generate synthetic data."""
    if _DEMO_FILE.exists():
        raw = json.loads(_DEMO_FILE.read_text(encoding="utf-8"))
        return [Competitor(**c) for c in raw]
    return _generate_synthetic_competitors(location, radius_km)


def _generate_synthetic_competitors(loc: Location, radius_km: float) -> list[Competitor]:
    """Create realistic-looking demo competitors (includes local + national brands)."""
    stores = [
        ("Pothys", 4.4, 3200),
        ("The Chennai Silks", 4.3, 2900),
        ("RMKV Silks", 4.2, 1800),
        ("Saravana Stores", 4.1, 4500),
        ("FabIndia", 4.3, 1820),
        ("Westside", 4.1, 2450),
        ("Zara", 4.4, 3100),
        ("H&M", 4.2, 2780),
        ("Pantaloons", 3.9, 1540),
        ("Max Fashion", 4.0, 1950),
        ("Lifestyle", 4.1, 2100),
        ("Biba", 4.1, 980),
    ]
    competitors = []
    for name, rating, reviews in stores:
        dist = round(random.uniform(0.2, radius_km), 2)
        competitors.append(
            Competitor(
                name=name,
                address=f"Near {loc.name}",
                rating=rating,
                review_count=reviews,
                distance_km=dist,
                business_type="clothing_store",
                is_open_now=random.choice([True, True, False]),
            )
        )
    competitors.sort(key=lambda c: c.distance_km)
    return competitors


# ── Utilities ───────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
