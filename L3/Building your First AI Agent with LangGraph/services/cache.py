"""
Simple on-disk cache using *diskcache* to respect API rate limits
and avoid redundant calls during the same session.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import diskcache

import config

logger = logging.getLogger(__name__)

_cache = diskcache.Cache(str(config.CACHE_DIR), eviction_policy="least-recently-used")


def get(key: str) -> Any | None:
    """Return cached value or None."""
    return _cache.get(key)


def put(key: str, value: Any, ttl: int | None = None) -> None:
    """Store *value* under *key* with an optional TTL (seconds)."""
    _cache.set(key, value, expire=ttl or config.CACHE_TTL_SECONDS)


def make_key(*parts: str) -> str:
    """Produce a stable cache key from arbitrary string parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def clear() -> None:
    """Wipe the entire cache."""
    _cache.clear()
    logger.info("Cache cleared.")
