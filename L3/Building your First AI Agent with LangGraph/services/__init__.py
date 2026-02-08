from .places_service import geocode, search_nearby_competitors
from .cache import get as cache_get, put as cache_put, make_key, clear as cache_clear


def get_llm(*args, **kwargs):
    """Lazy import wrapper to avoid heavy transitive imports at module load."""
    from .llm_service import get_llm as _get_llm
    return _get_llm(*args, **kwargs)


__all__ = [
    "get_llm",
    "geocode",
    "search_nearby_competitors",
    "cache_get",
    "cache_put",
    "make_key",
    "cache_clear",
]
