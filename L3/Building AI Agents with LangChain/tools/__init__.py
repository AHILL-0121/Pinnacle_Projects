"""tools/__init__.py – Expose all agent tools from a single import."""

from tools.weather import get_weather  # noqa: F401
from tools.attractions import get_attractions  # noqa: F401

ALL_TOOLS = [get_weather, get_attractions]
