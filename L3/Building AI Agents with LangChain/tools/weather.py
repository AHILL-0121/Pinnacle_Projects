"""
tools/weather.py – Real-time weather retrieval via WeatherAPI.com

LangChain @tool wrapper so the agent can call it autonomously.
"""

import requests
from langchain_core.tools import tool

from config import WEATHER_API_KEY, WEATHER_API_BASE


@tool
def get_weather(city: str) -> str:
    """Fetch current weather for a city.

    Returns temperature, condition, humidity, and wind speed.
    Use this tool when the user asks about weather or climate
    for a specific destination.

    Args:
        city: The name of the city (e.g. "Paris", "Tokyo").
    """
    if not WEATHER_API_KEY:
        return (
            "⚠️  Weather API key is not configured. "
            "Please set WEATHER_API_KEY in your .env file. "
            "Get a free key at https://www.weatherapi.com/"
        )

    try:
        response = requests.get(
            WEATHER_API_BASE,
            params={"key": WEATHER_API_KEY, "q": city, "aqi": "no"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        location = data["location"]
        current = data["current"]

        return (
            f"📍 Weather for {location['name']}, {location['country']}:\n"
            f"  🌡️  Temperature : {current['temp_c']}°C / {current['temp_f']}°F\n"
            f"  ☁️  Condition   : {current['condition']['text']}\n"
            f"  💧 Humidity    : {current['humidity']}%\n"
            f"  💨 Wind        : {current['wind_kph']} km/h {current['wind_dir']}\n"
            f"  🌡️  Feels like  : {current['feelslike_c']}°C"
        )

    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 400:
            return f"❌ City '{city}' not found. Please check the spelling."
        return f"❌ Weather API error: {exc}"
    except requests.exceptions.ConnectionError:
        return "❌ Could not connect to the Weather API. Check your internet."
    except Exception as exc:  # noqa: BLE001
        return f"❌ Unexpected error fetching weather: {exc}"
