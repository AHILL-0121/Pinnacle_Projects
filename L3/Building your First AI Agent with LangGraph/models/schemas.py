"""Pydantic data models for the Competitor Intelligence System."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────

class FootfallLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class AnalysisIntent(str, Enum):
    LIST = "list"
    COMPARE = "compare"
    REPORT = "report"
    INSIGHT = "insight"


# ── Core Models ─────────────────────────────────────────────

class Location(BaseModel):
    """Represents a geographic location."""
    name: str = Field(description="Human-readable location name")
    latitude: float = Field(description="Latitude coordinate")
    longitude: float = Field(description="Longitude coordinate")
    formatted_address: str = Field(default="", description="Full address string")


class BusyHourSlot(BaseModel):
    """A time slot with its estimated footfall level."""
    day: str = Field(description="Day of week (e.g., Monday)")
    hour_start: str = Field(description="Start hour (e.g., 10:00)")
    hour_end: str = Field(description="End hour (e.g., 12:00)")
    footfall_level: FootfallLevel = Field(description="Estimated footfall level")
    relative_busyness: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Busyness percentage (0-100)",
    )


class Competitor(BaseModel):
    """A competing clothing store."""
    name: str = Field(description="Store name")
    address: str = Field(description="Full address")
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="Google rating")
    review_count: int = Field(default=0, ge=0, description="Number of reviews")
    distance_km: float = Field(default=0.0, ge=0.0, description="Distance from query location (km)")
    phone: str = Field(default="", description="Phone number")
    place_id: str = Field(default="", description="Google Place ID")
    business_type: str = Field(default="clothing_store", description="Business type")
    is_open_now: Optional[bool] = Field(default=None, description="Whether the store is currently open")


class FootfallEstimate(BaseModel):
    """Footfall and busy-hour data for a competitor."""
    competitor_name: str
    peak_hours: list[BusyHourSlot] = Field(default_factory=list)
    average_footfall: FootfallLevel = FootfallLevel.MEDIUM
    busiest_day: str = Field(default="Saturday")
    quietest_day: str = Field(default="Tuesday")
    weekly_pattern_summary: str = Field(default="")


class CompetitorAnalysis(BaseModel):
    """Full analysis output for a set of competitors."""
    location_queried: str
    radius_km: float
    competitors: list[Competitor] = Field(default_factory=list)
    footfall_estimates: list[FootfallEstimate] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    report_markdown: str = Field(default="")


# ── Query Parsing ───────────────────────────────────────────

class ParsedQuery(BaseModel):
    """Extracted intent and parameters from a user's natural language query."""
    raw_query: str
    location: str = Field(default="")
    business_type: str = Field(default="clothing_store")
    intent: AnalysisIntent = AnalysisIntent.LIST
    radius_km: float = Field(default=2.0)
