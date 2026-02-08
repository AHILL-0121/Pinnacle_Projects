"""
Tool: footfall_estimator_tool

Estimates footfall / busy-hour patterns for a set of competitors.

In demo mode the data is synthesised from review counts, ratings, and
heuristic time-of-day distributions.  With live API access the tool would
query Google Maps "Popular Times" metadata.
"""

from __future__ import annotations

import json
import random
from langchain_core.tools import tool

from models.schemas import BusyHourSlot, FootfallEstimate, FootfallLevel


_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_TIME_SLOTS = [
    ("09:00", "11:00"),
    ("11:00", "13:00"),
    ("13:00", "15:00"),
    ("15:00", "17:00"),
    ("17:00", "19:00"),
    ("19:00", "21:00"),
]

# Heuristic weight per day (weekday vs weekend pattern)
_DAY_WEIGHTS = {
    "Monday": 0.6, "Tuesday": 0.5, "Wednesday": 0.65,
    "Thursday": 0.7, "Friday": 0.85,
    "Saturday": 1.0, "Sunday": 0.9,
}

# Heuristic weight per time slot
_SLOT_WEIGHTS = [0.3, 0.6, 0.5, 0.7, 1.0, 0.8]


def _level(score: float) -> FootfallLevel:
    if score >= 55:
        return FootfallLevel.HIGH
    if score >= 30:
        return FootfallLevel.MEDIUM
    return FootfallLevel.LOW


def _estimate_one(name: str, rating: float, review_count: int) -> FootfallEstimate:
    """Generate a plausible footfall estimate for a single competitor."""
    # Popularity proxy: normalise review_count to 0-100 scale
    pop = min(review_count / 35.0, 100)
    rating_boost = (rating - 3.0) * 10  # ±20 range

    peak_hours: list[BusyHourSlot] = []
    day_scores: dict[str, float] = {}

    for day in _DAYS:
        day_w = _DAY_WEIGHTS[day]
        total_day = 0.0
        for (start, end), slot_w in zip(_TIME_SLOTS, _SLOT_WEIGHTS):
            base = pop * day_w * slot_w + rating_boost
            noise = random.uniform(-5, 5)
            score = max(0, min(100, base + noise))
            total_day += score
            peak_hours.append(
                BusyHourSlot(
                    day=day,
                    hour_start=start,
                    hour_end=end,
                    footfall_level=_level(score),
                    relative_busyness=int(score),
                )
            )
        day_scores[day] = total_day

    busiest = max(day_scores, key=day_scores.get)  # type: ignore[arg-type]
    quietest = min(day_scores, key=day_scores.get)  # type: ignore[arg-type]
    avg_score = sum(day_scores.values()) / len(day_scores)
    avg_level = _level(avg_score / len(_TIME_SLOTS))

    # Keep only the 3 highest-traffic slots for brevity
    top_peaks = sorted(peak_hours, key=lambda s: s.relative_busyness, reverse=True)[:6]

    return FootfallEstimate(
        competitor_name=name,
        peak_hours=top_peaks,
        average_footfall=avg_level,
        busiest_day=busiest,
        quietest_day=quietest,
        weekly_pattern_summary=(
            f"{name} sees {avg_level.value} average footfall. "
            f"Busiest on {busiest}, quietest on {quietest}. "
            f"Evening hours (17:00–19:00) tend to be peak."
        ),
    )


@tool
def footfall_estimator_tool(location_name: str = "") -> str:
    """Estimate footfall and busy-hour patterns for competitors.

    Call this AFTER competitor_fetch_tool. It automatically uses the
    competitor data from the most recent search.

    Args:
        location_name: Optional location to get footfall for.
                       If empty, uses the last searched location.

    Returns:
        A formatted Markdown table of footfall estimates with peak hours,
        busiest/quietest days, and a summary for each store.
    """
    from tools._store import get_search, get_last_location, save_search

    data = get_search(location_name)
    if not data:
        loc = get_last_location()
        if loc:
            return f"No competitor data found for '{location_name}'. The last search was for '{loc}'. Try that instead."
        return "No competitor data available. Please call competitor_fetch_tool first."

    competitors = data.get("competitors", [])
    if not competitors:
        return f"No competitors were found for '{data.get('search_area', location_name)}'. Nothing to estimate."

    estimates = []
    for c in competitors:
        # OSM data has no ratings/reviews; use distance as a proxy for popularity
        # Closer to center = likely busier commercial area
        distance = c.get("distance_km", 5.0)
        # Aggressive distance-based scaling: very close stores are busy
        synth_reviews = max(300, int(5000 / (1 + distance * 1.5)) + random.randint(-400, 400))
        synth_rating = round(min(4.8, random.uniform(3.5, 4.5) + max(0, 0.5 - distance * 0.1)), 1)

        est = _estimate_one(
            name=c.get("name", "Unknown"),
            rating=c.get("rating") or synth_rating,
            review_count=c.get("review_count") or synth_reviews,
        )
        estimates.append(est.model_dump(mode="json"))

    # Save estimates back into the store for report generation
    data["footfall_estimates"] = estimates
    save_search(data.get("search_area", location_name), data)

    # Format as Markdown
    area = data.get("search_area", location_name)
    lines = [
        f"### Footfall Estimates for stores near {area}",
        "",
        "| # | Store | Avg Footfall | Busiest Day | Quietest Day |",
        "|---|-------|-------------|-------------|-------------|",
    ]
    for i, est in enumerate(estimates, 1):
        avg = est['average_footfall']
        if hasattr(avg, 'value'):
            avg = avg.value
        lines.append(
            f"| {i} | {est['competitor_name']} | {avg} | "
            f"{est['busiest_day']} | {est['quietest_day']} |"
        )

    lines.append("")
    lines.append("#### Peak Hours (top stores)")
    lines.append("")

    for est in estimates[:8]:  # Top 8 for brevity
        lines.append(f"**{est['competitor_name']}:** {est['weekly_pattern_summary']}")
        if est.get("peak_hours"):
            lines.append("")
            lines.append("| Day | Time | Busyness | Level |")
            lines.append("|-----|------|----------|-------|")
            for slot in est["peak_hours"][:4]:
                bar = chr(9608) * (slot['relative_busyness'] // 10)
                level = slot['footfall_level']
                if hasattr(level, 'value'):
                    level = level.value
                lines.append(
                    f"| {slot['day']} | {slot['hour_start']}-{slot['hour_end']} | "
                    f"{bar} {slot['relative_busyness']}% | {level} |"
                )
            lines.append("")

    lines.append("*Note: Footfall data is estimated based on heuristic indicators.*")
    return "\n".join(lines)
