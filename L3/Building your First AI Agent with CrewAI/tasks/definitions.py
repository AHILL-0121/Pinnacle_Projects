"""
Task definitions for the Logistics Optimization System.

SRS Section 6 — Two sequential tasks:
1. Logistics Analysis Task  → assigned to Logistics Analyst
2. Optimization Strategy Task → assigned to Optimization Strategist

Key design decisions (post-review fixes):
- All numeric metrics are PRE-COMPUTED in Python to prevent LLM hallucination.
- The LLM is explicitly told not to recalculate any numbers.
- Holding cost uses a per-unit basis to remove ambiguity.
- Threshold: < 1.0 overstocked, > 6.0 stockout risk, = 6.0 borderline.
"""

from crewai import Agent, Task
from models.schemas import LogisticsData


# ── FIX 1: Pre-compute all metrics in Python ────────────────────────

def _precompute_route_metrics(logistics_data: LogisticsData) -> str:
    """Compute speed, cost/km, and detect overlaps — return formatted text."""
    lines: list[str] = []
    speeds: list[float] = []
    cost_per_kms: dict[str, float] = {}
    dest_map: dict[str, list[str]] = {}  # "origin→dest" → [route_ids]

    for r in logistics_data.routes:
        speed = round(r.distance_km / r.delivery_time_hr, 2)
        cpk = round(r.fuel_cost_usd / r.distance_km, 2) if r.fuel_cost_usd else 0
        speeds.append(speed)
        cost_per_kms[r.route_id] = cpk

        key = f"{r.origin} → {r.destination}"
        dest_map.setdefault(key, []).append(r.route_id)

        lines.append(
            f"  - {r.route_id}: {r.origin} → {r.destination} | "
            f"Distance: {r.distance_km} km | Time: {r.delivery_time_hr} hr | "
            f"Speed: {speed} km/hr | Cost/km: ${cpk} | "
            f"Products: {', '.join(r.product_ids)}"
        )

    avg_speed = round(sum(speeds) / len(speeds), 2) if speeds else 0
    worst_route = max(cost_per_kms, key=cost_per_kms.get) if cost_per_kms else "N/A"

    overlap_lines: list[str] = []
    for key, rids in dest_map.items():
        if len(rids) > 1:
            overlap_lines.append(f"  - OVERLAP: {key} served by routes {', '.join(rids)}")

    section = "PRE-COMPUTED ROUTE METRICS:\n" + "\n".join(lines)
    section += f"\n\n  Average route speed: {avg_speed} km/hr"
    section += f"\n  Highest cost-per-km route: {worst_route} (${cost_per_kms.get(worst_route, 0)}/km)"
    if overlap_lines:
        section += "\n\n  OVERLAPPING ROUTES:\n" + "\n".join(overlap_lines)
    return section


def _precompute_inventory_metrics(logistics_data: LogisticsData) -> str:
    """Compute excess stock, waste, and classify each product — return formatted text."""
    product_names = {p.product_id: p.name for p in logistics_data.products}
    lines: list[str] = []
    total_waste = 0.0
    most_overstocked_id = None
    most_overstocked_excess = 0

    for inv in logistics_data.inventory:
        name = product_names.get(inv.product_id, inv.product_id)
        rp = inv.reorder_point or 0
        hcpu = inv.holding_cost_per_unit_usd or 0

        excess = round(inv.stock_level - (inv.turnover_rate * rp), 2)
        waste = round(max(excess, 0) * hcpu, 2)

        # FIX 4: Threshold logic — strict inequalities, = 6.0 is borderline
        if inv.turnover_rate < 1.0:
            classification = "OVERSTOCKED / slow-moving"
        elif inv.turnover_rate > 6.0:
            classification = "STOCKOUT RISK / fast-moving"
        elif inv.turnover_rate == 6.0:
            classification = "BORDERLINE — monitor closely"
        else:
            classification = "Normal"

        total_waste += waste
        if excess > most_overstocked_excess:
            most_overstocked_excess = excess
            most_overstocked_id = inv.product_id

        lines.append(
            f"  - {inv.product_id} ({name}): Classification: {classification} | "
            f"Turnover: {inv.turnover_rate} | Stock: {inv.stock_level} | "
            f"Excess stock: {excess} units | "
            f"Holding cost/unit: ${hcpu} | Waste: ${waste}/month"
        )

    total_waste = round(total_waste, 2)
    annual_waste = round(total_waste * 12, 2)
    section = "PRE-COMPUTED INVENTORY METRICS:\n" + "\n".join(lines)
    section += f"\n\n  Most overstocked product: {most_overstocked_id} ({most_overstocked_excess} excess units)"
    section += f"\n  Total monthly holding cost waste: ${total_waste}"
    section += f"\n  Total annualized holding cost waste: ${annual_waste}"
    return section


# ─────────────────────────────────────────────────────────────────────


def create_analysis_task(
    agent: Agent,
    logistics_data: LogisticsData,
) -> Task:
    """
    Task 1 — Logistics Analysis

    All numeric metrics are pre-computed in Python so the LLM only
    interprets and structures them — never recalculates.
    """
    route_metrics = _precompute_route_metrics(logistics_data)
    inventory_metrics = _precompute_inventory_metrics(logistics_data)

    description = f"""\
You are given PRE-COMPUTED logistics metrics below. Your job is to interpret
them and produce a structured analysis report.

⚠️ CRITICAL: Do NOT recalculate any numbers. Use ONLY the pre-computed values
provided below. Every number in your output must match exactly.

=== {route_metrics} ===

=== {inventory_metrics} ===

**Classification thresholds (already applied above):**
- Turnover < 1.0 → OVERSTOCKED / slow-moving
- Turnover > 6.0 → STOCKOUT RISK / fast-moving
- Turnover = 6.0 → BORDERLINE — monitor closely
- Routes with speed < 80 km/hr → flag as slow / possible detour
- Routes with cost/km > $0.45 → flag as cost-inefficient

⚠️ IMPORTANT: List ALL routes and products violating any threshold.
Do NOT summarize, group, or omit entries. Every single violation must appear.

**Your report MUST follow this EXACT format:**

ROUTE INEFFICIENCIES:
(List EVERY route where speed < 80 km/hr OR cost/km > $0.45. Do NOT skip any.)
- [Route ID]: [Issue description] | Speed: [use pre-computed] km/hr | Cost/km: $[use pre-computed]

OVERLAPPING ROUTES:
(List any origin→destination pairs served by multiple routes. Flag as consolidation opportunity.)
- [Route IDs]: [Origin] → [Destination] — consolidation candidate

INVENTORY INEFFICIENCIES:
- [Product ID] ([Name]): [Classification from above] | Turnover: [exact] | Excess stock: [exact] units | Waste: $[exact]/month

KEY METRICS:
- Average route speed: [use pre-computed] km/hr
- Most cost-inefficient route: [use pre-computed]
- Most overstocked product: [use pre-computed]
- Total estimated monthly holding cost waste: $[use pre-computed]
- Total estimated annual holding cost waste: $[use pre-computed]

OBSERVATIONS:
- [Numbered list of 3-5 top-level observations based on the data]
"""

    return Task(
        description=description,
        expected_output=(
            "A structured logistics analysis report with sections: "
            "ROUTE INEFFICIENCIES (exhaustive — every violation listed), "
            "OVERLAPPING ROUTES, INVENTORY INEFFICIENCIES, KEY METRICS, "
            "and OBSERVATIONS. All numbers must match the pre-computed values exactly."
        ),
        agent=agent,
    )


def create_optimization_task(
    agent: Agent,
    analysis_task: Task,
    logistics_data: LogisticsData,
) -> Task:
    """
    Task 2 — Optimization Strategy

    Consumes the Logistics Analyst's structured insights and generates
    prioritized, actionable optimization recommendations.
    """
    product_names = ", ".join(p.name for p in logistics_data.products)

    # Pre-compute total annual waste so the strategist doesn't hallucinate it
    product_name_map = {p.product_id: p.name for p in logistics_data.products}
    monthly_waste = 0.0
    for inv in logistics_data.inventory:
        rp = inv.reorder_point or 0
        hcpu = inv.holding_cost_per_unit_usd or 0
        excess = inv.stock_level - (inv.turnover_rate * rp)
        monthly_waste += max(excess, 0) * hcpu
    monthly_waste = round(monthly_waste, 2)
    annual_waste = round(monthly_waste * 12, 2)

    description = f"""\
Using the Logistics Analyst's findings, develop a comprehensive optimization
strategy for the following products: {product_names}.

⚠️ CRITICAL: Do NOT invent or recalculate any numbers. Use ONLY the figures
from the Analyst's report and the reference values below.

**Reference values (pre-computed):**
- Total monthly holding cost waste: ${monthly_waste}
- Total annualized holding cost waste: ${annual_waste}

Your strategy MUST address every inefficiency identified in the analysis.

**Deliverables:**

1. **ROUTE OPTIMIZATION STRATEGIES**
   For EVERY route inefficiency AND overlapping route pair from the analysis:
   - Recommended action (consolidate, re-route, schedule change, etc.)
   - For overlapping routes: propose consolidation with expected savings
   - Implementation steps (1-2-3 format)
   - Expected impact (cost saving %, time saving %)
   - Priority: HIGH / MEDIUM / LOW

2. **INVENTORY OPTIMIZATION STRATEGIES**
   For each inventory inefficiency found:
   - Recommended action (reduce stock, increase reorder frequency, liquidate, etc.)
   - Implementation steps
   - Expected impact (holding cost reduction $/month, stockout risk change)
   - Priority: HIGH / MEDIUM / LOW

3. **QUICK WINS** (can be implemented within 1 week)
   - List 2-3 easiest optimizations with highest ROI

4. **SUMMARY**
   - Total estimated annual savings (must not exceed ${annual_waste} for inventory items)
   - Top 3 priorities
   - Next steps for implementation

**Output format:** Structured text with clear headers and bullet points.
"""

    return Task(
        description=description,
        expected_output=(
            "A prioritized optimization strategy report with sections: "
            "ROUTE OPTIMIZATION STRATEGIES, INVENTORY OPTIMIZATION STRATEGIES, "
            "QUICK WINS, and SUMMARY. Each recommendation includes implementation "
            "steps and expected impact metrics. All numbers must be grounded in "
            "the Analyst's pre-computed figures."
        ),
        agent=agent,
        context=[analysis_task],   # chain: receives Task 1 output
    )
