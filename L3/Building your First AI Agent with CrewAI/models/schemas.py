"""
Data models for the Logistics Optimization System.

Matches SRS Section 11 - Data Model (MVP):
- Product
- Route
- Inventory
- LogisticsData (composite input)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class Product:
    """A product in the logistics pipeline."""
    product_id: str
    name: str
    category: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Route:
    """A delivery route between locations."""
    route_id: str
    origin: str
    destination: str
    distance_km: float
    delivery_time_hr: float
    product_ids: List[str] = field(default_factory=list)
    fuel_cost_usd: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Inventory:
    """Inventory record for a product at a warehouse."""
    product_id: str
    warehouse: str
    stock_level: int
    turnover_rate: float               # units sold / avg inventory per period
    reorder_point: Optional[int] = None
    holding_cost_per_unit_usd: Optional[float] = None  # cost to hold ONE unit/month

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LogisticsData:
    """
    Composite input for the analysis pipeline.
    Parametrized per SRS FR-1 / FR-2.
    """
    products: List[Product]
    routes: List[Route]
    inventory: List[Inventory]

    def to_dict(self) -> dict:
        return {
            "products": [p.to_dict() for p in self.products],
            "routes": [r.to_dict() for r in self.routes],
            "inventory": [i.to_dict() for i in self.inventory],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "LogisticsData":
        products = [Product(**p) for p in data.get("products", [])]
        routes = [Route(**r) for r in data.get("routes", [])]
        inventory = [Inventory(**i) for i in data.get("inventory", [])]
        return cls(products=products, routes=routes, inventory=inventory)

    @classmethod
    def from_json_file(cls, path: str) -> "LogisticsData":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
