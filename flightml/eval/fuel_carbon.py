from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.geo.trajectory import analyze_trajectory
from flightml.eval.jet_fuel_prices import get_weekly_fuel_price

# Approximate nominal cruise & holding fuel flow rates (kg / hour) by aircraft category
DEFAULT_FUEL_FLOW_KG_HR = {
    # Narrowbody
    "A320": 2400.0,
    "A321": 2700.0,
    "A319": 2250.0,
    "B738": 2450.0,
    "B737": 2400.0,
    "B739": 2600.0,
    # Widebody
    "B77W": 6800.0,
    "B772": 6200.0,
    "B763": 4800.0,
    "A332": 5200.0,
    "A333": 5600.0,
    "A359": 5400.0,
    "A388": 11500.0,
    "B744": 10200.0,
    "B788": 4900.0,
    "B789": 5300.0,
    # Regional Turboprop / Jet
    "DH8D": 620.0,
    "AT72": 580.0,
    "AT76": 600.0,
    "E190": 1750.0,
    "CRJ9": 1600.0,
    "F50": 600.0,
    # Default fallback
    "DEFAULT_JET": 2500.0,
}

# Standard ICAO carbon emission factor: 3.16 kg CO2 per kg Jet-A fuel
CO2_FACTOR = 3.16


@dataclass
class EnvironmentalImpact:
    ectrl_id: int
    ac_type: str
    planned_distance_nm: float
    actual_distance_nm: float
    excess_distance_nm: float
    holding_duration_min: float
    nominal_fuel_flow_kg_hr: float
    excess_holding_fuel_kg: float
    excess_routing_fuel_kg: float
    total_excess_fuel_kg: float
    excess_co2_kg: float
    estimated_excess_fuel_cost_usd: float
    fuel_price_usd_per_kg: float = 0.85
    fuel_price_usd_per_gallon: float = 2.58
    fuel_price_week: str = "N/A"
    fuel_price_source: str = "EIA Weekly Spot Price (USGC/IATA)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ectrl_id": self.ectrl_id,
            "ac_type": self.ac_type,
            "planned_distance_nm": round(self.planned_distance_nm, 1),
            "actual_distance_nm": round(self.actual_distance_nm, 1),
            "excess_distance_nm": round(self.excess_distance_nm, 1),
            "holding_duration_min": round(self.holding_duration_min, 1),
            "nominal_fuel_flow_kg_hr": self.nominal_fuel_flow_kg_hr,
            "excess_holding_fuel_kg": round(self.excess_holding_fuel_kg, 1),
            "excess_routing_fuel_kg": round(self.excess_routing_fuel_kg, 1),
            "total_excess_fuel_kg": round(self.total_excess_fuel_kg, 1),
            "excess_co2_kg": round(self.excess_co2_kg, 1),
            "estimated_excess_fuel_cost_usd": round(self.estimated_excess_fuel_cost_usd, 2),
            "fuel_price_usd_per_kg": round(self.fuel_price_usd_per_kg, 4),
            "fuel_price_usd_per_gallon": round(self.fuel_price_usd_per_gallon, 3),
            "fuel_price_week": self.fuel_price_week,
            "fuel_price_source": self.fuel_price_source,
        }


def compute_flight_fuel_and_emissions(
    dataset: FlightDataset,
    ectrl_id: int,
    fuel_price_per_kg: float | None = None,
) -> EnvironmentalImpact:
    """
    Compute excess fuel burn and CO2 emissions caused by horizontal route extensions
    and terminal holding patterns.
    Automatically resolves the official EIA weekly jet fuel spot price for the flight's departure date.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        raise ValueError(f"Flight {ectrl_id} not found")

    # Resolve dynamic weekly jet fuel spot price from EIA if not explicitly overridden
    if fuel_price_per_kg is None:
        price_rec = get_weekly_fuel_price(
            flight.get("actual_off_block_time") or flight.get("filed_off_block_time"),
            period_hint=dataset.period,
        )
        active_price_kg = price_rec.price_usd_per_kg
        price_gal = price_rec.price_usd_per_gallon
        price_week = f"{price_rec.year}-W{price_rec.week_number:02d} ({price_rec.week_start} to {price_rec.week_end})"
        price_source = price_rec.source
    else:
        active_price_kg = float(fuel_price_per_kg)
        price_gal = round(active_price_kg * 3.043, 3)
        price_week = "User Override"
        price_source = "User Specified Parameter"

    ac_type = str(flight.get("ac_type") or "DEFAULT_JET").upper()
    fuel_flow = DEFAULT_FUEL_FLOW_KG_HR.get(ac_type, DEFAULT_FUEL_FLOW_KG_HR["DEFAULT_JET"])

    trajs = dataset.get_trajectory(ectrl_id)
    metrics = analyze_trajectory(trajs["actual"], trajs["planned"])

    # 1. Fuel burn in holding loops
    # Holding speed ~ 210-250 kts; fuel flow in holding ~ 85% of nominal cruise
    holding_hours = metrics.total_holding_duration_min / 60.0
    holding_fuel_kg = holding_hours * (fuel_flow * 0.85)

    # 2. Fuel burn from route extension (actual distance - planned distance)
    excess_dist_nm = max(0.0, metrics.actual_distance_nm - metrics.planned_distance_nm)
    # Average ground speed ~ 450 kts
    excess_cruise_hours = excess_dist_nm / 450.0
    routing_fuel_kg = excess_cruise_hours * fuel_flow

    total_excess_fuel = holding_fuel_kg + routing_fuel_kg
    excess_co2 = total_excess_fuel * CO2_FACTOR
    excess_cost = total_excess_fuel * active_price_kg

    return EnvironmentalImpact(
        ectrl_id=ectrl_id,
        ac_type=ac_type,
        planned_distance_nm=metrics.planned_distance_nm,
        actual_distance_nm=metrics.actual_distance_nm,
        excess_distance_nm=excess_dist_nm,
        holding_duration_min=metrics.total_holding_duration_min,
        nominal_fuel_flow_kg_hr=fuel_flow,
        excess_holding_fuel_kg=holding_fuel_kg,
        excess_routing_fuel_kg=routing_fuel_kg,
        total_excess_fuel_kg=total_excess_fuel,
        excess_co2_kg=excess_co2,
        estimated_excess_fuel_cost_usd=excess_cost,
        fuel_price_usd_per_kg=active_price_kg,
        fuel_price_usd_per_gallon=price_gal,
        fuel_price_week=price_week,
        fuel_price_source=price_source,
    )

