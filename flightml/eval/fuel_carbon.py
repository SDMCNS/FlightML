from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.geo.trajectory import analyze_trajectory
from flightml.eval.jet_fuel_prices import get_weekly_fuel_price
from flightml.eval.aircraft_engine_perf import (
    AircraftEnginePerformance,
    CruiseAltitudeEfficiency,
    compute_cruise_altitude_efficiency,
    resolve_aircraft_engine_perf,
)

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
    estimated_excess_fuel_cost_eur: float
    fuel_price_eur_per_kg: float = 0.78
    fuel_price_eur_per_gallon: float = 2.38
    estimated_excess_fuel_cost_usd: float = 0.0
    fuel_price_usd_per_kg: float = 0.85
    fuel_price_usd_per_gallon: float = 2.58
    fuel_price_week: str = "N/A"
    fuel_price_source: str = "EIA Weekly Spot Price (EUR Converted)"
    # Aircraft & Engine Cruise Altitude Performance
    selected_engine: str = "DEFAULT"
    engine_options: list[str] = field(default_factory=list)
    is_engine_override: bool = False
    engine_type_class: str = "Turbofan"
    bypass_ratio: float = 6.0
    pressure_ratio: float = 30.0
    optimal_cruise_fl: float = 360.0
    requested_fl: float = 360.0
    actual_cruise_fl: float = 360.0
    cruise_hours: float = 1.0
    fl_delta_actual_vs_optimal: float = 0.0
    fl_delta_requested_vs_optimal: float = 0.0
    fl_delta_actual_vs_requested: float = 0.0
    actual_burn_penalty_pct: float = 0.0
    requested_burn_penalty_pct: float = 0.0
    excess_altitude_fuel_kg: float = 0.0
    excess_altitude_fuel_cost_eur: float = 0.0
    planning_excess_fuel_kg: float = 0.0
    planning_excess_fuel_cost_eur: float = 0.0
    atc_tactical_excess_fuel_kg: float = 0.0
    atc_tactical_excess_fuel_cost_eur: float = 0.0
    potential_fuel_savings_kg: float = 0.0
    potential_fuel_savings_eur: float = 0.0
    potential_fuel_savings_pct: float = 0.0

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
            "excess_altitude_fuel_kg": round(self.excess_altitude_fuel_kg, 1),
            "excess_altitude_fuel_cost_eur": round(self.excess_altitude_fuel_cost_eur, 2),
            "total_excess_fuel_kg": round(self.total_excess_fuel_kg, 1),
            "excess_co2_kg": round(self.excess_co2_kg, 1),
            "estimated_excess_fuel_cost_eur": round(self.estimated_excess_fuel_cost_eur, 2),
            "fuel_price_eur_per_kg": round(self.fuel_price_eur_per_kg, 4),
            "fuel_price_eur_per_gallon": round(self.fuel_price_eur_per_gallon, 3),
            "fuel_price_week": self.fuel_price_week,
            "fuel_price_source": self.fuel_price_source,
            # Engine & Altitude Efficiency Decomposition
            "selected_engine": self.selected_engine,
            "engine_options": self.engine_options,
            "is_engine_override": self.is_engine_override,
            "engine_type_class": self.engine_type_class,
            "bypass_ratio": round(self.bypass_ratio, 2),
            "pressure_ratio": round(self.pressure_ratio, 1),
            "optimal_cruise_fl": round(self.optimal_cruise_fl, 1),
            "requested_fl": round(self.requested_fl, 1),
            "actual_cruise_fl": round(self.actual_cruise_fl, 1),
            "cruise_hours": round(self.cruise_hours, 2),
            "fl_delta_actual_vs_optimal": round(self.fl_delta_actual_vs_optimal, 1),
            "fl_delta_requested_vs_optimal": round(self.fl_delta_requested_vs_optimal, 1),
            "fl_delta_actual_vs_requested": round(self.fl_delta_actual_vs_requested, 1),
            "actual_burn_penalty_pct": round(self.actual_burn_penalty_pct, 2),
            "requested_burn_penalty_pct": round(self.requested_burn_penalty_pct, 2),
            "planning_excess_fuel_kg": round(self.planning_excess_fuel_kg, 1),
            "planning_excess_fuel_cost_eur": round(self.planning_excess_fuel_cost_eur, 2),
            "atc_tactical_excess_fuel_kg": round(self.atc_tactical_excess_fuel_kg, 1),
            "atc_tactical_excess_fuel_cost_eur": round(self.atc_tactical_excess_fuel_cost_eur, 2),
            "potential_fuel_savings_kg": round(self.potential_fuel_savings_kg, 1),
            "potential_fuel_savings_eur": round(self.potential_fuel_savings_eur, 2),
            "potential_fuel_savings_pct": round(self.potential_fuel_savings_pct, 2),
        }


def compute_flight_fuel_and_emissions(
    dataset: FlightDataset,
    ectrl_id: int,
    fuel_price_per_kg: float | None = None,
    engine_override: str | None = None,
) -> EnvironmentalImpact:
    """
    Compute excess fuel burn, CO2 emissions, and Euro (€) cost caused by horizontal route extensions,
    terminal holding patterns, and suboptimal cruise flight level (altitude deviation vs. ideal aircraft/engine FL).
    Automatically resolves the official EIA weekly jet fuel spot price (in EUR) for the flight's departure date.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        raise ValueError(f"Flight {ectrl_id} not found")

    # Resolve dynamic weekly jet fuel spot price in EUR from EIA if not explicitly overridden
    if fuel_price_per_kg is None:
        price_rec = get_weekly_fuel_price(
            flight.get("actual_off_block_time") or flight.get("filed_off_block_time"),
            period_hint=dataset.period,
        )
        active_price_kg_eur = price_rec.price_eur_per_kg
        price_gal_eur = price_rec.price_eur_per_gallon
        active_price_kg_usd = price_rec.price_usd_per_kg
        price_gal_usd = price_rec.price_usd_per_gallon
        price_week = f"{price_rec.year}-W{price_rec.week_number:02d} ({price_rec.week_start} to {price_rec.week_end})"
        price_source = price_rec.source
    else:
        active_price_kg_eur = float(fuel_price_per_kg)
        price_gal_eur = round(active_price_kg_eur * 3.043, 3)
        active_price_kg_usd = round(active_price_kg_eur / 0.925, 4)
        price_gal_usd = round(price_gal_eur / 0.925, 3)
        price_week = "User Override"
        price_source = "User Specified Parameter (EUR)"

    ac_type = str(flight.get("ac_type") or "DEFAULT_JET").upper()
    
    # Resolve aircraft & engine combination performance and altitude efficiency
    alt_eff = compute_cruise_altitude_efficiency(
        dataset,
        ectrl_id,
        engine_override=engine_override,
        fuel_price_per_kg=active_price_kg_eur,
    )
    eng_perf = resolve_aircraft_engine_perf(dataset, ac_type, engine_override=engine_override)
    fuel_flow = eng_perf.nominal_fuel_flow_kg_hr

    trajs = dataset.get_trajectory(ectrl_id)
    metrics = analyze_trajectory(trajs["actual"], trajs["planned"])

    # 1. Fuel burn in holding loops
    # Holding speed ~ 210-250 kts; fuel flow in holding ~ 85% of nominal cruise
    holding_hours = metrics.total_holding_duration_min / 60.0
    holding_fuel_kg = holding_hours * (fuel_flow * 0.85)

    # 2. Fuel burn from route extension (actual distance - planned distance)
    excess_dist_nm = max(0.0, metrics.actual_distance_nm - metrics.planned_distance_nm)
    excess_cruise_hours = excess_dist_nm / 450.0
    routing_fuel_kg = excess_cruise_hours * fuel_flow

    # 3. Fuel burn from suboptimal cruising altitude (vertical profile inefficiency vs ideal FL)
    altitude_fuel_kg = alt_eff.excess_altitude_fuel_kg

    total_excess_fuel = holding_fuel_kg + routing_fuel_kg + altitude_fuel_kg
    excess_co2 = total_excess_fuel * CO2_FACTOR
    excess_cost_eur = total_excess_fuel * active_price_kg_eur
    excess_cost_usd = total_excess_fuel * active_price_kg_usd

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
        estimated_excess_fuel_cost_eur=excess_cost_eur,
        fuel_price_eur_per_kg=active_price_kg_eur,
        fuel_price_eur_per_gallon=price_gal_eur,
        estimated_excess_fuel_cost_usd=excess_cost_usd,
        fuel_price_usd_per_kg=active_price_kg_usd,
        fuel_price_usd_per_gallon=price_gal_usd,
        fuel_price_week=price_week,
        fuel_price_source=price_source,
        # Extended engine & cruise altitude efficiency
        selected_engine=eng_perf.selected_engine,
        engine_options=eng_perf.engine_options,
        is_engine_override=eng_perf.is_engine_override,
        engine_type_class=eng_perf.engine_type_class,
        bypass_ratio=eng_perf.bypass_ratio,
        pressure_ratio=eng_perf.pressure_ratio,
        optimal_cruise_fl=alt_eff.optimal_cruise_fl,
        requested_fl=alt_eff.requested_fl,
        actual_cruise_fl=alt_eff.actual_cruise_fl,
        cruise_hours=alt_eff.cruise_hours,
        fl_delta_actual_vs_optimal=alt_eff.fl_delta_actual_vs_optimal,
        fl_delta_requested_vs_optimal=alt_eff.fl_delta_requested_vs_optimal,
        fl_delta_actual_vs_requested=alt_eff.fl_delta_actual_vs_requested,
        actual_burn_penalty_pct=alt_eff.actual_burn_penalty_pct,
        requested_burn_penalty_pct=alt_eff.requested_burn_penalty_pct,
        excess_altitude_fuel_kg=alt_eff.excess_altitude_fuel_kg,
        excess_altitude_fuel_cost_eur=alt_eff.excess_altitude_fuel_cost_eur,
        planning_excess_fuel_kg=alt_eff.planning_excess_fuel_kg,
        planning_excess_fuel_cost_eur=alt_eff.planning_excess_fuel_cost_eur,
        atc_tactical_excess_fuel_kg=alt_eff.atc_tactical_excess_fuel_kg,
        atc_tactical_excess_fuel_cost_eur=alt_eff.atc_tactical_excess_fuel_cost_eur,
        potential_fuel_savings_kg=alt_eff.potential_fuel_savings_kg,
        potential_fuel_savings_eur=alt_eff.potential_fuel_savings_eur,
        potential_fuel_savings_pct=alt_eff.potential_fuel_savings_pct,
    )

