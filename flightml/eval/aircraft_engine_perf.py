"""
Aircraft & Engine Performance, Ideal Cruising Altitude & Vertical Fuel Optimization Engine.

Quantifies the aerodynamic and thermodynamic cruise efficiency of flights based on:
1. Specific aircraft airframe specifications (OpenAP aircraft & WRAP kinematic models).
2. Engine options and installed turbofan / turboprop specifications (OpenAP engines & ICAO EDB):
   - Bypass ratio (BPR)
   - Pressure ratio (PR)
   - Specific Fuel Consumption (SFC)
3. Three-way cruise flight level decomposition:
   - Optimal Aerodynamic / Engine Cruise Flight Level (FL_opt)
   - Filed / Requested Flight Level (FL_req)
   - Actual Flown Cruise Flight Level (FL_actual)
4. Breakdown of altitude-induced fuel spend / savings:
   - Flight Planning Inefficiency (Requested vs. Optimal)
   - ATC Tactical Inefficiency / Level-Capping (Actual vs. Requested)
   - Potential Fuel & Cost Savings achievable by cruising at the aircraft's ideal altitude.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from flightml.data.dataset import FlightDataset


# Turboprop and special airframe ideal altitude fallbacks (where openap has generic synonyms)
TURBOPROP_SPECS = {
    "AT72": {"opt_fl": 200.0, "ceil_fl": 250.0, "engine": "PW127F", "bpr": 0.0, "pr": 15.0, "ff_kg_hr": 580.0},
    "AT75": {"opt_fl": 200.0, "ceil_fl": 250.0, "engine": "PW127M", "bpr": 0.0, "pr": 15.0, "ff_kg_hr": 590.0},
    "AT76": {"opt_fl": 200.0, "ceil_fl": 250.0, "engine": "PW127XT", "bpr": 0.0, "pr": 16.0, "ff_kg_hr": 560.0},
    "DH8D": {"opt_fl": 230.0, "ceil_fl": 270.0, "engine": "PW150A", "bpr": 0.0, "pr": 18.0, "ff_kg_hr": 620.0},
    "DH8C": {"opt_fl": 210.0, "ceil_fl": 250.0, "engine": "PW123D", "bpr": 0.0, "pr": 15.0, "ff_kg_hr": 580.0},
    "F50":  {"opt_fl": 200.0, "ceil_fl": 250.0, "engine": "PW125B", "bpr": 0.0, "pr": 15.0, "ff_kg_hr": 600.0},
    "SF34": {"opt_fl": 190.0, "ceil_fl": 250.0, "engine": "CT7-9B",  "bpr": 0.0, "pr": 15.0, "ff_kg_hr": 510.0},
    "ATP":  {"opt_fl": 190.0, "ceil_fl": 250.0, "engine": "PW126",  "bpr": 0.0, "pr": 15.0, "ff_kg_hr": 550.0},
}


@dataclass
class AircraftEnginePerformance:
    ac_type: str
    aircraft_name: str
    wake_category: str
    selected_engine: str
    engine_options: list[str]
    is_engine_override: bool
    engine_type_class: str
    bypass_ratio: float
    pressure_ratio: float
    cruise_sfc: float | None
    nominal_fuel_flow_kg_hr: float
    optimal_cruise_fl: float
    operational_ceiling_fl: float
    cruise_mach: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CruiseAltitudeEfficiency:
    ectrl_id: int
    ac_type: str
    selected_engine: str
    optimal_cruise_fl: float
    requested_fl: float
    actual_cruise_fl: float
    max_flown_fl: float
    cruise_hours: float
    base_cruise_fuel_kg: float
    # Altitude deltas (FL)
    fl_delta_actual_vs_optimal: float     # actual - optimal
    fl_delta_requested_vs_optimal: float  # requested - optimal
    fl_delta_actual_vs_requested: float    # actual - requested (ATC tactical deviation)
    # Fuel penalty percentages
    actual_burn_penalty_pct: float
    requested_burn_penalty_pct: float
    # Absolute excess fuel burn (kg)
    excess_altitude_fuel_kg: float
    excess_altitude_fuel_cost_eur: float
    planning_excess_fuel_kg: float
    planning_excess_fuel_cost_eur: float
    atc_tactical_excess_fuel_kg: float
    atc_tactical_excess_fuel_cost_eur: float
    # Potential savings by cruising at optimal altitude
    potential_fuel_savings_kg: float
    potential_fuel_savings_eur: float
    potential_fuel_savings_pct: float
    fuel_price_eur_per_kg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ectrl_id": self.ectrl_id,
            "ac_type": self.ac_type,
            "selected_engine": self.selected_engine,
            "optimal_cruise_fl": round(self.optimal_cruise_fl, 1),
            "requested_fl": round(self.requested_fl, 1),
            "actual_cruise_fl": round(self.actual_cruise_fl, 1),
            "max_flown_fl": round(self.max_flown_fl, 1),
            "cruise_hours": round(self.cruise_hours, 2),
            "base_cruise_fuel_kg": round(self.base_cruise_fuel_kg, 1),
            "fl_delta_actual_vs_optimal": round(self.fl_delta_actual_vs_optimal, 1),
            "fl_delta_requested_vs_optimal": round(self.fl_delta_requested_vs_optimal, 1),
            "fl_delta_actual_vs_requested": round(self.fl_delta_actual_vs_requested, 1),
            "actual_burn_penalty_pct": round(self.actual_burn_penalty_pct, 2),
            "requested_burn_penalty_pct": round(self.requested_burn_penalty_pct, 2),
            "excess_altitude_fuel_kg": round(self.excess_altitude_fuel_kg, 1),
            "excess_altitude_fuel_cost_eur": round(self.excess_altitude_fuel_cost_eur, 2),
            "planning_excess_fuel_kg": round(self.planning_excess_fuel_kg, 1),
            "planning_excess_fuel_cost_eur": round(self.planning_excess_fuel_cost_eur, 2),
            "atc_tactical_excess_fuel_kg": round(self.atc_tactical_excess_fuel_kg, 1),
            "atc_tactical_excess_fuel_cost_eur": round(self.atc_tactical_excess_fuel_cost_eur, 2),
            "potential_fuel_savings_kg": round(self.potential_fuel_savings_kg, 1),
            "potential_fuel_savings_eur": round(self.potential_fuel_savings_eur, 2),
            "potential_fuel_savings_pct": round(self.potential_fuel_savings_pct, 2),
            "fuel_price_eur_per_kg": round(self.fuel_price_eur_per_kg, 4),
        }


def resolve_aircraft_engine_perf(
    dataset: FlightDataset,
    ac_type: str,
    engine_override: str | None = None,
) -> AircraftEnginePerformance:
    """
    Resolve aircraft airframe and engine combination performance, including bypass ratio,
    pressure ratio, nominal fuel flow, and optimal aerodynamic cruising altitude (FL_opt).
    """
    code = ac_type.strip().upper()

    # 1. Turboprop fast-path override for realistic operational ceilings & fuel flow
    if code in TURBOPROP_SPECS:
        tb = TURBOPROP_SPECS[code]
        selected_eng = engine_override.strip() if engine_override else tb["engine"]
        return AircraftEnginePerformance(
            ac_type=code,
            aircraft_name=f"{code} Regional Turboprop",
            wake_category="M" if code in ("AT72", "DH8D", "AT75", "AT76") else "L",
            selected_engine=selected_eng,
            engine_options=[tb["engine"]],
            is_engine_override=bool(engine_override and engine_override != tb["engine"]),
            engine_type_class="Turboprop",
            bypass_ratio=tb["bpr"],
            pressure_ratio=tb["pr"],
            cruise_sfc=None,
            nominal_fuel_flow_kg_hr=tb["ff_kg_hr"],
            optimal_cruise_fl=tb["opt_fl"],
            operational_ceiling_fl=tb["ceil_fl"],
            cruise_mach=0.52,
        )

    # 2. Query openap_aircraft
    ac_row = dataset.cursor.execute(
        "SELECT * FROM openap_aircraft WHERE ac_type = ?", [code]
    ).df()

    aircraft_name = f"{code} Aircraft"
    wake_cat = "M"
    default_eng = "DEFAULT_TURBOFAN"
    engine_opts: list[str] = []
    ceiling_fl = 410.0
    cruise_mach = 0.78
    nominal_ff = 2500.0

    if not ac_row.empty:
        r = ac_row.iloc[0]
        aircraft_name = str(r.get("aircraft_name") or aircraft_name)
        wake_cat = str(r.get("wake_category") or "M")
        default_eng = str(r.get("default_engine") or default_eng)
        cruise_mach = float(r.get("cruise_mach") or 0.78)
        if r.get("ceiling_m") and not pd.isna(r["ceiling_m"]):
            ceiling_fl = round(float(r["ceiling_m"]) * 3.28084 / 100 / 10) * 10
        raw_opts = str(r.get("engine_options") or "")
        if raw_opts and raw_opts != "None":
            # de-duplicate preserving order
            seen = set()
            for opt in raw_opts.split(","):
                clean = opt.strip()
                if clean and clean not in seen:
                    seen.add(clean)
                    engine_opts.append(clean)
        if default_eng not in engine_opts and default_eng != "DEFAULT_TURBOFAN":
            engine_opts.insert(0, default_eng)

    if not engine_opts:
        engine_opts = [default_eng]

    # Selected engine selection
    is_override = False
    if engine_override and engine_override.strip() in engine_opts:
        selected_eng = engine_override.strip()
        is_override = True
    elif engine_override and engine_override.strip():
        # User specified an engine not in list: accept as custom override
        selected_eng = engine_override.strip()
        is_override = True
    else:
        selected_eng = engine_opts[0] if engine_opts else default_eng

    # 3. Query engine thermodynamic parameters (BPR, PR, cruise SFC)
    bpr = 6.0
    pr = 30.0
    cruise_sfc: float | None = None
    engine_class = "Turbofan (High Bypass)"

    eng_row = dataset.cursor.execute(
        "SELECT name, bpr, pr, cruise_sfc, cruise_alt FROM openap_engines WHERE name = ?",
        [selected_eng],
    ).df()
    if eng_row.empty:
        eng_row = dataset.cursor.execute(
            "SELECT name, bpr, pr, cruise_sfc, cruise_alt FROM openap_engines WHERE name LIKE ?",
            [selected_eng[:6] + "%"],
        ).df()

    if not eng_row.empty:
        er = eng_row.iloc[0]
        if er.get("bpr") is not None and not pd.isna(er["bpr"]):
            bpr = float(er["bpr"])
        if er.get("pr") is not None and not pd.isna(er["pr"]):
            pr = float(er["pr"])
        if er.get("cruise_sfc") is not None and not pd.isna(er["cruise_sfc"]):
            cruise_sfc = float(er["cruise_sfc"])

    if bpr >= 9.5:
        engine_class = "Turbofan (Ultra-High / Geared Bypass)"
    elif bpr >= 5.0:
        engine_class = "Turbofan (High Bypass)"
    elif bpr > 0.0:
        engine_class = "Turbofan (Medium / Low Bypass)"
    else:
        engine_class = "Turboprop / Piston"

    # 4. Resolve baseline aerodynamic cruise flight level from WRAP kinematics
    opt_fl = 360.0
    wrap_row = dataset.cursor.execute(
        "SELECT opt_val FROM openap_wrap WHERE ac_type = ? AND variable = ?",
        [code, "cr_h_mean"],
    ).df()
    if not wrap_row.empty and not pd.isna(wrap_row.iloc[0]["opt_val"]):
        km = float(wrap_row.iloc[0]["opt_val"])
        opt_fl = round((km * 1000.0 * 3.28084) / 100 / 10) * 10
    elif not ac_row.empty and ac_row.iloc[0].get("cruise_height_m") and not pd.isna(ac_row.iloc[0]["cruise_height_m"]):
        opt_fl = round(float(ac_row.iloc[0]["cruise_height_m"]) * 3.28084 / 100 / 10) * 10

    # Modern high-efficiency aircraft aerodynamic calibrations
    if code in ("A359", "A35K", "B788", "B789", "B78X"):
        opt_fl = max(opt_fl, 390.0)
    elif code in ("B38M", "B39M", "A20N", "A21N", "A19N"):
        opt_fl = max(opt_fl, 380.0)
    elif code in ("GLF5", "GLF6", "GL5T"):
        opt_fl = max(opt_fl, 450.0)

    # Engine-specific optimal altitude adjustment:
    # High pressure ratio engines (PR > 36) or ultra-high bypass (BPR >= 9.0) operate
    # most efficiently at higher, colder flight levels (+10 to +20 FL)
    if pr >= 38.0 and bpr >= 8.0:
        opt_fl = min(ceiling_fl - 10.0, opt_fl + 10.0)
    elif pr < 24.0 and bpr < 5.5 and opt_fl > 340.0:
        # Older low-pressure engines reach optimal specific air range slightly lower
        opt_fl = max(320.0, opt_fl - 10.0)

    opt_fl = min(opt_fl, ceiling_fl)

    # 5. Tailor baseline fuel flow to aircraft category & engine efficiency
    base_category_ff = {
        "A320": 2400.0, "A321": 2700.0, "A319": 2250.0, "A318": 2200.0,
        "A20N": 2050.0, "A21N": 2300.0, "A19N": 1950.0,
        "B738": 2450.0, "B737": 2400.0, "B739": 2600.0, "B733": 2550.0, "B734": 2600.0,
        "B38M": 2100.0, "B39M": 2250.0, "B37M": 2000.0,
        "B77W": 6800.0, "B772": 6200.0, "B773": 6600.0, "B763": 4800.0, "B762": 4500.0,
        "A332": 5200.0, "A333": 5600.0, "A343": 6500.0, "A359": 5400.0, "A388": 11500.0,
        "B744": 10200.0, "B748": 9600.0, "B788": 4900.0, "B789": 5300.0,
        "E190": 1750.0, "E195": 1850.0, "E170": 1600.0, "E145": 1200.0,
        "CRJ9": 1600.0, "CRJ2": 1100.0,
    }
    nominal_ff = base_category_ff.get(code, 2500.0 if wake_cat == "M" else 6500.0 if wake_cat == "H" else 11500.0 if wake_cat == "J" else 1200.0)

    # Adjust fuel flow by engine efficiency: modern high BPR engines burn ~8% to 14% less
    if bpr >= 10.0:
        nominal_ff *= 0.88
    elif bpr >= 8.0:
        nominal_ff *= 0.93
    elif bpr < 5.0 and engine_class != "Turboprop":
        nominal_ff *= 1.05

    return AircraftEnginePerformance(
        ac_type=code,
        aircraft_name=aircraft_name,
        wake_category=wake_cat,
        selected_engine=selected_eng,
        engine_options=engine_opts,
        is_engine_override=is_override,
        engine_type_class=engine_class,
        bypass_ratio=round(bpr, 2),
        pressure_ratio=round(pr, 1),
        cruise_sfc=cruise_sfc,
        nominal_fuel_flow_kg_hr=round(nominal_ff, 1),
        optimal_cruise_fl=float(opt_fl),
        operational_ceiling_fl=float(ceiling_fl),
        cruise_mach=round(cruise_mach, 2),
    )


def compute_altitude_burn_penalty(
    flown_fl: float,
    optimal_fl: float,
    pressure_ratio: float = 30.0,
) -> float:
    """
    Calculate the fractional cruise fuel burn penalty for flying at `flown_fl` instead of `optimal_fl`.
    - Below optimal: Denser air increases parasite drag, engine operates off design pressure ratio.
      ~1.5% extra fuel burn per 1,000 ft (10 FL) below optimal FL, scaled by engine pressure ratio.
    - Above optimal: Thin air forces higher angle of attack, increasing induced drag (k*CL^2).
      ~2.0% extra fuel burn per 1,000 ft (10 FL) above optimal FL.
    """
    if flown_fl <= 0.0 or optimal_fl <= 0.0:
        return 0.0

    delta_fl = flown_fl - optimal_fl

    if delta_fl < 0.0:
        # Cruising below optimal altitude
        deficit_thousands_ft = abs(delta_fl) / 10.0
        # Scaled by engine pressure ratio (higher PR engines suffer slightly steeper penalties when kept low)
        pr_sensitivity = 1.0 + max(0.0, pressure_ratio - 25.0) / 100.0
        return 0.015 * deficit_thousands_ft * pr_sensitivity
    elif delta_fl > 0.0:
        # Cruising above optimal altitude
        excess_thousands_ft = delta_fl / 10.0
        return 0.020 * excess_thousands_ft

    return 0.0


def compute_cruise_altitude_efficiency(
    dataset: FlightDataset,
    ectrl_id: int,
    engine_override: str | None = None,
    fuel_price_per_kg: float = 0.78,
) -> CruiseAltitudeEfficiency:
    """
    Evaluate the full 3-way vertical cruise efficiency and fuel spend/savings:
    Optimal FL vs. Requested FL vs. Actual Trajectory Flown FL.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        raise ValueError(f"Flight {ectrl_id} not found")

    ac_type = str(flight.get("ac_type") or "A320").upper()
    perf = resolve_aircraft_engine_perf(dataset, ac_type, engine_override=engine_override)

    trajs = dataset.get_trajectory(ectrl_id)
    actual_df = trajs["actual"].sort_values("seq_num").reset_index(drop=True)
    planned_df = trajs["planned"].sort_values("seq_num").reset_index(drop=True)

    max_act_fl = float(actual_df["flight_level"].max()) if not actual_df.empty else 0.0
    req_fl = float(flight.get("requested_fl") or 0.0)
    if req_fl <= 0.0 and not planned_df.empty:
        req_fl = float(planned_df["flight_level"].max())
    if req_fl <= 0.0:
        req_fl = max_act_fl if max_act_fl > 0.0 else perf.optimal_cruise_fl

    # Extract actual cruise altitude (median of upper cruise band: top 30 FLs)
    if not actual_df.empty and max_act_fl >= 80.0:
        cruise_pts = actual_df[actual_df["flight_level"] >= max(80.0, max_act_fl - 30.0)]
        actual_cruise_fl = float(cruise_pts["flight_level"].median()) if not cruise_pts.empty else max_act_fl
    else:
        actual_cruise_fl = max_act_fl if max_act_fl > 0.0 else req_fl

    # Cruise duration calculation (excluding climb and descent phases ~ 75% of air time or derived from cruise points)
    air_time_min = float(flight.get("actual_air_time_min") or 60.0)
    if air_time_min <= 0.0:
        air_time_min = 60.0
    cruise_hours = max(0.2, (air_time_min / 60.0) * 0.75)

    base_cruise_fuel = cruise_hours * perf.nominal_fuel_flow_kg_hr

    # Compute burn penalties
    actual_penalty = compute_altitude_burn_penalty(actual_cruise_fl, perf.optimal_cruise_fl, perf.pressure_ratio)
    requested_penalty = compute_altitude_burn_penalty(req_fl, perf.optimal_cruise_fl, perf.pressure_ratio)

    excess_altitude_fuel = base_cruise_fuel * actual_penalty
    excess_altitude_cost_eur = excess_altitude_fuel * fuel_price_per_kg

    planning_excess_fuel = base_cruise_fuel * requested_penalty
    planning_excess_cost_eur = planning_excess_fuel * fuel_price_per_kg

    # Tactical ATC deviation = difference between actual flown burn and filed requested burn
    atc_excess_fuel = excess_altitude_fuel - planning_excess_fuel
    atc_excess_cost_eur = atc_excess_fuel * fuel_price_per_kg

    # Potential savings by flying at optimal flight level
    potential_savings_fuel = max(0.0, excess_altitude_fuel)
    potential_savings_cost_eur = potential_savings_fuel * fuel_price_per_kg
    total_cruise_burn = base_cruise_fuel + excess_altitude_fuel
    potential_savings_pct = (potential_savings_fuel / total_cruise_burn * 100.0) if total_cruise_burn > 0.0 else 0.0

    return CruiseAltitudeEfficiency(
        ectrl_id=int(ectrl_id),
        ac_type=ac_type,
        selected_engine=perf.selected_engine,
        optimal_cruise_fl=perf.optimal_cruise_fl,
        requested_fl=req_fl,
        actual_cruise_fl=actual_cruise_fl,
        max_flown_fl=max_act_fl,
        cruise_hours=cruise_hours,
        base_cruise_fuel_kg=base_cruise_fuel,
        fl_delta_actual_vs_optimal=actual_cruise_fl - perf.optimal_cruise_fl,
        fl_delta_requested_vs_optimal=req_fl - perf.optimal_cruise_fl,
        fl_delta_actual_vs_requested=actual_cruise_fl - req_fl,
        actual_burn_penalty_pct=actual_penalty * 100.0,
        requested_burn_penalty_pct=requested_penalty * 100.0,
        excess_altitude_fuel_kg=excess_altitude_fuel,
        excess_altitude_fuel_cost_eur=excess_altitude_cost_eur,
        planning_excess_fuel_kg=planning_excess_fuel,
        planning_excess_fuel_cost_eur=planning_excess_cost_eur,
        atc_tactical_excess_fuel_kg=atc_excess_fuel,
        atc_tactical_excess_fuel_cost_eur=atc_excess_cost_eur,
        potential_fuel_savings_kg=potential_savings_fuel,
        potential_fuel_savings_eur=potential_savings_cost_eur,
        potential_fuel_savings_pct=potential_savings_pct,
        fuel_price_eur_per_kg=fuel_price_per_kg,
    )
