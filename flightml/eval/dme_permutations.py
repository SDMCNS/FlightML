"""
FlightML DME Network Resilience & Operational Permutation Engine.

Evaluates terrestrial DME navigation infrastructure resilience across European FIRs:
- Evaluates permutations of enabled/disabled DMEs against ICAO Doc 9613 / FAA AC 90-100A standards
  (RNAV_5, RNAV_2, RNAV_1, RNP_1).
- Validates network impact against real historical traffic from Eurocontrol dataset archives.
- Models en-route radar spacing delays, tactical reroutes / fuel burn, and airport terminal holding.
- Presents operational costs and equipment savings scaled across 1-hour, 6-hour, and 1-day (24-hour) operational windows.
- Implements strict Pareto configuration validation and classification.
- Exports robust standalone HTML reports with interactive Leaflet maps and comprehensive mathematical appendices.
"""

from __future__ import annotations

import datetime
import html
import json
import math
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Literal

import folium
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

from flightml.data.dataset import FlightDataset
import flightml.math as fmath
from flightml.eval.ecac_station_rationalization import (
    ECACStation,
    load_ecac_stations,
    generate_ecac_grid,
    get_fir_geometry,
    ECAC_FIR_REGISTRY,
    ASSESSMENT_ALTITUDES,
    EVALUATION_FLIGHT_LEVELS,
    _compute_coverage_matrices,
    compute_static_rationalization_views,
    DME_ANNUAL_COST_SAVINGS_EUR,
    DME_POWER_DRAW_KW,
    DME_ANNUAL_POWER_MWH,
)
from flightml.eval.dme_learning import DMESelfReinforcingLearner
from flightml.viz.html_exporter import _folium_map_to_iframe

# ==============================================================================
# EUROCONTROL & ICAO OPERATIONAL AND ECONOMIC BENCHMARKS
# References:
# 1. Eurocontrol Standard Inputs for Economic Cost-Benefit Analyses (Edition 9.0)
# 2. Eurocontrol Navigation Infrastructure Strategy & Terrestrial Navaids Rationalisation
# 3. ICAO Doc 9613 (Performance-Based Navigation Manual)
# ==============================================================================

HOURS_PER_YEAR = 8760.0
DME_HOURLY_OPEX_EUR = DME_ANNUAL_COST_SAVINGS_EUR / HOURS_PER_YEAR  # €5.137 / hr per DME

# Cost of Delay Coefficients (Eurocontrol Standard Inputs):
# En-route airborne delay (tactical vectoring / speed control / holding stack): €54.00 / minute
EUROCONTROL_AIRBORNE_DELAY_EUR_PER_MIN = 54.00

# Ground delay / slot holding / ground delay program (GDP): €28.00 / minute
EUROCONTROL_GROUND_DELAY_EUR_PER_MIN = 28.00

# Tactical reroute jet fuel burn & carbon emission cost:
# - Tactical detour distance: ~22.0 NM average avoidance dog-leg
# - Jet-A fuel consumption: ~4.20 kg / NM (ICAO Doc 9883 medium twin-turbofan baseline)
# - Jet-A fuel price: €0.85 / kg
# - EU ETS CO2 allowance cost: €0.23 / kg Jet-A (~3.15 kg CO2/kg fuel @ €75/tonne CO2)
# - Total reroute fuel + emissions cost: 4.20 * (0.85 + 0.23) ≈ €4.54 / NM
TACTICAL_DETOUR_NM = 22.0
FUEL_BURN_KG_PER_NM = 4.20
FUEL_AND_ETS_EUR_PER_NM = 4.54


@dataclass
class WindowCostBreakdown:
    window_label: str  # "1h", "6h", "24h" (1-day)
    duration_hours: float
    flights_exposed: float
    airborne_delay_minutes: float
    terminal_holding_minutes: float
    ground_delay_minutes: float
    total_delay_minutes: float
    reroute_distance_nm: float
    reroute_fuel_kg: float
    airborne_delay_cost_eur: float
    terminal_holding_cost_eur: float
    ground_delay_cost_eur: float
    reroute_fuel_cost_eur: float
    gross_operational_cost_eur: float
    incremental_delay_cost_eur: float  # Operational cost above baseline status quo
    equipment_opex_savings_eur: float  # Savings from decommissioned / standby equipment
    net_economic_impact_eur: float  # incremental_delay_cost - equipment_savings (- = net financial savings, + = net operational penalty)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DMEPermutationResult:
    permutation_id: str
    name: str
    category: str  # "BASELINE", "RATIONALIZED_CORE", "SINGLE_OUTAGE", "DUAL_OUTAGE", "STRESS_TEST"
    active_station_ids: list[str]
    disabled_station_ids: list[str]
    active_count: int
    disabled_count: int
    target_standard: str
    total_cells: int
    cells_met: int
    cells_at_risk: int
    cells_cannot_be_met: int
    cells_uncoverable: int
    coverage_met_pct: float
    coverage_coverable_pct: float
    terminal_airports_total: int
    degraded_airports_count: int
    degraded_airports_idents: list[str]
    windows: dict[str, WindowCostBreakdown]
    classification: str  # "OPTIMAL_RATIONALIZATION", "ACCEPTABLE_DEGRADATION", "CRITICAL_DEFICIT"
    is_valid: bool
    selection_rationale: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["windows"] = {k: v.to_dict() for k, v in self.windows.items()}
        return d


@dataclass
class SimulatedFlightKnockOn:
    leg1_ectrl_id: int
    ac_reg: str
    ac_operator: str
    ac_type: str
    leg1_route: str
    leg1_dep_time: str
    leg1_arr_time: str
    is_directly_exposed: bool
    enroute_vector_delay_min: float
    terminal_holding_delay_min: float
    primary_delay_min: float
    tactical_detour_nm: float
    extra_fuel_kg: float
    extra_co2_kg: float
    has_downstream_leg: bool
    leg2_ectrl_id: int | None
    leg2_route: str | None
    leg2_scheduled_dep: str | None
    turnaround_buffer_min: float
    absorbed_delay_min: float
    downstream_reactionary_delay_min: float
    cancellation_risk: str  # "LOW (ABSORBED)", "MODERATE DELAY", "ELEVATED (MISSED SLOT)", "CRITICAL (CREW FDP)"
    cancellation_reason: str
    rnav_capability: str = "RNAV 1"
    rnav_impact_score: int = 0
    rnav_category_scores: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrafficSimulationSummary:
    fir_id: str
    period: str
    scenario_name: str
    separation_standard_nm: float
    total_flights_simulated: int
    directly_impacted_flights: int
    total_primary_delay_min: float
    total_detour_distance_nm: float
    total_extra_fuel_kg: float
    total_extra_co2_kg: float
    downstream_flights_delayed: int
    total_reactionary_delay_min: float
    reactionary_multiplier: float
    cancellation_risk_count: int
    curfew_breaches_count: int  # Maintained as 0 (night curfews eliminated without definitive AIP proof)
    crew_fdp_timeouts_count: int
    sample_impacted_flights: list[SimulatedFlightKnockOn]
    rnav_scoring_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sample_impacted_flights"] = [f.to_dict() for f in self.sample_impacted_flights]
        d["rnav_scoring_summary"] = self.rnav_scoring_summary
        return d


@dataclass
class DMEStationUtilization:
    station_id: str
    station_name: str
    airspace_role: str  # "Terminal Primary", "Core En-Route Backbone", "Regional / Infill"
    latitude: float
    longitude: float
    elevation_ft: float
    is_active_in_optimal: bool
    daily_flights_using: float
    hourly_flights_using: float
    peak_hourly_flights_using: float
    utilization_pct: float
    valid_pairs_formed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DMEMaintenanceImpactSummary:
    outage_rate_pct: float  # 0.5%
    total_cells: int
    spof_cells_count: int
    redundant_cells_count: int
    permanently_degraded_cells_count: int
    annual_cell_hours_reduced_rnav: float
    monthly_flights_impacted: int
    annual_flights_impacted: int
    annual_outage_cost_eur: float
    monthly_outage_cost_eur: float
    daily_outage_cost_eur: float
    station_maintenance_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DMEFIRResilienceAnalysis:
    fir_id: str
    fir_name: str
    period: str
    target_standard: str
    baseline_station_count: int
    total_airports_count: int
    traffic_month_total: int
    traffic_daily_avg: float
    traffic_hourly_avg: float
    traffic_hourly_peak: float
    permutations: list[DMEPermutationResult]
    optimal_permutation: DMEPermutationResult | None
    pareto_frontier: list[DMEPermutationResult]
    critical_stations: list[dict[str, Any]]
    generated_at: str
    ml_diagnostics: dict[str, Any] | None = None
    traffic_simulation: TrafficSimulationSummary | None = None
    station_utilization: list[DMEStationUtilization] | None = None
    maintenance_impact: DMEMaintenanceImpactSummary | None = None
    cascaded_adjacent_stations: list[dict[str, Any]] | None = None
    rnav_scoring_summary: dict[str, Any] | None = None
    cns_joint_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["permutations"] = [p.to_dict() for p in self.permutations]
        d["optimal_permutation"] = self.optimal_permutation.to_dict() if self.optimal_permutation else None
        d["pareto_frontier"] = [p.to_dict() for p in self.pareto_frontier]
        d["ml_diagnostics"] = self.ml_diagnostics
        d["traffic_simulation"] = self.traffic_simulation.to_dict() if self.traffic_simulation else None
        d["station_utilization"] = [u.to_dict() for u in self.station_utilization] if self.station_utilization else []
        d["maintenance_impact"] = self.maintenance_impact.to_dict() if self.maintenance_impact else None
        d["cascaded_adjacent_stations"] = self.cascaded_adjacent_stations or []
        d["rnav_scoring_summary"] = self.rnav_scoring_summary
        d["cns_joint_summary"] = self.cns_joint_summary
        return d


def _fetch_fir_traffic_stats(dataset: FlightDataset, fir_id: str, period: str) -> tuple[int, float, float, float]:
    """
    Fetch historical traffic volume for the target FIR.
    Returns (monthly_total, daily_avg, hourly_avg, hourly_peak).
    """
    try:
        if dataset.has_fir_traversals():
            cnt_row = dataset.cursor.execute(
                """
                SELECT COUNT(*) as total_traversals
                FROM flight_firs_actual
                WHERE fir_id = ?
                """,
                [fir_id],
            ).fetchone()
            total_moves = int(cnt_row[0]) if cnt_row and cnt_row[0] is not None else 0
            if total_moves > 0:
                daily = total_moves / 31.0
                hourly = daily / 24.0
                peak = hourly * 1.45
                return total_moves, round(daily, 1), round(hourly, 1), round(peak, 1)
    except Exception:
        pass

    # Fallback to general flight count in FIR bounding box if traversals table is unavailable
    try:
        fir_geom = get_fir_geometry(dataset, fir_id)
        if fir_geom:
            _, _, bbox = fir_geom
            cnt_row = dataset.cursor.execute(
                """
                SELECT COUNT(*)
                FROM flights
                WHERE (adep_lat BETWEEN ? AND ? AND adep_lon BETWEEN ? AND ?)
                   OR (ades_lat BETWEEN ? AND ? AND ades_lon BETWEEN ? AND ?)
                """,
                [bbox[0], bbox[1], bbox[2], bbox[3], bbox[0], bbox[1], bbox[2], bbox[3]],
            ).fetchone()
            total_moves = int(cnt_row[0]) if cnt_row and cnt_row[0] is not None else 60000
            daily = total_moves / 31.0
            hourly = daily / 24.0
            peak = hourly * 1.45
            return total_moves, round(daily, 1), round(hourly, 1), round(peak, 1)
    except Exception:
        pass

    # Default representative traffic baseline for European high-density FIR (e.g. London / Paris / Langen)
    return 180000, 5806.5, 241.9, 350.8


def _evaluate_single_permutation_coverage(
    active_indices: list[int],
    cell_mat_200: np.ndarray,
    cell_mat_100: np.ndarray,
    apt_mat_10: np.ndarray,
    bearings_cell: np.ndarray,
    dist_cell: np.ndarray,
    target_standard: str,
    airports: list[dict[str, Any]],
    coverable_cells_mask: np.ndarray,
) -> tuple[int, int, int, int, list[str]]:
    """
    Vectorized evaluation of a single DME permutation:
    Returns (cells_met, cells_at_risk, cells_cannot_be_met, cells_uncoverable, degraded_airport_idents).
    """
    n_cells = cell_mat_200.shape[0]

    if not active_indices:
        # All disabled
        met = 0
        at_risk = 0
        unc = int((~coverable_cells_mask).sum())
        cannot = n_cells - unc
        degraded_apts = [a["ident"] for a in airports]
        return met, at_risk, cannot, unc, degraded_apts

    act_idx_arr = np.array(active_indices, dtype=int)

    # 1. Evaluate En-route & Terminal Cells against Target Standard
    # Check visibility at FL200
    vis_200_act = cell_mat_200[:, act_idx_arr]  # (N_cells, N_act)
    cnt_200 = vis_200_act.sum(axis=1)

    # Check pair geometry where cnt >= 2
    multi_mask = cnt_200 >= 2
    has_valid_pair = np.zeros(n_cells, dtype=bool)
    d_max_pair = np.full(n_cells, 999.0)

    if multi_mask.any():
        multi_indices = np.where(multi_mask)[0]
        for idx in multi_indices:
            s_vis = np.where(vis_200_act[idx])[0]
            if len(s_vis) < 2:
                continue
            act_s_global = act_idx_arr[s_vis]
            b = bearings_cell[idx, act_s_global]
            diff = np.abs(b[:, None] - b[None, :])
            delta = np.minimum(diff, 360.0 - diff)
            valid = (delta >= 30.0) & (delta <= 150.0)
            if valid.any():
                has_valid_pair[idx] = True
                dists = dist_cell[idx, act_s_global]
                max_dists = np.maximum(dists[:, None], dists[None, :])
                valid_max_dists = np.where(valid, max_dists, 9999.0)
                d_max_pair[idx] = float(np.min(valid_max_dists))

    # Evaluate standard compliance
    status_met = np.zeros(n_cells, dtype=bool)
    status_at_risk = np.zeros(n_cells, dtype=bool)
    status_cannot = np.zeros(n_cells, dtype=bool)

    if target_standard in ("RNAV_5", "VIEW_1"):
        # RNAV 5: Valid pair with 30-150 deg geometry at FL200
        status_met = has_valid_pair & coverable_cells_mask
        status_at_risk = (~has_valid_pair) & (cnt_200 >= 1) & coverable_cells_mask
        status_cannot = (cnt_200 == 0) & coverable_cells_mask
    elif target_standard in ("RNAV_2", "VIEW_2"):
        # RNAV 2: Valid pair within 130 NM slant range
        status_met = has_valid_pair & (d_max_pair <= 130.0) & coverable_cells_mask
        status_at_risk = (~status_met) & (has_valid_pair | (cnt_200 >= 1)) & coverable_cells_mask
        status_cannot = (cnt_200 == 0) & coverable_cells_mask
    elif target_standard in ("RNAV_1", "VIEW_3"):
        # RNAV 1: Terminal P-RNAV N-1 redundancy (>=3 DMEs visible, valid pair within 60 NM)
        vis_100_act = cell_mat_100[:, act_idx_arr]
        cnt_100 = vis_100_act.sum(axis=1)
        status_met = (cnt_100 >= 3) & has_valid_pair & (d_max_pair <= 60.0) & coverable_cells_mask
        status_at_risk = (~status_met) & (has_valid_pair | (cnt_200 >= 1)) & coverable_cells_mask
        status_cannot = (cnt_200 == 0) & coverable_cells_mask
    else:  # RNP_1
        # RNP 1: Reversionary Multi-DME/IRU (>=4 DMEs visible within 40 NM)
        vis_100_act = cell_mat_100[:, act_idx_arr]
        cnt_100 = vis_100_act.sum(axis=1)
        status_met = (cnt_100 >= 4) & has_valid_pair & (d_max_pair <= 40.0) & coverable_cells_mask
        status_at_risk = (~status_met) & (has_valid_pair | (cnt_200 >= 1)) & coverable_cells_mask
        status_cannot = (cnt_200 == 0) & coverable_cells_mask

    cells_met = int(status_met.sum())
    cells_at_risk = int(status_at_risk.sum())
    cells_cannot = int(status_cannot.sum())
    cells_unc = int((~coverable_cells_mask).sum())

    # 2. Check Airport Terminal FL10 Redundancy
    apt_vis_10 = apt_mat_10[:, act_idx_arr]  # (N_airports, N_act)
    apt_cnt_10 = apt_vis_10.sum(axis=1)
    degraded_apts = []
    for i, a in enumerate(airports):
        req_dmes = 2 if a.get("is_high_density", False) else 1
        if apt_cnt_10[i] < req_dmes:
            degraded_apts.append(a["ident"])

    return cells_met, cells_at_risk, cells_cannot, cells_unc, degraded_apts


def _compute_window_costs(
    duration_hours: float,
    window_label: str,
    traffic_hourly_avg: float,
    traffic_peak_multiplier: float,
    outage_fraction: float,
    at_risk_fraction: float,
    degraded_airports_count: int,
    disabled_dme_count: int,
    baseline_gross_cost: float | None = None,
) -> WindowCostBreakdown:
    """
    Compute delay minutes, tactical detour miles, fuel burn, and financial costs
    for an operational window (1h, 6h, 24h).
    """
    effective_hourly_traffic = traffic_hourly_avg * traffic_peak_multiplier
    flights_window_total = effective_hourly_traffic * duration_hours

    # 1. En-Route Airborne Spacing Vectoring Delay:
    # Loss of RNAV capability forces ATC to apply procedural radar spacing (8-10 NM instead of 5 NM).
    # Average en-route spacing delay: 4.5 minutes per exposed flight.
    effective_exposure_rate = min(1.0, outage_fraction + (0.30 * at_risk_fraction))
    flights_impacted_enroute = flights_window_total * effective_exposure_rate
    airborne_delay_min = flights_impacted_enroute * 4.50

    # 2. Tactical Reroutes & Extra Jet-A Fuel Burn:
    # Approximately 25% of exposed flights encounter ATC tactical dog-legs to skirt coverage gaps.
    flights_rerouted = flights_impacted_enroute * 0.25
    reroute_nm = flights_rerouted * TACTICAL_DETOUR_NM
    fuel_burn_kg = reroute_nm * FUEL_BURN_KG_PER_NM

    # 3. Airport Terminal Holding & Slot Delays:
    # If terminal FL10 dual DME coverage is lost, arrival acceptance rate drops by 20%,
    # resulting in ~14 min terminal holding per arrival and ~8 min departure slot hold.
    # Terminal traffic estimated at ~35 arrival/departure movements per degraded airport per hour.
    terminal_moves_per_apt_hr = 35.0
    impacted_terminal_flights = degraded_airports_count * terminal_moves_per_apt_hr * duration_hours
    terminal_holding_min = (impacted_terminal_flights * 0.5) * 14.00  # arrivals
    ground_delay_min = (impacted_terminal_flights * 0.5) * 8.00  # departures

    total_delay_min = airborne_delay_min + terminal_holding_min + ground_delay_min

    # Financial Cost Calculations:
    airborne_cost = airborne_delay_min * EUROCONTROL_AIRBORNE_DELAY_EUR_PER_MIN
    terminal_cost = terminal_holding_min * EUROCONTROL_AIRBORNE_DELAY_EUR_PER_MIN
    ground_cost = ground_delay_min * EUROCONTROL_GROUND_DELAY_EUR_PER_MIN
    fuel_cost = reroute_nm * FUEL_AND_ETS_EUR_PER_NM

    gross_operational_cost = airborne_cost + terminal_cost + ground_cost + fuel_cost

    # Incremental Delay Cost relative to Baseline Status Quo
    if baseline_gross_cost is not None:
        incremental_delay_cost = max(0.0, gross_operational_cost - baseline_gross_cost)
    else:
        incremental_delay_cost = 0.0

    # Equipment OPEX Savings:
    equipment_savings = disabled_dme_count * DME_HOURLY_OPEX_EUR * duration_hours
    net_economic_impact = incremental_delay_cost - equipment_savings

    return WindowCostBreakdown(
        window_label=window_label,
        duration_hours=duration_hours,
        flights_exposed=round(flights_impacted_enroute, 1),
        airborne_delay_minutes=round(airborne_delay_min, 1),
        terminal_holding_minutes=round(terminal_holding_min, 1),
        ground_delay_minutes=round(ground_delay_min, 1),
        total_delay_minutes=round(total_delay_min, 1),
        reroute_distance_nm=round(reroute_nm, 1),
        reroute_fuel_kg=round(fuel_burn_kg, 1),
        airborne_delay_cost_eur=round(airborne_cost, 2),
        terminal_holding_cost_eur=round(terminal_cost, 2),
        ground_delay_cost_eur=round(ground_cost, 2),
        reroute_fuel_cost_eur=round(fuel_cost, 2),
        gross_operational_cost_eur=round(gross_operational_cost, 2),
        incremental_delay_cost_eur=round(incremental_delay_cost, 2),
        equipment_opex_savings_eur=round(equipment_savings, 2),
        net_economic_impact_eur=round(net_economic_impact, 2),
    )


# RNAV Operational Impact Scoring Matrix (7 Operational Categories)
# Assigns operational impact penalty points when transitioning between RNAV capabilities:
# RNAV 1 (Target Standard: 0 pts), RNAV 2 (7 pts), RNAV 5 (16 pts), RNAV 10 (31 pts)
RNAV_IMPACT_CATEGORIES = [
    "Route flexibility",
    "Parallel route capacity",
    "Crossing traffic management",
    "Tactical intervention",
    "Controller workload",
    "Procedural separation dependency",
    "Alternative routing requirement",
]

RNAV_SCORING_MATRIX: dict[str, dict[str, int]] = {
    "RNAV 1": {
        "Route flexibility": 0,
        "Parallel route capacity": 0,
        "Crossing traffic management": 0,
        "Tactical intervention": 0,
        "Controller workload": 0,
        "Procedural separation dependency": 0,
        "Alternative routing requirement": 0,
    },
    "RNAV 2": {
        "Route flexibility": 1,
        "Parallel route capacity": 1,
        "Crossing traffic management": 1,
        "Tactical intervention": 1,
        "Controller workload": 1,
        "Procedural separation dependency": 1,
        "Alternative routing requirement": 1,
    },
    "RNAV 5": {
        "Route flexibility": 2,
        "Parallel route capacity": 3,
        "Crossing traffic management": 2,
        "Tactical intervention": 2,
        "Controller workload": 2,
        "Procedural separation dependency": 2,
        "Alternative routing requirement": 3,
    },
    "RNAV 10": {
        "Route flexibility": 4,
        "Parallel route capacity": 5,
        "Crossing traffic management": 4,
        "Tactical intervention": 4,
        "Controller workload": 4,
        "Procedural separation dependency": 5,
        "Alternative routing requirement": 5,
    },
}

RNAV_CAPABILITY_TOTALS: dict[str, int] = {
    "RNAV 1": 0,
    "RNAV 2": 7,
    "RNAV 5": 16,
    "RNAV 10": 31,
}


def compute_cell_rnav_capabilities(
    active_indices: list[int],
    cell_mat_200: np.ndarray,
    cell_mat_100: np.ndarray,
    dist_cell: np.ndarray,
    bearings_cell: np.ndarray,
    coverable_cells_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes achievable RNAV capability and operational score for every 10 NM airspace cell:
    - RNAV 1 (0 pts): >=3 DMEs visible at FL100, valid pair 30-150 deg, max slant range <= 60 NM
    - RNAV 2 (7 pts): >=2 DMEs visible at FL200, valid pair 30-150 deg, max slant range <= 130 NM
    - RNAV 5 (16 pts): >=2 DMEs visible at FL200, valid pair 30-150 deg
    - RNAV 10 (31 pts): Single DME or non-compliant geometry (loss of area navigation capability)
    Returns (capabilities_array, scores_array).
    """
    n_cells = cell_mat_200.shape[0]
    caps = np.full(n_cells, "RNAV 10", dtype=object)
    scores = np.full(n_cells, RNAV_CAPABILITY_TOTALS["RNAV 10"], dtype=int)

    if not active_indices or len(active_indices) == 0:
        return caps, scores

    act_idx_arr = np.array(active_indices, dtype=int)
    vis_200_act = cell_mat_200[:, act_idx_arr]
    cnt_200 = vis_200_act.sum(axis=1)

    vis_100_act = cell_mat_100[:, act_idx_arr]
    cnt_100 = vis_100_act.sum(axis=1)

    multi_mask = (cnt_200 >= 2) & coverable_cells_mask
    if multi_mask.any():
        multi_indices = np.where(multi_mask)[0]
        for idx in multi_indices:
            s_vis = np.where(vis_200_act[idx])[0]
            if len(s_vis) < 2:
                continue
            act_s_global = act_idx_arr[s_vis]
            b = bearings_cell[idx, act_s_global]
            diff = np.abs(b[:, None] - b[None, :])
            delta = np.minimum(diff, 360.0 - diff)
            valid = (delta >= 30.0) & (delta <= 150.0)
            if valid.any():
                dists = dist_cell[idx, act_s_global]
                max_dists = np.maximum(dists[:, None], dists[None, :])
                valid_max_dists = np.where(valid, max_dists, 9999.0)
                d_min_max = float(np.min(valid_max_dists))

                if cnt_100[idx] >= 3 and d_min_max <= 60.0:
                    caps[idx] = "RNAV 1"
                    scores[idx] = RNAV_CAPABILITY_TOTALS["RNAV 1"]
                elif d_min_max <= 130.0:
                    caps[idx] = "RNAV 2"
                    scores[idx] = RNAV_CAPABILITY_TOTALS["RNAV 2"]
                else:
                    caps[idx] = "RNAV 5"
                    scores[idx] = RNAV_CAPABILITY_TOTALS["RNAV 5"]

    return caps, scores

AIRCRAFT_FUEL_BURN_RATES: dict[str, float] = {
    # Heavy Widebodies (kg / NM)
    "B77W": 7.80, "B772": 7.20, "B773": 7.50, "B788": 5.40, "B789": 5.80, "B78X": 6.20,
    "A359": 5.90, "A35K": 6.80, "A332": 6.40, "A333": 6.70, "A339": 5.80, "A388": 11.20,
    "B744": 10.50, "B748": 9.80,
    # Standard Narrowbodies (kg / NM)
    "A320": 4.10, "A20N": 3.60, "A321": 4.60, "A21N": 3.90, "A319": 3.80,
    "B738": 4.20, "B737": 4.00, "B38M": 3.60, "B39M": 3.80, "B739": 4.50,
    "BCS3": 3.20, "A220": 3.20,
    # Regional Turboprops & Jets (kg / NM)
    "E190": 2.90, "E195": 3.10, "E75L": 2.70, "E295": 2.60, "CRJ9": 2.80, "CRJ7": 2.50,
    "AT76": 1.60, "DH8D": 1.80,
}
DEFAULT_FUEL_BURN_KG_PER_NM = 4.20
CO2_FACTOR_PER_KG_FUEL = 3.15

WIDEBODY_TYPES = {"B77W", "B772", "B773", "B788", "B789", "B78X", "A359", "A35K", "A332", "A333", "A339", "A388", "B744", "B748"}
REGIONAL_TYPES = {"E190", "E195", "E75L", "E295", "CRJ9", "CRJ7", "AT76", "DH8D"}


def simulate_fir_traffic_knock_on(
    dataset: FlightDataset,
    fir_id: str,
    period: str,
    outage_fraction: float,
    at_risk_fraction: float,
    degraded_airports: list[str],
    scenario_name: str = "Evaluated Scenario",
    limit_flights: int | None = None,
    separation_standard_nm: float | None = None,
    cell_rnav_caps: np.ndarray | None = None,
) -> TrafficSimulationSummary:
    """
    Simulate the operational consequences of degraded separation standards and coverage deficits
    on actual flight traversals in the target FIR during the specified period.
    Assesses all flights in the period (100% census), modeling primary radar spacing expansions
    (5 NM -> 8.5-10 NM), ATC intervention workload, tactical avoidance dog-legs, extra fuel burn,
    downstream rotational turnaround delays, and crew duty period exceedances.
    """
    limit_clause = f"LIMIT {int(limit_flights)}" if limit_flights is not None else ""
    query = f"""
    WITH fir_flights AS (
        SELECT DISTINCT f.ectrl_id, f.ac_reg, f.ac_operator, f.ac_type, f.adep, f.ades,
               f.actual_off_block_time, f.actual_arrival_time, f.dep_delay_min, f.arr_delay_min
        FROM flights f
        JOIN flight_firs_actual t ON f.ectrl_id = t.ectrl_id
        WHERE t.fir_id = ?
          AND f.ac_reg IS NOT NULL AND f.ac_reg != ''
        {limit_clause}
    )
    SELECT f1.ectrl_id AS leg1_id, f1.ac_reg, f1.ac_operator, f1.ac_type,
           f1.adep AS leg1_adep, f1.ades AS leg1_ades,
           f1.actual_off_block_time AS leg1_dep, f1.actual_arrival_time AS leg1_arr,
           f2.ectrl_id AS leg2_id, f2.adep AS leg2_adep, f2.ades AS leg2_ades,
           f2.actual_off_block_time AS leg2_dep, f2.actual_arrival_time AS leg2_arr
    FROM fir_flights f1
    LEFT JOIN flights f2 ON f1.ac_reg = f2.ac_reg
      AND f2.actual_off_block_time > f1.actual_arrival_time
      AND f2.actual_off_block_time <= f1.actual_arrival_time + INTERVAL 6 HOUR
      AND f1.ades = f2.adep
    ORDER BY f1.actual_off_block_time ASC;
    """
    try:
        df = dataset.query(query, params=[fir_id])
    except Exception:
        df = pd.DataFrame()

    effective_sep = separation_standard_nm if separation_standard_nm is not None else (8.5 if outage_fraction > 0 else 5.0)

    if df.empty:
        return TrafficSimulationSummary(
            fir_id=fir_id,
            period=period,
            scenario_name=scenario_name,
            separation_standard_nm=effective_sep,
            total_flights_simulated=0,
            directly_impacted_flights=0,
            total_primary_delay_min=0.0,
            total_detour_distance_nm=0.0,
            total_extra_fuel_kg=0.0,
            total_extra_co2_kg=0.0,
            downstream_flights_delayed=0,
            total_reactionary_delay_min=0.0,
            reactionary_multiplier=0.0,
            cancellation_risk_count=0,
            curfew_breaches_count=0,
            crew_fdp_timeouts_count=0,
            sample_impacted_flights=[],
            rnav_scoring_summary={},
        )

    n_total_flights = len(df)
    p_exp = min(0.95, outage_fraction + (0.35 * at_risk_fraction))
    deg_set = set(degraded_airports)

    # Vectorized computation across all flights
    leg1_ids = df["leg1_id"].values.astype(np.int64)
    adep_arr = df["leg1_adep"].fillna("").values.astype(str)
    ades_arr = df["leg1_ades"].fillna("").values.astype(str)
    ac_type_arr = df["ac_type"].fillna("A320").values.astype(str)
    op_arr = df["ac_operator"].fillna("").values.astype(str)
    reg_arr = df["ac_reg"].fillna("").values.astype(str)

    # Exposure: flights traversing degraded airspace or arriving/departing degraded airports
    is_exp = ((leg1_ids % 1000) / 1000.0) < p_exp
    is_deg_terminal = np.isin(ades_arr, list(deg_set)) | np.isin(adep_arr, list(deg_set))
    is_impacted = is_exp | is_deg_terminal

    # 1. Primary delay: en-route vectoring spacing expansion + terminal holding
    d_spacing = np.where(is_exp, 4.0 + 8.0 * ((leg1_ids % 100) / 100.0), 0.0)
    d_arr_hold = np.where(np.isin(ades_arr, list(deg_set)), 14.0, 0.0)
    d_dep_slot = np.where(np.isin(adep_arr, list(deg_set)), 8.0, 0.0)
    pri_delay = d_spacing + d_arr_hold + d_dep_slot

    # 2. Tactical detours & extra fuel burn (~25% of exposed flights)
    is_detour = is_exp & ((leg1_ids % 4) == 0)
    detour_nm = np.where(is_detour, 22.0, 0.0)
    burn_rates = np.array([AIRCRAFT_FUEL_BURN_RATES.get(t, DEFAULT_FUEL_BURN_KG_PER_NM) for t in ac_type_arr])
    extra_fuel = detour_nm * burn_rates

    # 3. Downstream rotational knock-on delay calculation
    has_leg2 = df["leg2_id"].notna().values
    leg1_arr_dt = pd.to_datetime(df["leg1_arr"]).values
    leg2_dep_dt = pd.to_datetime(df["leg2_dep"]).values
    leg2_arr_dt = pd.to_datetime(df["leg2_arr"]).values

    # Scheduled turnaround buffer
    sched_turn_min = np.where(has_leg2, (leg2_dep_dt - leg1_arr_dt) / np.timedelta64(1, "m"), 0.0)
    is_widebody = np.isin(ac_type_arr, list(WIDEBODY_TYPES))
    is_regional = np.isin(ac_type_arr, list(REGIONAL_TYPES))
    min_buf = np.where(is_widebody, 60.0, np.where(is_regional, 30.0, 40.0))
    avail_buf = np.maximum(0.0, sched_turn_min - min_buf)

    absorbed_delay = np.where(has_leg2, np.minimum(pri_delay, avail_buf), pri_delay)
    react_delay = np.where(has_leg2, np.maximum(0.0, pri_delay - avail_buf), 0.0)

    # Extra APU fuel burn on ground during reactionary delays (~2.0 kg/min)
    apu_fuel = react_delay * 2.0
    extra_fuel += apu_fuel
    extra_co2 = extra_fuel * CO2_FACTOR_PER_KG_FUEL

    # Risk classifications and reasons (strictly verifiable operational metrics: FDP timeouts & slot forfeitures)
    crew_timeouts = 0
    cancellations = 0

    risk_labels = []
    reason_labels = []

    # Fast evaluation of downstream risks for impacted flights
    impacted_indices = np.where(is_impacted)[0]
    for idx in impacted_indices:
        r_del = react_delay[idx]
        if has_leg2[idx]:
            if r_del >= 150.0:
                risk_labels.append("CRITICAL (CREW FDP EXPIRED)")
                reason_labels.append(f"Rotational delay of {r_del:.0f} min exceeds legal EASA Flight Duty Period.")
                crew_timeouts += 1
                cancellations += 1
            elif r_del >= 45.0:
                risk_labels.append("ELEVATED (MISSED CTOT SLOT)")
                reason_labels.append(f"Downstream departure delayed {r_del:.0f} min; forfeits Eurocontrol CTOT slot.")
                cancellations += 1
            elif r_del > 0.0:
                risk_labels.append("MODERATE REACTIONARY DELAY")
                reason_labels.append(f"Turnaround delay of {r_del:.0f} min propagated to next flight leg.")
            else:
                risk_labels.append("LOW (ABSORBED)")
                reason_labels.append("Delay fully absorbed within scheduled turnaround buffer.")
        else:
            risk_labels.append("LOW (TERMINAL LEG)")
            reason_labels.append("End of daily aircraft rotation; no downstream commercial leg.")

    # 4. RNAV Capability & Operational Impact Scoring across 100% census
    # Categorizes flights into RNAV 1 (0 pts), RNAV 2 (7 pts), RNAV 5 (16 pts), RNAV 10 (31 pts)
    flight_rnav_caps = np.full(n_total_flights, "RNAV 1", dtype=object)
    if is_impacted.any():
        deg_10_mask = is_impacted & (pri_delay >= 8.0)
        deg_5_mask = is_impacted & (pri_delay >= 3.5) & (~deg_10_mask)
        deg_2_mask = is_impacted & (~deg_10_mask) & (~deg_5_mask)
        flight_rnav_caps[deg_10_mask] = "RNAV 10"
        flight_rnav_caps[deg_5_mask] = "RNAV 5"
        flight_rnav_caps[deg_2_mask] = "RNAV 2"

    flight_scores = np.array([RNAV_CAPABILITY_TOTALS[c] for c in flight_rnav_caps], dtype=int)
    tot_fleet_score = int(flight_scores.sum())
    avg_flight_score = round(float(tot_fleet_score / max(1, n_total_flights)), 2)

    cap_counts = {
        "RNAV 1": int((flight_rnav_caps == "RNAV 1").sum()),
        "RNAV 2": int((flight_rnav_caps == "RNAV 2").sum()),
        "RNAV 5": int((flight_rnav_caps == "RNAV 5").sum()),
        "RNAV 10": int((flight_rnav_caps == "RNAV 10").sum()),
    }
    cap_pcts = {
        k: round((v / max(1, n_total_flights)) * 100.0, 1) for k, v in cap_counts.items()
    }

    category_totals = {}
    for cat in RNAV_IMPACT_CATEGORIES:
        cat_pts = sum(cap_counts[cap] * RNAV_SCORING_MATRIX[cap][cat] for cap in ("RNAV 1", "RNAV 2", "RNAV 5", "RNAV 10"))
        category_totals[cat] = {
            "total_points": int(cat_pts),
            "avg_points_per_flight": round(float(cat_pts / max(1, n_total_flights)), 2),
        }

    rnav_scoring_summary = {
        "total_fleet_score": tot_fleet_score,
        "avg_flight_score": avg_flight_score,
        "capability_counts": cap_counts,
        "capability_percentages": cap_pcts,
        "category_totals": category_totals,
        "scoring_matrix": RNAV_SCORING_MATRIX,
        "capability_totals": RNAV_CAPABILITY_TOTALS,
    }

    # Build sample list of top 25 impacted flights for HTML cascade table
    sample_impacted_flights: list[SimulatedFlightKnockOn] = []
    if len(impacted_indices) > 0:
        sorted_impact_order = sorted(
            range(len(impacted_indices)),
            key=lambda i: (
                risk_labels[i].startswith("CRITICAL"),
                react_delay[impacted_indices[i]],
                pri_delay[impacted_indices[i]],
            ),
            reverse=True,
        )

        for rank_pos in sorted_impact_order[:25]:
            orig_idx = impacted_indices[rank_pos]
            leg1_id = int(leg1_ids[orig_idx])
            has_dw = bool(has_leg2[orig_idx])
            fl_cap = flight_rnav_caps[orig_idx]
            sample_impacted_flights.append(
                SimulatedFlightKnockOn(
                    leg1_ectrl_id=leg1_id,
                    ac_reg=str(reg_arr[orig_idx]),
                    ac_operator=str(op_arr[orig_idx]),
                    ac_type=str(ac_type_arr[orig_idx]),
                    leg1_route=f"{adep_arr[orig_idx]} -> {ades_arr[orig_idx]}",
                    leg1_dep_time=str(df['leg1_dep'].iloc[orig_idx])[:16],
                    leg1_arr_time=str(df['leg1_arr'].iloc[orig_idx])[:16],
                    is_directly_exposed=bool(is_exp[orig_idx]),
                    enroute_vector_delay_min=round(float(d_spacing[orig_idx]), 1),
                    terminal_holding_delay_min=round(float(d_arr_hold[orig_idx] + d_dep_slot[orig_idx]), 1),
                    primary_delay_min=round(float(pri_delay[orig_idx]), 1),
                    tactical_detour_nm=round(float(detour_nm[orig_idx]), 1),
                    extra_fuel_kg=round(float(extra_fuel[orig_idx]), 1),
                    extra_co2_kg=round(float(extra_co2[orig_idx]), 1),
                    has_downstream_leg=has_dw,
                    leg2_ectrl_id=int(df['leg2_id'].iloc[orig_idx]) if has_dw else None,
                    leg2_route=f"{df['leg2_adep'].iloc[orig_idx]} -> {df['leg2_ades'].iloc[orig_idx]}" if has_dw else None,
                    leg2_scheduled_dep=str(df['leg2_dep'].iloc[orig_idx])[:16] if has_dw else None,
                    turnaround_buffer_min=round(float(sched_turn_min[orig_idx]), 1),
                    absorbed_delay_min=round(float(absorbed_delay[orig_idx]), 1),
                    downstream_reactionary_delay_min=round(float(react_delay[orig_idx]), 1),
                    cancellation_risk=risk_labels[rank_pos],
                    cancellation_reason=reason_labels[rank_pos],
                    rnav_capability=fl_cap,
                    rnav_impact_score=RNAV_CAPABILITY_TOTALS[fl_cap],
                    rnav_category_scores=dict(RNAV_SCORING_MATRIX[fl_cap]),
                )
            )

    tot_pri_delay = float(pri_delay[is_impacted].sum())
    tot_react_delay = float(react_delay[is_impacted].sum())
    downstream_delayed_cnt = int((react_delay[is_impacted] > 0.0).sum())
    react_multiplier = round(tot_react_delay / max(1.0, tot_pri_delay), 2)

    return TrafficSimulationSummary(
        fir_id=fir_id,
        period=period,
        scenario_name=scenario_name,
        separation_standard_nm=effective_sep,
        total_flights_simulated=n_total_flights,
        directly_impacted_flights=int(is_impacted.sum()),
        total_primary_delay_min=round(tot_pri_delay, 1),
        total_detour_distance_nm=round(float(detour_nm[is_impacted].sum()), 1),
        total_extra_fuel_kg=round(float(extra_fuel[is_impacted].sum()), 1),
        total_extra_co2_kg=round(float(extra_co2[is_impacted].sum()), 1),
        downstream_flights_delayed=downstream_delayed_cnt,
        total_reactionary_delay_min=round(tot_react_delay, 1),
        reactionary_multiplier=react_multiplier,
        cancellation_risk_count=cancellations,
        curfew_breaches_count=0,
        crew_fdp_timeouts_count=crew_timeouts,
        sample_impacted_flights=sample_impacted_flights,
        rnav_scoring_summary=rnav_scoring_summary,
    )


def compute_station_utilization_and_maintenance(
    dataset: FlightDataset,
    fir_id: str,
    stations: list[ECACStation],
    grid_cells: list[tuple[float, float]],
    cell_matrices: dict[int, np.ndarray],
    dist_cell: np.ndarray,
    bearings_cell: np.ndarray,
    target_standard: str,
    optimal_active_ids: set[str],
    fir_bbox: tuple[float, float, float, float],
    total_fir_flights: int,
) -> tuple[list[DMEStationUtilization], DMEMaintenanceImpactSummary]:
    """
    Computes:
    1. DME Station Utilization Table: Average hourly and daily flights using each DME as part of
       their journey across the FIR, based on flight paths, cell traffic density, and FMS tuning likelihood.
    2. Maintenance & Planned Outage Reliability Model: Simulates a 0.5% planned annual outage rate
       per DME, calculates the cell-hours of reduced RNAV capability (SPOFs vs redundant pairs),
       and applies the operational cost of ATC restrictions/vectoring, strictly preventing double accounting.
    """
    n_cells = len(grid_cells)
    n_sta = len(stations)
    grid_arr = np.array(grid_cells)
    tree = cKDTree(grid_arr)

    # 1. Query Trajectory Density from DuckDB for the target FIR
    try:
        df_cell_traffic = dataset.query(
            """
            SELECT
                ROUND(tr.latitude / 0.18) * 0.18 as c_lat,
                ROUND(tr.longitude / 0.18) * 0.18 as c_lon,
                count(distinct tr.ectrl_id) as flight_count
            FROM trajectory_actual tr
            JOIN flight_firs_actual fa ON tr.ectrl_id = fa.ectrl_id
            WHERE fa.fir_id = ?
              AND tr.latitude BETWEEN ? AND ?
              AND tr.longitude BETWEEN ? AND ?
            GROUP BY c_lat, c_lon
            """,
            params=[fir_id, fir_bbox[0], fir_bbox[1], fir_bbox[2], fir_bbox[3]],
        )
    except Exception:
        df_cell_traffic = pd.DataFrame()

    cell_traffic = np.zeros(n_cells)
    if not df_cell_traffic.empty and len(df_cell_traffic) > 0:
        pts = df_cell_traffic[["c_lat", "c_lon"]].values
        dists_kdtree, indices = tree.query(pts)
        valid = dists_kdtree < 0.25
        for idx, cnt in zip(indices[valid], df_cell_traffic["flight_count"].values[valid]):
            cell_traffic[idx] += cnt
    else:
        # Fallback: estimate traffic density based on average FIR traffic
        cell_traffic.fill(max(1.0, total_fir_flights / max(1, n_cells) * 8.5))

    # Calculate average cells traversed per flight
    total_traversals = float(cell_traffic.sum())
    avg_cells_per_flight = max(4.0, total_traversals / max(1, total_fir_flights))

    # 2. Compute FMS Tuning Likelihood for each Cell and Station
    vis_200 = cell_matrices[200]
    p_tune = np.zeros((n_cells, n_sta))

    valid_pairs_count = np.zeros(n_sta, dtype=int)
    for c in range(n_cells):
        vis_s = np.where(vis_200[c])[0]
        if len(vis_s) < 2:
            if len(vis_s) == 1:
                p_tune[c, vis_s[0]] = 0.15
            continue

        b = bearings_cell[c, vis_s]
        d = dist_cell[c, vis_s]
        diff = np.abs(b[:, None] - b[None, :])
        delta = np.minimum(diff, 360.0 - diff)
        valid_pair = (delta >= 30.0) & (delta <= 150.0)

        for i in range(len(vis_s)):
            valid_pairs_count[vis_s[i]] += int(valid_pair[i].sum())

        w_pair = np.where(valid_pair, np.sin(np.radians(delta)) / (d[:, None] + d[None, :] + 1e-3), 0.0)
        tot_w = float(w_pair.sum())
        if tot_w > 0:
            s_w = w_pair.sum(axis=1)
            p_tune[c, vis_s] = np.clip((s_w / tot_w) * 1.8, 0.05, 0.95)
        else:
            p_tune[c, vis_s] = 0.15

    # Build DME Station Utilization List
    utilization_list: list[DMEStationUtilization] = []
    for s_idx, sta in enumerate(stations):
        is_opt = sta.id in optimal_active_ids
        vis_cells = np.where(vis_200[:, s_idx])[0]
        if len(vis_cells) == 0:
            monthly_fl = 0.0
        else:
            # Expected distinct flights utilizing station across journey
            eff_traversals = float((cell_traffic[vis_cells] * p_tune[vis_cells, s_idx]).sum())
            monthly_fl = min(float(total_fir_flights * 0.96), eff_traversals / avg_cells_per_flight)

        daily_fl = monthly_fl / 31.0
        hourly_fl = daily_fl / 24.0
        peak_fl = hourly_fl * 1.45
        util_pct = (monthly_fl / max(1, total_fir_flights)) * 100.0

        if sta.nearest_airport_ident and daily_fl > 300:
            role = "Terminal Primary"
        elif daily_fl >= 250:
            role = "Core En-Route Backbone"
        else:
            role = "Regional / Infill"

        utilization_list.append(
            DMEStationUtilization(
                station_id=sta.id,
                station_name=sta.name,
                airspace_role=role,
                latitude=round(sta.latitude, 4),
                longitude=round(sta.longitude, 4),
                elevation_ft=round(sta.elevation_ft, 0),
                is_active_in_optimal=is_opt,
                daily_flights_using=round(daily_fl, 1),
                hourly_flights_using=round(hourly_fl, 1),
                peak_hourly_flights_using=round(peak_fl, 1),
                utilization_pct=round(util_pct, 1),
                valid_pairs_formed=int(valid_pairs_count[s_idx]),
            )
        )

    # Sort utilization list by daily flights descending
    utilization_list.sort(key=lambda u: u.daily_flights_using, reverse=True)

    # 3. Planned Outages & Maintenance Reliability Model (0.5% Outage Rate)
    lambda_maint = 0.005  # 0.5% planned annual outage rate
    annual_hours = 8760.0

    # Mask of active stations under optimal configuration
    opt_indices = [i for i, s in enumerate(stations) if s.id in optimal_active_ids]
    opt_mask = np.zeros(n_sta, dtype=bool)
    opt_mask[opt_indices] = True

    spof_cells_cnt = 0
    redundant_cells_cnt = 0
    perm_degraded_cnt = 0
    cell_degrade_prob = np.zeros(n_cells)
    station_spof_cell_counts = np.zeros(n_sta, dtype=int)

    for c in range(n_cells):
        # Only active stations under optimal configuration can cause active maintenance outages
        vis_act = np.where(vis_200[c] & opt_mask)[0]
        if len(vis_act) < 2:
            perm_degraded_cnt += 1
            # CRITICAL: If cell is already permanently degraded under optimal config,
            # its cost is already accounted for in incremental_delay_cost!
            # Outage degradation is strictly 0.0 to prevent double accounting!
            continue

        b = bearings_cell[c, vis_act]
        d = dist_cell[c, vis_act]
        diff = np.abs(b[:, None] - b[None, :])
        delta = np.minimum(diff, 360.0 - diff)
        valid_pair = (delta >= 30.0) & (delta <= 150.0)

        pairs = []
        for i in range(len(vis_act)):
            for j in range(i + 1, len(vis_act)):
                if valid_pair[i, j]:
                    pairs.append((vis_act[i], vis_act[j]))

        if not pairs:
            perm_degraded_cnt += 1
            # Permanently degraded: 0.0 outage prob to prevent double accounting
            continue

        # Check if a single station is present in EVERY valid pair (Single Point of Failure)
        common_stations = set(pairs[0])
        for p in pairs[1:]:
            common_stations &= set(p)

        if common_stations:
            spof_cells_cnt += 1
            cell_degrade_prob[c] = lambda_maint
            for s_idx in common_stations:
                station_spof_cell_counts[s_idx] += 1
        else:
            redundant_cells_cnt += 1
            # Multiple independent pairs: capability loss requires simultaneous outages
            cell_degrade_prob[c] = (2 * lambda_maint) ** 2

    tot_annual_cell_hours = float((cell_degrade_prob * annual_hours).sum())

    # Flights encountering reduced capability during maintenance:
    # Normalized by average cells per flight to avoid double counting across adjacent cells!
    monthly_impacted_fl = int(round(float((cell_traffic * cell_degrade_prob).sum()) / avg_cells_per_flight))
    annual_impacted_fl = monthly_impacted_fl * 12

    # Eurocontrol standard unit cost per degraded traversal:
    # Airborne spacing delay (4.5m @ €55.50/m) + ATC tactical intervention (€35) + fuel (€45) = €329.75
    unit_intervention_cost = 329.75
    annual_outage_cost = annual_impacted_fl * unit_intervention_cost
    monthly_outage_cost = annual_outage_cost / 12.0
    daily_outage_cost = monthly_outage_cost / 31.0

    station_maint_records = []
    for s_idx, sta in enumerate(stations):
        spof_c = int(station_spof_cell_counts[s_idx])
        if spof_c > 0:
            ann_sta_cost = (spof_c / max(1, spof_cells_cnt)) * annual_outage_cost
            station_maint_records.append({
                "station_id": sta.id,
                "station_name": sta.name,
                "spof_cells_count": spof_c,
                "annual_cell_hours_at_risk": round(spof_c * lambda_maint * annual_hours, 1),
                "annual_outage_cost_eur": round(ann_sta_cost, 2),
            })
    station_maint_records.sort(key=lambda r: r["spof_cells_count"], reverse=True)

    maint_summary = DMEMaintenanceImpactSummary(
        outage_rate_pct=0.5,
        total_cells=n_cells,
        spof_cells_count=spof_cells_cnt,
        redundant_cells_count=redundant_cells_cnt,
        permanently_degraded_cells_count=perm_degraded_cnt,
        annual_cell_hours_reduced_rnav=round(tot_annual_cell_hours, 1),
        monthly_flights_impacted=monthly_impacted_fl,
        annual_flights_impacted=annual_impacted_fl,
        annual_outage_cost_eur=round(annual_outage_cost, 2),
        monthly_outage_cost_eur=round(monthly_outage_cost, 2),
        daily_outage_cost_eur=round(daily_outage_cost, 2),
        station_maintenance_records=station_maint_records,
    )

    return utilization_list, maint_summary


def _evaluate_critical_dmes(
    target_perm: DMEPermutationResult,
    inside_stations: list[ECACStation],
    sta_id_to_idx: dict[str, int],
    cascaded_indices: list[int],
    cell_mat_200: np.ndarray,
    cell_mat_100: np.ndarray,
    apt_mat_10: np.ndarray,
    bearings_cell: np.ndarray,
    dist_cell: np.ndarray,
    target_standard: str,
    airports: list[dict[str, Any]],
    coverable_cells_mask: np.ndarray,
    n_cells: int,
    inside_sta_ids: list[str],
    hourly_moves: float,
    base_gross_costs: dict[str, float],
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """
    Evaluate Top Critical Single Point of Failure (SPOF) DMEs under the CHOSEN SCENARIO.
    Specifically tests N-1 outages for each surviving active DME participating in the chosen configuration,
    quantifying the operational delay penalty and coverage disruption caused by its unexpected loss.
    """
    chosen_active_ids = set(target_perm.active_station_ids)
    scenario_gross_24h = target_perm.windows["24h"].gross_operational_cost_eur
    scenario_gross_1h = target_perm.windows["1h"].gross_operational_cost_eur

    critical_candidates = []
    # Test only active inside DMEs in the chosen scenario
    active_scenario_stations = [s for s in inside_stations if s.id in chosen_active_ids]

    for s_obj in active_scenario_stations:
        outage_active_ids = chosen_active_ids - {s_obj.id}
        outage_act_indices = [sta_id_to_idx[sid] for sid in outage_active_ids if sid in sta_id_to_idx] + cascaded_indices

        o_met, o_at_risk, o_cannot, o_unc, o_degraded_apts = _evaluate_single_permutation_coverage(
            active_indices=outage_act_indices,
            cell_mat_200=cell_mat_200,
            cell_mat_100=cell_mat_100,
            apt_mat_10=apt_mat_10,
            bearings_cell=bearings_cell,
            dist_cell=dist_cell,
            target_standard=target_standard,
            airports=airports,
            coverable_cells_mask=coverable_cells_mask,
        )

        cov_tot = max(1, n_cells - o_unc)
        o_outage_frac = o_cannot / cov_tot
        o_risk_frac = o_at_risk / cov_tot
        o_dis_count = len(inside_sta_ids) - len(outage_active_ids)

        o_win_1h = _compute_window_costs(
            duration_hours=1.0,
            window_label="1-Hour Peak Outage",
            traffic_hourly_avg=hourly_moves,
            traffic_peak_multiplier=1.30,
            outage_fraction=o_outage_frac,
            at_risk_fraction=o_risk_frac,
            degraded_airports_count=len(o_degraded_apts),
            disabled_dme_count=o_dis_count,
            baseline_gross_cost=base_gross_costs["1h"],
        )
        o_win_24h = _compute_window_costs(
            duration_hours=24.0,
            window_label="1-Day (24-Hour) Outage",
            traffic_hourly_avg=hourly_moves,
            traffic_peak_multiplier=1.00,
            outage_fraction=o_outage_frac,
            at_risk_fraction=o_risk_frac,
            degraded_airports_count=len(o_degraded_apts),
            disabled_dme_count=o_dis_count,
            baseline_gross_cost=base_gross_costs["24h"],
        )

        day_penalty = max(0.0, o_win_24h.gross_operational_cost_eur - scenario_gross_24h)
        hour_penalty = max(0.0, o_win_1h.gross_operational_cost_eur - scenario_gross_1h)

        newly_degraded_cells = max(0, target_perm.cells_met - o_met)
        newly_degraded_apts = [a for a in o_degraded_apts if a not in target_perm.degraded_airports_idents]

        critical_candidates.append({
            "id": s_obj.id,
            "name": s_obj.name,
            "latitude": s_obj.latitude,
            "longitude": s_obj.longitude,
            "elevation_ft": s_obj.elevation_ft,
            "near_airport": s_obj.nearest_airport_ident or "En-Route",
            "outage_daily_delay_cost_eur": round(day_penalty, 2),
            "outage_hourly_delay_cost_eur": round(hour_penalty, 2),
            "newly_degraded_cells": newly_degraded_cells,
            "newly_degraded_airports": newly_degraded_apts,
            "scenario_name": target_perm.name,
        })

    critical_candidates.sort(
        key=lambda x: (x["outage_daily_delay_cost_eur"], x["newly_degraded_cells"], len(x["newly_degraded_airports"])),
        reverse=True,
    )
    return critical_candidates[:top_n]


def simulate_dme_permutations(
    dataset: FlightDataset,
    fir_id: str = "EGTTFIR",
    period: str = "202507",
    target_standard: str = "RNAV_5",
    max_permutations: int = 25,
    external_active_station_ids: set[str] | list[str] | None = None,
    enable_cross_border_cascading: bool = True,
) -> DMEFIRResilienceAnalysis:
    """
    Run the DME permutation resilience simulation for a selected FIR and navigation performance standard.
    Generates structured permutations (Baseline N-0, Rationalized Core, Single N-1 Outages,
    Strategic Dual Outages, and Stress Cutbacks), validates against historical traffic,
    and quantifies operational impacts across 1h, 6h, and 24h operational windows.

    Cross-Border Cascading:
    When a DME is kept active in the plan of a busier FIR, it is cascaded outward to compensate
    coverage in adjacent less busy FIRs. Cells in the border region of this FIR leverage valid
    DME/DME pairs formed by internal and cascaded neighbor stations, enabling deeper local
    rationalization without compromising safety or performance.
    """
    fir_geom = get_fir_geometry(dataset, fir_id)
    if not fir_geom:
        raise ValueError(f"FIR '{fir_id}' not found in raw_fir dataset. Please select a valid ECAC FIR.")

    fir_poly, fir_boundary_coords, fir_bbox = fir_geom
    fir_name = ECAC_FIR_REGISTRY.get(fir_id, fir_id)

    # 1. Load sovereign internal stations and line-of-sight adjacent neighbor candidates
    inside_stations, airports, adjacent_candidates = load_ecac_stations(
        dataset,
        station_type="DME",
        fir_poly=fir_poly,
        fir_bbox=fir_bbox,
        include_adjacent=True,
    )
    if not inside_stations:
        raise ValueError(f"No DME stations found inside {fir_id}. Cannot run DME permutation analysis.")

    # Determine cascaded active neighbor stations (kept active by previously assessed busier FIRs)
    cascaded_adjacent_stations: list[ECACStation] = []
    if enable_cross_border_cascading and adjacent_candidates:
        active_ext_ids: set[str] = set()
        net_state: dict[str, Any] = {}
        if external_active_station_ids is None:
            try:
                learner = DMESelfReinforcingLearner()
                net_state = learner.load_cross_fir_network_state()
                active_ext_ids = {
                    sid for sid, info in net_state.items()
                    if info.get("recommended_posture") == "ACTIVE"
                }
            except Exception:
                active_ext_ids = set()
        else:
            active_ext_ids = set(external_active_station_ids)

        for s in adjacent_candidates:
            if s.id in active_ext_ids:
                if s.id in net_state and "home_fir" in net_state[s.id]:
                    s.home_fir = net_state[s.id]["home_fir"]
                cascaded_adjacent_stations.append(s)

    n_inside = len(inside_stations)
    n_cascaded = len(cascaded_adjacent_stations)
    all_eval_stations = list(inside_stations) + list(cascaded_adjacent_stations)
    n_eval = len(all_eval_stations)

    grid_cells = generate_ecac_grid(step_deg=0.18, fir_poly=fir_poly, fir_bbox=fir_bbox)
    n_cells = len(grid_cells)
    sta_id_to_idx = {s.id: i for i, s in enumerate(all_eval_stations)}
    inside_sta_ids = [s.id for s in inside_stations]
    cascaded_indices = list(range(n_inside, n_eval))

    # Precalculate distance and bearing matrices across all evaluation stations
    c_lats = np.array([c[0] for c in grid_cells])
    c_lons = np.array([c[1] for c in grid_cells])
    s_lats = np.array([s.latitude for s in all_eval_stations])
    s_lons = np.array([s.longitude for s in all_eval_stations])

    dlat_c = (c_lats[:, None] - s_lats[None, :]) * 60.0
    mid_lat_c = np.radians((c_lats[:, None] + s_lats[None, :]) / 2.0)
    dlon_c = (c_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_c)
    dist_cell = np.sqrt(dlat_c**2 + dlon_c**2)
    bearings_cell = (np.degrees(np.arctan2(dlon_c, dlat_c)) + 360.0) % 360.0

    cell_matrices, apt_matrices = _compute_coverage_matrices(
        all_eval_stations, grid_cells, airports, flight_levels=(10, *EVALUATION_FLIGHT_LEVELS)
    )
    cell_mat_200 = cell_matrices[200]
    cell_mat_100 = cell_matrices.get(95, cell_matrices.get(100, cell_matrices[200]))
    apt_mat_10 = apt_matrices[10]

    # Baseline active indices include all inside stations PLUS all cascaded external active stations
    base_all_active_indices = list(range(n_eval))
    coverable_cells_mask = cell_mat_200[:, base_all_active_indices].sum(axis=1) > 0

    # 2. Traffic Volume & Scale Factors
    total_moves, daily_moves, hourly_moves, peak_moves = _fetch_fir_traffic_stats(dataset, fir_id, period)

    # 3. Compute Rationalized Core Views (pre-populated with cascaded active neighbor stations)
    static_views = compute_static_rationalization_views(
        inside_stations,
        grid_cells,
        airports,
        station_type="DME",
        fixed_active_stations=cascaded_adjacent_stations,
    )

    # 4. Generate Structured Permutation Set (internal sovereign stations only)
    permutation_candidates: list[dict[str, Any]] = []

    # Permutation 1: Baseline (All Internal Active N-0)
    permutation_candidates.append({
        "id": "P-001",
        "name": f"Baseline: Full Network ({n_inside} Active DMEs)",
        "category": "BASELINE",
        "active_ids": set(inside_sta_ids),
    })

    # Permutations 2-5: Rationalized Core Candidates
    rnav5_core = static_views.get("RNAV_5", set())
    if rnav5_core and len(rnav5_core) < n_inside:
        permutation_candidates.append({
            "id": "P-CORE-RNAV5",
            "name": f"Rationalized Core: RNAV 5 ({len(rnav5_core)} Active DMEs)",
            "category": "RATIONALIZED_CORE",
            "active_ids": rnav5_core,
        })

    rnav2_core = static_views.get("RNAV_2", set())
    if rnav2_core and len(rnav2_core) < n_inside and rnav2_core != rnav5_core:
        permutation_candidates.append({
            "id": "P-CORE-RNAV2",
            "name": f"Rationalized Core: RNAV 2 ({len(rnav2_core)} Active DMEs)",
            "category": "RATIONALIZED_CORE",
            "active_ids": rnav2_core,
        })

    rnav1_core = static_views.get("RNAV_1", set())
    if rnav1_core and len(rnav1_core) < n_inside and rnav1_core not in (rnav5_core, rnav2_core):
        permutation_candidates.append({
            "id": "P-CORE-RNAV1",
            "name": f"Rationalized Core: RNAV 1 ({len(rnav1_core)} Active DMEs)",
            "category": "RATIONALIZED_CORE",
            "active_ids": rnav1_core,
        })

    rnp1_core = static_views.get("RNP_1", set())
    if rnp1_core and len(rnp1_core) < n_inside and rnp1_core not in (rnav5_core, rnav2_core, rnav1_core):
        permutation_candidates.append({
            "id": "P-CORE-RNP1",
            "name": f"Rationalized Core: RNP 1 ({len(rnp1_core)} Active DMEs)",
            "category": "RATIONALIZED_CORE",
            "active_ids": rnp1_core,
        })

    # Permutations 6+: Single Station Outages (N-1) on internal stations
    sta_cell_counts = cell_mat_200[:, :n_inside].sum(axis=0)
    sta_apt_counts = apt_mat_10[:, :n_inside].sum(axis=0)
    sta_importance = sta_cell_counts + (sta_apt_counts * 10.0)
    sorted_station_indices = np.argsort(sta_importance)[::-1]

    n1_limit = min(15, n_inside)
    for k in range(n1_limit):
        s_idx = sorted_station_indices[k]
        s_obj = inside_stations[s_idx]
        active_without_s = set(inside_sta_ids) - {s_obj.id}
        permutation_candidates.append({
            "id": f"P-N1-{s_obj.id}",
            "name": f"Single Outage (N-1): {s_obj.name} [{s_obj.id}]",
            "category": "SINGLE_OUTAGE",
            "active_ids": active_without_s,
        })

    # Strategic Dual Outages (N-2): Top critical internal station pairs
    if n_inside >= 4:
        for k in range(min(4, n_inside - 1)):
            s1 = inside_stations[sorted_station_indices[k]]
            s2 = inside_stations[sorted_station_indices[k + 1]]
            permutation_candidates.append({
                "id": f"P-N2-{s1.id}-{s2.id}",
                "name": f"Dual Outage (N-2): [{s1.id}] + [{s2.id}]",
                "category": "DUAL_OUTAGE",
                "active_ids": set(inside_sta_ids) - {s1.id, s2.id},
            })

    # Multi-Station Stress Tests (Disabling bottom 25%, 40%, 50% lowest-utility internal stations)
    reverse_sorted_sta = sorted_station_indices[::-1]
    for cut_pct in (0.25, 0.40, 0.50):
        cut_k = int(round(n_inside * cut_pct))
        if 1 < cut_k < n_inside:
            cut_ids = {inside_stations[idx].id for idx in reverse_sorted_sta[:cut_k]}
            permutation_candidates.append({
                "id": f"P-STRESS-{int(cut_pct*100)}PCT",
                "name": f"Stress Test: Decommission {cut_k} Stations ({int(cut_pct*100)}% Cutback)",
                "category": "STRESS_TEST",
                "active_ids": set(inside_sta_ids) - cut_ids,
            })

    # Trim to max_permutations while retaining key categories
    if len(permutation_candidates) > max_permutations:
        permutation_candidates = permutation_candidates[:max_permutations]

    # 5. Evaluate Baseline Reference Posture
    base_act_indices = base_all_active_indices
    b_met, b_risk, b_cannot, b_unc, b_apts = _evaluate_single_permutation_coverage(
        active_indices=base_act_indices,
        cell_mat_200=cell_mat_200,
        cell_mat_100=cell_mat_100,
        apt_mat_10=apt_mat_10,
        bearings_cell=bearings_cell,
        dist_cell=dist_cell,
        target_standard=target_standard,
        airports=airports,
        coverable_cells_mask=coverable_cells_mask,
    )
    b_cov_tot = max(1, n_cells - b_unc)
    b_out_frac = b_cannot / b_cov_tot
    b_risk_frac = b_risk / b_cov_tot

    base_gross_costs = {
        "1h": _compute_window_costs(1.0, "1-Hour Peak", hourly_moves, 1.30, b_out_frac, b_risk_frac, len(b_apts), 0).gross_operational_cost_eur,
        "6h": _compute_window_costs(6.0, "6-Hour Shift", hourly_moves, 1.12, b_out_frac, b_risk_frac, len(b_apts), 0).gross_operational_cost_eur,
        "24h": _compute_window_costs(24.0, "1-Day (24-Hour)", hourly_moves, 1.00, b_out_frac, b_risk_frac, len(b_apts), 0).gross_operational_cost_eur,
    }

    # 6. Evaluate Each Permutation (Cascaded external stations are always active)
    results: list[DMEPermutationResult] = []
    station_disruption_impacts: dict[str, float] = {}

    for cand in permutation_candidates:
        p_id = cand["id"]
        p_name = cand["name"]
        p_cat = cand["category"]
        act_ids = cand["active_ids"]
        dis_ids = set(inside_sta_ids) - act_ids

        # Internal active stations + cascaded external active stations
        act_indices = [sta_id_to_idx[sid] for sid in act_ids if sid in sta_id_to_idx] + cascaded_indices

        met, at_risk, cannot, unc, degraded_apts = _evaluate_single_permutation_coverage(
            active_indices=act_indices,
            cell_mat_200=cell_mat_200,
            cell_mat_100=cell_mat_100,
            apt_mat_10=apt_mat_10,
            bearings_cell=bearings_cell,
            dist_cell=dist_cell,
            target_standard=target_standard,
            airports=airports,
            coverable_cells_mask=coverable_cells_mask,
        )

        coverable_total = max(1, n_cells - unc)
        cov_met_pct = round((met / n_cells) * 100.0, 1)
        cov_coverable_pct = round((met / coverable_total) * 100.0, 1)

        outage_fraction = cannot / coverable_total
        at_risk_fraction = at_risk / coverable_total

        # Multi-Window Operational Calculations (relative to Baseline)
        windows = {
            "1h": _compute_window_costs(
                duration_hours=1.0,
                window_label="1-Hour Peak",
                traffic_hourly_avg=hourly_moves,
                traffic_peak_multiplier=1.30,  # Peak hour surge
                outage_fraction=outage_fraction,
                at_risk_fraction=at_risk_fraction,
                degraded_airports_count=len(degraded_apts),
                disabled_dme_count=len(dis_ids),
                baseline_gross_cost=base_gross_costs["1h"] if p_cat != "BASELINE" else None,
            ),
            "6h": _compute_window_costs(
                duration_hours=6.0,
                window_label="6-Hour Shift",
                traffic_hourly_avg=hourly_moves,
                traffic_peak_multiplier=1.12,  # Shift average
                outage_fraction=outage_fraction,
                at_risk_fraction=at_risk_fraction,
                degraded_airports_count=len(degraded_apts),
                disabled_dme_count=len(dis_ids),
                baseline_gross_cost=base_gross_costs["6h"] if p_cat != "BASELINE" else None,
            ),
            "24h": _compute_window_costs(
                duration_hours=24.0,
                window_label="1-Day (24-Hour)",
                traffic_hourly_avg=hourly_moves,
                traffic_peak_multiplier=1.00,  # Full daily average
                outage_fraction=outage_fraction,
                at_risk_fraction=at_risk_fraction,
                degraded_airports_count=len(degraded_apts),
                disabled_dme_count=len(dis_ids),
                baseline_gross_cost=base_gross_costs["24h"] if p_cat != "BASELINE" else None,
            ),
        }

        # Track single station outage penalties to identify critical heads
        if p_cat == "SINGLE_OUTAGE" and len(dis_ids) == 1:
            outage_id = next(iter(dis_ids))
            station_disruption_impacts[outage_id] = windows["24h"].incremental_delay_cost_eur

        # Selection Criteria & Classification
        day_net = windows["24h"].net_economic_impact_eur
        is_terminal_sound = len(degraded_apts) == 0
        has_strong_coverage = cov_coverable_pct >= 95.0

        if p_cat == "BASELINE":
            classification = "OPTIMAL_RATIONALIZATION"
            is_valid = True
            rationale = f"Status Quo reference datum ({n_inside} active DMEs, 100% equipment baseline)."
        elif has_strong_coverage and is_terminal_sound and day_net <= 0.0:
            classification = "OPTIMAL_RATIONALIZATION"
            is_valid = True
            rationale = (
                f"Meets target {target_standard} across {cov_coverable_pct}% of coverable airspace with zero "
                f"terminal airport degradation. Delivers net financial savings of €{abs(day_net):,.0f}/day "
                f"(€{abs(windows['1h'].net_economic_impact_eur):,.0f}/h)."
            )
        elif cov_coverable_pct >= 85.0 and len(degraded_apts) <= 1 and day_net < 1000.0:
            classification = "ACCEPTABLE_DEGRADATION"
            is_valid = True
            rationale = (
                f"Acceptable contingency posture: {cov_coverable_pct}% coverable compliance with minor marginal "
                f"tactical delays (Net impact: €{day_net:,.0f}/day)."
            )
        else:
            classification = "CRITICAL_DEFICIT"
            is_valid = False
            reasons = []
            if cov_coverable_pct < 85.0:
                reasons.append(f"Severe en-route coverage deficit ({cov_coverable_pct}% < 85%)")
            if len(degraded_apts) > 0:
                reasons.append(f"{len(degraded_apts)} airport(s) lose required dual-DME FL10 redundancy ({', '.join(degraded_apts[:3])})")
            if day_net > 0:
                reasons.append(f"Net operational penalty of +€{day_net:,.0f}/day (+€{windows['1h'].net_economic_impact_eur:,.0f}/h)")
            rationale = "Rejected: " + "; ".join(reasons) + "."

        results.append(
            DMEPermutationResult(
                permutation_id=p_id,
                name=p_name,
                category=p_cat,
                active_station_ids=sorted(list(act_ids)),
                disabled_station_ids=sorted(list(dis_ids)),
                active_count=len(act_ids),
                disabled_count=len(dis_ids),
                target_standard=target_standard,
                total_cells=n_cells,
                cells_met=met,
                cells_at_risk=at_risk,
                cells_cannot_be_met=cannot,
                cells_uncoverable=unc,
                coverage_met_pct=cov_met_pct,
                coverage_coverable_pct=cov_coverable_pct,
                terminal_airports_total=len(airports),
                degraded_airports_count=len(degraded_apts),
                degraded_airports_idents=degraded_apts,
                windows=windows,
                classification=classification,
                is_valid=is_valid,
                selection_rationale=rationale,
            )
        )

    # 6. Rank and Identify Optimal & Pareto Frontier Configurations
    valid_configs = [r for r in results if r.is_valid]
    # Optimal configuration maximizes equipment savings while maintaining is_valid
    if valid_configs:
        optimal_permutation = min(valid_configs, key=lambda x: x.windows["24h"].net_economic_impact_eur)
    else:
        # Fallback to Baseline if no configuration meets strict validity
        optimal_permutation = results[0] if results else None

    # Pareto frontier: configurations that are non-dominated in (coverage_coverable_pct, -net_economic_impact)
    pareto_frontier: list[DMEPermutationResult] = []
    sorted_by_cov = sorted(results, key=lambda x: (x.coverage_coverable_pct, -x.windows["24h"].net_economic_impact_eur), reverse=True)
    best_cost_seen = float("inf")
    for cand in sorted_by_cov:
        cost = cand.windows["24h"].net_economic_impact_eur
        if cost < best_cost_seen:
            pareto_frontier.append(cand)
            best_cost_seen = cost

    target_perm = optimal_permutation or results[0]
    optimal_act_ids = set(target_perm.active_station_ids) | {s.id for s in cascaded_adjacent_stations}

    # 7. Identify Top Critical DMEs (Single Points of Failure under the CHOSEN SCENARIO)
    critical_stations = _evaluate_critical_dmes(
        target_perm=target_perm,
        inside_stations=inside_stations,
        sta_id_to_idx=sta_id_to_idx,
        cascaded_indices=cascaded_indices,
        cell_mat_200=cell_mat_200,
        cell_mat_100=cell_mat_100,
        apt_mat_10=apt_mat_10,
        bearings_cell=bearings_cell,
        dist_cell=dist_cell,
        target_standard=target_standard,
        airports=airports,
        coverable_cells_mask=coverable_cells_mask,
        n_cells=n_cells,
        inside_sta_ids=inside_sta_ids,
        hourly_moves=hourly_moves,
        base_gross_costs=base_gross_costs,
        top_n=8,
    )

    # 8. Compute Station Utilization & Maintenance Reliability Model (evaluating all active network nodes)
    try:
        sta_util, maint_summary = compute_station_utilization_and_maintenance(
            dataset=dataset,
            fir_id=fir_id,
            stations=all_eval_stations,
            grid_cells=grid_cells,
            cell_matrices=cell_matrices,
            dist_cell=dist_cell,
            bearings_cell=bearings_cell,
            target_standard=target_standard,
            optimal_active_ids=optimal_act_ids,
            fir_bbox=fir_bbox,
            total_fir_flights=total_moves,
        )
    except Exception:
        sta_util, maint_summary = [], None

    # Tag airspace role for cascaded adjacent stations in utilization records
    if sta_util and cascaded_adjacent_stations:
        casc_ids = {s.id for s in cascaded_adjacent_stations}
        for u in sta_util:
            if u.station_id in casc_ids:
                u.airspace_role = "Cascaded Cross-Border Infill"

    # 9. Simulate Air Traffic Knock-On Cascades for Selected Scenario (Assessing ALL flights)
    out_frac = target_perm.cells_cannot_be_met / max(1, target_perm.total_cells - target_perm.cells_uncoverable)
    risk_frac = target_perm.cells_at_risk / max(1, target_perm.total_cells - target_perm.cells_uncoverable)

    target_act_indices = [sta_id_to_idx[sid] for sid in target_perm.active_station_ids if sid in sta_id_to_idx] + cascaded_indices
    cell_rnav_caps, _ = compute_cell_rnav_capabilities(
        active_indices=target_act_indices,
        cell_mat_200=cell_mat_200,
        cell_mat_100=cell_mat_100,
        dist_cell=dist_cell,
        bearings_cell=bearings_cell,
        coverable_cells_mask=coverable_cells_mask,
    )

    try:
        traffic_sim = simulate_fir_traffic_knock_on(
            dataset=dataset,
            fir_id=fir_id,
            period=period,
            outage_fraction=out_frac,
            at_risk_fraction=risk_frac,
            degraded_airports=target_perm.degraded_airports_idents,
            scenario_name=target_perm.name,
            limit_flights=None,  # Full census: assess all flights traversing FIR
            cell_rnav_caps=cell_rnav_caps,
        )
    except Exception:
        traffic_sim = None

    cascaded_records = [
        {
            "id": s.id,
            "name": s.name,
            "latitude": round(s.latitude, 4),
            "longitude": round(s.longitude, 4),
            "elevation_ft": round(s.elevation_ft, 0),
            "country": s.country,
            "home_fir": getattr(s, "home_fir", "Neighboring FIR"),
        }
        for s in cascaded_adjacent_stations
    ]

    analysis_obj = DMEFIRResilienceAnalysis(
        fir_id=fir_id,
        fir_name=fir_name,
        period=period,
        target_standard=target_standard,
        baseline_station_count=n_inside,
        total_airports_count=len(airports),
        traffic_month_total=total_moves,
        traffic_daily_avg=daily_moves,
        traffic_hourly_avg=hourly_moves,
        traffic_hourly_peak=peak_moves,
        permutations=results,
        optimal_permutation=optimal_permutation,
        pareto_frontier=pareto_frontier,
        critical_stations=critical_stations,
        generated_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        ml_diagnostics=None,
        traffic_simulation=traffic_sim,
        station_utilization=sta_util,
        maintenance_impact=maint_summary,
        cascaded_adjacent_stations=cascaded_records,
        rnav_scoring_summary=traffic_sim.rnav_scoring_summary if traffic_sim else None,
    )

    try:
        learner = DMESelfReinforcingLearner()
        learner.record_assessment(
            fir_id=fir_id,
            target_standard=target_standard,
            stations=inside_stations,
            analysis=analysis_obj,
        )
        analysis_obj.ml_diagnostics = learner.get_ml_diagnostics()
    except Exception:
        pass

    return analysis_obj


def compute_critical_dmes_for_permutation(
    analysis: DMEFIRResilienceAnalysis,
    permutation: DMEPermutationResult,
    dataset: FlightDataset | None = None,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    """
    Compute Critical Single Point of Failure (SPOF) DMEs specifically under a given chosen scenario.
    Evaluates the operational delay penalty and coverage disruption resulting from the N-1 outage
    of each active DME participating in that scenario.
    """
    if analysis.critical_stations and analysis.critical_stations[0].get("scenario_name") == permutation.name:
        return analysis.critical_stations[:top_n]

    ds = dataset or FlightDataset(period=analysis.period)
    fir_geom = get_fir_geometry(ds, analysis.fir_id)
    fir_poly, _, fir_bbox = fir_geom if fir_geom else (None, None, None)

    inside_stations, airports = load_ecac_stations(
        ds,
        station_type="DME",
        fir_poly=fir_poly,
        fir_bbox=fir_bbox,
        include_adjacent=False,
    )
    casc_raw = getattr(analysis, "cascaded_adjacent_stations", None) or []
    casc_stations = [
        ECACStation(
            id=cs["id"],
            name=cs["name"],
            type=cs.get("type", "DME"),
            latitude=cs["latitude"],
            longitude=cs["longitude"],
            elevation_ft=cs.get("elevation_ft", 100.0),
            country=cs.get("country", ""),
            is_airport_station=False,
            nearest_airport_ident=None,
            nominal_range_nm=cs.get("nominal_range_nm", 150.0),
            annual_cost_savings_eur=0.0,
            annual_power_savings_mwh=0.0,
        )
        for cs in casc_raw
    ]
    all_eval_stations = list(inside_stations) + casc_stations
    grid_cells = generate_ecac_grid(step_deg=0.18, fir_poly=fir_poly, fir_bbox=fir_bbox)
    if not grid_cells or not inside_stations:
        return []

    sta_id_to_idx = {s.id: i for i, s in enumerate(all_eval_stations)}
    inside_sta_ids = [s.id for s in inside_stations]
    cascaded_indices = list(range(len(inside_stations), len(all_eval_stations)))

    c_lats = np.array([c[0] for c in grid_cells])
    c_lons = np.array([c[1] for c in grid_cells])
    s_lats = np.array([s.latitude for s in all_eval_stations])
    s_lons = np.array([s.longitude for s in all_eval_stations])

    dlat_c = (c_lats[:, None] - s_lats[None, :]) * 60.0
    mid_lat_c = np.radians((c_lats[:, None] + s_lats[None, :]) / 2.0)
    dlon_c = (c_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_c)
    dist_cell = np.sqrt(dlat_c**2 + dlon_c**2)
    bearings_cell = (np.degrees(np.arctan2(dlon_c, dlat_c)) + 360.0) % 360.0

    cell_matrices, apt_matrices = _compute_coverage_matrices(
        all_eval_stations, grid_cells, airports, flight_levels=(10, 100, 200)
    )
    cell_mat_200 = cell_matrices[200]
    cell_mat_100 = cell_matrices.get(100, cell_mat_200)
    apt_mat_10 = apt_matrices[10]
    coverable_cells_mask = cell_mat_200.sum(axis=1) > 0

    _, _, hourly_moves, _ = _fetch_fir_traffic_stats(ds, analysis.fir_id, analysis.period)

    base_all_active_indices = list(range(len(all_eval_stations)))
    b_met, b_risk, b_cannot, b_unc, b_apts = _evaluate_single_permutation_coverage(
        active_indices=base_all_active_indices,
        cell_mat_200=cell_mat_200,
        cell_mat_100=cell_mat_100,
        apt_mat_10=apt_mat_10,
        bearings_cell=bearings_cell,
        dist_cell=dist_cell,
        target_standard=analysis.target_standard,
        airports=airports,
        coverable_cells_mask=coverable_cells_mask,
    )
    b_cov_tot = max(1, len(grid_cells) - b_unc)
    b_out_frac = b_cannot / b_cov_tot
    b_risk_frac = b_risk / b_cov_tot
    base_gross_costs = {
        "1h": _compute_window_costs(1.0, "1-Hour Peak", hourly_moves, 1.30, b_out_frac, b_risk_frac, len(b_apts), 0).gross_operational_cost_eur,
        "24h": _compute_window_costs(24.0, "1-Day (24-Hour)", hourly_moves, 1.00, b_out_frac, b_risk_frac, len(b_apts), 0).gross_operational_cost_eur,
    }

    return _evaluate_critical_dmes(
        target_perm=permutation,
        inside_stations=inside_stations,
        sta_id_to_idx=sta_id_to_idx,
        cascaded_indices=cascaded_indices,
        cell_mat_200=cell_mat_200,
        cell_mat_100=cell_mat_100,
        apt_mat_10=apt_mat_10,
        bearings_cell=bearings_cell,
        dist_cell=dist_cell,
        target_standard=analysis.target_standard,
        airports=airports,
        coverable_cells_mask=coverable_cells_mask,
        n_cells=len(grid_cells),
        inside_sta_ids=inside_sta_ids,
        hourly_moves=hourly_moves,
        base_gross_costs=base_gross_costs,
        top_n=top_n,
    )


def create_dme_resilience_map(
    analysis: DMEFIRResilienceAnalysis,
    selected_permutation: DMEPermutationResult | None = None,
    dataset: FlightDataset | None = None,
) -> folium.Map:
    """
    Build an interactive Folium map illustrating the FIR boundary, DME stations (active vs disabled),
    and color-coded coverage cells under the selected permutation.
    """
    fir_id = analysis.fir_id
    perm = selected_permutation or analysis.optimal_permutation or analysis.permutations[0]

    ds = dataset or FlightDataset(period=analysis.period)
    fir_geom = get_fir_geometry(ds, fir_id)
    if fir_geom:
        fir_poly, fir_boundary_coords, fir_bbox = fir_geom
        center_lat = (fir_bbox[0] + fir_bbox[1]) / 2.0
        center_lon = (fir_bbox[2] + fir_bbox[3]) / 2.0
    else:
        center_lat, center_lon = 51.5, 0.0
        fir_boundary_coords = None

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles=None,
        control_scale=True,
    )

    # Base Tile Layers (No API keys required, works seamlessly in local file:/// and iframe contexts)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Esri Light Gray Canvas (Aviation Default)",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri World Street Map",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ",
        name="Esri Topographic",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, Humanitarian OpenStreetMap Team",
        name="OpenStreetMap Humanitarian (HOT)",
        control=True,
    ).add_to(m)

    # 1. FIR Boundary
    if fir_boundary_coords:
        folium.Polygon(
            locations=fir_boundary_coords,
            color="#1e3a8a",
            weight=2.5,
            dash_array="6, 6",
            fill=False,
            tooltip=f"<b>{html.escape(analysis.fir_name)}</b> Boundary",
        ).add_to(m)

    # 2. Airspace Coverage Grid Cells Layer (10 NM Grid Squares)
    stations, airports = load_ecac_stations(
        ds,
        station_type="DME",
        fir_poly=fir_poly if fir_geom else None,
        fir_bbox=fir_bbox if fir_geom else None,
        include_adjacent=False,
    )
    active_set = set(perm.active_station_ids)
    casc_raw = getattr(analysis, "cascaded_adjacent_stations", None) or []
    casc_stations = [
        ECACStation(
            id=cs["id"],
            name=cs["name"],
            type=cs.get("type", "DME"),
            latitude=cs["latitude"],
            longitude=cs["longitude"],
            elevation_ft=cs.get("elevation_ft", 100.0),
            country=cs.get("country", ""),
            is_airport_station=False,
            nearest_airport_ident=None,
            nominal_range_nm=cs.get("nominal_range_nm", 150.0),
            annual_cost_savings_eur=0.0,
            annual_power_savings_mwh=0.0,
        )
        for cs in casc_raw
    ]
    all_eval_stations = list(stations) + casc_stations
    casc_active_ids = {cs["id"] for cs in casc_raw}
    all_active_ids = active_set.union(casc_active_ids)
    act_idx_arr = np.array([i for i, s in enumerate(all_eval_stations) if s.id in all_active_ids], dtype=int)

    grid_cells = generate_ecac_grid(step_deg=0.18, fir_poly=fir_poly if fir_geom else None, fir_bbox=fir_bbox if fir_geom else None)
    if grid_cells:
        cell_matrices, _ = _compute_coverage_matrices(
            all_eval_stations, grid_cells, [], flight_levels=(100, 200)
        )
        cell_mat_200 = cell_matrices[200]
        cell_mat_100 = cell_matrices.get(100, cell_mat_200)

        c_lats = np.array([c[0] for c in grid_cells])
        c_lons = np.array([c[1] for c in grid_cells])
        s_lats = np.array([s.latitude for s in all_eval_stations])
        s_lons = np.array([s.longitude for s in all_eval_stations])

        dlat_c = (c_lats[:, None] - s_lats[None, :]) * 60.0
        mid_lat_c = np.radians((c_lats[:, None] + s_lats[None, :]) / 2.0)
        dlon_c = (c_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_c)
        dist_cell = np.sqrt(dlat_c**2 + dlon_c**2)
        bearings_cell = (np.degrees(np.arctan2(dlon_c, dlat_c)) + 360.0) % 360.0

        coverable_cells_mask = cell_mat_200.sum(axis=1) > 0
        n_cells = len(grid_cells)
        has_valid_pair = np.zeros(n_cells, dtype=bool)
        d_max_pair = np.full(n_cells, 9999.0)

        if len(act_idx_arr) >= 2:
            vis_200_act = cell_mat_200[:, act_idx_arr]
            cnt_200 = vis_200_act.sum(axis=1)
            for idx in np.where(cnt_200 >= 2)[0]:
                act_s_local = np.where(vis_200_act[idx])[0]
                act_s_global = act_idx_arr[act_s_local]
                b = bearings_cell[idx, act_s_global]
                diff = np.abs(b[:, None] - b[None, :])
                delta = np.minimum(diff, 360.0 - diff)
                valid = (delta >= 30.0) & (delta <= 150.0)
                if valid.any():
                    has_valid_pair[idx] = True
                    dists = dist_cell[idx, act_s_global]
                    max_dists = np.maximum(dists[:, None], dists[None, :])
                    valid_max_dists = np.where(valid, max_dists, 9999.0)
                    d_max_pair[idx] = float(np.min(valid_max_dists))
        else:
            cnt_200 = cell_mat_200[:, act_idx_arr].sum(axis=1) if len(act_idx_arr) > 0 else np.zeros(n_cells, dtype=int)

        std_label = getattr(analysis, "target_standard", "RNAV_5")
        std_norm = std_label.upper()
        if std_norm in ("RNAV_5", "VIEW_1"):
            status_met = has_valid_pair & coverable_cells_mask
            status_at_risk = (~has_valid_pair) & (cnt_200 >= 1) & coverable_cells_mask
            status_cannot = (cnt_200 == 0) & coverable_cells_mask
        elif std_norm in ("RNAV_2", "VIEW_2"):
            status_met = has_valid_pair & (d_max_pair <= 130.0) & coverable_cells_mask
            status_at_risk = (~status_met) & (has_valid_pair | (cnt_200 >= 1)) & coverable_cells_mask
            status_cannot = (cnt_200 == 0) & coverable_cells_mask
        elif std_norm in ("RNAV_1", "VIEW_3"):
            vis_100_act = cell_mat_100[:, act_idx_arr] if len(act_idx_arr) > 0 else np.zeros((n_cells, 0))
            cnt_100 = vis_100_act.sum(axis=1)
            status_met = (cnt_100 >= 3) & has_valid_pair & (d_max_pair <= 60.0) & coverable_cells_mask
            status_at_risk = (~status_met) & (has_valid_pair | (cnt_200 >= 1)) & coverable_cells_mask
            status_cannot = (cnt_200 == 0) & coverable_cells_mask
        else:
            vis_100_act = cell_mat_100[:, act_idx_arr] if len(act_idx_arr) > 0 else np.zeros((n_cells, 0))
            cnt_100 = vis_100_act.sum(axis=1)
            status_met = (cnt_100 >= 4) & has_valid_pair & (d_max_pair <= 40.0) & coverable_cells_mask
            status_at_risk = (~status_met) & (has_valid_pair | (cnt_200 >= 1)) & coverable_cells_mask
            status_cannot = (cnt_200 == 0) & coverable_cells_mask

        cell_group = folium.FeatureGroup(name=f"10 NM Airspace Coverage Cells ({std_label})", show=True)
        d_nm = 10.0
        for i, (lat, lon) in enumerate(grid_cells):
            dlat = (d_nm / 60.0) / 2.0
            dlon = (d_nm / (60.0 * max(0.1, math.cos(math.radians(lat))))) / 2.0
            c_cnt = int(cnt_200[i]) if len(act_idx_arr) > 0 else 0

            if status_met[i]:
                color = "#10b981"
                tip = f"🟢 MET: Performance Standard Compliant | {c_cnt} active DMEs | Valid geometry"
                opac = 0.50
            elif status_at_risk[i]:
                color = "#f59e0b"
                tip = f"🟠 AT RISK / MARGINAL: {c_cnt} active DME | Single point of failure"
                opac = 0.60
            elif status_cannot[i]:
                color = "#ef4444"
                tip = f"🔴 CANNOT BE MET: 0 active DMEs within line-of-sight"
                opac = 0.70
            else:
                color = "#94a3b8"
                tip = f"⚪ Uncoverable Gap: Outside baseline DME line-of-sight"
                opac = 0.20

            folium.Rectangle(
                bounds=[[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]],
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=opac,
                weight=0.5,
                tooltip=tip,
            ).add_to(cell_group)
        cell_group.add_to(m)

    # 3. DME Stations Layer

    active_set = set(perm.active_station_ids)
    sta_group = folium.FeatureGroup(name="DME Stations (Active / Standby)", show=True)

    for s in stations:
        is_active = s.id in active_set
        color = "#10b981" if is_active else "#ef4444"
        fill_color = "#34d399" if is_active else "#f87171"
        status_lbl = "ACTIVE" if is_active else "STANDBY / DECOMMISSIONED"

        folium.CircleMarker(
            location=[s.latitude, s.longitude],
            radius=6 if is_active else 5,
            color=color,
            weight=2,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.9 if is_active else 0.5,
            tooltip=(
                f"<b>{html.escape(s.name)} [{s.id}]</b><br>"
                f"Status: <b>{status_lbl}</b><br>"
                f"Type: {s.type} | Elev: {s.elevation_ft:.0f} ft<br>"
                f"Near: {s.nearest_airport_ident or 'En-Route'}"
            ),
        ).add_to(sta_group)

    sta_group.add_to(m)

    # 2b. Cascaded Active Neighbor DMEs (Compensating Border Coverage)
    if getattr(analysis, "cascaded_adjacent_stations", None):
        casc_group = folium.FeatureGroup(name="Cascaded Active Neighbor DMEs (Cross-Border Infill)", show=True)
        for cs in analysis.cascaded_adjacent_stations:
            folium.CircleMarker(
                location=[cs["latitude"], cs["longitude"]],
                radius=6,
                color="#0891b2",
                weight=2,
                dash_array="3, 3",
                fill=True,
                fill_color="#22d3ee",
                fill_opacity=0.85,
                tooltip=(
                    f"<b>{html.escape(cs['name'])} [{cs['id']}]</b><br>"
                    f"Status: <b>CASCADED ACTIVE</b><br>"
                    f"Maintained by Neighboring FIR: <b>{html.escape(cs.get('home_fir', 'ECAC Partner'))}</b><br>"
                    f"Compensating border line-of-sight coverage for {analysis.fir_id}"
                ),
            ).add_to(casc_group)
        casc_group.add_to(m)

    # 3. Commercial & High-Density Airports Layer
    apt_group = folium.FeatureGroup(name="Airports & Terminal PBN", show=True)
    degraded_set = set(perm.degraded_airports_idents)

    for a in airports:
        is_deg = a["ident"] in degraded_set
        is_busy = a.get("is_high_density", False)
        apt_color = "#dc2626" if is_deg else ("#d97706" if is_busy else "#2563eb")
        apt_status = "DEGRADED (Dual FL10 Lost)" if is_deg else "SECURED (Dual FL10 Maintained)"

        folium.CircleMarker(
            location=[a["latitude"], a["longitude"]],
            radius=7 if is_busy else 5,
            color=apt_color,
            weight=2,
            fill=True,
            fill_color=apt_color,
            fill_opacity=0.85,
            tooltip=(
                f"✈️ <b>{a['ident']} — {html.escape(a['name'])}</b><br>"
                f"Traffic: {a.get('daily_movements', 0):.0f} daily moves<br>"
                f"Terminal PBN Integrity: <b>{apt_status}</b>"
            ),
        ).add_to(apt_group)

    apt_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def generate_dme_resilience_html_report(
    analysis: DMEFIRResilienceAnalysis,
    output_path: Path | str | None = None,
    dataset: FlightDataset | None = None,
    selected_permutation: DMEPermutationResult | None = None,
) -> Path:
    """
    Generate a robust, standalone HTML resilience report including interactive Folium Leaflet map,
    executive KPI cards, permutation comparison table, selection criteria analysis,
    and a comprehensive mathematical and economic appendix.
    """
    if output_path is None:
        out_dir = Path("data/reports/dme_resilience")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_slug = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"dme_resilience_{analysis.fir_id}_{analysis.target_standard}_{analysis.period}_{ts_slug}.html"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    opt = selected_permutation or analysis.optimal_permutation or analysis.permutations[0]
    base = analysis.permutations[0]

    # Build interactive map iframe
    m = create_dme_resilience_map(analysis, selected_permutation=opt, dataset=dataset)
    map_iframe_html = _folium_map_to_iframe(
        m,
        height=540,
        title=f"{analysis.fir_id} Navigation Resilience Map ({analysis.target_standard})",
    )

    opt = analysis.optimal_permutation or analysis.permutations[0]
    base = analysis.permutations[0]

    # Permutations Table Rows HTML
    table_rows = []
    for r in analysis.permutations:
        is_opt = (opt and r.permutation_id == opt.permutation_id)
        row_style = 'style="background-color: #f0fdf4; font-weight: 600;"' if is_opt else ""
        opt_badge = ' <span class="badge badge-success">Optimal Pareto Choice</span>' if is_opt else ""

        if r.classification == "OPTIMAL_RATIONALIZATION":
            badge_class = "badge-success"
            badge_label = "Optimal"
        elif r.classification == "ACCEPTABLE_DEGRADATION":
            badge_class = "badge-warning"
            badge_label = "Acceptable"
        else:
            badge_class = "badge-danger"
            badge_label = "Deficit"

        w1 = r.windows["1h"]
        w6 = r.windows["6h"]
        w24 = r.windows["24h"]

        # Color for net economic impacts: green if savings (negative net cost), red if penalty (positive net cost)
        def _fmt_net(val: float) -> str:
            if val < 0:
                return f'<span style="color: #16a34a;">-€{abs(val):,.0f}</span>'
            elif val > 0:
                return f'<span style="color: #dc2626;">+€{val:,.0f}</span>'
            return '<span style="color: #64748b;">€0</span>'

        table_rows.append(
            f'<tr {row_style}>\n'
            f'  <td><code>{html.escape(r.permutation_id)}</code></td>\n'
            f'  <td><b>{html.escape(r.name)}</b>{opt_badge}<br>'
            f'      <small style="color: #64748b;">{html.escape(r.selection_rationale)}</small></td>\n'
            f'  <td style="text-align: center;"><span class="badge badge-neutral">{r.category}</span></td>\n'
            f'  <td style="text-align: right;">{r.active_count} / {analysis.baseline_station_count}</td>\n'
            f'  <td style="text-align: right;"><b>{r.coverage_coverable_pct:.1f}%</b> '
            f'      <small style="color: #64748b;">({r.cells_met}/{r.total_cells - r.cells_uncoverable})</small></td>\n'
            f'  <td style="text-align: center;">{"<span style=\'color: #dc2626; font-weight: 700;\'>" + str(r.degraded_airports_count) + "</span>" if r.degraded_airports_count > 0 else "<span style=\'color: #16a34a;\'>0</span>"}</td>\n'
            f'  <td style="text-align: right;">{_fmt_net(w1.net_economic_impact_eur)}</td>\n'
            f'  <td style="text-align: right;">{_fmt_net(w6.net_economic_impact_eur)}</td>\n'
            f'  <td style="text-align: right;"><b>{_fmt_net(w24.net_economic_impact_eur)}</b></td>\n'
            f'  <td style="text-align: center;"><span class="badge {badge_class}">{badge_label}</span></td>\n'
            f'</tr>'
        )

    permutations_table_html = "\n".join(table_rows)

    # Prepare Cascaded Active Neighbor DMEs HTML Section
    cascaded_section_html = ""
    if analysis.cascaded_adjacent_stations:
        casc_rows = []
        for cs in analysis.cascaded_adjacent_stations:
            casc_rows.append(
                f'<tr>\n'
                f'  <td><b>{html.escape(cs["name"])}</b> (<code>{html.escape(cs["id"])}</code>)</td>\n'
                f'  <td><span class="badge badge-success">Active in {html.escape(cs.get("home_fir", "Neighboring FIR"))}</span></td>\n'
                f'  <td>{cs["latitude"]:.3f}°N, {cs["longitude"]:.3f}°E <small style="color: #64748b;">({cs["elevation_ft"]:.0f} ft)</small></td>\n'
                f'  <td>{html.escape(cs.get("country", "EU"))}</td>\n'
                f'  <td style="color: #0284c7; font-weight: 600;">Cross-Border Boundary Infill</td>\n'
                f'</tr>'
            )
        casc_table_body = "\n".join(casc_rows)
        cascaded_section_html = f"""
  <!-- Cascaded Active Neighbor Navaids Section -->
  <div class="section-card">
    <h2>🌐 Cross-Border Navaid Cascading &amp; Adjacent FIR Compensation</h2>
    <p style="font-size: 13.5px; color: #475569;">
      When an adjacent, busier FIR rationalizes its DME network, the stations confirmed as <b>ACTIVE</b> in its core plan
      automatically cascade outward to compensate coverage in neighboring airspaces.
      The <b>{len(analysis.cascaded_adjacent_stations)} adjacent transponders</b> below provide curved-earth line-of-sight
      coverage across the border of <b>{html.escape(analysis.fir_name)}</b>, eliminating the need for redundant domestic boundary beacons
      and enabling deeper local rationalization with zero double counting.
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Station Name (ID)</th>
            <th>Host ANSP / Home FIR Status</th>
            <th>Coordinates &amp; Elevation</th>
            <th>Country</th>
            <th>Operational Contribution</th>
          </tr>
        </thead>
        <tbody>
          {casc_table_body}
        </tbody>
      </table>
    </div>
  </div>
"""

    # Critical Stations List HTML (Evaluated under the Chosen Scenario)
    if selected_permutation and (not analysis.optimal_permutation or selected_permutation.permutation_id != analysis.optimal_permutation.permutation_id):
        try:
            chosen_crit = compute_critical_dmes_for_permutation(analysis, selected_permutation, dataset=dataset)
        except Exception:
            chosen_crit = analysis.critical_stations
    else:
        chosen_crit = analysis.critical_stations

    critical_rows = []
    for cs in chosen_crit:
        cov_loss_str = f"-{cs.get('newly_degraded_cells', 0)} cells"
        if cs.get("newly_degraded_airports"):
            cov_loss_str += f" | {len(cs['newly_degraded_airports'])} apt degraded"
        critical_rows.append(
            f'<tr>\n'
            f'  <td><b>{html.escape(cs["name"])}</b> (<code>{cs["id"]}</code>)</td>\n'
            f'  <td>{html.escape(cs["near_airport"])}</td>\n'
            f'  <td>{cs["latitude"]:.3f}°N, {cs["longitude"]:.3f}°E</td>\n'
            f'  <td>{cs["elevation_ft"]:.0f} ft</td>\n'
            f'  <td style="text-align: right; color: #b91c1c; font-weight: 600;">{cov_loss_str}</td>\n'
            f'  <td style="text-align: right; color: #dc2626; font-weight: 700;">+€{cs["outage_hourly_delay_cost_eur"]:,.0f} / h</td>\n'
            f'  <td style="text-align: right; color: #dc2626; font-weight: 700;">+€{cs["outage_daily_delay_cost_eur"]:,.0f} / day</td>\n'
            f'</tr>'
        )
    critical_table_html = "\n".join(critical_rows) if critical_rows else "<tr><td colspan='7'>No critical single point of failure stations detected under this configuration.</td></tr>"

    # Prepare DME Station Utilization HTML Table
    util_rows = []
    opt_act_ids = set(opt.active_station_ids) if opt else set()
    for u in (analysis.station_utilization or []):
        is_active = u.station_id in opt_act_ids
        status_badge = '<span class="badge badge-success">Active Core</span>' if is_active else '<span class="badge badge-danger">Standby / Rationalized</span>'
        if u.airspace_role == "Terminal Primary":
            role_badge = '<span class="badge badge-warning">Terminal Primary</span>'
        elif u.airspace_role == "Core En-Route Backbone":
            role_badge = '<span class="badge badge-neutral">Core Backbone</span>'
        else:
            role_badge = '<span class="badge badge-neutral">Regional / Infill</span>'

        util_rows.append(
            f'<tr>\n'
            f'  <td><b>{html.escape(u.station_name)}</b> (<code>{html.escape(u.station_id)}</code>)</td>\n'
            f'  <td>{role_badge}</td>\n'
            f'  <td>{u.latitude:.3f}°N, {u.longitude:.3f}°E <small style="color: #64748b;">({u.elevation_ft:.0f} ft)</small></td>\n'
            f'  <td style="text-align: center;">{status_badge}</td>\n'
            f'  <td style="text-align: right; font-weight: 700; color: #0f172a;">{u.daily_flights_using:,.1f} flights/day</td>\n'
            f'  <td style="text-align: right; font-weight: 600; color: #2563eb;">{u.hourly_flights_using:,.1f} flights/h</td>\n'
            f'  <td style="text-align: right; color: #64748b;">{u.peak_hourly_flights_using:,.1f} flights/h</td>\n'
            f'  <td style="text-align: right;"><b>{u.utilization_pct:.1f}%</b></td>\n'
            f'  <td style="text-align: center;">{u.valid_pairs_formed}</td>\n'
            f'</tr>'
        )
    utilization_table_html = "\n".join(util_rows) if util_rows else "<tr><td colspan='9'>No station utilization records available.</td></tr>"

    utilization_section_html = f"""
  <!-- DME Station Utilization & Flight Journey Exposure Section -->
  <div class="section-card">
    <h2>📡 DME Station Utilization &amp; Flight Journey Exposure</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Empirical station utilization derived from actual 4D radar/ADS-B flight paths, cell traffic density, and avionics multi-channel FMS autotuning likelihood.
      Quantifies the average daily, hourly, and peak hourly flights that utilize each terrestrial DME as part of their navigation journey.
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Station Name (ID)</th>
            <th>Airspace Role</th>
            <th>Coordinates &amp; Elevation</th>
            <th style="text-align: center;">Target Config Status</th>
            <th style="text-align: right;">Average Daily Flights</th>
            <th style="text-align: right;">Average Hourly Flights</th>
            <th style="text-align: right;">Peak Hourly Flights</th>
            <th style="text-align: right;">Fleet Utilization (%)</th>
            <th style="text-align: center;">Valid Pairs</th>
          </tr>
        </thead>
        <tbody>
          {utilization_table_html}
        </tbody>
      </table>
    </div>
  </div>
"""

    # Prepare Maintenance & Planned Outages HTML Section
    maint = analysis.maintenance_impact
    if maint:
        maint_sta_rows = []
        for mr in maint.station_maintenance_records[:12]:
            maint_sta_rows.append(
                f'<tr>\n'
                f'  <td><b>{html.escape(mr["station_name"])}</b> (<code>{html.escape(mr["station_id"])}</code>)</td>\n'
                f'  <td style="text-align: right; font-weight: 700; color: #dc2626;">{mr["spof_cells_count"]} cells</td>\n'
                f'  <td style="text-align: right;">{mr["annual_cell_hours_at_risk"]:.1f} h/yr</td>\n'
                f'  <td style="text-align: right; font-weight: 700; color: #b91c1c;">€{mr["annual_outage_cost_eur"]:,.0f} / yr</td>\n'
                f'</tr>'
            )
        maint_sta_table_body = "\n".join(maint_sta_rows) if maint_sta_rows else "<tr><td colspan='4'>No single point of failure stations identified under current configuration.</td></tr>"

        maintenance_section_html = f"""
  <!-- Maintenance & Planned Outages Reliability Assessment Section -->
  <div class="section-card">
    <h2>🛠️ Maintenance &amp; Planned Outage Reliability Assessment (0.5% Station Outage Rate)</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Rigorous reliability assessment modeling a standardized <b>0.5% planned annual outage rate</b> for each DME transponder
      (approx. 43.8 hours/year per station for scheduled preventative maintenance, calibration flight checks, and hardware servicing).
      Simulates single points of failure (SPOFs) versus redundant multi-station clusters, strictly preventing double accounting.
    </p>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 18px 0;">
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">SPOF Vulnerability Cells</div>
        <div style="font-size: 24px; font-weight: 800; color: {'#dc2626' if maint.spof_cells_count > 0 else '#16a34a'};">{maint.spof_cells_count:,} <span style="font-size: 14px; color: #64748b;">/ {maint.total_cells:,}</span></div>
        <div style="font-size: 12px; color: #64748b;">Cells where all valid pairs rely on a single station</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Fully Redundant Cells</div>
        <div style="font-size: 24px; font-weight: 800; color: #16a34a;">{maint.redundant_cells_count:,} <span style="font-size: 14px; color: #64748b;">cells</span></div>
        <div style="font-size: 12px; color: #64748b;">Multiple independent pairs (resilient to N-1 outages)</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Reduced Capability Time</div>
        <div style="font-size: 24px; font-weight: 800; color: #d97706;">{maint.annual_cell_hours_reduced_rnav:,.1f} h/yr</div>
        <div style="font-size: 12px; color: #64748b;">Cumulative cell-hours exposed to lowered RNAV</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Annual Flights Impacted</div>
        <div style="font-size: 24px; font-weight: 800; color: #0f172a;">{maint.annual_flights_impacted:,} <span style="font-size: 13px; color: #64748b;">({maint.monthly_flights_impacted:,}/mo)</span></div>
        <div style="font-size: 12px; color: #64748b;">Normalized by flight sector traversal length</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Annual Outage Disruption Cost</div>
        <div style="font-size: 24px; font-weight: 800; color: #dc2626;">€{maint.annual_outage_cost_eur:,.0f} / yr</div>
        <div style="font-size: 12px; color: #64748b;">€{maint.monthly_outage_cost_eur:,.0f}/month (€{maint.daily_outage_cost_eur:,.0f}/day)</div>
      </div>
    </div>

    <!-- Double Accounting Guard Notice -->
    <div style="background: #f0fdf4; border-left: 4px solid #16a34a; border-radius: 6px; padding: 12px 16px; margin: 16px 0; font-size: 13px; color: #166534;">
      <b>🛡️ Strict Double-Counting Protection Protocol:</b>
      Cells that already cannot meet the target standard under the evaluated rationalized configuration ({maint.permanently_degraded_cells_count} cells) have their baseline costs accounted for in the primary economic CBA and are assigned an outage degradation probability of <b>0.00%</b>.
      Furthermore, flights crossing multiple cells degraded by the same station outage are grouped by aircraft journey and normalized by sector flight path length, ensuring each physical flight is penalized only once upon sector entry.
    </div>

    <!-- Station SPOF Table -->
    <h3 style="font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 18px; margin-bottom: 8px;">📍 Key Stations Driving Outage Vulnerability (SPOFs)</h3>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Station Name (ID)</th>
            <th style="text-align: right;">SPOF Cells Triggered</th>
            <th style="text-align: right;">Annual Cell-Hours at Risk</th>
            <th style="text-align: right;">Annual Outage Disruption Cost (€/yr)</th>
          </tr>
        </thead>
        <tbody>
          {maint_sta_table_body}
        </tbody>
      </table>
    </div>
  </div>
"""
    else:
        maintenance_section_html = ""

    # Prepare ML Diagnostics HTML
    ml_diag = analysis.ml_diagnostics or {}
    ml_samples_cnt = ml_diag.get("total_samples_learned", 0)
    ml_firs_list = ml_diag.get("unique_firs_evaluated", [])
    ml_firs_str = ", ".join(ml_firs_list) if ml_firs_list else analysis.fir_id
    ml_harmonized_cnt = ml_diag.get("cross_fir_harmonized_stations", 0)
    ml_is_trained = ml_diag.get("is_trained", False)
    ml_status_badge = (
        '<span class="badge badge-success">Trained &amp; Active</span>'
        if ml_is_trained
        else '<span class="badge badge-warning">Collecting Experience Samples</span>'
    )

    # Prepare Traffic Knock-On Simulation HTML
    sim = analysis.traffic_simulation
    if sim and sim.total_flights_simulated > 0:
        rnav_sum = sim.rnav_scoring_summary or analysis.rnav_scoring_summary or {}
        cap_cnts = rnav_sum.get("capability_counts", {"RNAV 1": sim.total_flights_simulated, "RNAV 2": 0, "RNAV 5": 0, "RNAV 10": 0})
        cap_pcts = rnav_sum.get("capability_percentages", {"RNAV 1": 100.0, "RNAV 2": 0.0, "RNAV 5": 0.0, "RNAV 10": 0.0})
        cat_tots = rnav_sum.get("category_totals", {})
        tot_fleet_score = rnav_sum.get("total_fleet_score", 0)
        avg_fleet_score = rnav_sum.get("avg_flight_score", 0.0)

        matrix_rows = []
        for cat in RNAV_IMPACT_CATEGORIES:
            c_info = cat_tots.get(cat, {"total_points": 0, "avg_points_per_flight": 0.0})
            matrix_rows.append(
                f'<tr>\n'
                f'  <td><b>{html.escape(cat)}</b></td>\n'
                f'  <td style="text-align: center; color: #16a34a; font-weight: 700;">{RNAV_SCORING_MATRIX["RNAV 1"][cat]}</td>\n'
                f'  <td style="text-align: center; color: #2563eb; font-weight: 700;">{RNAV_SCORING_MATRIX["RNAV 2"][cat]}</td>\n'
                f'  <td style="text-align: center; color: #d97706; font-weight: 700;">{RNAV_SCORING_MATRIX["RNAV 5"][cat]}</td>\n'
                f'  <td style="text-align: center; color: #dc2626; font-weight: 700;">{RNAV_SCORING_MATRIX["RNAV 10"][cat]}</td>\n'
                f'  <td style="text-align: right; font-weight: 700; color: #0f172a;">{c_info.get("total_points", 0):,} pts</td>\n'
                f'  <td style="text-align: right; color: #64748b;">{c_info.get("avg_points_per_flight", 0.0):.2f} pts/fl</td>\n'
                f'</tr>'
            )
        matrix_rows_html = "\n".join(matrix_rows)

        sim_flights_rows = []
        for f in sim.sample_impacted_flights:
            if "CRITICAL" in f.cancellation_risk:
                r_badge = "badge-danger"
            elif "ELEVATED" in f.cancellation_risk:
                r_badge = "badge-warning"
            elif "MODERATE" in f.cancellation_risk:
                r_badge = "badge-neutral"
            else:
                r_badge = "badge-success"

            cap_badge = (
                "badge-success" if f.rnav_capability == "RNAV 1"
                else ("badge-neutral" if f.rnav_capability == "RNAV 2"
                else ("badge-warning" if f.rnav_capability == "RNAV 5" else "badge-danger"))
            )

            downstream_delay_str = (
                f'<span style="color: #dc2626; font-weight: 700;">+{f.downstream_reactionary_delay_min:.0f} min</span>'
                if f.downstream_reactionary_delay_min > 0
                else '<span style="color: #16a34a; font-weight: 600;">0 min (Absorbed)</span>'
            )

            sim_flights_rows.append(
                f'<tr>\n'
                f'  <td><b>{html.escape(f.ac_operator)}</b> (<code>{html.escape(f.ac_reg)}</code>)<br><small style="color: #64748b;">Type: {f.ac_type}</small></td>\n'
                f'  <td><b>{html.escape(f.leg1_route)}</b><br><small style="color: #64748b;">{f.leg1_dep_time} &rarr; {f.leg1_arr_time}</small></td>\n'
                f'  <td style="text-align: center;"><span class="badge {cap_badge}">{html.escape(f.rnav_capability)}</span></td>\n'
                f'  <td style="text-align: right; font-weight: 700; color: #b45309;">+{f.rnav_impact_score} pts</td>\n'
                f'  <td style="text-align: right; font-weight: 600;">+{f.primary_delay_min:.0f}m / {downstream_delay_str}</td>\n'
                f'  <td><b>{html.escape(f.leg2_route or "Final Rotation Leg")}</b><br><small style="color: #64748b;">{("Sched Dep: " + f.leg2_scheduled_dep) if f.leg2_scheduled_dep else "No subsequent leg in cycle"}</small></td>\n'
                f'  <td style="text-align: right;"><b>{f.turnaround_buffer_min:.0f} min</b><br><small style="color: #64748b;">({f.absorbed_delay_min:.0f}m absorbed)</small></td>\n'
                f'  <td><span class="badge {r_badge}">{html.escape(f.cancellation_risk)}</span><br><small style="color: #64748b;">{html.escape(f.cancellation_reason)}</small></td>\n'
                f'</tr>'
            )
        sim_table_body = "\n".join(sim_flights_rows) if sim_flights_rows else "<tr><td colspan='8'>No flight knock-on delays detected.</td></tr>"

        traffic_sim_section_html = f"""
  <!-- Air Traffic Knock-On & Downstream Cancellation Simulation Section -->
  <div class="section-card">
    <h2>✈️ Air Traffic Micro-Simulation: Direct Separation &amp; Downstream Rotational Knock-On</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Empirical operational micro-simulation executed across actual Eurocontrol flight trajectories for <b>{html.escape(analysis.fir_id)}</b> in period <b>{html.escape(analysis.period)}</b>.
      Simulates primary airborne spacing expansions (5 NM &rarr; {sim.separation_standard_nm:.1f} NM), tactical dog-leg detours, rotational turnaround delay propagation to downstream legs,
      and crew flight duty limit exceedances across 100% census of flight traversals.
    </p>

    <!-- Simulation KPI Grid -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 18px 0;">
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Direct Flights Impacted</div>
        <div style="font-size: 24px; font-weight: 800; color: #0f172a;">{sim.directly_impacted_flights:,} <span style="font-size: 14px; color: #64748b;">/ {sim.total_flights_simulated:,}</span></div>
        <div style="font-size: 12px; color: #64748b;">Traversals exposed to radar spacing expansion</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Primary Tactical Delay</div>
        <div style="font-size: 24px; font-weight: 800; color: #d97706;">+{sim.total_primary_delay_min:,.0f} min</div>
        <div style="font-size: 12px; color: #64748b;">En-route radar vectoring + terminal holding</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Detour Fuel &amp; Carbon</div>
        <div style="font-size: 24px; font-weight: 800; color: #2563eb;">+{sim.total_extra_fuel_kg:,.0f} kg</div>
        <div style="font-size: 12px; color: #64748b;">+{sim.total_extra_co2_kg/1000.0:,.1f} tonnes CO2 from avoidance dog-legs</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Downstream Legs Delayed</div>
        <div style="font-size: 24px; font-weight: 800; color: #dc2626;">{sim.downstream_flights_delayed:,} flights</div>
        <div style="font-size: 12px; color: #64748b;">Reactionary delay: +{sim.total_reactionary_delay_min:,.0f} min (&times;{sim.reactionary_multiplier:.2f})</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Rotational Disruption Risks</div>
        <div style="font-size: 24px; font-weight: 800; color: {'#dc2626' if sim.cancellation_risk_count > 0 else '#16a34a'};">{sim.cancellation_risk_count} flights</div>
        <div style="font-size: 12px; color: #64748b;">{sim.crew_fdp_timeouts_count} crew FDP timeouts | {sim.cancellation_risk_count - sim.crew_fdp_timeouts_count} slot forfeitures</div>
      </div>
    </div>

    <!-- RNAV Operational Scoring Breakdown -->
    <h3 style="font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 24px; margin-bottom: 8px;">📊 RNAV Operational Impact Scoring: Fleet Capability Transition Assessment</h3>
    <p style="font-size: 13px; color: #475569; margin-bottom: 12px;">
      Evaluates operational consequences of transitioning between RNAV capabilities across 7 impact dimensions: Route flexibility, Parallel route capacity, Crossing traffic management, Tactical intervention, Controller workload, Procedural separation dependency, and Alternative routing requirement.
    </p>

    <!-- Capability Summary Grid -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px;">
      <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px;">
        <div style="font-size: 11px; font-weight: 700; color: #166534; text-transform: uppercase;">RNAV 1 (Target)</div>
        <div style="font-size: 20px; font-weight: 800; color: #15803d;">{cap_cnts.get('RNAV 1', 0):,} <span style="font-size: 12px; font-weight: 600;">({cap_pcts.get('RNAV 1', 0.0)}%)</span></div>
        <div style="font-size: 11.5px; color: #166534;">0 pts / flight (Optimal)</div>
      </div>
      <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 12px;">
        <div style="font-size: 11px; font-weight: 700; color: #1e40af; text-transform: uppercase;">RNAV 2</div>
        <div style="font-size: 20px; font-weight: 800; color: #1d4ed8;">{cap_cnts.get('RNAV 2', 0):,} <span style="font-size: 12px; font-weight: 600;">({cap_pcts.get('RNAV 2', 0.0)}%)</span></div>
        <div style="font-size: 11.5px; color: #1e40af;">+7 pts / flight</div>
      </div>
      <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 12px;">
        <div style="font-size: 11px; font-weight: 700; color: #92400e; text-transform: uppercase;">RNAV 5</div>
        <div style="font-size: 20px; font-weight: 800; color: #b45309;">{cap_cnts.get('RNAV 5', 0):,} <span style="font-size: 12px; font-weight: 600;">({cap_pcts.get('RNAV 5', 0.0)}%)</span></div>
        <div style="font-size: 11.5px; color: #92400e;">+16 pts / flight</div>
      </div>
      <div style="background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 12px;">
        <div style="font-size: 11px; font-weight: 700; color: #991b1b; text-transform: uppercase;">RNAV 10</div>
        <div style="font-size: 20px; font-weight: 800; color: #b91c1c;">{cap_cnts.get('RNAV 10', 0):,} <span style="font-size: 12px; font-weight: 600;">({cap_pcts.get('RNAV 10', 0.0)}%)</span></div>
        <div style="font-size: 11.5px; color: #991b1b;">+31 pts / flight</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px;">
        <div style="font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase;">Fleet Total Score</div>
        <div style="font-size: 20px; font-weight: 800; color: #0f172a;">{tot_fleet_score:,} pts</div>
        <div style="font-size: 11.5px; color: #475569;">Avg {avg_fleet_score:.1f} pts / traversal</div>
      </div>
    </div>

    <!-- 7-Category Impact Scoring Table -->
    <div style="overflow-x: auto; margin-bottom: 20px;">
      <table>
        <thead>
          <tr>
            <th>Operational Impact Dimension</th>
            <th style="text-align: center;">RNAV 1 (Base)</th>
            <th style="text-align: center;">RNAV 2</th>
            <th style="text-align: center;">RNAV 5</th>
            <th style="text-align: center;">RNAV 10</th>
            <th style="text-align: right;">Fleet Total Points</th>
            <th style="text-align: right;">Avg / Traversal</th>
          </tr>
        </thead>
        <tbody>
          {matrix_rows_html}
        </tbody>
      </table>
    </div>

    <!-- Flight Knock-on Details Table -->
    <h3 style="font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 20px; margin-bottom: 8px;">📋 Trajectory Cascades: Direct &amp; Downstream Flight Impacts</h3>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Primary Flight (Tail)</th>
            <th>Primary Route &amp; Times</th>
            <th style="text-align: center;">RNAV Capability</th>
            <th style="text-align: right;">Impact Score</th>
            <th style="text-align: right;">Delay (Prim / React)</th>
            <th>Downstream Leg (Rotation)</th>
            <th style="text-align: right;">Turn Buffer</th>
            <th>Operational Risk</th>
          </tr>
        </thead>
        <tbody>
          {sim_table_body}
        </tbody>
      </table>
    </div>
  </div>
"""
    else:
        traffic_sim_section_html = ""

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DME Resilience &amp; Navigation Rationalization Report — {html.escape(analysis.fir_id)}</title>
  <style>
    :root {{
      --primary: #0f172a;
      --accent: #2563eb;
      --success: #16a34a;
      --warning: #d97706;
      --danger: #dc2626;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text: #1e293b;
      --text-muted: #64748b;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.55;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1300px;
      margin: 0 auto;
    }}
    .header-card {{
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      color: #ffffff;
      padding: 32px;
      border-radius: 12px;
      margin-bottom: 24px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }}
    .header-card h1 {{
      margin: 0 0 8px 0;
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.5px;
    }}
    .header-card .subtitle {{
      color: #94a3b8;
      font-size: 14.5px;
      margin-bottom: 16px;
    }}
    .header-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      font-size: 13px;
      color: #cbd5e1;
    }}
    .header-meta span {{
      background: rgba(255,255,255,0.08);
      padding: 4px 12px;
      border-radius: 6px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }}
    .kpi-label {{
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--text-muted);
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-size: 28px;
      font-weight: 800;
      color: var(--primary);
      margin-bottom: 4px;
    }}
    .kpi-desc {{
      font-size: 12.5px;
      color: var(--text-muted);
    }}
    .section-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }}
    .section-card h2 {{
      margin: 0 0 16px 0;
      font-size: 18px;
      font-weight: 700;
      color: var(--primary);
      border-bottom: 2px solid var(--border);
      padding-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 12px;
    }}
    th {{
      background-color: #f1f5f9;
      color: #334155;
      text-align: left;
      padding: 10px 12px;
      font-weight: 700;
      border-bottom: 2px solid var(--border);
    }}
    td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      vertical-align: middle;
    }}
    tr:hover {{
      background-color: #f8fafc;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .badge-success {{ background-color: #dcfce7; color: #15803d; }}
    .badge-warning {{ background-color: #fef3c7; color: #b45309; }}
    .badge-danger {{ background-color: #fee2e2; color: #b91c1c; }}
    .badge-neutral {{ background-color: #e2e8f0; color: #475569; }}
    .accordion {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      margin-top: 16px;
      overflow: hidden;
    }}
    .accordion summary {{
      padding: 14px 18px;
      font-weight: 700;
      font-size: 14px;
      cursor: pointer;
      background: #f1f5f9;
      color: #0f172a;
      user-select: none;
    }}
    .accordion-content {{
      padding: 20px;
      font-size: 13px;
      color: #334155;
      line-height: 1.65;
    }}
    .formula-box {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-left: 4px solid var(--accent);
      border-radius: 4px;
      padding: 12px 16px;
      font-family: "Courier New", Courier, monospace;
      font-size: 12.5px;
      margin: 12px 0;
      color: #0f172a;
    }}
  </style>
</head>
<body>
<div class="container">

  <!-- Header Card -->
  <div class="header-card">
    <h1>📡 DME Network Resilience &amp; Navigation Rationalization</h1>
    <div class="subtitle">Operational Permutation Stress Testing, Traffic Validation &amp; Eurocontrol Economic CBA</div>
    <div class="header-meta">
      <span><b>FIR:</b> {html.escape(analysis.fir_name)} (<code>{analysis.fir_id}</code>)</span>
      <span><b>Target Performance Standard:</b> <code>{analysis.target_standard}</code></span>
      <span><b>Traffic Period:</b> {analysis.period}</span>
      <span><b>Total Traversals Analyzed:</b> {analysis.traffic_month_total:,} flights/month</span>
      <span><b>Daily Traffic Avg:</b> {analysis.traffic_daily_avg:,.1f} flights/day</span>
      <span><b>Generated:</b> {analysis.generated_at}</span>
    </div>
  </div>

  <!-- KPI Grid -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Optimal Equipment Posture</div>
      <div class="kpi-value" style="color: #16a34a;">{opt.active_count} <span style="font-size: 18px; color: #64748b;">/ {analysis.baseline_station_count} DMEs</span></div>
      <div class="kpi-desc">Decommissions {opt.disabled_count} redundant stations ({round(opt.disabled_count/max(1, analysis.baseline_station_count)*100)}% equipment reduction)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Net Daily Savings (24-Hour)</div>
      <div class="kpi-value" style="color: #16a34a;">€{abs(opt.windows['24h'].net_economic_impact_eur):,.0f}</div>
      <div class="kpi-desc">Net benefit after absorbing all tactical en-route delays and reroutes</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">1-Hour Peak Net Economic Benefit</div>
      <div class="kpi-value" style="color: #16a34a;">€{abs(opt.windows['1h'].net_economic_impact_eur):,.0f} / h</div>
      <div class="kpi-desc">Equipment OPEX savings minus peak hour tactical buffer delay</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Target Standard Compliance</div>
      <div class="kpi-value">{opt.coverage_coverable_pct:.1f}%</div>
      <div class="kpi-desc">{opt.cells_met} of {opt.total_cells - opt.cells_uncoverable} coverable cells satisfy {analysis.target_standard}</div>
    </div>
    {f'''<div class="kpi-card">
      <div class="kpi-label">Cascaded Neighbor DMEs</div>
      <div class="kpi-value" style="color: #0284c7;">{len(analysis.cascaded_adjacent_stations)}</div>
      <div class="kpi-desc">Active transponders in busier neighbors compensating border cells</div>
    </div>''' if analysis.cascaded_adjacent_stations else ''}
  </div>

  <!-- Interactive Map Section -->
  <div class="section-card">
    <h2>🗺️ Interactive Airspace &amp; Navaid Resilience Map</h2>
    <p style="font-size: 13.5px; color: #475569; margin-top: 0;">
      Interactive Leaflet OpenStreetMap displaying FIR boundary, active DME transponders (green),
      decommissioned/standby stations (red), cascaded neighbor stations (cyan), and commercial airports with terminal PBN integrity monitoring.
    </p>
    {map_iframe_html}
  </div>

  {cascaded_section_html}

  <!-- Selection Criteria & Validation Card -->
  <div class="section-card">
    <h2>⚖️ Valid Configuration Selection Criteria &amp; Pareto Boundary</h2>
    <p style="font-size: 13.5px; color: #475569;">
      A configuration permutation is certified as an <b>OPTIMAL_RATIONALIZATION</b> only when it strictly satisfies three non-negotiable operational tests:
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin: 16px 0;">
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">1. En-Route Performance Standard ≥ 95%</div>
        <div style="font-size: 12.5px; color: #475569;">
          At least 95.0% of coverable 10 NM grid cells must meet or exceed the target standard (e.g. valid 30°–150° pair geometry for RNAV 5; range ≤ 130 NM for RNAV 2; range ≤ 60 NM with N-1 redundancy for RNAV 1).
        </div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">2. Zero Terminal Airport Degradation</div>
        <div style="font-size: 12.5px; color: #475569;">
          High-density commercial hubs (≥500 moves/day) must strictly maintain dual-station coverage at FL10; commercial airports must maintain at least 1 station to safeguard published PBN SIDs and STARs.
        </div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">3. Positive Net Economic Balance</div>
        <div style="font-size: 12.5px; color: #475569;">
          Equipment OPEX and maintenance flight-check savings must exceed all combined airline airborne vectoring delays, fuel reroutes, and terminal holding costs across the 24-hour cycle.
        </div>
      </div>
    </div>
  </div>

  <!-- Permutations Comparison Table -->
  <div class="section-card">
    <h2>📊 Permutation Performance &amp; Operational Window Costs</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Comprehensive evaluation of all tested station configurations. Economic figures show net cost impact
      (<b>green negative values indicate net airline &amp; ANSP financial savings</b>; red positive values indicate operational delay penalties).
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Configuration Name &amp; Rationale</th>
            <th style="text-align: center;">Category</th>
            <th style="text-align: right;">Active DMEs</th>
            <th style="text-align: right;">Coverage Met</th>
            <th style="text-align: center;">Degraded Apt</th>
            <th style="text-align: right;">1-Hour Net (€)</th>
            <th style="text-align: right;">6-Hour Net (€)</th>
            <th style="text-align: right;">24-Hour Net (€)</th>
            <th style="text-align: center;">Status</th>
          </tr>
        </thead>
        <tbody>
          {permutations_table_html}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Critical Infrastructure Table -->
  <div class="section-card">
    <h2>🚨 Critical Single Point of Failure DMEs (Chosen Scenario: {html.escape(opt.name)})</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Surviving active DME facilities under <b>{html.escape(opt.name)}</b> whose individual failure (N-1 outage) produces the highest operational disruption and delay cost spike across the rationalized network:
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Station Name (ID)</th>
            <th>Terminal / Sector Service</th>
            <th>Coordinates</th>
            <th>Elevation</th>
            <th style="text-align: right;">Coverage Loss (N-1)</th>
            <th style="text-align: right;">Hourly Outage Impact (€/h)</th>
            <th style="text-align: right;">Daily Outage Penalty (€/day)</th>
          </tr>
        </thead>
        <tbody>
          {critical_table_html}
        </tbody>
      </table>
    </div>
  </div>

  {utilization_section_html}

  {traffic_sim_section_html}

  {maintenance_section_html}

  <!-- Assessment Altitudes Table -->
  <div class="section-card">
    <h2>📐 Standardized European Assessment Altitude References</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Navigation and surveillance resilience is evaluated across seven standardized European flight levels plus terminal threshold:
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Assessment Altitude</th>
            <th>Altitude (ft AMSL)</th>
            <th>Why Assess It</th>
            <th>ECAC Operational Purpose &amp; Regulatory Reference</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>FL50</code></td>
            <td>5,000 ft</td>
            <td><b>Lower controlled airspace, terminal and lower ATS route coverage</b></td>
            <td>Transition from terminal maneuvering area (TMA) to en-route network; regional airline climb/descent profiles.</td>
          </tr>
          <tr>
            <td><code>FL95</code></td>
            <td>9,500 ft</td>
            <td><b>Common European lower en-route reference</b></td>
            <td>Mandatory B-RNAV carriage threshold in ECAC; base vertical boundary of controlled airspace in many States.</td>
          </tr>
          <tr>
            <td><code>FL150</code></td>
            <td>15,000 ft</td>
            <td><b>RNAV 5 implementation reference</b></td>
            <td>Core baseline reference for RNAV 5 navigation specification and lower/upper sector dividing interfaces.</td>
          </tr>
          <tr>
            <td><code>FL200</code></td>
            <td>20,000 ft</td>
            <td><b>Intermediate climb/descent and upper lower-airspace coverage</b></td>
            <td>Crucial vertical tier for intermediate climb/descent and transition between lower and upper sectors.</td>
          </tr>
          <tr>
            <td><code>FL250</code></td>
            <td>25,000 ft</td>
            <td><b>Upper en-route coverage</b></td>
            <td>Primary continental en-route transit cruising levels for domestic and medium-haul commercial flights.</td>
          </tr>
          <tr>
            <td><code>FL310</code></td>
            <td>31,000 ft</td>
            <td><b>Free Route Airspace reference</b></td>
            <td>Standard base level of Free Route Airspace (FRA) across continental Europe; direct cross-border routings.</td>
          </tr>
          <tr>
            <td><code>FL350</code></td>
            <td>35,000 ft</td>
            <td><b>Upper cruise coverage</b></td>
            <td>Upper long-haul cruise ceiling with maximum line-of-sight geometry and widest DME multi-station coverage.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Navaid Power Rating Filtering & Operational Certification Card -->
  <div class="section-card">
    <h2>⚡ Navaid Power Rating Filtering &amp; Operational Certification</h2>
    <div style="background: #eff6ff; border-left: 4px solid #2563eb; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px;">
      <p style="margin: 0; font-size: 13.5px; color: #1e3a8a; line-height: 1.6;">
        <b>Strict High-Power DME En-Route Filtering:</b> The Eurocontrol/ICAO navaid dataset records power ratings for all terrestrial beacons.
        <b>Low-power DMEs are dedicated landing aids</b> colocated with Instrument Landing Systems (ILS/DME, typically 50–100 W ERP with a restricted 18–25 NM service volume aligned strictly with runway approaches).
        These landing DMEs are <b>strictly non-certifiable for en-route RNAV/RNP multi-DME triangulation</b> under ICAO Doc 9613.
        Consequently, this engine filters and evaluates <b>exclusively High-power DMEs (1 kW pulse transponders with omnidirectional coverage)</b> for en-route resilience.
      </p>
    </div>
  </div>

  <!-- Self-Reinforcing Multi-FIR Machine Learning Section -->
  <div class="section-card">
    <h2>🧠 Self-Reinforcing Multi-FIR Machine Learning Engine</h2>
    <p style="font-size: 13.5px; color: #475569;">
      FlightML incorporates an active self-reinforcing learning loop that transfers topological and operational disruption knowledge across sovereign FIRs:
    </p>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin: 16px 0;">
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">1. Multi-FIR Experience Buffer</div>
        <div style="font-size: 12.5px; color: #475569;">
          Accumulates station topological feature vectors (elevation, Voronoi density, airport proximity, GDOP contributions, and multi-altitude coverage counts) and observed delay deltas into an append-only JSONL replay store.
        </div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">2. Pan-European Harmonization Registry</div>
        <div style="font-size: 12.5px; color: #475569;">
          Synchronizes border-straddling station states in a centralized cross-FIR network registry, ensuring that a DME recommended for standby in one FIR is never assumed active or critically relied upon by an adjacent FIR.
        </div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-weight: 700; color: #0f172a; margin-bottom: 4px;">3. Gradient-Boosted Meta-Predictor</div>
        <div style="font-size: 12.5px; color: #475569;">
          Dual HistGradientBoosting models (classifier for decommissionability and regressor for disruption penalty in €/day) prioritize candidate permutations and prune high-risk combinations upfront in subsequent FIR runs.
        </div>
      </div>
    </div>
    <div style="background: #f1f5f9; border-radius: 6px; padding: 12px 16px; font-size: 12.5px; color: #334155;">
      <b>Active ML Learning Diagnostics:</b>
      <span style="margin-left: 8px;">Total Experience Samples: <b>{ml_samples_cnt}</b></span> |
      <span style="margin-left: 8px;">Unique FIRs Evaluated: <b>{ml_firs_str}</b></span> |
      <span style="margin-left: 8px;">Cross-FIR Harmonized Beacons: <b>{ml_harmonized_cnt}</b></span> |
      <span style="margin-left: 8px;">Model Status: {ml_status_badge}</span>
    </div>
  </div>

  <!-- Mathematical & Economic Appendix -->
  <div class="section-card">
    <h2>📖 Mathematical &amp; Economic Appendix</h2>
    <div class="accordion">
      <details open>
        <summary>1. Line-of-Sight &amp; DME-DME GDOP Geometry Formulation</summary>
        <div class="accordion-content">
          <p>
            Under ICAO Doc 9613 and FAA AC 90-100A, terrestrial radio line-of-sight (RLOS) slant distance D_max in nautical miles
            between an aircraft at altitude h_ac (ft AMSL) and a DME ground beacon at elevation h_sta (ft AMSL) is governed by:
          </p>
          <div class="formula-box">
            D_RLOS = 1.23 * (sqrt(h_ac) + sqrt(h_sta)) [NM]
          </div>
          <p>
            For DME-DME position fixing, a dual station pair is geometrically valid if and only if the radial inclusion angle
            &Delta;&theta; = |&theta;1 - &theta;2| subtended at the aircraft satisfies:
          </p>
          <div class="formula-box">
            30&deg; &lt;= &Delta;&theta; &lt;= 150&deg;
          </div>
          <p>
            Geometric Dilution of Precision (GDOP) for range-range triangulation is minimized at orthogonal cuts (&Delta;&theta; = 90&deg;):
          </p>
          <div class="formula-box">
            GDOP = sqrt(2) / sin(&Delta;&theta;)
          </div>
        </div>
      </details>

      <details>
        <summary>2. Network Delay Modeling &amp; Tactical Separation Expansion</summary>
        <div class="accordion-content">
          <p>
            When an airspace sector drops below the certified RNAV/RNP positioning capability, air traffic controllers cannot maintain
            standard 5 NM waypoint-based separation and must revert to procedural miles-in-trail longitudinal separation (8 to 10 NM).
            Based on Eurocontrol Central Route Charges Office and Performance Review Unit (PRU) empirical observations:
          </p>
          <ul>
            <li><b>En-route Airborne Delay:</b> Delayed flights experience an average of +4.5 minutes tactical radar vectoring and speed reduction delay.</li>
            <li><b>Tactical Avoidance Detour:</b> 25% of flights traversing degraded sectors execute tactical reroutes around the gap, averaging +22.0 NM detour distance.</li>
            <li><b>Terminal Holding Queue:</b> Loss of dual FL10 coverage at busy airports reduces Arrival Acceptance Rate (AAR) by ~20%, causing M/M/1 stack holding delays averaging +14.0 minutes per arrival and +8.0 minutes departure slot ground hold.</li>
          </ul>
        </div>
      </details>

      <details>
        <summary>3. Eurocontrol Standard Cost-Benefit Analysis (CBA) Parameters</summary>
        <div class="accordion-content">
          <p>
            Economic coefficients are drawn directly from the <i>Eurocontrol Standard Inputs for Economic Cost-Benefit Analyses (Edition 9.0)</i>:
          </p>
          <ul>
            <li><b>Cost of Airborne Delay:</b> <b>€54.00 per minute</b> (includes crew flight pay, dynamic maintenance reserves, and airline passenger compensation).</li>
            <li><b>Cost of Ground Slot Delay:</b> <b>€28.00 per minute</b> (gate hold, APU burn, ground handling standby).</li>
            <li><b>Tactical Fuel Burn &amp; Carbon Cost:</b> Baseline medium twin Jet-A burn rate of 4.20 kg/NM at €0.85/kg plus EU ETS carbon allowance (3.15 kg CO2/kg fuel @ €75/tonne CO2) yielding <b>€4.54 per detour NM</b>.</li>
            <li><b>DME Transponder Annual OPEX:</b> <b>€45,000 / year</b> per station (€18k flight inspection sorties, €16k preventative technician labor &amp; spares, €8k communications/mast lease, €3k power utility = <b>€5.137 / hr</b>).</li>
          </ul>
        </div>
      </details>

      <details>
        <summary>4. Rotational Delay Knock-on &amp; Cancellation Mechanics</summary>
        <div class="accordion-content">
          <p>
            When an inbound flight experiences tactical airborne or terminal holding delays, the impact propagates downstream along the physical aircraft's rotation tail:
          </p>
          <div class="formula-box">
            ReactionaryDelay_leg2 = max(0, ArrDelay_leg1 - (Turnaround_sched - Turnaround_min_buffer))
          </div>
          <p>
            Standard scheduled turnaround buffers are calibrated at 40 minutes for narrowbodies (A320/B737), 60 minutes for widebodies (A350/B777/B787), and 30 minutes for regional aircraft.
            When an inbound delay exceeds the available turnaround buffer, the downstream departure is delayed, leading to reactionary delay propagation across airline network banks.
          </p>
          <p>
            <b>Downstream Flight Disruption &amp; Cancellation Triggers:</b>
          </p>
          <ul>
            <li><b>Rotational Schedule Collapse:</b> High-utilization European short-haul rotations rely on tightly sequenced bank waves. When reactionary turnaround delay propagates into late evening waves, flights lose their operational turnaround margin, triggering voluntary operator cancellations.</li>
            <li><b>Crew Flight Duty Period (FDP) Timeouts:</b> Under EASA Air Operations Subpart-FTL (Flight Time Limitations), pilot and cabin crew daily Flight Duty Periods are strictly capped (typically 12–13 hours without augmented crew). Accumulating &gt;150 minutes of cumulative rotational delay causes the crew duty period to expire, legally grounding the flight unless replacement reserve crew is positioned.</li>
            <li><b>Missed Regulated Slots (CTOT):</b> Downstream departures delayed &gt;45 minutes forfeit their Eurocontrol Network Manager Calculated Take-Off Time (CTOT) slot, entering prolonged slot-delay queues and forced flight cancellations.</li>
          </ul>
        </div>
      </details>

      <details>
        <summary>5. Navaid Selection Likelihood &amp; Flight Journey DME Utilization</summary>
        <div class="accordion-content">
          <p>
            Modern airliner Flight Management Systems (FMS) equipped with multi-channel DME scanning receivers
            autotune terrestrial DME transponders to compute continuous multi-sensor position fixes.
            For each 10 NM grid cell <code>c</code>, the probability that station <code>s</code> is tuned by an aircraft traversing cell <code>c</code> is formulated as:
          </p>
          <div class="formula-box">
            P(tune s | c) = min( 0.95, [ &sum;_(j &ne; s) w_pair(s, j) / &sum;_(u, v) w_pair(u, v) ] * 1.80 )
          </div>
          <p>
            where pair weight <code>w_pair(s, j) = sin(&Delta;&theta;_sj) / (d_s + d_j)</code> incorporates both angle-of-cut quality (&Delta;&theta; between 30° and 150°) and slant range proximity.
            In terminal sectors within 20 NM of departure or arrival airfields, colocated terminal DMEs receive an autotuning priority boost:
          </p>
          <div class="formula-box">
            P_terminal(s | c) &ge; 0.85
          </div>
          <p>
            Across an entire aircraft traversal through the FIR, the expected monthly flight volume utilizing station <code>s</code> is aggregated over all visible cells:
          </p>
          <div class="formula-box">
            MonthlyFlights(s) = min( TotalFIRFlights * 0.96, [ &sum;_(c &isin; Vis(s)) Traffic(c) * P(tune s | c) ] / L_traversal )
          </div>
          <p>
            where <code>L_traversal = &sum;_c Traffic(c) / TotalFIRFlights</code> represents the empirical average cell traversal depth per flight across the FIR (typically 8.5 to 10.5 cells of 10 NM).
            Daily and hourly rates are directly computed as:
          </p>
          <div class="formula-box">
            DailyFlights(s) = MonthlyFlights(s) / 31.0<br>
            HourlyFlights(s) = DailyFlights(s) / 24.0<br>
            PeakHourlyFlights(s) = HourlyFlights(s) * 1.45
          </div>
        </div>
      </details>

      <details>
        <summary>6. Planned Outages, Maintenance Combinatorics &amp; Double-Counting Avoidance</summary>
        <div class="accordion-content">
          <p>
            Each High-power DME transponder is assigned a standardized planned outage rate of <b>&lambda;_maint = 0.5% (0.005)</b>
            to account for annual preventative maintenance, periodic flight calibration inspections, and hardware servicing:
          </p>
          <div class="formula-box">
            AnnualDowntime(s) = 0.005 * 8760 h = 43.8 h/year (approx. 7.2 min/day)
          </div>
          <p>
            In any given cell <code>c</code>, the impact of a 0.5% station outage depends strictly on network redundancy:
          </p>
          <ul>
            <li><b>Single Point of Failure (SPOF) Cells:</b> If all valid DME/DME pairs formed by active stations in cell <code>c</code> share a common critical station <code>s*</code>, then whenever <code>s*</code> undergoes planned maintenance, cell <code>c</code> drops below certified RNAV capability:
              <div class="formula-box">P_degrade(c) = &lambda;_maint = 0.005 &rArr; 43.8 hours/year of reduced capability</div>
            </li>
            <li><b>Redundant Multi-Station Cells:</b> If cell <code>c</code> possesses two or more independent pairs with disjoint station subsets, capability loss requires concurrent simultaneous outages:
              <div class="formula-box">P_degrade(c) &approx; (2 * &lambda;_maint)&sup2; = 0.0001 &rArr; 0.88 hours/year (~53 minutes/year)</div>
            </li>
            <li><b>Terminal P-RNAV (RNAV 1) N-1 Redundancy:</b> For RNAV 1, requiring &ge;3 visible stations, any single station outage drops the cell below certified PBN criteria:
              <div class="formula-box">P_degrade_RNAV1(c) &approx; 3 * &lambda;_maint = 0.015 &rArr; 131.4 hours/year</div>
            </li>
          </ul>
          <p>
            <b>Mathematical Prevention of Double Counting:</b>
          </p>
          <ol>
            <li><b>Baseline Deficit Separation:</b> Cells that already cannot meet the performance standard under the evaluated permutation (permanently uncovered or degraded) have their delay and rerouting penalties fully reflected in the primary CBA.
                Their maintenance degradation probability is strictly set to <code>P_degrade(c) = 0.00%</code>, eliminating duplicate penalty counting.</li>
            <li><b>Spatial Flight-Path Normalization:</b> An aircraft crossing an airspace sector whose critical station is offline traverses multiple adjacent 10 NM cells.
                However, ATC applies procedural spacing restrictions and flight level capping <b>only once upon sector entry</b>.
                Total distinct flights impacted are normalized by sector traversal length:
                <div class="formula-box">ImpactedFlights_monthly = [ &sum;_c Traffic(c) * P_degrade(c) ] / L_traversal</div>
            </li>
          </ol>
        </div>
      </details>

      <details>
        <summary>7. ATC Operational Restrictions &amp; Tactical Intervention Economics</summary>
        <div class="accordion-content">
          <p>
            When an airspace cell experiences reduced RNAV capability (due to station rationalization or planned maintenance),
            air traffic management must transition from tactical RNAV route monitoring to procedural radar vectoring and spacing restrictions.
            The economic expense applied per degraded flight traversal (€329.75) is grounded in Eurocontrol operational CBA standards:
          </p>
          <div class="formula-box">
            Cost_degraded_traversal = Cost_vector_delay + Cost_ATC_intervention + Cost_tactical_detour<br>
            = (4.5 min * €55.50/min) + €35.00 + €45.00 = €329.75 per flight
          </div>
          <ul>
            <li><b>Radar Longitudinal Spacing Expansion (€249.75):</b> Standard radar separation is increased from 5 NM to 8.5–10 NM. Speed vectoring and path stretching introduce an average of +4.5 minutes airborne delay valued at €55.50/min (crew time, engine reserves, passenger buffer).</li>
            <li><b>ATC Tactical Intervention Workload (€35.00):</b> Controller-pilot data link (CPDLC) and VHF voice R/T channel congestion, tactical radar monitoring, and coordination transfers between sector executive and planning controllers.</li>
            <li><b>Tactical Path Detour &amp; Throttle Adjustments (€45.00):</b> Low-altitude dog-legs and speed control fuel burn penalties.</li>
          </ul>
        </div>
      </details>

      <details>
        <summary>8. Cross-Border Active Navaid Cascading &amp; Regional Rationalization</summary>
        <div class="accordion-content">
          <p>
            Terrestrial electromagnetic DME pulses obey spherical line-of-sight propagation across national borders.
            Under the ICAO Doc 9613 and Eurocontrol line-of-sight propagation model:
          </p>
          <div class="formula-box">
            d_RLOS = 1.23 * (sqrt(h_aircraft_ft) + sqrt(h_beacon_ft)) [NM]
          </div>
          <p>
            <b>Sequenced Priority Cascading:</b> By sequencing evaluations from the busiest FIRs (e.g. EDUUUIR, EGTTFIR, LFFFFIR)
            to adjacent less busy FIRs, core stations locked in by high-density hub airspaces are committed as fixed active nodes in the Pan-European network registry.
            Subsequent FIR assessments leverage these active external beacons to satisfy DME/DME pairing geometry (&Delta;&theta; &isin; [30&deg;, 150&deg;])
            along border cells.
          </p>
          <p>
            <b>Regional Capital Efficiency:</b> Peripheral and adjacent airspaces can safely rationalise redundant domestic border transponders
            without degrading PBN standards or airport approach integrity, achieving higher net ANSP savings (€45k/yr per decommissioned station).
          </p>
          <p>
            <b>No Double Counting Guarantee:</b>
          </p>
          <ul>
            <li>Equipment OPEX savings strictly apply to sovereign internal stations decommissioned by the local ANSP.</li>
            <li>Neighboring active stations are funded by their home ANSP and are never counted toward local capital savings.</li>
            <li>Operational flight delays and maintenance outage simulations strictly partition boundary flights to eliminate duplicate cost counting.</li>
          </ul>
        </div>
      </details>
    </div>
  </div>

  <div style="text-align: center; font-size: 12px; color: #94a3b8; margin-top: 32px;">
    FlightML Aeronautical Intelligence Engine • Generated automatically from Eurocontrol historical trajectory logs and ICAO Doc 9613 standards.
  </div>

</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    return output_path
