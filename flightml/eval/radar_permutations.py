"""
FlightML Radar Network Resilience & Operational Permutation Engine.

Evaluates terrestrial primary and secondary surveillance radar (PSR/SSR/Mode-S) resilience across European FIRs:
- Evaluates permutations of enabled/disabled radars against European surveillance standards
  (VIEW_1: Minimum Baseline FL200 single coverage; VIEW_2: Dual En-Route Redundancy; VIEW_3: Low-Level FL50 Infill & Busy Hubs).
- Validates network impact against real historical traffic from Eurocontrol dataset archives (100% census).
- Models non-radar separation expansion (5 NM -> 10-20 NM procedural / miles-in-trail), tactical reroutes, and terminal holding.
- Presents operational costs and equipment savings across 1-hour, 6-hour, and 1-day (24-hour) operational windows.
- Models 0.5% planned annual maintenance outage rate, evaluating single points of failure (SPOF) vs redundant dual coverage.
- Computes empirical radar station utilization: average daily, hourly, and peak hourly flights tracked across their journey.
- Implements cross-border active radar cascading: active radars in busier neighbor FIRs compensate border surveillance.
- Exports robust standalone HTML reports with interactive Leaflet maps and comprehensive mathematical appendices.
"""

from __future__ import annotations

import datetime
import html
import json
import math
import os
from dataclasses import dataclass, asdict
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
    RADAR_ANNUAL_COST_SAVINGS_EUR,
    RADAR_POWER_DRAW_KW,
    RADAR_ANNUAL_POWER_MWH,
)
from flightml.eval.dme_learning import DMESelfReinforcingLearner
from flightml.eval.dme_permutations import (
    WindowCostBreakdown,
    SimulatedFlightKnockOn,
    TrafficSimulationSummary,
    simulate_fir_traffic_knock_on,
    _fetch_fir_traffic_stats,
    EUROCONTROL_AIRBORNE_DELAY_EUR_PER_MIN,
    EUROCONTROL_GROUND_DELAY_EUR_PER_MIN,
    TACTICAL_DETOUR_NM,
    FUEL_BURN_KG_PER_NM,
    FUEL_AND_ETS_EUR_PER_NM,
)
from flightml.viz.html_exporter import _folium_map_to_iframe

HOURS_PER_YEAR = 8760.0
RADAR_HOURLY_OPEX_EUR = RADAR_ANNUAL_COST_SAVINGS_EUR / HOURS_PER_YEAR  # €9.703 / hr per radar head


@dataclass
class RadarPermutationResult:
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
class RadarStationUtilization:
    station_id: str
    station_name: str
    airspace_role: str  # "Terminal Primary", "Core En-Route Backbone", "Regional / Infill", "Cascaded Cross-Border Infill"
    latitude: float
    longitude: float
    elevation_ft: float
    is_active_in_optimal: bool
    daily_flights_tracked: float
    hourly_flights_tracked: float
    peak_hourly_flights_tracked: float
    utilization_pct: float
    cells_covered_fl200: int

    @property
    def daily_flights_using(self) -> float:
        return self.daily_flights_tracked

    @property
    def hourly_flights_using(self) -> float:
        return self.hourly_flights_tracked

    @property
    def peak_hourly_flights_using(self) -> float:
        return self.peak_hourly_flights_tracked

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RadarMaintenanceImpactSummary:
    outage_rate_pct: float  # 0.5%
    total_cells: int
    spof_cells_count: int  # Cells where surveillance relies strictly on a single radar
    redundant_cells_count: int  # Cells covered by >=2 independent radars
    permanently_degraded_cells_count: int  # Cells degraded under permanent rationalization posture
    annual_cell_hours_reduced_surveillance: float
    monthly_flights_impacted: int
    annual_flights_impacted: int
    annual_maintenance_disruption_cost_eur: float
    daily_maintenance_disruption_cost_eur: float
    station_maintenance_records: list[dict[str, Any]]

    @property
    def annual_cell_hours_reduced_rnav(self) -> float:
        return self.annual_cell_hours_reduced_surveillance

    @property
    def annual_outage_cost_eur(self) -> float:
        return self.annual_maintenance_disruption_cost_eur

    @property
    def daily_outage_cost_eur(self) -> float:
        return self.daily_maintenance_disruption_cost_eur

    @property
    def monthly_outage_cost_eur(self) -> float:
        return self.annual_maintenance_disruption_cost_eur / 12.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RadarFIRResilienceAnalysis:
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
    permutations: list[RadarPermutationResult]
    optimal_permutation: RadarPermutationResult | None
    pareto_frontier: list[RadarPermutationResult]
    critical_stations: list[dict[str, Any]]
    generated_at: str
    ml_diagnostics: dict[str, Any] | None = None
    traffic_simulation: TrafficSimulationSummary | None = None
    station_utilization: list[RadarStationUtilization] | None = None
    maintenance_impact: RadarMaintenanceImpactSummary | None = None
    cascaded_adjacent_stations: list[dict[str, Any]] | None = None

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
        return d


def _evaluate_single_radar_permutation_coverage(
    active_indices: list[int],
    cell_mat_200: np.ndarray,
    cell_mat_50: np.ndarray,
    apt_mat_10: np.ndarray,
    target_standard: str,
    airports: list[dict[str, Any]],
    coverable_cells_mask: np.ndarray,
) -> tuple[int, int, int, int, list[str]]:
    """
    Vectorized evaluation of a single Radar permutation against target surveillance standard:
    - VIEW_1: En-Route FL200 >= 1 radar, commercial airports FL10 >= 1 radar.
    - VIEW_2: Dual En-Route FL200 >= 2 radars, busy hubs FL10 >= 2 radars (N-1 redundancy).
    - VIEW_3: Low-Level FL50 >= 1 radar, busy hubs FL10 >= 2 radars.
    Returns (cells_met, cells_at_risk, cells_cannot_be_met, cells_uncoverable, degraded_airport_idents).
    """
    n_cells = cell_mat_200.shape[0]

    if not active_indices:
        met = 0
        at_risk = 0
        unc = int((~coverable_cells_mask).sum())
        cannot = n_cells - unc
        degraded_apts = [a["ident"] for a in airports]
        return met, at_risk, cannot, unc, degraded_apts

    act_idx_arr = np.array(active_indices, dtype=int)
    cnt_200 = cell_mat_200[:, act_idx_arr].sum(axis=1)
    cnt_50 = cell_mat_50[:, act_idx_arr].sum(axis=1)

    status_met = np.zeros(n_cells, dtype=bool)
    status_at_risk = np.zeros(n_cells, dtype=bool)
    status_cannot = np.zeros(n_cells, dtype=bool)

    std_norm = target_standard.upper()
    if std_norm in ("VIEW_1", "SURV_1", "FL200"):
        # Single Radar Baseline: >= 1 radar at FL200
        status_met = (cnt_200 >= 1) & coverable_cells_mask
        status_at_risk = np.zeros(n_cells, dtype=bool)
        status_cannot = (cnt_200 == 0) & coverable_cells_mask
    elif std_norm in ("VIEW_2", "SURV_2", "DUAL_ENROUTE"):
        # Dual Radar Redundancy: >= 2 radars at FL200
        status_met = (cnt_200 >= 2) & coverable_cells_mask
        status_at_risk = (cnt_200 == 1) & coverable_cells_mask
        status_cannot = (cnt_200 == 0) & coverable_cells_mask
    elif std_norm in ("VIEW_3", "SURV_3", "FL50", "LOW_LEVEL"):
        # Low-Level Infill: >= 1 radar at FL50
        status_met = (cnt_50 >= 1) & coverable_cells_mask
        status_at_risk = (cnt_50 == 0) & (cnt_200 >= 1) & coverable_cells_mask
        status_cannot = (cnt_50 == 0) & (cnt_200 == 0) & coverable_cells_mask
    else:
        # Default to VIEW_1
        status_met = (cnt_200 >= 1) & coverable_cells_mask
        status_cannot = (cnt_200 == 0) & coverable_cells_mask

    # Evaluate Airport Terminal Surveillance
    degraded_apts: list[str] = []
    if airports:
        apt_counts = apt_mat_10[:, act_idx_arr].sum(axis=1)
        for idx, a in enumerate(airports):
            is_busy = a.get("is_high_density", False)
            req = 2 if (is_busy and std_norm in ("VIEW_2", "VIEW_3", "SURV_2", "SURV_3")) else 1
            if apt_counts[idx] < req:
                degraded_apts.append(a["ident"])

    unc_count = int((~coverable_cells_mask).sum())
    met_count = int(status_met.sum())
    at_risk_count = int(status_at_risk.sum())
    cannot_count = int(status_cannot.sum())

    return met_count, at_risk_count, cannot_count, unc_count, degraded_apts


def _compute_radar_window_costs(
    duration_hours: float,
    window_label: str,
    traffic_hourly_avg: float,
    traffic_peak_multiplier: float,
    outage_fraction: float,
    at_risk_fraction: float,
    degraded_airports_count: int,
    disabled_radar_count: int,
    baseline_gross_cost: float | None = None,
) -> WindowCostBreakdown:
    """
    Calculate operational delay and fuel penalties for radar degradation vs equipment savings.
    When radar surveillance is lost, ATC expands separation from 5 NM to 10-20 NM procedural / MIT,
    imposing average airborne vector delays (+6.5m @ €54/m), tactical reroutes (+25 NM @ €4.54/NM),
    and terminal approach holding stacks (+16.0m arrival hold).
    """
    total_window_traffic = traffic_hourly_avg * traffic_peak_multiplier * duration_hours
    effective_exposure_rate = (outage_fraction * 1.0) + (at_risk_fraction * 0.28)
    flights_exposed = total_window_traffic * effective_exposure_rate

    # Operational Delay Penalties
    enroute_vector_delay_min = flights_exposed * 6.5  # +6.5 min radar vector spacing delay
    terminal_holding_min = degraded_airports_count * 16.0 * (duration_hours / 24.0) * (total_window_traffic * 0.12)
    ground_delay_min = degraded_airports_count * 8.0 * (duration_hours / 24.0) * (total_window_traffic * 0.08)
    total_delay_min = enroute_vector_delay_min + terminal_holding_min + ground_delay_min

    # Tactical Detours & Fuel Burn
    reroute_flights = flights_exposed * 0.25
    reroute_distance_nm = reroute_flights * 25.0
    reroute_fuel_kg = reroute_distance_nm * FUEL_BURN_KG_PER_NM

    # Monetary Valuations (Eurocontrol Standard Inputs)
    airborne_delay_cost = enroute_vector_delay_min * EUROCONTROL_AIRBORNE_DELAY_EUR_PER_MIN
    terminal_holding_cost = terminal_holding_min * EUROCONTROL_AIRBORNE_DELAY_EUR_PER_MIN
    ground_delay_cost = ground_delay_min * EUROCONTROL_GROUND_DELAY_EUR_PER_MIN
    reroute_fuel_cost = reroute_distance_nm * FUEL_AND_ETS_EUR_PER_NM

    gross_cost = airborne_delay_cost + terminal_holding_cost + ground_delay_cost + reroute_fuel_cost

    if baseline_gross_cost is not None:
        incremental_delay_cost = max(0.0, gross_cost - baseline_gross_cost)
    else:
        incremental_delay_cost = 0.0

    # Equipment OPEX Savings (€85k/yr = €9.703/hr per decommissioned radar head)
    equipment_savings = disabled_radar_count * RADAR_HOURLY_OPEX_EUR * duration_hours
    net_impact = incremental_delay_cost - equipment_savings

    return WindowCostBreakdown(
        window_label=window_label,
        duration_hours=duration_hours,
        flights_exposed=round(flights_exposed, 1),
        airborne_delay_minutes=round(enroute_vector_delay_min, 1),
        terminal_holding_minutes=round(terminal_holding_min, 1),
        ground_delay_minutes=round(ground_delay_min, 1),
        total_delay_minutes=round(total_delay_min, 1),
        reroute_distance_nm=round(reroute_distance_nm, 1),
        reroute_fuel_kg=round(reroute_fuel_kg, 1),
        airborne_delay_cost_eur=round(airborne_delay_cost, 2),
        terminal_holding_cost_eur=round(terminal_holding_cost, 2),
        ground_delay_cost_eur=round(ground_delay_cost, 2),
        reroute_fuel_cost_eur=round(reroute_fuel_cost, 2),
        gross_operational_cost_eur=round(gross_cost, 2),
        incremental_delay_cost_eur=round(incremental_delay_cost, 2),
        equipment_opex_savings_eur=round(equipment_savings, 2),
        net_economic_impact_eur=round(net_impact, 2),
    )


def compute_radar_utilization_and_maintenance(
    dataset: FlightDataset,
    fir_id: str,
    stations: list[ECACStation],
    grid_cells: list[tuple[float, float]],
    cell_matrices: dict[int, np.ndarray],
    dist_cell: np.ndarray,
    target_standard: str,
    optimal_active_ids: set[str],
    fir_bbox: tuple[float, float, float, float],
    total_fir_flights: int,
) -> tuple[list[RadarStationUtilization], RadarMaintenanceImpactSummary]:
    """
    Computes:
    1. Radar Station Utilization Table: Average hourly, daily, and peak flights whose 4D trajectory
       is actively tracked by each primary/secondary radar head across the FIR.
    2. Planned Outages & Maintenance Reliability Model: Simulates 0.5% planned annual outage rate
       per radar head (43.8 h/year), evaluating single points of failure (SPOF) vs redundant dual coverage.
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
        cell_traffic.fill(max(1.0, total_fir_flights / max(1, n_cells) * 8.5))

    total_traversals = float(cell_traffic.sum())
    avg_cells_per_flight = max(4.0, total_traversals / max(1, total_fir_flights))

    vis_200 = cell_matrices[200]
    vis_50 = cell_matrices.get(50, vis_200)

    # 2. Compute Radar Utilization
    utilization_list: list[RadarStationUtilization] = []
    for s_idx, sta in enumerate(stations):
        is_opt = sta.id in optimal_active_ids
        vis_cells = np.where(vis_200[:, s_idx])[0]
        cells_cov_cnt = len(vis_cells)

        if cells_cov_cnt == 0:
            monthly_fl = 0.0
        else:
            eff_traversals = float(cell_traffic[vis_cells].sum())
            monthly_fl = min(float(total_fir_flights * 0.98), eff_traversals / avg_cells_per_flight)

        daily_fl = monthly_fl / 31.0
        hourly_fl = daily_fl / 24.0
        peak_fl = hourly_fl * 1.45
        util_pct = (monthly_fl / max(1, total_fir_flights)) * 100.0

        if sta.nearest_airport_ident and daily_fl > 400:
            role = "Terminal Primary"
        elif daily_fl >= 300:
            role = "Core En-Route Backbone"
        else:
            role = "Regional / Infill"

        utilization_list.append(
            RadarStationUtilization(
                station_id=sta.id,
                station_name=sta.name,
                airspace_role=role,
                latitude=round(sta.latitude, 4),
                longitude=round(sta.longitude, 4),
                elevation_ft=round(sta.elevation_ft, 0),
                is_active_in_optimal=is_opt,
                daily_flights_tracked=round(daily_fl, 1),
                hourly_flights_tracked=round(hourly_fl, 1),
                peak_hourly_flights_tracked=round(peak_fl, 1),
                utilization_pct=round(util_pct, 1),
                cells_covered_fl200=cells_cov_cnt,
            )
        )

    utilization_list.sort(key=lambda u: u.daily_flights_tracked, reverse=True)

    # 3. Planned Outages & Maintenance Reliability Model (0.5% Outage Rate)
    lambda_maint = 0.005  # 0.5% planned annual outage rate (43.8 h/year)
    annual_hours = 8760.0

    opt_indices = [i for i, s in enumerate(stations) if s.id in optimal_active_ids]
    opt_mask = np.zeros(n_sta, dtype=bool)
    opt_mask[opt_indices] = True

    spof_cells_cnt = 0
    redundant_cells_cnt = 0
    perm_degraded_cnt = 0
    cell_degrade_prob = np.zeros(n_cells)
    station_spof_counts = np.zeros(n_sta, dtype=int)

    for c in range(n_cells):
        vis_act = np.where(vis_200[c] & opt_mask)[0]
        cnt = len(vis_act)
        if cnt == 0:
            perm_degraded_cnt += 1
            # Permanently uncovered: 0.0 outage prob to prevent double accounting!
            continue
        elif cnt == 1:
            # Single Point of Failure (SPOF): loss of this single radar drops surveillance completely
            spof_cells_cnt += 1
            cell_degrade_prob[c] = lambda_maint
            station_spof_counts[vis_act[0]] += 1
        else:
            # Redundant dual coverage: loss of surveillance requires simultaneous multi-station outages
            redundant_cells_cnt += 1
            cell_degrade_prob[c] = (2 * lambda_maint) ** 2

    tot_annual_cell_hours = float((cell_degrade_prob * annual_hours).sum())
    monthly_impacted_fl = int(round(float((cell_traffic * cell_degrade_prob).sum()) / avg_cells_per_flight))
    annual_impacted_fl = monthly_impacted_fl * 12

    # Eurocontrol standard unit intervention cost for radar surveillance loss:
    # Airborne vectoring (+6.5m @ €54/m = €351) + ATC intervention (€45) + detour fuel (€65) = €461.00
    unit_intervention_cost = 461.00
    annual_outage_cost = annual_impacted_fl * unit_intervention_cost
    daily_outage_cost = annual_outage_cost / 365.0

    station_maint_records = []
    for s_idx, sta in enumerate(stations):
        spof_c = int(station_spof_counts[s_idx])
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

    maint_summary = RadarMaintenanceImpactSummary(
        outage_rate_pct=0.5,
        total_cells=n_cells,
        spof_cells_count=spof_cells_cnt,
        redundant_cells_count=redundant_cells_cnt,
        permanently_degraded_cells_count=perm_degraded_cnt,
        annual_cell_hours_reduced_surveillance=round(tot_annual_cell_hours, 1),
        monthly_flights_impacted=monthly_impacted_fl,
        annual_flights_impacted=annual_impacted_fl,
        annual_maintenance_disruption_cost_eur=round(annual_outage_cost, 2),
        daily_maintenance_disruption_cost_eur=round(daily_outage_cost, 2),
        station_maintenance_records=station_maint_records,
    )

    return utilization_list, maint_summary


def simulate_radar_permutations(
    dataset: FlightDataset,
    fir_id: str = "EDUUUIR",
    period: str = "202510",
    target_standard: str = "VIEW_1",
    max_permutations: int = 25,
    external_active_station_ids: set[str] | list[str] | None = None,
    enable_cross_border_cascading: bool = True,
) -> RadarFIRResilienceAnalysis:
    """
    Run the Radar permutation resilience simulation for a selected FIR and surveillance standard.
    Generates structured permutations (Baseline N-0, Rationalized Core, Single N-1 Outages,
    Strategic Dual Outages, and Stress Cutbacks), validates against historical traffic,
    and quantifies operational impacts across 1h, 6h, and 24h operational windows.
    """
    fir_geom = get_fir_geometry(dataset, fir_id)
    if not fir_geom:
        raise ValueError(f"FIR '{fir_id}' not found in raw_fir dataset. Please select a valid ECAC FIR.")

    fir_poly, fir_boundary_coords, fir_bbox = fir_geom
    fir_name = ECAC_FIR_REGISTRY.get(fir_id, fir_id)

    # 1. Load sovereign internal radar stations and line-of-sight adjacent neighbor candidates
    inside_stations, airports, adjacent_candidates = load_ecac_stations(
        dataset,
        station_type="RADAR",
        fir_poly=fir_poly,
        fir_bbox=fir_bbox,
        include_adjacent=True,
    )
    if not inside_stations:
        raise ValueError(f"No radar stations found inside {fir_id}. Cannot run radar permutation analysis.")

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

    # Precalculate distance matrices
    c_lats = np.array([c[0] for c in grid_cells])
    c_lons = np.array([c[1] for c in grid_cells])
    s_lats = np.array([s.latitude for s in all_eval_stations])
    s_lons = np.array([s.longitude for s in all_eval_stations])

    dlat_c = (c_lats[:, None] - s_lats[None, :]) * 60.0
    mid_lat_c = np.radians((c_lats[:, None] + s_lats[None, :]) / 2.0)
    dlon_c = (c_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_c)
    dist_cell = np.sqrt(dlat_c**2 + dlon_c**2)

    cell_matrices, apt_matrices = _compute_coverage_matrices(
        all_eval_stations, grid_cells, airports, flight_levels=(10, 50, *EVALUATION_FLIGHT_LEVELS)
    )
    cell_mat_200 = cell_matrices[200]
    cell_mat_50 = cell_matrices.get(50, cell_mat_200)
    apt_mat_10 = apt_matrices[10]

    # Baseline active mask (all internal + cascaded neighbor active)
    base_all_active_indices = list(range(n_eval))
    coverable_cells_mask = cell_mat_200[:, base_all_active_indices].sum(axis=1) > 0

    # 2. Traffic Volume & Scale Factors
    total_moves, daily_moves, hourly_moves, peak_moves = _fetch_fir_traffic_stats(dataset, fir_id, period)

    # 3. Compute Rationalized Core Views (pre-populated with cascaded active neighbor stations)
    static_views = compute_static_rationalization_views(
        inside_stations,
        grid_cells,
        airports,
        station_type="RADAR",
        fixed_active_stations=cascaded_adjacent_stations,
    )

    # 4. Generate Structured Permutation Set (internal sovereign stations only)
    permutation_candidates: list[dict[str, Any]] = []

    # Permutation 1: Baseline (All Internal Active N-0)
    permutation_candidates.append({
        "id": "P-001",
        "name": f"Baseline: Full Radar Network ({n_inside} Active Radars)",
        "category": "BASELINE",
        "active_ids": set(inside_sta_ids),
    })

    # Permutations 2-4: Rationalized Core Views
    v1_core = static_views.get("VIEW_1", set())
    if v1_core and len(v1_core) < n_inside:
        permutation_candidates.append({
            "id": "P-CORE-VIEW1",
            "name": f"Rationalized Core: Single Baseline ({len(v1_core)} Active Radars)",
            "category": "RATIONALIZED_CORE",
            "active_ids": v1_core,
        })

    v2_core = static_views.get("VIEW_2", set())
    if v2_core and len(v2_core) < n_inside and v2_core != v1_core:
        permutation_candidates.append({
            "id": "P-CORE-VIEW2",
            "name": f"Rationalized Core: Dual Redundancy ({len(v2_core)} Active Radars)",
            "category": "RATIONALIZED_CORE",
            "active_ids": v2_core,
        })

    v3_core = static_views.get("VIEW_3", set())
    if v3_core and len(v3_core) < n_inside and v3_core not in (v1_core, v2_core):
        permutation_candidates.append({
            "id": "P-CORE-VIEW3",
            "name": f"Rationalized Core: Low-Level Infill ({len(v3_core)} Active Radars)",
            "category": "RATIONALIZED_CORE",
            "active_ids": v3_core,
        })

    # Permutations 5+: Single Radar Outages (N-1) on internal stations
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
                "name": f"Stress Test: Decommission {cut_k} Radars ({int(cut_pct*100)}% Cutback)",
                "category": "STRESS_TEST",
                "active_ids": set(inside_sta_ids) - cut_ids,
            })

    if len(permutation_candidates) > max_permutations:
        permutation_candidates = permutation_candidates[:max_permutations]

    # 5. Evaluate Baseline Reference Posture
    base_act_indices = base_all_active_indices
    b_met, b_risk, b_cannot, b_unc, b_apts = _evaluate_single_radar_permutation_coverage(
        active_indices=base_act_indices,
        cell_mat_200=cell_mat_200,
        cell_mat_50=cell_mat_50,
        apt_mat_10=apt_mat_10,
        target_standard=target_standard,
        airports=airports,
        coverable_cells_mask=coverable_cells_mask,
    )
    b_cov_tot = max(1, n_cells - b_unc)
    b_out_frac = b_cannot / b_cov_tot
    b_risk_frac = b_risk / b_cov_tot

    base_gross_costs = {
        "1h": _compute_radar_window_costs(1.0, "1-Hour Peak", hourly_moves, 1.30, b_out_frac, b_risk_frac, 0, 0).gross_operational_cost_eur,
        "6h": _compute_radar_window_costs(6.0, "6-Hour Shift", hourly_moves, 1.12, b_out_frac, b_risk_frac, 0, 0).gross_operational_cost_eur,
        "24h": _compute_radar_window_costs(24.0, "1-Day (24-Hour)", hourly_moves, 1.00, b_out_frac, b_risk_frac, 0, 0).gross_operational_cost_eur,
    }

    # 6. Evaluate Each Permutation (Cascaded external stations are always active)
    results: list[RadarPermutationResult] = []
    station_disruption_impacts: dict[str, float] = {}

    for cand in permutation_candidates:
        p_id = cand["id"]
        p_name = cand["name"]
        p_cat = cand["category"]
        act_ids = cand["active_ids"]
        dis_ids = set(inside_sta_ids) - act_ids

        # Internal active stations + cascaded external active stations
        act_indices = [sta_id_to_idx[sid] for sid in act_ids if sid in sta_id_to_idx] + cascaded_indices

        met, at_risk, cannot, unc, degraded_apts = _evaluate_single_radar_permutation_coverage(
            active_indices=act_indices,
            cell_mat_200=cell_mat_200,
            cell_mat_50=cell_mat_50,
            apt_mat_10=apt_mat_10,
            target_standard=target_standard,
            airports=airports,
            coverable_cells_mask=coverable_cells_mask,
        )

        b_apts_set = set(b_apts)
        newly_degraded_apts = [a for a in degraded_apts if a not in b_apts_set]

        coverable_total = max(1, n_cells - unc)
        cov_met_pct = round((met / n_cells) * 100.0, 1)
        cov_coverable_pct = round((met / coverable_total) * 100.0, 1)

        outage_fraction = cannot / coverable_total
        at_risk_fraction = at_risk / coverable_total

        windows = {
            "1h": _compute_radar_window_costs(
                duration_hours=1.0,
                window_label="1-Hour Peak",
                traffic_hourly_avg=hourly_moves,
                traffic_peak_multiplier=1.30,
                outage_fraction=outage_fraction,
                at_risk_fraction=at_risk_fraction,
                degraded_airports_count=len(newly_degraded_apts),
                disabled_radar_count=len(dis_ids),
                baseline_gross_cost=base_gross_costs["1h"] if p_cat != "BASELINE" else None,
            ),
            "6h": _compute_radar_window_costs(
                duration_hours=6.0,
                window_label="6-Hour Shift",
                traffic_hourly_avg=hourly_moves,
                traffic_peak_multiplier=1.12,
                outage_fraction=outage_fraction,
                at_risk_fraction=at_risk_fraction,
                degraded_airports_count=len(newly_degraded_apts),
                disabled_radar_count=len(dis_ids),
                baseline_gross_cost=base_gross_costs["6h"] if p_cat != "BASELINE" else None,
            ),
            "24h": _compute_radar_window_costs(
                duration_hours=24.0,
                window_label="1-Day (24-Hour)",
                traffic_hourly_avg=hourly_moves,
                traffic_peak_multiplier=1.00,
                outage_fraction=outage_fraction,
                at_risk_fraction=at_risk_fraction,
                degraded_airports_count=len(newly_degraded_apts),
                disabled_radar_count=len(dis_ids),
                baseline_gross_cost=base_gross_costs["24h"] if p_cat != "BASELINE" else None,
            ),
        }

        if p_cat == "SINGLE_OUTAGE" and len(dis_ids) == 1:
            outage_id = next(iter(dis_ids))
            station_disruption_impacts[outage_id] = windows["24h"].incremental_delay_cost_eur

        day_net = windows["24h"].net_economic_impact_eur
        is_terminal_sound = len(newly_degraded_apts) == 0
        has_strong_coverage = cov_coverable_pct >= 95.0

        if p_cat == "BASELINE":
            classification = "OPTIMAL_RATIONALIZATION"
            is_valid = True
            rationale = f"Status Quo reference datum ({n_inside} active Radars, 100% equipment baseline)."
        elif has_strong_coverage and is_terminal_sound and day_net <= 0.0:
            classification = "OPTIMAL_RATIONALIZATION"
            is_valid = True
            rationale = (
                f"Meets surveillance standard across {cov_coverable_pct}% of airspace with zero newly degraded hubs. "
                f"Delivers net financial savings of €{abs(day_net):,.0f}/day (€{abs(windows['1h'].net_economic_impact_eur):,.0f}/h)."
            )
        elif cov_coverable_pct >= 85.0 and len(newly_degraded_apts) <= 1 and day_net < 1500.0:
            classification = "ACCEPTABLE_DEGRADATION"
            is_valid = True
            rationale = (
                f"Acceptable contingency posture: {cov_coverable_pct}% coverable compliance with minor procedural separation delays "
                f"(Net impact: €{day_net:,.0f}/day)."
            )
        else:
            classification = "CRITICAL_DEFICIT"
            is_valid = False
            reasons = []
            if cov_coverable_pct < 85.0:
                reasons.append(f"Severe surveillance deficit ({cov_coverable_pct}% < 85%)")
            if len(newly_degraded_apts) > 0:
                reasons.append(f"{len(newly_degraded_apts)} airport(s) lose required radar coverage ({', '.join(newly_degraded_apts[:3])})")
            if day_net > 0:
                reasons.append(f"Net operational penalty of +€{day_net:,.0f}/day")
            rationale = "Rejected: " + "; ".join(reasons) + "."

        results.append(
            RadarPermutationResult(
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
                degraded_airports_count=len(newly_degraded_apts),
                degraded_airports_idents=newly_degraded_apts,
                windows=windows,
                classification=classification,
                is_valid=is_valid,
                selection_rationale=rationale,
            )
        )

    # 6. Optimal & Pareto Frontier Selection
    valid_configs = [r for r in results if r.is_valid and r.category != "BASELINE"]
    if not valid_configs:
        valid_configs = [r for r in results if r.is_valid]
    if valid_configs:
        optimal_permutation = min(valid_configs, key=lambda x: x.windows["24h"].net_economic_impact_eur)
    else:
        optimal_permutation = results[0] if results else None

    pareto_frontier: list[RadarPermutationResult] = []
    sorted_by_cov = sorted(results, key=lambda x: (x.coverage_coverable_pct, -x.windows["24h"].net_economic_impact_eur), reverse=True)
    best_cost_seen = float("inf")
    for cand in sorted_by_cov:
        cost = cand.windows["24h"].net_economic_impact_eur
        if cost < best_cost_seen:
            pareto_frontier.append(cand)
            best_cost_seen = cost

    # 7. Identify Critical Radar Heads
    critical_stations: list[dict[str, Any]] = []
    for sid, day_pen in sorted(station_disruption_impacts.items(), key=lambda x: x[1], reverse=True)[:6]:
        matching_sta = next((s for s in inside_stations if s.id == sid), None)
        if matching_sta:
            critical_stations.append({
                "id": sid,
                "name": matching_sta.name,
                "latitude": matching_sta.latitude,
                "longitude": matching_sta.longitude,
                "elevation_ft": matching_sta.elevation_ft,
                "near_airport": matching_sta.nearest_airport_ident or "En-Route",
                "outage_daily_delay_cost_eur": round(day_pen, 2),
                "outage_hourly_delay_cost_eur": round(day_pen / 24.0, 2),
            })

    target_perm = optimal_permutation or results[0]
    optimal_act_ids = set(target_perm.active_station_ids) | {s.id for s in cascaded_adjacent_stations}

    # 8. Compute Radar Station Utilization & Maintenance Impact
    try:
        sta_util, maint_summary = compute_radar_utilization_and_maintenance(
            dataset=dataset,
            fir_id=fir_id,
            stations=all_eval_stations,
            grid_cells=grid_cells,
            cell_matrices=cell_matrices,
            dist_cell=dist_cell,
            target_standard=target_standard,
            optimal_active_ids=optimal_act_ids,
            fir_bbox=fir_bbox,
            total_fir_flights=total_moves,
        )
    except Exception:
        sta_util, maint_summary = [], None

    if sta_util and cascaded_adjacent_stations:
        casc_ids = {s.id for s in cascaded_adjacent_stations}
        for u in sta_util:
            if u.station_id in casc_ids:
                u.airspace_role = "Cascaded Cross-Border Infill"

    # 9. Micro-simulation of Air Traffic Knock-On Cascades (100% census)
    out_frac = target_perm.cells_cannot_be_met / max(1, target_perm.total_cells - target_perm.cells_uncoverable)
    risk_frac = target_perm.cells_at_risk / max(1, target_perm.total_cells - target_perm.cells_uncoverable)
    try:
        traffic_sim = simulate_fir_traffic_knock_on(
            dataset=dataset,
            fir_id=fir_id,
            period=period,
            outage_fraction=out_frac,
            at_risk_fraction=risk_frac,
            degraded_airports=target_perm.degraded_airports_idents,
            scenario_name=target_perm.name,
            limit_flights=None,
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

    analysis_obj = RadarFIRResilienceAnalysis(
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


def create_radar_resilience_map(
    analysis: RadarFIRResilienceAnalysis,
    selected_permutation: RadarPermutationResult | None = None,
    dataset: FlightDataset | None = None,
) -> folium.Map:
    """
    Build an interactive Folium map illustrating FIR boundary, active radars (green),
    standby radars (red), cascaded neighbor radars (cyan), and commercial airports.
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
        center_lat, center_lon = 50.0, 10.0
        fir_boundary_coords = None

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles=None,
        control_scale=True,
    )

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

    # 2. Sovereign Internal Radars Layer
    stations, airports = load_ecac_stations(
        ds,
        station_type="RADAR",
        fir_poly=fir_poly if fir_geom else None,
        fir_bbox=fir_bbox if fir_geom else None,
        include_adjacent=False,
    )

    active_set = set(perm.active_station_ids)
    sta_group = folium.FeatureGroup(name="Surveillance Radars (Active / Standby)", show=True)

    for s in stations:
        is_active = s.id in active_set
        color = "#10b981" if is_active else "#ef4444"
        fill_color = "#34d399" if is_active else "#f87171"
        status_lbl = "ACTIVE SURVEILLANCE" if is_active else "STANDBY / DECOMMISSIONED"

        folium.CircleMarker(
            location=[s.latitude, s.longitude],
            radius=7 if is_active else 5,
            color=color,
            weight=2,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.9 if is_active else 0.5,
            tooltip=(
                f"📡 <b>{html.escape(s.name)} [{s.id}]</b><br>"
                f"Status: <b>{status_lbl}</b><br>"
                f"Nominal Range: {s.nominal_range_nm:.0f} NM | Elev: {s.elevation_ft:.0f} ft<br>"
                f"Colocated: {s.nearest_airport_ident or 'En-Route Facility'}"
            ),
        ).add_to(sta_group)

    sta_group.add_to(m)

    # 2b. Cascaded Active Neighbor Radars (Cross-Border Infill)
    if getattr(analysis, "cascaded_adjacent_stations", None):
        casc_group = folium.FeatureGroup(name="Cascaded Active Neighbor Radars (Cross-Border Infill)", show=True)
        for cs in analysis.cascaded_adjacent_stations:
            folium.CircleMarker(
                location=[cs["latitude"], cs["longitude"]],
                radius=7,
                color="#0891b2",
                weight=2,
                dash_array="3, 3",
                fill=True,
                fill_color="#22d3ee",
                fill_opacity=0.85,
                tooltip=(
                    f"📡 <b>{html.escape(cs['name'])} [{cs['id']}]</b><br>"
                    f"Status: <b>CASCADED ACTIVE RADAR</b><br>"
                    f"Maintained by Neighboring FIR: <b>{html.escape(cs.get('home_fir', 'ECAC Partner'))}</b><br>"
                    f"Compensating border line-of-sight surveillance for {analysis.fir_id}"
                ),
            ).add_to(casc_group)
        casc_group.add_to(m)

    # 3. Commercial & High-Density Airports Layer
    apt_group = folium.FeatureGroup(name="Airports & Terminal Surveillance", show=True)
    degraded_set = set(perm.degraded_airports_idents)

    for a in airports:
        is_deg = a["ident"] in degraded_set
        is_busy = a.get("is_high_density", False)
        apt_color = "#dc2626" if is_deg else ("#d97706" if is_busy else "#2563eb")
        apt_status = "DEGRADED (Surveillance Lost)" if is_deg else "SECURED (Radar Maintained)"

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
                f"Terminal Radar Status: <b>{apt_status}</b>"
            ),
        ).add_to(apt_group)

    apt_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def generate_radar_resilience_html_report(
    analysis: RadarFIRResilienceAnalysis,
    output_path: Path | str | None = None,
    dataset: FlightDataset | None = None,
) -> Path:
    """
    Generate a robust, standalone HTML radar resilience report including interactive Folium Leaflet map,
    executive KPI cards, permutation comparison table, selection criteria analysis,
    and a comprehensive mathematical and economic appendix.
    """
    if output_path is None:
        out_dir = Path("data/reports/radar_resilience")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_slug = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"radar_resilience_{analysis.fir_id}_{analysis.target_standard}_{analysis.period}_{ts_slug}.html"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    m = create_radar_resilience_map(analysis, dataset=dataset)
    map_iframe_html = _folium_map_to_iframe(
        m,
        height=540,
        title=f"{analysis.fir_id} Radar Surveillance Resilience Map ({analysis.target_standard})",
    )

    opt = analysis.optimal_permutation or analysis.permutations[0]

    # Permutations Table Rows HTML
    table_rows = []
    for r in analysis.permutations:
        is_opt = (opt and r.permutation_id == opt.permutation_id)
        row_style = 'style="background-color: #f0fdf4; font-weight: 600;"' if is_opt else ""
        opt_badge = ' <span class="badge badge-success">Optimal Choice</span>' if is_opt else ""

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

    # Critical Stations List HTML
    critical_rows = []
    for cs in analysis.critical_stations:
        critical_rows.append(
            f'<tr>\n'
            f'  <td><b>{html.escape(cs["name"])}</b> (<code>{cs["id"]}</code>)</td>\n'
            f'  <td>{html.escape(cs["near_airport"])}</td>\n'
            f'  <td>{cs["latitude"]:.3f}°N, {cs["longitude"]:.3f}°E</td>\n'
            f'  <td>{cs["elevation_ft"]:.0f} ft</td>\n'
            f'  <td style="text-align: right; color: #dc2626; font-weight: 700;">+€{cs["outage_hourly_delay_cost_eur"]:,.0f} / h</td>\n'
            f'  <td style="text-align: right; color: #dc2626; font-weight: 700;">+€{cs["outage_daily_delay_cost_eur"]:,.0f} / day</td>\n'
            f'</tr>'
        )
    critical_table_html = "\n".join(critical_rows) if critical_rows else "<tr><td colspan='6'>No critical single point of failure radars detected.</td></tr>"

    # Cascaded Active Neighbor Radars Section HTML
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
                f'  <td style="color: #0284c7; font-weight: 600;">Cross-Border Radar Infill</td>\n'
                f'</tr>'
            )
        casc_table_body = "\n".join(casc_rows)
        cascaded_section_html = f"""
  <!-- Cascaded Active Neighbor Radars Section -->
  <div class="section-card">
    <h2>🌐 Cross-Border Radar Cascading &amp; Adjacent FIR Compensation</h2>
    <p style="font-size: 13.5px; color: #475569;">
      When an adjacent, busier FIR rationalizes its surveillance infrastructure, radar heads confirmed as <b>ACTIVE</b> in its core plan
      cascade outward up to 250 NM line of sight to compensate surveillance in neighboring airspaces.
      The <b>{len(analysis.cascaded_adjacent_stations)} adjacent radar heads</b> below provide curved-earth line-of-sight
      surveillance across the border of <b>{html.escape(analysis.fir_name)}</b>, eliminating the need for redundant domestic boundary beacons
      and enabling deeper local rationalization with zero double counting.
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Radar Head (ID)</th>
            <th>Host ANSP / Home FIR Status</th>
            <th>Coordinates &amp; Elevation</th>
            <th>Country</th>
            <th>Surveillance Contribution</th>
          </tr>
        </thead>
        <tbody>
          {casc_table_body}
        </tbody>
      </table>
    </div>
  </div>
"""

    # Radar Station Utilization HTML Table
    util_rows = []
    opt_act_ids = set(opt.active_station_ids) if opt else set()
    for u in (analysis.station_utilization or []):
        is_active = u.station_id in opt_act_ids
        status_badge = '<span class="badge badge-success">Active Core</span>' if is_active else '<span class="badge badge-danger">Standby / Rationalized</span>'
        if u.airspace_role == "Terminal Primary":
            role_badge = '<span class="badge badge-warning">Terminal Primary</span>'
        elif u.airspace_role == "Core En-Route Backbone":
            role_badge = '<span class="badge badge-neutral">Core Backbone</span>'
        elif u.airspace_role == "Cascaded Cross-Border Infill":
            role_badge = '<span class="badge badge-success">Cascaded Infill</span>'
        else:
            role_badge = '<span class="badge badge-neutral">Regional / Infill</span>'

        util_rows.append(
            f'<tr>\n'
            f'  <td><b>{html.escape(u.station_name)}</b> (<code>{html.escape(u.station_id)}</code>)</td>\n'
            f'  <td>{role_badge}</td>\n'
            f'  <td>{u.latitude:.3f}°N, {u.longitude:.3f}°E <small style="color: #64748b;">({u.elevation_ft:.0f} ft)</small></td>\n'
            f'  <td style="text-align: center;">{status_badge}</td>\n'
            f'  <td style="text-align: right; font-weight: 700; color: #0f172a;">{u.daily_flights_tracked:,.1f} flights/day</td>\n'
            f'  <td style="text-align: right; font-weight: 600; color: #2563eb;">{u.hourly_flights_tracked:,.1f} flights/h</td>\n'
            f'  <td style="text-align: right; color: #64748b;">{u.peak_hourly_flights_tracked:,.1f} flights/h</td>\n'
            f'  <td style="text-align: right;"><b>{u.utilization_pct:.1f}%</b></td>\n'
            f'  <td style="text-align: center;">{u.cells_covered_fl200}</td>\n'
            f'</tr>'
        )
    utilization_table_html = "\n".join(util_rows) if util_rows else "<tr><td colspan='9'>No radar utilization records available.</td></tr>"

    # Maintenance Reliability Assessment HTML Section
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
        maint_sta_table_body = "\n".join(maint_sta_rows) if maint_sta_rows else "<tr><td colspan='4'>No single point of failure radars identified under current configuration.</td></tr>"

        maintenance_section_html = f"""
  <!-- Maintenance & Planned Outages Reliability Assessment Section -->
  <div class="section-card">
    <h2>🛠️ Maintenance &amp; Planned Outage Reliability Assessment (0.5% Radar Outage Rate)</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Models standardized <b>0.5% planned annual maintenance outage rate</b> (43.8 h/year, 7.2 min/day) per radar head.
      Quantifies cell-hours of reduced surveillance capability, separates Single Points of Failure (SPOFs) from redundant dual-radar cells,
      and calculates airline delay costs without double counting.
    </p>

    <!-- Maintenance KPI Grid -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 18px 0;">
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Annual Cell-Hours at Risk</div>
        <div style="font-size: 24px; font-weight: 800; color: #d97706;">{maint.annual_cell_hours_reduced_surveillance:,.1f} h/yr</div>
        <div style="font-size: 12px; color: #64748b;">Cumulative hours across all 10 NM cells</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Single Point of Failure Cells</div>
        <div style="font-size: 24px; font-weight: 800; color: #dc2626;">{maint.spof_cells_count:,} <span style="font-size: 14px; color: #64748b;">/ {maint.total_cells:,}</span></div>
        <div style="font-size: 12px; color: #64748b;">Cells relying on 1 radar (P = 0.5%)</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Redundant Multi-Radar Cells</div>
        <div style="font-size: 24px; font-weight: 800; color: #16a34a;">{maint.redundant_cells_count:,} cells</div>
        <div style="font-size: 12px; color: #64748b;">Dual-radar coverage (P &approx; 0.01%)</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Flights Exposed to Outages</div>
        <div style="font-size: 24px; font-weight: 800; color: #0f172a;">{maint.monthly_flights_impacted:,} <span style="font-size: 14px; color: #64748b;">/ mo</span></div>
        <div style="font-size: 12px; color: #64748b;">{maint.annual_flights_impacted:,} flights/year normalized by path length</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Annual Maintenance Delay Cost</div>
        <div style="font-size: 24px; font-weight: 800; color: #dc2626;">€{maint.annual_maintenance_disruption_cost_eur:,.0f} / yr</div>
        <div style="font-size: 12px; color: #64748b;">€{maint.daily_maintenance_disruption_cost_eur:,.0f} / day in tactical spacing delays</div>
      </div>
    </div>

    <!-- SPOF Station Maintenance Table -->
    <h3 style="font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 20px; margin-bottom: 8px;">🚨 Radars with Single Points of Failure</h3>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Radar Head Name (ID)</th>
            <th style="text-align: right;">SPOF Cells Dependent</th>
            <th style="text-align: right;">Annual Cell-Hours at Risk</th>
            <th style="text-align: right;">Attributed Outage Cost</th>
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

    # Air Traffic Knock-On Micro-Simulation HTML Section
    sim = analysis.traffic_simulation
    if sim and getattr(sim, "total_flights_simulated", 0) > 0:
        sim_flights_rows = []
        for f in getattr(sim, "sample_impacted_flights", [])[:15]:
            r_badge = "badge-danger" if f.cancellation_risk != "LOW_RISK" else "badge-success"
            downstream_delay_str = f"+{f.reactionary_delay_min:.0f} min" if f.reactionary_delay_min > 0 else "<span style='color: #16a34a;'>On-time (0m)</span>"
            sim_flights_rows.append(
                f'<tr>\n'
                f'  <td><b>{html.escape(f.ac_operator)}</b> (<code>{html.escape(f.ac_reg)}</code>)<br><small style="color: #64748b;">Type: {f.ac_type}</small></td>\n'
                f'  <td><b>{html.escape(f.leg1_route)}</b><br><small style="color: #64748b;">{f.leg1_dep_time} &rarr; {f.leg1_arr_time}</small></td>\n'
                f'  <td style="text-align: right; font-weight: 700; color: #b45309;">+{f.primary_delay_min:.0f} min</td>\n'
                f'  <td style="text-align: right;"><b>+{f.extra_fuel_kg:,.0f} kg</b><br><small style="color: #64748b;">(+{f.extra_co2_kg:,.0f} kg CO2)</small></td>\n'
                f'  <td><b>{html.escape(f.leg2_route or "Final Rotation Leg")}</b><br><small style="color: #64748b;">{("Sched Dep: " + f.leg2_scheduled_dep) if f.leg2_scheduled_dep else "No subsequent leg"}</small></td>\n'
                f'  <td style="text-align: right;"><b>{f.turnaround_buffer_min:.0f} min</b></td>\n'
                f'  <td style="text-align: right;">{downstream_delay_str}</td>\n'
                f'  <td><span class="badge {r_badge}">{html.escape(f.cancellation_risk)}</span></td>\n'
                f'</tr>'
            )
        sim_table_body = "\n".join(sim_flights_rows) if sim_flights_rows else "<tr><td colspan='8'>No flight knock-on delays detected.</td></tr>"

        traffic_sim_section_html = f"""
  <!-- Air Traffic Knock-On & Downstream Cancellation Simulation Section -->
  <div class="section-card">
    <h2>✈️ Air Traffic Micro-Simulation: Non-Radar Separation &amp; Downstream Rotational Knock-On</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Empirical operational micro-simulation executed across actual Eurocontrol flight trajectories for <b>{html.escape(analysis.fir_id)}</b>.
      Simulates primary separation expansions (5 NM &rarr; {sim.separation_standard_nm:.1f} NM procedural separation),
      rotational turnaround delay propagation, and rotational cancellation / curfew breach risks across 100% census of flights.
    </p>

    <!-- Simulation KPI Grid -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 18px 0;">
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Direct Flights Impacted</div>
        <div style="font-size: 24px; font-weight: 800; color: #0f172a;">{sim.directly_impacted_flights:,} <span style="font-size: 14px; color: #64748b;">/ {sim.total_flights_simulated:,}</span></div>
        <div style="font-size: 12px; color: #64748b;">Traversals exposed to radar separation expansion</div>
      </div>
      <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Primary Tactical Delay</div>
        <div style="font-size: 24px; font-weight: 800; color: #d97706;">+{sim.total_primary_delay_min:,.0f} min</div>
        <div style="font-size: 12px; color: #64748b;">Speed reduction + miles-in-trail vectoring</div>
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
        <div style="font-size: 11.5px; font-weight: 700; color: #64748b; text-transform: uppercase;">Cancellation / Curfew Risks</div>
        <div style="font-size: 24px; font-weight: 800; color: {'#dc2626' if sim.cancellation_risk_count > 0 else '#16a34a'};">{sim.cancellation_risk_count} flights</div>
        <div style="font-size: 12px; color: #64748b;">{sim.curfew_breaches_count} curfew breaches | {sim.crew_fdp_timeouts_count} crew timeouts</div>
      </div>
    </div>

    <!-- Flight Knock-on Details Table -->
    <h3 style="font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 20px; margin-bottom: 8px;">📋 Trajectory Cascades: Direct &amp; Downstream Flight Impacts</h3>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Primary Flight (Tail)</th>
            <th>Primary Route &amp; Times</th>
            <th style="text-align: right;">Primary Delay</th>
            <th style="text-align: right;">Detour Fuel &amp; CO2</th>
            <th>Downstream Leg (Rotation)</th>
            <th style="text-align: right;">Turn Buffer</th>
            <th style="text-align: right;">Downstream Delay</th>
            <th>Cancellation / Curfew Risk</th>
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
  <meta name="referrer" content="no-referrer-when-downgrade">
  <title>Radar Resilience &amp; Surveillance Rationalization Report — {html.escape(analysis.fir_id)}</title>
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
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
      color: white;
      border-radius: 12px;
      padding: 28px 32px;
      margin-bottom: 24px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }}
    .header-card h1 {{
      margin: 0 0 8px 0;
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.5px;
    }}
    .header-card .subtitle {{
      color: #93c5fd;
      font-size: 15px;
      margin-bottom: 16px;
    }}
    .header-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      font-size: 13px;
      color: #e0e7ff;
    }}
    .header-meta span {{
      background: rgba(255,255,255,0.12);
      padding: 4px 12px;
      border-radius: 6px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
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
      font-size: 11.5px;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--text-muted);
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-size: 26px;
      font-weight: 800;
      color: var(--primary);
      margin-bottom: 4px;
    }}
    .kpi-desc {{
      font-size: 12px;
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
    <h1>📡 Radar Network Resilience &amp; Surveillance Rationalization</h1>
    <div class="subtitle">Surveillance Permutation Stress Testing, Traffic Validation &amp; Eurocontrol Economic CBA</div>
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
      <div class="kpi-value" style="color: #16a34a;">{opt.active_count} <span style="font-size: 18px; color: #64748b;">/ {analysis.baseline_station_count} Radars</span></div>
      <div class="kpi-desc">Decommissions {opt.disabled_count} redundant stations ({round(opt.disabled_count/max(1, analysis.baseline_station_count)*100)}% equipment reduction)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Net Daily Savings (24-Hour)</div>
      <div class="kpi-value" style="color: #16a34a;">€{abs(opt.windows['24h'].net_economic_impact_eur):,.0f}</div>
      <div class="kpi-desc">Net benefit after absorbing all tactical en-route delays and reroutes</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Target Standard Compliance</div>
      <div class="kpi-value">{opt.coverage_coverable_pct:.1f}%</div>
      <div class="kpi-desc">{opt.cells_met} of {opt.total_cells - opt.cells_uncoverable} coverable cells satisfy {analysis.target_standard}</div>
    </div>
    {f'''<div class="kpi-card">
      <div class="kpi-label">Cascaded Neighbor Radars</div>
      <div class="kpi-value" style="color: #0284c7;">{len(analysis.cascaded_adjacent_stations)}</div>
      <div class="kpi-desc">Active radar heads in busier neighbors compensating border cells</div>
    </div>''' if analysis.cascaded_adjacent_stations else ''}
  </div>

  <!-- Interactive Map Section -->
  <div class="section-card">
    <h2>🗺️ Interactive Airspace &amp; Radar Surveillance Map</h2>
    <p style="font-size: 13.5px; color: #475569; margin-top: 0;">
      Interactive Leaflet OpenStreetMap displaying FIR boundary, active radar heads (green),
      decommissioned/standby stations (red), cascaded neighbor stations (cyan), and commercial airports.
    </p>
    {map_iframe_html}
  </div>

  {cascaded_section_html}

  <!-- Permutations Comparison Table -->
  <div class="section-card">
    <h2>📊 Permutation Performance &amp; Operational Window Costs</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Comprehensive evaluation of all tested radar station configurations. Economic figures show net cost impact
      (<b>green negative values indicate net airline &amp; ANSP financial savings</b>; red positive values indicate operational delay penalties).
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Config ID</th>
            <th>Permutation Name &amp; Selection Rationale</th>
            <th style="text-align: center;">Category</th>
            <th style="text-align: right;">Active / Total</th>
            <th style="text-align: right;">Surveillance %</th>
            <th style="text-align: center;">Degraded Apt</th>
            <th style="text-align: right;">1-Hour Peak Net</th>
            <th style="text-align: right;">6-Hour Shift Net</th>
            <th style="text-align: right;">24-Hour Day Net</th>
            <th style="text-align: center;">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {permutations_table_html}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Critical Radar Stations Section -->
  <div class="section-card">
    <h2>🚨 Top Critical Single Point of Failure Radars (Highest N-1 Outage Disruption)</h2>
    <p style="font-size: 13.5px; color: #475569;">
      These radar heads produce the highest operational disruption penalties if taken offline,
      representing vital surveillance backbones that must be protected.
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Radar Head (ID)</th>
            <th>Airspace Role / Colocated Facility</th>
            <th>Coordinates</th>
            <th>Elevation</th>
            <th style="text-align: right;">Peak Hourly Outage Penalty</th>
            <th style="text-align: right;">Daily Outage Disruption Cost</th>
          </tr>
        </thead>
        <tbody>
          {critical_table_html}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Radar Station Utilization Section -->
  <div class="section-card">
    <h2>📡 Radar Station Utilization &amp; Flight Journey Exposure</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Empirical station utilization derived from actual 4D radar/ADS-B flight paths and cell traffic density.
      Quantifies the average daily, hourly, and peak hourly flights tracked by each radar head as part of their journey.
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th>Station Name (ID)</th>
            <th>Airspace Role</th>
            <th>Coordinates &amp; Elevation</th>
            <th style="text-align: center;">Target Config Status</th>
            <th style="text-align: right;">Average Daily Flights Tracked</th>
            <th style="text-align: right;">Average Hourly Flights Tracked</th>
            <th style="text-align: right;">Peak Hourly Flights Tracked</th>
            <th style="text-align: right;">Fleet Utilization (%)</th>
            <th style="text-align: center;">Cells Covered FL200</th>
          </tr>
        </thead>
        <tbody>
          {utilization_table_html}
        </tbody>
      </table>
    </div>
  </div>

  {maintenance_section_html}

  {traffic_sim_section_html}

  <!-- Mathematical & Economic Appendix -->
  <div class="section-card">
    <h2>📖 Mathematical &amp; Economic Appendix</h2>
    <div class="accordion">
      <details open>
        <summary>1. Line-of-Sight &amp; Radio Horizon Geometry Formulation</summary>
        <div class="accordion-content">
          <p>
            Under ICAO Doc 9684 and Doc 8071, primary and secondary surveillance radar line-of-sight (RLOS)
            distance D_max in nautical miles between an aircraft at altitude h_ac (ft AMSL) and a radar head at elevation h_sta (ft AMSL) is:
          </p>
          <div class="formula-box">
            D_RLOS = 1.23 * (sqrt(h_ac) + sqrt(h_sta)) [NM]
          </div>
          <p>
            The maximum instrumented en-route surveillance range for en-route Secondary Surveillance Radar (SSR) / Mode-S transponder interrogators
            is nominally capped at 250 NM.
          </p>
        </div>
      </details>

      <details>
        <summary>2. Procedural &amp; Non-Radar Longitudinal Separation Expansion</summary>
        <div class="accordion-content">
          <p>
            When radar surveillance is lost or degraded, air traffic controllers cannot maintain
            standard 5 NM radar separation and must transition to procedural miles-in-trail longitudinal separation (10 to 20 NM).
            Based on Eurocontrol Performance Review Unit (PRU) empirical observations:
          </p>
          <ul>
            <li><b>En-route Vectoring Delay:</b> Flights traversing degraded sectors experience an average of +6.5 minutes tactical vectoring and speed reduction delay.</li>
            <li><b>Tactical Detour:</b> 25% of flights execute tactical dog-legs around surveillance gaps, averaging +25.0 NM detour distance.</li>
            <li><b>Terminal Radar Approach Holding:</b> Loss of approach radar monitoring increases runway spacing from 3 NM to 6–8 NM, reducing Arrival Acceptance Rate (AAR) by ~30%, causing terminal stack holding delays averaging +16.0 minutes per arrival.</li>
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
            <li><b>Cost of Airborne Delay:</b> <b>€54.00 per minute</b>.</li>
            <li><b>Cost of Ground Slot Delay:</b> <b>€28.00 per minute</b>.</li>
            <li><b>Tactical Fuel Burn &amp; Carbon Cost:</b> 4.20 kg Jet-A/NM at €0.85/kg plus EU ETS carbon allowance yielding <b>€4.54 per detour NM</b>.</li>
            <li><b>Surveillance Radar Annual OPEX:</b> <b>€85,000 / year</b> per radar head (€42k maintenance, €22k flight calibration check, €13k comms &amp; licensing, €8k power utility = <b>€9.703 / hr</b>).</li>
          </ul>
        </div>
      </details>

      <details>
        <summary>4. Planned Outages &amp; Surveillance Reliability Mechanics</summary>
        <div class="accordion-content">
          <p>
            Each radar head is assigned a standardized planned outage rate of <b>&lambda;_maint = 0.5% (0.005)</b>
            to account for annual preventative maintenance, periodic flight calibration inspections, and hardware servicing:
          </p>
          <div class="formula-box">
            AnnualDowntime(s) = 0.005 * 8760 h = 43.8 h/year (approx. 7.2 min/day)
          </div>
          <p>
            <b>Single Points of Failure (SPOF):</b> In cells relying strictly on a single radar, any outage causes immediate capability loss (P = 0.005).
            In redundant cells with dual radar coverage, capability loss requires concurrent simultaneous outages (P &approx; (2 * &lambda;)&sup2; = 0.0001).
          </p>
        </div>
      </details>

      <details>
        <summary>5. Cross-Border Active Radar Cascading &amp; Regional Rationalization</summary>
        <div class="accordion-content">
          <p>
            Radar line-of-sight coverage extends up to 250 NM across sovereign boundaries.
            By sequencing evaluations from the busiest FIRs to adjacent less busy FIRs, core radar stations locked in
            by high-density hub airspaces are committed as fixed active nodes in the Pan-European network registry.
            Adjacent airspaces leverage these active external beacons to satisfy border surveillance requirements,
            enabling deeper local equipment rationalization (€85k/yr per decommissioned station) without compromising safety.
          </p>
        </div>
      </details>
    </div>
  </div>

  <div style="text-align: center; font-size: 12px; color: #94a3b8; margin-top: 32px;">
    FlightML Aeronautical Intelligence Engine • Generated automatically from Eurocontrol historical trajectory logs and ICAO Doc 9684 standards.
  </div>

</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    return output_path
