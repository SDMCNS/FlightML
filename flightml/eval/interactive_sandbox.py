"""
FlightML Interactive FIR Configuration Sandbox Engine.

Provides instant what-if experimentation for both DME (Terrestrial Navigation)
and Radar (Surveillance) networks across European sovereign FIRs:
1. Loads the ideal rationalized baseline configuration for any sovereign FIR.
2. Interactively simulates the activation, deactivation, or failure of any individual ground beacon/radar.
3. Computes real-time deltas in sub-milliseconds:
   - En-route coverage percentage (FL50 / FL200)
   - Terminal/Airport coverage integrity (FL10 down to ground level)
   - Newly degraded aerodromes
   - Annual OPEX savings delta (€45k/DME, €85k/Radar)
   - Operational delay penalty and Single Point of Failure (SPOF) vulnerabilities
4. Renders interactive Folium maps with color-coded status and deficit callouts.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import folium
import numpy as np
from shapely.geometry import Point

from flightml.data.dataset import FlightDataset
from flightml.eval.ecac_station_rationalization import (
    ECAC_FIR_REGISTRY,
    ECACStation,
    get_fir_geometry,
    load_ecac_stations,
    generate_ecac_grid,
    _compute_coverage_matrices,
    DME_ANNUAL_COST_SAVINGS_EUR,
    RADAR_ANNUAL_COST_SAVINGS_EUR,
)
from flightml.eval.radar_permutations import _evaluate_single_radar_permutation_coverage
from flightml.eval.dme_permutations import _evaluate_single_permutation_coverage

RADAR_CONFIGS_DIR = Path("data/reports/radar_resilience/configurations")
DME_CONFIGS_DIR = Path("data/reports/dme_resilience/configurations")


def get_available_fir_list(system_type: str, dataset: FlightDataset | None = None) -> list[tuple[str, str]]:
    """Return sorted list of (fir_id, display_label) available for the chosen system."""
    system_norm = system_type.upper()
    if system_norm == "RADAR":
        from flightml.eval.radar_ecac_pipeline import get_ranked_radar_firs
        ds = dataset or FlightDataset()
        ranked = get_ranked_radar_firs(ds)
        return [(r["fir_id"], f"{r['fir_id']} — {r['fir_name']} ({r['radar_count']} radars)") for r in ranked]
    else:
        from flightml.eval.dme_ecac_pipeline import get_ranked_ecac_firs
        ds = dataset or FlightDataset()
        ranked = get_ranked_ecac_firs(ds)
        return [(r["fir_id"], f"{r['fir_id']} — {r['fir_name']} ({r['dme_count']} DMEs)") for r in ranked]


def load_fir_ideal_configuration(
    system_type: str,
    fir_id: str,
    target_standard: str | None = None,
    period: str = "202507",
    dataset: FlightDataset | None = None,
) -> dict[str, Any]:
    """
    Load precomputed ideal configuration for the specified FIR and system type.
    Falls back to dynamic simulation if no precomputed configuration exists yet.
    """
    system_norm = system_type.upper()
    default_std = "VIEW_3" if system_norm == "RADAR" else "RNAV_5"
    std = target_standard or default_std

    cfg_dir = RADAR_CONFIGS_DIR if system_norm == "RADAR" else DME_CONFIGS_DIR
    cfg_file = cfg_dir / f"{fir_id}_{std}_{period}.json"

    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # If file not found, dynamically compute baseline stations and candidate configuration
    ds = dataset or FlightDataset(period=period)
    fir_geom = get_fir_geometry(ds, fir_id)
    fir_poly, b_coords, fir_bbox = fir_geom if fir_geom else (None, None, None)
    sta_type = "RADAR" if system_norm == "RADAR" else "DME"
    stations, airports = load_ecac_stations(ds, station_type=sta_type, fir_poly=fir_poly, fir_bbox=fir_bbox, include_adjacent=False)
    sta_ids = [s.id for s in stations]

    # Quick heuristic or simulation to generate default ideal configuration
    if system_norm == "RADAR":
        from flightml.eval.radar_permutations import simulate_radar_permutations
        analysis = simulate_radar_permutations(ds, fir_id=fir_id, period=period, target_standard=std, max_permutations=3)
        opt = analysis.optimal_permutation
        act_ids = list(opt.active_station_ids) if opt else sta_ids
        dis_ids = list(opt.disabled_station_ids) if opt else []
        spofs = analysis.critical_stations or []
        annual_sav = len(dis_ids) * RADAR_ANNUAL_COST_SAVINGS_EUR
        cov_pct = opt.coverage_coverable_pct if opt else 100.0
        deg_apts = opt.degraded_airports_idents if opt else []
    else:
        from flightml.eval.dme_permutations import simulate_dme_permutations
        analysis = simulate_dme_permutations(ds, fir_id=fir_id, period=period, target_standard=std, max_permutations=3)
        opt = analysis.optimal_permutation
        act_ids = list(opt.active_station_ids) if opt else sta_ids
        dis_ids = list(opt.disabled_station_ids) if opt else []
        spofs = analysis.critical_stations or []
        annual_sav = len(dis_ids) * DME_ANNUAL_COST_SAVINGS_EUR
        cov_pct = opt.coverage_coverable_pct if opt else 100.0
        deg_apts = opt.degraded_airports_idents if opt else []

    fir_name = ECAC_FIR_REGISTRY.get(fir_id, fir_id)
    payload = {
        "fir_id": fir_id,
        "fir_name": fir_name,
        "period": period,
        "system_type": system_norm,
        "target_standard": std,
        "baseline_station_count": len(stations),
        "optimal_permutation_id": "P-IDEAL",
        "optimal_permutation_name": f"Optimal Rationalized {system_norm} Core",
        "optimal_active_station_ids": act_ids,
        "decommissioned_station_ids": dis_ids,
        "all_stations": [
            {
                "id": s.id,
                "name": s.name,
                "role": "Terminal Primary" if s.is_airport_station else "Core En-Route Backbone",
                "lat": s.latitude,
                "lon": s.longitude,
                "elevation_ft": s.elevation_ft,
                "range_nm": s.nominal_range_nm,
                "is_active_in_optimal": (s.id in act_ids),
                "is_airport_station": s.is_airport_station,
                "nearest_airport": s.nearest_airport_ident,
            }
            for s in stations
        ],
        "coverage_compliance_pct": cov_pct,
        "degraded_airports_count": len(deg_apts),
        "degraded_airports_idents": deg_apts,
        "annual_savings_eur": annual_sav,
        "daily_net_savings_eur": 0.0,
        "critical_spofs": spofs,
    }

    # Cache payload
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

    return payload


# In-memory evaluation cache for instant what-if reruns
_SANDBOX_CACHE: dict[str, dict[str, Any]] = {}


def get_sandbox_evaluation_cache(
    system_type: str,
    fir_id: str,
    target_standard: str,
    period: str = "202507",
    dataset: FlightDataset | None = None,
) -> dict[str, Any]:
    """Retrieve or precompute vectorized LOS matrices and geometry for sub-millisecond what-if evaluations."""
    cache_key = f"{system_type}_{fir_id}_{target_standard}_{period}"
    if cache_key in _SANDBOX_CACHE:
        return _SANDBOX_CACHE[cache_key]

    ds = dataset or FlightDataset(period=period)
    fir_geom = get_fir_geometry(ds, fir_id)
    fir_poly, b_coords, fir_bbox = fir_geom if fir_geom else (None, None, None)
    sta_type = "RADAR" if system_type.upper() == "RADAR" else "DME"
    stations, airports = load_ecac_stations(ds, station_type=sta_type, fir_poly=fir_poly, fir_bbox=fir_bbox, include_adjacent=False)
    grid_cells = generate_ecac_grid(step_deg=0.18, fir_poly=fir_poly, fir_bbox=fir_bbox)

    sta_id_to_idx = {s.id: idx for idx, s in enumerate(stations)}

    c_lats = np.array([c[0] for c in grid_cells])
    c_lons = np.array([c[1] for c in grid_cells])
    s_lats = np.array([s.latitude for s in stations])
    s_lons = np.array([s.longitude for s in stations])

    dlat_c = (c_lats[:, None] - s_lats[None, :]) * 60.0
    mid_lat_c = np.radians((c_lats[:, None] + s_lats[None, :]) / 2.0)
    dlon_c = (c_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_c)
    dist_cell = np.sqrt(dlat_c**2 + dlon_c**2)
    bearings_cell = (np.degrees(np.arctan2(dlon_c, dlat_c)) + 360.0) % 360.0

    cell_matrices, apt_matrices = _compute_coverage_matrices(
        stations, grid_cells, airports, flight_levels=(10, 50, 95, 200)
    )
    cell_mat_200 = cell_matrices[200]
    cell_mat_50 = cell_matrices.get(50, cell_mat_200)
    cell_mat_100 = cell_matrices.get(95, cell_matrices.get(100, cell_mat_200))
    apt_mat_10 = apt_matrices[10]

    sys_norm = system_type.upper()
    if sys_norm == "RADAR":
        std_norm = target_standard.upper()
        if std_norm in ("VIEW_3", "SURV_3", "FL50", "LOW_LEVEL"):
            coverable_mask = cell_mat_50.sum(axis=1) > 0
        else:
            coverable_mask = cell_mat_200.sum(axis=1) > 0
    else:
        coverable_mask = cell_mat_200.sum(axis=1) > 0

    cache_data = {
        "system_type": sys_norm,
        "fir_id": fir_id,
        "stations": stations,
        "sta_id_to_idx": sta_id_to_idx,
        "grid_cells": grid_cells,
        "airports": airports,
        "cell_mat_200": cell_mat_200,
        "cell_mat_50": cell_mat_50,
        "cell_mat_100": cell_mat_100,
        "apt_mat_10": apt_mat_10,
        "dist_cell": dist_cell,
        "bearings_cell": bearings_cell,
        "coverable_mask": coverable_mask,
        "poly_coords": b_coords,
    }

    _SANDBOX_CACHE[cache_key] = cache_data
    return cache_data


def evaluate_custom_fir_posture(
    system_type: str,
    fir_id: str,
    active_station_ids: set[str],
    target_standard: str,
    ideal_active_station_ids: set[str],
    period: str = "202507",
    dataset: FlightDataset | None = None,
) -> dict[str, Any]:
    """
    Sub-millisecond vectorized evaluation of an arbitrary active/deactivated station selection.
    Quantifies coverage met %, degraded airports, OPEX financial balance, and operational delay penalties.
    """
    system_norm = system_type.upper()
    cache = get_sandbox_evaluation_cache(system_norm, fir_id, target_standard, period=period, dataset=dataset)

    stations = cache["stations"]
    sta_map = cache["sta_id_to_idx"]
    airports = cache["airports"]
    coverable_mask = cache["coverable_mask"]

    n_baseline = len(stations)
    active_indices = [sta_map[sid] for sid in active_station_ids if sid in sta_map]
    active_cnt = len(active_indices)
    disabled_cnt = n_baseline - active_cnt

    annual_unit_cost = RADAR_ANNUAL_COST_SAVINGS_EUR if system_norm == "RADAR" else DME_ANNUAL_COST_SAVINGS_EUR

    if system_norm == "RADAR":
        met, at_risk, cannot, unc, degraded_apts = _evaluate_single_radar_permutation_coverage(
            active_indices=active_indices,
            cell_mat_200=cache["cell_mat_200"],
            cell_mat_50=cache["cell_mat_50"],
            apt_mat_10=cache["apt_mat_10"],
            target_standard=target_standard,
            airports=airports,
            coverable_cells_mask=coverable_mask,
        )
    else:
        met, at_risk, cannot, unc, degraded_apts = _evaluate_single_permutation_coverage(
            active_indices=active_indices,
            cell_mat_200=cache["cell_mat_200"],
            cell_mat_100=cache["cell_mat_100"],
            apt_mat_10=cache["apt_mat_10"],
            bearings_cell=cache["bearings_cell"],
            dist_cell=cache["dist_cell"],
            target_standard=target_standard,
            airports=airports,
            coverable_cells_mask=coverable_mask,
        )

    n_coverable = int(coverable_mask.sum())
    cov_met_pct = round((met / max(1, len(cache["grid_cells"]))) * 100.0, 1)
    cov_cov_pct = round((met / max(1, n_coverable)) * 100.0, 1) if n_coverable > 0 else 100.0

    # Financial OPEX savings vs baseline
    annual_opex_savings_eur = disabled_cnt * annual_unit_cost
    daily_opex_savings_eur = annual_opex_savings_eur / 365.0

    # Compare with ideal configuration
    ideal_active_cnt = len(ideal_active_station_ids)
    delta_active_vs_ideal = active_cnt - ideal_active_cnt
    toggled_off_from_ideal = sorted(list(ideal_active_station_ids - active_station_ids))
    toggled_on_from_ideal = sorted(list(active_station_ids - ideal_active_station_ids))

    # Calculate operational penalty if degraded airports or coverage deficits exist
    daily_operational_penalty_eur = 0.0
    hourly_operational_penalty_eur = 0.0
    degraded_apt_details = []

    apt_dict = {a["ident"]: a for a in airports}
    for apt_id in degraded_apts:
        apt_info = apt_dict.get(apt_id, {})
        mv = apt_info.get("daily_movements", 250.0)
        # Tactical non-radar / dead-reckoning holding and separation penalty
        pen_day = mv * (2200.0 if system_norm == "RADAR" else 1400.0)
        daily_operational_penalty_eur += pen_day
        degraded_apt_details.append({
            "ident": apt_id,
            "name": apt_info.get("name", apt_id),
            "movements": mv,
            "penalty_daily_eur": pen_day,
        })

    hourly_operational_penalty_eur = daily_operational_penalty_eur / 24.0
    net_daily_economic_impact_eur = daily_operational_penalty_eur - daily_opex_savings_eur

    # Identify Single Points of Failure (SPOFs) among currently active stations
    active_spofs = []
    if active_cnt > 0:
        for s_idx in active_indices:
            s_obj = stations[s_idx]
            test_without_s = [i for i in active_indices if i != s_idx]
            if system_norm == "RADAR":
                _, _, _, _, test_deg = _evaluate_single_radar_permutation_coverage(
                    active_indices=test_without_s,
                    cell_mat_200=cache["cell_mat_200"],
                    cell_mat_50=cache["cell_mat_50"],
                    apt_mat_10=cache["apt_mat_10"],
                    target_standard=target_standard,
                    airports=airports,
                    coverable_cells_mask=coverable_mask,
                )
            else:
                _, _, _, _, test_deg = _evaluate_single_permutation_coverage(
                    active_indices=test_without_s,
                    cell_mat_200=cache["cell_mat_200"],
                    cell_mat_100=cache["cell_mat_100"],
                    apt_mat_10=cache["apt_mat_10"],
                    bearings_cell=cache["bearings_cell"],
                    dist_cell=cache["dist_cell"],
                    target_standard=target_standard,
                    airports=airports,
                    coverable_cells_mask=coverable_mask,
                )
            new_losses = [a for a in test_deg if a not in degraded_apts]
            if len(new_losses) > 0:
                active_spofs.append({
                    "station_id": s_obj.id,
                    "station_name": s_obj.name,
                    "airports_vulnerable": new_losses,
                })

    return {
        "system_type": system_norm,
        "fir_id": fir_id,
        "target_standard": target_standard,
        "baseline_station_count": n_baseline,
        "active_station_count": active_cnt,
        "disabled_station_count": disabled_cnt,
        "active_station_ids": active_station_ids,
        "disabled_station_ids": set(s.id for s in stations) - active_station_ids,
        "delta_active_vs_ideal": delta_active_vs_ideal,
        "toggled_off_from_ideal": toggled_off_from_ideal,
        "toggled_on_from_ideal": toggled_on_from_ideal,
        "coverage_met_pct": cov_met_pct,
        "coverage_coverable_pct": cov_cov_pct,
        "cells_met": met,
        "cells_at_risk": at_risk,
        "cells_cannot": cannot,
        "total_cells": len(cache["grid_cells"]),
        "degraded_airports_count": len(degraded_apts),
        "degraded_airports_idents": degraded_apts,
        "degraded_airports_details": degraded_apt_details,
        "annual_opex_savings_eur": annual_opex_savings_eur,
        "daily_opex_savings_eur": daily_opex_savings_eur,
        "daily_operational_penalty_eur": daily_operational_penalty_eur,
        "hourly_operational_penalty_eur": hourly_operational_penalty_eur,
        "net_daily_economic_impact_eur": net_daily_economic_impact_eur,
        "active_spofs": active_spofs,
    }


def create_sandbox_map(
    eval_result: dict[str, Any],
    ideal_active_station_ids: set[str],
    dataset: FlightDataset | None = None,
    show_cells: bool = False,
) -> folium.Map:
    """
    Build an interactive Folium Map showing the current sandbox configuration state:
    - Sovereign FIR boundary
    - Stations color-coded by posture & what-if action:
      🟢 Green: Active Core (Ideal & Active)
      🔴 Red: Standby / Decommissioned (Ideal & Standby)
      🟠 Amber/Orange: Toggled Outage (Was active in ideal, deactivated by user!)
      🔵 Blue: Toggled Reactivated (Was standby in ideal, brought online by user!)
    - Aerodrome pins:
      🟢 Green pin: Terminal surveillance / dual DME requirement met down to FL10
      🔴 Red pin: CRITICAL DEFICIT! Terminal coverage lost at FL10!
    """
    fir_id = eval_result["fir_id"]
    system_type = eval_result["system_type"]
    target_std = eval_result["target_standard"]
    active_ids = eval_result["active_station_ids"]
    degraded_idents = set(eval_result.get("degraded_airports_idents", []))

    ds = dataset or FlightDataset()
    cache = get_sandbox_evaluation_cache(system_type, fir_id, target_std, dataset=ds)
    stations = cache["stations"]
    airports = cache["airports"]
    poly_coords = cache["poly_coords"]

    # Calculate center
    lats = [s.latitude for s in stations] or [50.0]
    lons = [s.longitude for s in stations] or [10.0]
    c_lat, c_lon = float(np.mean(lats)), float(np.mean(lons))

    m = folium.Map(
        location=[c_lat, c_lon],
        zoom_start=6,
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, Humanitarian OpenStreetMap Team",
        name="Humanitarian OSM (Default)",
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Esri World Topo",
    ).add_to(m)

    # 1. Sovereign FIR Boundary
    if poly_coords:
        folium.Polygon(
            locations=poly_coords,
            color="#2563eb",
            weight=2.5,
            dash_array="6, 6",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.04,
            tooltip=f"<b>{html.escape(ECAC_FIR_REGISTRY.get(fir_id, fir_id))} [{fir_id}]</b>",
        ).add_to(m)

    # 2. Station Layers
    sta_group_active = folium.FeatureGroup(name="Active Core Stations", show=True)
    sta_group_standby = folium.FeatureGroup(name="Standby / Decommissioned", show=True)
    sta_group_toggled = folium.FeatureGroup(name="What-If Toggled Elements", show=True)

    for s in stations:
        is_active = s.id in active_ids
        was_ideal_active = s.id in ideal_active_station_ids

        # Determine visual style
        if is_active and was_ideal_active:
            status_text = "ACTIVE CORE"
            color = "#10b981"
            fill_color = "#34d399"
            target_grp = sta_group_active
            radius = 7
        elif not is_active and not was_ideal_active:
            status_text = "STANDBY (Decommissioned)"
            color = "#ef4444"
            fill_color = "#f87171"
            target_grp = sta_group_standby
            radius = 6
        elif not is_active and was_ideal_active:
            status_text = "⚠️ TOGGLED OUTAGE (What-If Deactivation)"
            color = "#ea580c"
            fill_color = "#f97316"
            target_grp = sta_group_toggled
            radius = 9
        else:
            status_text = "⚡ TOGGLED REACTIVATED (What-If Online)"
            color = "#2563eb"
            fill_color = "#60a5fa"
            target_grp = sta_group_toggled
            radius = 9

        folium.CircleMarker(
            location=[s.latitude, s.longitude],
            radius=radius,
            color=color,
            weight=2.5,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.9,
            tooltip=(
                f"<b>{html.escape(s.name)} [{s.id}]</b><br>"
                f"Type: {s.type} | Elevation: {s.elevation_ft:.0f} ft<br>"
                f"Current Posture: <b>{status_text}</b><br>"
                f"Range: {s.nominal_range_nm:.0f} NM"
            ),
        ).add_to(target_grp)

    sta_group_active.add_to(m)
    sta_group_standby.add_to(m)
    sta_group_toggled.add_to(m)

    # 3. Airport Terminal Coverage Layer
    apt_group = folium.FeatureGroup(name="Terminal Hubs (FL10 Integrity)", show=True)
    for a in airports:
        is_deg = a["ident"] in degraded_idents
        color = "#dc2626" if is_deg else "#16a34a"
        icon_name = "remove" if is_deg else "ok"
        status_label = "🚨 TERMINAL RADAR / DUAL DME DEFICIT" if is_deg else "✅ FL10 COVERAGE MET"

        folium.Marker(
            location=[a["latitude"], a["longitude"]],
            icon=folium.Icon(color="red" if is_deg else "green", icon=icon_name, prefix="glyphicon"),
            tooltip=(
                f"<b>{html.escape(a['name'])} [{a['ident']}]</b><br>"
                f"Status: <b>{status_label}</b><br>"
                f"Daily Traffic: {a.get('daily_movements', 0):.0f} mov/day"
            ),
        ).add_to(apt_group)

    apt_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m
