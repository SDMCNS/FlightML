"""
FlightML ECAC Station Rationalization & Coverage Architecture Engine.

Specialized for European Civil Aviation Conference (ECAC) airspace:
- Evaluates Primary/Secondary Radar and Terrestrial DME networks.
- Precomputes 3 progressive static rationalization views:
    * View 1: Minimum Required (>= 1 station in each cell at FL200+ AND >= 1 station at FL10 at airports)
    * View 2: Dual En-Route Redundancy (>= 2 stations in each cell at FL200+ AND >= 1 station at airports)
    * View 3: Low-Level Infill & High-Density Terminal (>= 1 station at FL50+, >= 2 at FL200+, and >= 2 stations at busy airports >= 500 moves/day at FL10)
- Identifies unserviceable baseline gaps as UNCOVERABLE cells (preventing infinite solver loops).
- Computes an ideal synthetic station list (Facility Location problem) to cover gaps with minimum equipment.
- Allows full interactive user augmentation (activating / deactivating individual stations).
- Delivers multi-level coverage maps for FL50, FL100, FL200, and FL300.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Literal
import numpy as np
import pandas as pd

from flightml.data.dataset import FlightDataset
import flightml.math as fmath
from flightml.utils.cache import disk_cache

# ECAC Continental Airspace Bounding Box (Lat: 34N to 68N, Lon: -12W to 35E)
ECAC_LAT_MIN = 35.0
ECAC_LAT_MAX = 65.0
ECAC_LON_MIN = -11.0
ECAC_LON_MAX = 32.0

ECAC_COUNTRIES = {
    "AL", "AM", "AT", "AZ", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "GE", "DE", "GR", "HU", "IS", "IE", "IT", "LV", "LT", "LU", "MT", "MD", "MC", "ME",
    "NL", "MK", "NO", "PL", "PT", "RO", "SM", "RS", "SK", "SI", "ES", "SE", "CH", "TR", "UA", "GB"
}


@dataclass
class ECACStation:
    id: str
    name: str
    type: str  # "RADAR" or "DME" / "VOR-DME"
    latitude: float
    longitude: float
    elevation_ft: float
    country: str
    is_airport_station: bool
    nearest_airport_ident: str | None
    nominal_range_nm: float
    annual_cost_savings_eur: float
    annual_power_savings_mwh: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SyntheticGapFiller:
    id: str
    name: str
    latitude: float
    longitude: float
    elevation_ft: float
    target_flight_level: int
    service_range_nm: float
    cells_covered_count: int
    region_note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ecac_stations(
    dataset: FlightDataset,
    station_type: Literal["RADAR", "DME"] = "RADAR",
    radars_path: str = "data/radars.json",
) -> tuple[list[ECACStation], list[dict[str, Any]]]:
    """
    Load all ECAC stations (Radars or DMEs) along with ECAC commercial airports
    and high-density hubs (>= 500 movements/day).
    """
    # 1. Fetch ECAC airports
    airports_df = dataset.cursor.execute(
        """
        WITH moves AS (
            SELECT adep AS apt FROM flights
            UNION ALL
            SELECT ades AS apt FROM flights
        ),
        counts AS (
            SELECT apt, ROUND(COUNT(*) / 31.0, 1) as daily_moves
            FROM moves
            WHERE apt IS NOT NULL
            GROUP BY apt
        )
        SELECT a.ident, a.name, a.latitude_deg as latitude, a.longitude_deg as longitude,
               COALESCE(a.elevation_ft, 100.0) as elevation_ft, a.iso_country,
               COALESCE(c.daily_moves, 0.0) as daily_movements
        FROM raw_airports a
        LEFT JOIN counts c ON a.ident = c.apt
        WHERE a.latitude_deg BETWEEN ? AND ?
          AND a.longitude_deg BETWEEN ? AND ?
          AND a.type IN ('large_airport', 'medium_airport')
        ORDER BY daily_movements DESC
        """,
        [ECAC_LAT_MIN, ECAC_LAT_MAX, ECAC_LON_MIN, ECAC_LON_MAX],
    ).df()

    airports: list[dict[str, Any]] = []
    for _, row in airports_df.iterrows():
        airports.append({
            "ident": str(row["ident"]),
            "name": str(row["name"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "elevation_ft": float(row["elevation_ft"]),
            "country": str(row.get("iso_country") or "EU"),
            "daily_movements": float(row["daily_movements"]),
            "is_high_density": float(row["daily_movements"]) >= 500.0,
        })

    # 2. Load Stations
    stations: list[ECACStation] = []
    seen_ids = set()

    if station_type == "RADAR":
        if os.path.exists(radars_path):
            with open(radars_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for feat in data.get("features", []):
                props = feat.get("properties", {})
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [0.0, 0.0])
                lon, lat = float(coords[0]), float(coords[1])
                if not (ECAC_LAT_MIN - 1.0 <= lat <= ECAC_LAT_MAX + 1.0 and ECAC_LON_MIN - 1.0 <= lon <= ECAC_LON_MAX + 1.0):
                    continue

                r_id = str(props.get("id") or props.get("name") or f"RAD_{len(stations)+1}").strip()
                if r_id in seen_ids:
                    r_id = f"{r_id}_{len(stations)+1}"
                seen_ids.add(r_id)

                # Check proximity to nearest airport (<= 5 NM)
                min_dist = 99999.0
                nearest_apt_ident = None
                for a in airports:
                    if abs(lat - a["latitude"]) > 0.15 or abs(lon - a["longitude"]) > 0.25:
                        continue
                    d = fmath.haversine_distance_nm(lat, lon, a["latitude"], a["longitude"])
                    if d < min_dist:
                        min_dist = d
                        nearest_apt_ident = a["ident"]

                is_apt = min_dist <= 5.0
                stations.append(
                    ECACStation(
                        id=r_id,
                        name=str(props.get("name") or r_id),
                        type="RADAR",
                        latitude=lat,
                        longitude=lon,
                        elevation_ft=150.0,
                        country="EU",
                        is_airport_station=is_apt,
                        nearest_airport_ident=nearest_apt_ident if is_apt else None,
                        nominal_range_nm=250.0,  # Max line of sight 250 NM for radar
                        annual_cost_savings_eur=85000.0,
                        annual_power_savings_mwh=135.0,
                    )
                )
    else:  # DME
        df_dme = dataset.cursor.execute(
            """
            SELECT ident, name, type, latitude_deg, longitude_deg, elevation_ft, iso_country
            FROM raw_navaids
            WHERE (type LIKE '%DME%' OR type = 'VOR-DME' OR type = 'TACAN')
              AND latitude_deg BETWEEN ? AND ?
              AND longitude_deg BETWEEN ? AND ?
            ORDER BY ident ASC
            """,
            [ECAC_LAT_MIN, ECAC_LAT_MAX, ECAC_LON_MIN, ECAC_LON_MAX],
        ).df()

        for _, row in df_dme.iterrows():
            d_id = str(row["ident"]).strip().upper()
            if not d_id or d_id in seen_ids:
                continue
            seen_ids.add(d_id)
            lat = float(row["latitude_deg"])
            lon = float(row["longitude_deg"])
            elev = float(row["elevation_ft"]) if pd.notna(row["elevation_ft"]) else 100.0

            min_dist = 99999.0
            nearest_apt_ident = None
            for a in airports:
                if abs(lat - a["latitude"]) > 0.15 or abs(lon - a["longitude"]) > 0.25:
                    continue
                d = fmath.haversine_distance_nm(lat, lon, a["latitude"], a["longitude"])
                if d < min_dist:
                    min_dist = d
                    nearest_apt_ident = a["ident"]

            is_apt = min_dist <= 5.0
            stations.append(
                ECACStation(
                    id=d_id,
                    name=str(row.get("name") or d_id),
                    type=str(row.get("type") or "DME"),
                    latitude=lat,
                    longitude=lon,
                    elevation_ft=elev,
                    country=str(row.get("iso_country") or "EU"),
                    is_airport_station=is_apt,
                    nearest_airport_ident=nearest_apt_ident if is_apt else None,
                    nominal_range_nm=160.0,  # Max service range 160 NM for DME
                    annual_cost_savings_eur=45000.0,
                    annual_power_savings_mwh=25.0,
                )
            )

    return stations, airports


def generate_ecac_grid(step_deg: float = 0.35) -> list[tuple[float, float]]:
    """
    Generate 2D spatial grid covering ECAC continental airspace (~15-20 NM cells).
    """
    lats = np.arange(ECAC_LAT_MIN, ECAC_LAT_MAX + 1e-5, step_deg)
    lons = np.arange(ECAC_LON_MIN, ECAC_LON_MAX + 1e-5, step_deg * 1.5)
    cells = []
    for lat in lats:
        for lon in lons:
            cells.append((round(float(lat), 3), round(float(lon), 3)))
    return cells


def _compute_coverage_matrices(
    stations: list[ECACStation],
    grid_cells: list[tuple[float, float]],
    airports: list[dict[str, Any]],
    flight_levels: tuple[int, ...] = (10, 50, 100, 200, 300),
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Vectorized computation of boolean visibility matrix:
    cell_matrix[FL]: shape (N_cells, N_stations)
    apt_matrix[FL]:  shape (N_airports, N_stations)
    """
    n_cells = len(grid_cells)
    n_sta = len(stations)
    n_apt = len(airports)

    c_lats = np.array([c[0] for c in grid_cells])
    c_lons = np.array([c[1] for c in grid_cells])

    s_lats = np.array([s.latitude for s in stations])
    s_lons = np.array([s.longitude for s in stations])
    s_elev = np.array([s.elevation_ft for s in stations])
    s_nom = np.array([s.nominal_range_nm for s in stations])

    a_lats = np.array([a["latitude"] for a in airports])
    a_lons = np.array([a["longitude"] for a in airports])

    # Distance matrices (approx NM)
    # Cell to station
    dlat_c = (c_lats[:, None] - s_lats[None, :]) * 60.0
    mid_lat_c = np.radians((c_lats[:, None] + s_lats[None, :]) / 2.0)
    dlon_c = (c_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_c)
    dist_cell = np.sqrt(dlat_c**2 + dlon_c**2)

    # Airport to station
    dlat_a = (a_lats[:, None] - s_lats[None, :]) * 60.0
    mid_lat_a = np.radians((a_lats[:, None] + s_lats[None, :]) / 2.0)
    dlon_a = (a_lons[:, None] - s_lons[None, :]) * 60.0 * np.cos(mid_lat_a)
    dist_apt = np.sqrt(dlat_a**2 + dlon_a**2)

    cell_matrices: dict[int, np.ndarray] = {}
    apt_matrices: dict[int, np.ndarray] = {}

    for fl in flight_levels:
        h_ac = fl * 100.0
        # Line of sight for each station
        rlos = 1.23 * (np.sqrt(h_ac) + np.sqrt(np.maximum(0.0, s_elev)))
        max_r = np.minimum(s_nom, rlos)  # shape (N_stations,)

        cell_matrices[fl] = dist_cell <= max_r[None, :]
        apt_matrices[fl] = dist_apt <= max_r[None, :]

    return cell_matrices, apt_matrices


def compute_static_rationalization_views(
    stations: list[ECACStation],
    grid_cells: list[tuple[float, float]],
    airports: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """
    Computes the 3 static rationalization views using greedy set-cover:
    - VIEW_1: >= 1 station at FL200+ in each coverable cell AND >= 1 station at FL10 at coverable airports.
    - VIEW_2: >= 2 stations at FL200+ in each coverable cell AND >= 1 station at FL10 at airports.
    - VIEW_3: Infill >= 1 station at FL50+ in each cell AND >= 2 stations at busy airports (>= 500 moves/day) at FL10.
    """
    cell_mat, apt_mat = _compute_coverage_matrices(stations, grid_cells, airports, flight_levels=(10, 50, 200))
    n_sta = len(stations)
    sta_ids = [s.id for s in stations]

    # Coverability masks (cells that have at least 1 station when ALL stations are active)
    coverable_c_fl200 = cell_mat[200].sum(axis=1) > 0
    coverable_c_fl50 = cell_mat[50].sum(axis=1) > 0
    coverable_a_fl10 = apt_mat[10].sum(axis=1) > 0

    busy_apt_indices = [i for i, a in enumerate(airports) if a["is_high_density"]]
    coverable_busy_fl10 = np.zeros(len(airports), dtype=bool)
    for i in busy_apt_indices:
        if apt_mat[10][i].sum() >= 2:
            coverable_busy_fl10[i] = True

    # ---------------- VIEW 1: Minimum Baseline ----------------
    view_1_set: set[int] = set()

    # Priority 1: Cover all coverable airports at FL10
    apt_cov_mask = np.zeros(len(airports), dtype=bool)
    while (~apt_cov_mask & coverable_a_fl10).any():
        uncovered = ~apt_cov_mask & coverable_a_fl10
        best_sta = -1
        best_gain = 0
        for s in range(n_sta):
            if s in view_1_set:
                continue
            gain = (apt_mat[10][:, s] & uncovered).sum()
            if gain > best_gain:
                best_gain = gain
                best_sta = s
        if best_gain == 0 or best_sta == -1:
            break
        view_1_set.add(best_sta)
        apt_cov_mask |= apt_mat[10][:, best_sta]

    # Priority 2: Cover all coverable FL200 cells with >= 1 station
    cell_cov_count = cell_mat[200][:, list(view_1_set)].sum(axis=1) if view_1_set else np.zeros(len(grid_cells))
    while ((cell_cov_count < 1) & coverable_c_fl200).any():
        uncovered = (cell_cov_count < 1) & coverable_c_fl200
        if not uncovered.any():
            break
        best_sta = -1
        best_gain = 0
        for s in range(n_sta):
            if s in view_1_set:
                continue
            gain = (cell_mat[200][:, s] & uncovered).sum()
            if gain > best_gain:
                best_gain = gain
                best_sta = s
        if best_gain == 0 or best_sta == -1:
            break
        view_1_set.add(best_sta)
        cell_cov_count += cell_mat[200][:, best_sta]

    # ---------------- VIEW 2: Dual En-Route Redundancy ----------------
    view_2_set: set[int] = set(view_1_set)
    max_fl200_potential = cell_mat[200].sum(axis=1)
    target_fl200_cov = np.minimum(2, max_fl200_potential)

    cell_cov_count_v2 = cell_mat[200][:, list(view_2_set)].sum(axis=1) if view_2_set else np.zeros(len(grid_cells))
    while (cell_cov_count_v2 < target_fl200_cov).any():
        needed = cell_cov_count_v2 < target_fl200_cov
        best_sta = -1
        best_gain = 0
        for s in range(n_sta):
            if s in view_2_set:
                continue
            gain = (cell_mat[200][:, s] & needed).sum()
            if gain > best_gain:
                best_gain = gain
                best_sta = s
        if best_gain == 0 or best_sta == -1:
            break
        view_2_set.add(best_sta)
        cell_cov_count_v2 += cell_mat[200][:, best_sta]

    # ---------------- VIEW 3: Low-Level Infill & Busy Hub Dual ----------------
    view_3_set: set[int] = set(view_2_set)

    # 1. Ensure dual FL10 coverage at busy hubs
    for i in busy_apt_indices:
        if not coverable_busy_fl10[i]:
            continue
        cur_apt_cov = apt_mat[10][i, list(view_3_set)].sum() if view_3_set else 0
        while cur_apt_cov < 2:
            best_s = -1
            for s in range(n_sta):
                if s not in view_3_set and apt_mat[10][i, s]:
                    best_s = s
                    break
            if best_s == -1:
                break
            view_3_set.add(best_s)
            cur_apt_cov += 1

    # 2. Infill FL50 cells to >= 1 station
    cell_cov_fl50 = cell_mat[50][:, list(view_3_set)].sum(axis=1) if view_3_set else np.zeros(len(grid_cells))
    while ((cell_cov_fl50 < 1) & coverable_c_fl50).any():
        uncovered = (cell_cov_fl50 < 1) & coverable_c_fl50
        if not uncovered.any():
            break
        best_sta = -1
        best_gain = 0
        for s in range(n_sta):
            if s in view_3_set:
                continue
            gain = (cell_mat[50][:, s] & uncovered).sum()
            if gain > best_gain:
                best_gain = gain
                best_sta = s
        if best_gain == 0 or best_s == -1:
            break
        view_3_set.add(best_sta)
        cell_cov_fl50 += cell_mat[50][:, best_sta]

    return {
        "VIEW_1": {sta_ids[idx] for idx in view_1_set},
        "VIEW_2": {sta_ids[idx] for idx in view_2_set},
        "VIEW_3": {sta_ids[idx] for idx in view_3_set},
    }


def compute_synthetic_gap_fillers(
    uncoverable_cells: list[tuple[float, float]],
    station_type: Literal["RADAR", "DME"],
    target_fl: int = 200,
) -> list[SyntheticGapFiller]:
    """
    Greedy Facility Location solver:
    Identifies the minimum number of ideal synthetic station locations
    required to cover 100% of the dataset gap cells.
    """
    if not uncoverable_cells:
        return []

    service_r = 160.0 if station_type == "DME" else 220.0
    u_lats = np.array([c[0] for c in uncoverable_cells])
    u_lons = np.array([c[1] for c in uncoverable_cells])
    rem_indices = set(range(len(uncoverable_cells)))

    synthetics: list[SyntheticGapFiller] = []
    gap_count = 0

    while rem_indices:
        gap_count += 1
        best_idx = -1
        best_covered: set[int] = set()

        rem_list = list(rem_indices)
        rem_lats = u_lats[rem_list]
        rem_lons = u_lons[rem_list]

        for c_idx in rem_list:
            lat_c, lon_c = u_lats[c_idx], u_lons[c_idx]
            dlat = (rem_lats - lat_c) * 60.0
            dlon = (rem_lons - lon_c) * 60.0 * np.cos(np.radians(lat_c))
            dists = np.sqrt(dlat**2 + dlon**2)
            covered_in_step = {rem_list[k] for k, d in enumerate(dists) if d <= service_r}

            if len(covered_in_step) > len(best_covered):
                best_covered = covered_in_step
                best_idx = c_idx

        if best_idx == -1 or not best_covered:
            break

        c_lat = float(u_lats[best_idx])
        c_lon = float(u_lons[best_idx])
        rem_indices -= best_covered

        # Geographic region descriptor
        if c_lat < 40.0 and c_lon > 20.0:
            reg = "Eastern Mediterranean / Aegean Gap"
        elif c_lat < 42.0 and c_lon < 0.0:
            reg = "Iberian Atlantic Approach Gap"
        elif c_lat > 60.0:
            reg = "Nordic / Oceanic Airspace Gap"
        elif c_lon > 25.0:
            reg = "Black Sea / Eastern Perimeter Gap"
        else:
            reg = "Central European High-Terrain Gap"

        synthetics.append(
            SyntheticGapFiller(
                id=f"SYN-{station_type}-{gap_count:02d}",
                name=f"Proposed {station_type} Infill #{gap_count}",
                latitude=round(c_lat, 3),
                longitude=round(c_lon, 3),
                elevation_ft=1000.0,
                target_flight_level=target_fl,
                service_range_nm=service_r,
                cells_covered_count=len(best_covered),
                region_note=reg,
            )
        )

    return synthetics


@disk_cache(namespace="ecac_rationalization")
def get_ecac_rationalization_report(
    dataset: FlightDataset,
    station_type: Literal["RADAR", "DME"] = "RADAR",
    active_view: Literal["VIEW_1", "VIEW_2", "VIEW_3"] = "VIEW_1",
    custom_inactive_ids: tuple[str, ...] | None = None,
    custom_active_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Main evaluation pipeline with disk caching.
    Computes active/deactivated stations, multi-altitude coverage maps (FL50, FL100, FL200, FL300),
    airport ground integrity, and synthetic gap filler placements.
    """
    stations, airports = load_ecac_stations(dataset, station_type=station_type)
    grid_cells = generate_ecac_grid(step_deg=0.35)

    # 1. Compute or retrieve static baseline views
    static_views = compute_static_rationalization_views(stations, grid_cells, airports)
    base_active_set = set(static_views.get(active_view, set()))

    # 2. Apply user augmentations
    if custom_inactive_ids:
        base_active_set -= set(custom_inactive_ids)
    if custom_active_ids:
        base_active_set |= set(custom_active_ids)

    # 3. Vectorized coverage evaluation across FL50, FL100, FL200, FL300
    levels = (10, 50, 100, 200, 300)
    cell_matrices, apt_matrices = _compute_coverage_matrices(stations, grid_cells, airports, flight_levels=levels)

    sta_id_to_idx = {s.id: i for i, s in enumerate(stations)}
    active_indices = [sta_id_to_idx[sid] for sid in base_active_set if sid in sta_id_to_idx]

    # Full network potential (for identifying uncoverable cells)
    uncoverable_cells_by_fl: dict[int, list[tuple[float, float]]] = {}
    for fl in (50, 100, 200, 300):
        max_potential = cell_matrices[fl].sum(axis=1)
        unc_mask = max_potential == 0
        uncoverable_cells_by_fl[fl] = [grid_cells[idx] for idx in np.where(unc_mask)[0]]

    # Active coverage evaluations per FL
    coverage_by_fl: dict[str, Any] = {}
    for fl in (50, 100, 200, 300):
        if active_indices:
            counts = cell_matrices[fl][:, active_indices].sum(axis=1)
        else:
            counts = np.zeros(len(grid_cells), dtype=int)

        max_cov = cell_matrices[fl].sum(axis=1)
        coverable_total = int((max_cov > 0).sum())
        total_cells = len(grid_cells)

        single_cov = int((counts >= 1).sum())
        dual_cov = int((counts >= 2).sum())
        outage = int(((counts == 0) & (max_cov > 0)).sum())
        uncoverable = int((max_cov == 0).sum())

        single_pct = round((single_cov / max(1, coverable_total)) * 100.0, 1)
        dual_pct = round((dual_cov / max(1, coverable_total)) * 100.0, 1)

        # Assemble compact cell details
        cell_details = []
        for i, (lat, lon) in enumerate(grid_cells):
            cnt = int(counts[i])
            m_pot = int(max_cov[i])
            if m_pot == 0:
                st = "UNCOVERABLE"
            elif cnt >= 2:
                st = "DUAL_OR_MORE"
            elif cnt == 1:
                st = "SINGLE"
            else:
                st = "OUTAGE"

            cell_details.append({
                "lat": lat,
                "lon": lon,
                "count": cnt,
                "status": st,
            })

        coverage_by_fl[f"fl{fl}"] = {
            "flight_level": fl,
            "altitude_ft": fl * 100,
            "total_cells": total_cells,
            "coverable_cells": coverable_total,
            "uncoverable_cells": uncoverable,
            "single_coverage_cells": single_cov,
            "single_coverage_pct": single_pct,
            "dual_coverage_cells": dual_cov,
            "dual_coverage_pct": dual_pct,
            "outage_cells": outage,
            "cells": cell_details,
        }

    # 4. Airport Coverage at FL10
    apt_cov_fl10 = apt_matrices[10][:, active_indices].sum(axis=1) if active_indices else np.zeros(len(airports))
    apt_reports = []
    busy_dual_met = 0
    busy_total = 0

    for i, apt in enumerate(airports):
        c_10 = int(apt_cov_fl10[i])
        is_busy = apt["is_high_density"]
        if is_busy:
            busy_total += 1
            if c_10 >= 2:
                busy_dual_met += 1

        target_met = (c_10 >= 2) if is_busy else (c_10 >= 1)
        apt_reports.append({
            "ident": apt["ident"],
            "name": apt["name"],
            "latitude": apt["latitude"],
            "longitude": apt["longitude"],
            "daily_movements": apt["daily_movements"],
            "is_high_density": is_busy,
            "coverage_count_fl10": c_10,
            "target_met": target_met,
        })

    # 5. Station Inventory with Active / Inactive states
    station_list = []
    active_count = 0
    deactivated_count = 0
    tot_cost_saved = 0.0
    tot_power_saved = 0.0

    for s in stations:
        is_act = s.id in base_active_set
        if is_act:
            active_count += 1
        else:
            deactivated_count += 1
            tot_cost_saved += s.annual_cost_savings_eur
            tot_power_saved += s.annual_power_savings_mwh

        station_list.append({
            "id": s.id,
            "name": s.name,
            "type": s.type,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "elevation_ft": s.elevation_ft,
            "country": s.country,
            "is_airport_station": s.is_airport_station,
            "nearest_airport": s.nearest_airport_ident,
            "is_active": is_act,
            "annual_savings_eur": s.annual_cost_savings_eur,
            "annual_power_mwh": s.annual_power_savings_mwh,
        })

    # 6. Ideal Synthetic Gap Fillers
    gap_fillers = compute_synthetic_gap_fillers(
        uncoverable_cells_by_fl[200],
        station_type=station_type,
        target_fl=200,
    )

    return {
        "station_type": station_type,
        "active_view": active_view,
        "total_stations_count": len(stations),
        "active_stations_count": active_count,
        "deactivated_stations_count": deactivated_count,
        "annual_cost_savings_eur": tot_cost_saved,
        "annual_power_savings_mwh": tot_power_saved,
        "high_density_airports_count": busy_total,
        "high_density_dual_target_met_count": busy_dual_met,
        "high_density_dual_pct": round((busy_dual_met / max(1, busy_total)) * 100.0, 1),
        "stations": station_list,
        "airports": apt_reports,
        "coverage_by_fl": coverage_by_fl,
        "synthetic_gap_fillers": [g.to_dict() for g in gap_fillers],
        "uncoverable_cells_count_fl200": len(uncoverable_cells_by_fl[200]),
    }
