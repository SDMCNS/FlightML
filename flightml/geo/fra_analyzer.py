"""
Free Route Airspace (FRA), Airway Snapping & RNAV Fix Adherence Engine (Sprint 3).

Snaps 4D flight trajectories against:
1. Published ATS Airway segments (`raw_routes` — Eurocontrol AIRAC Route_2509/2510/2511 & 201611).
2. Global 5-letter RNAV Waypoints (`nav_fixes` — 119,720 fixes from X-Plane `fix.dat`).
3. Global VHF/LF Radio Navaids (`nav_aids_xplane` & `raw_navaids` — VOR, VORTAC, DME, NDB).

Quantifies the transition from structured ATS airways to Free Route Airspace (FRA) Direct (`DCT`) routing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.geo.geodesy import haversine_nm, haversine_vectorized_nm


@dataclass
class SnappedWaypoint:
    seq_num: int
    time_over: str
    flight_level: float
    latitude: float
    longitude: float
    classification: str  # 'TERMINAL_SID_STAR', 'ATS_AIRWAY', 'FRA_PUBLISHED_FIX_DCT', 'TACTICAL_RADAR_DCT'
    matched_airway_id: str | None
    matched_fix_ident: str | None
    matched_fix_type: str | None
    snap_distance_nm: float | None


@dataclass
class FRAReport:
    ectrl_id: int
    period: str
    adep: str
    ades: str
    ac_operator: str
    ac_type: str
    total_enroute_waypoints: int
    airway_waypoints_count: int
    fra_fix_dct_waypoints_count: int
    tactical_dct_waypoints_count: int
    enroute_distance_nm: float
    ats_airway_distance_nm: float
    fra_direct_distance_nm: float
    fra_directness_index_pct: float
    ats_airway_adherence_pct: float
    distinct_airways_used: list[str]
    sample_snapped_fixes: list[str]
    waypoints: list[SnappedWaypoint]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_flight_fra_adherence(
    dataset: FlightDataset,
    ectrl_id: int,
    snap_tolerance_nm: float = 2.5,
) -> FRAReport | None:
    """
    Match each 4D waypoint of a flight against published ATS Airways (`raw_routes`),
    5-letter RNAV fixes (`nav_fixes`), and radio navaids (`nav_aids_xplane`), and compute
    the Free Route Airspace (FRA) Directness Index (%).
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return None

    trajs = dataset.get_trajectory(ectrl_id)
    actual_df = trajs["actual"]
    if len(actual_df) < 3:
        return None

    actual_df = actual_df.sort_values("seq_num").reset_index(drop=True)
    adep = str(flight.get("adep") or "")
    ades = str(flight.get("ades") or "")
    ac_op = str(flight.get("ac_operator") or "")
    ac_type = str(flight.get("ac_type") or "")
    adep_lat = float(flight.get("adep_lat") or actual_df.iloc[0]["latitude"])
    adep_lon = float(flight.get("adep_lon") or actual_df.iloc[0]["longitude"])
    ades_lat = float(flight.get("ades_lat") or actual_df.iloc[-1]["latitude"])
    ades_lon = float(flight.get("ades_lon") or actual_df.iloc[-1]["longitude"])

    lats = actual_df["latitude"].to_numpy(dtype=float)
    lons = actual_df["longitude"].to_numpy(dtype=float)
    min_lat, max_lat = float(np.min(lats)) - 0.5, float(np.max(lats)) + 0.5
    min_lon, max_lon = float(np.min(lons)) - 0.8, float(np.max(lons)) + 0.8

    # Fetch candidate ATS airway points within trajectory bounding box
    routes_df = dataset.cursor.execute(
        """
        SELECT "Route ID" AS route_id, Latitude AS lat, Longitude AS lon
        FROM raw_routes
        WHERE Latitude BETWEEN ? AND ?
          AND Longitude BETWEEN ? AND ?
        """,
        [min_lat, max_lat, min_lon, max_lon],
    ).df()

    # Fetch candidate 5-letter RNAV fixes & radio navaids within bounding box
    fixes_df = dataset.cursor.execute(
        """
        SELECT fix_ident AS ident, 'RNAV_FIX' AS kind, latitude AS lat, longitude AS lon
        FROM nav_fixes
        WHERE latitude BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?
        UNION ALL
        SELECT ident, nav_type AS kind, latitude AS lat, longitude AS lon
        FROM nav_aids_xplane
        WHERE nav_type IN ('VOR', 'NDB', 'DME')
          AND latitude BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?
        """,
        [min_lat, max_lat, min_lon, max_lon, min_lat, max_lat, min_lon, max_lon],
    ).df()

    r_lats = routes_df["lat"].to_numpy(dtype=float) if not routes_df.empty else np.array([])
    r_lons = routes_df["lon"].to_numpy(dtype=float) if not routes_df.empty else np.array([])
    r_ids = routes_df["route_id"].to_numpy(dtype=str) if not routes_df.empty else np.array([])

    f_lats = fixes_df["lat"].to_numpy(dtype=float) if not fixes_df.empty else np.array([])
    f_lons = fixes_df["lon"].to_numpy(dtype=float) if not fixes_df.empty else np.array([])
    f_ids = fixes_df["ident"].to_numpy(dtype=str) if not fixes_df.empty else np.array([])
    f_kinds = fixes_df["kind"].to_numpy(dtype=str) if not fixes_df.empty else np.array([])

    d_adep_arr = haversine_vectorized_nm(adep_lat, adep_lon, lats, lons)
    d_ades_arr = haversine_vectorized_nm(ades_lat, ades_lon, lats, lons)

    snapped_wps: list[SnappedWaypoint] = []
    airways_used: list[str] = []
    fixes_used: list[str] = []

    enroute_wp_count = 0
    airway_wp_count = 0
    fra_fix_wp_count = 0
    tactical_wp_count = 0

    for i, row in actual_df.iterrows():
        lat_i = float(row["latitude"])
        lon_i = float(row["longitude"])
        fl_i = float(row["flight_level"])
        seq_i = int(row["seq_num"])
        t_i = str(row["time_over"])

        is_terminal = (d_adep_arr[i] < 40.0) or (d_ades_arr[i] < 40.0) or (fl_i < 100.0)

        matched_airway = None
        airway_dist = None
        if len(r_lats) > 0:
            # Fast bounding pre-filter (~0.2 deg)
            near_mask = (np.abs(r_lats - lat_i) <= 0.15) & (np.abs(r_lons - lon_i) <= 0.25)
            if np.any(near_mask):
                sub_d = haversine_vectorized_nm(lat_i, lon_i, r_lats[near_mask], r_lons[near_mask])
                min_idx = int(np.argmin(sub_d))
                if float(sub_d[min_idx]) <= snap_tolerance_nm:
                    matched_airway = str(r_ids[near_mask][min_idx])
                    airway_dist = float(sub_d[min_idx])

        matched_fix = None
        matched_fix_kind = None
        fix_dist = None
        if len(f_lats) > 0:
            near_f = (np.abs(f_lats - lat_i) <= 0.15) & (np.abs(f_lons - lon_i) <= 0.25)
            if np.any(near_f):
                sub_fd = haversine_vectorized_nm(lat_i, lon_i, f_lats[near_f], f_lons[near_f])
                min_fidx = int(np.argmin(sub_fd))
                if float(sub_fd[min_fidx]) <= snap_tolerance_nm:
                    matched_fix = str(f_ids[near_f][min_fidx])
                    matched_fix_kind = str(f_kinds[near_f][min_fidx])
                    fix_dist = float(sub_fd[min_fidx])

        if is_terminal:
            cls_label = "TERMINAL_SID_STAR"
        else:
            enroute_wp_count += 1
            if matched_airway is not None:
                cls_label = "ATS_AIRWAY"
                airway_wp_count += 1
                if matched_airway not in airways_used:
                    airways_used.append(matched_airway)
            elif matched_fix is not None:
                cls_label = "FRA_PUBLISHED_FIX_DCT"
                fra_fix_wp_count += 1
            else:
                cls_label = "TACTICAL_RADAR_DCT"
                tactical_wp_count += 1

        if matched_fix and matched_fix not in fixes_used:
            fixes_used.append(matched_fix)

        best_snap_dist = (
            airway_dist
            if airway_dist is not None and (fix_dist is None or airway_dist <= fix_dist)
            else fix_dist
        )

        snapped_wps.append(
            SnappedWaypoint(
                seq_num=seq_i,
                time_over=t_i,
                flight_level=round(fl_i, 0),
                latitude=round(lat_i, 4),
                longitude=round(lon_i, 4),
                classification=cls_label,
                matched_airway_id=matched_airway,
                matched_fix_ident=matched_fix,
                matched_fix_type=matched_fix_kind,
                snap_distance_nm=round(best_snap_dist, 2) if best_snap_dist is not None else None,
            )
        )

    # Calculate en-route distance split between structured ATS airways vs. Free Route (DCT)
    enroute_dist = 0.0
    airway_dist_total = 0.0
    fra_dct_dist_total = 0.0

    for j in range(len(snapped_wps) - 1):
        w1 = snapped_wps[j]
        w2 = snapped_wps[j + 1]
        if w1.classification == "TERMINAL_SID_STAR" and w2.classification == "TERMINAL_SID_STAR":
            continue
        seg_nm = haversine_nm(w1.latitude, w1.longitude, w2.latitude, w2.longitude)
        enroute_dist += seg_nm
        if w1.classification == "ATS_AIRWAY" or w2.classification == "ATS_AIRWAY":
            airway_dist_total += seg_nm
        else:
            fra_dct_dist_total += seg_nm

    if enroute_dist <= 0.1:
        # Fallback for short-haul terminal flights
        seg_all = haversine_vectorized_nm(lats[:-1], lons[:-1], lats[1:], lons[1:])
        enroute_dist = max(1.0, float(np.sum(seg_all)))
        fra_dct_dist_total = enroute_dist - airway_dist_total

    fra_pct = (fra_dct_dist_total / enroute_dist) * 100.0
    airway_pct = (airway_dist_total / enroute_dist) * 100.0

    return FRAReport(
        ectrl_id=int(ectrl_id),
        period=dataset.period,
        adep=adep,
        ades=ades,
        ac_operator=ac_op,
        ac_type=ac_type,
        total_enroute_waypoints=enroute_wp_count,
        airway_waypoints_count=airway_wp_count,
        fra_fix_dct_waypoints_count=fra_fix_wp_count,
        tactical_dct_waypoints_count=tactical_wp_count,
        enroute_distance_nm=round(enroute_dist, 1),
        ats_airway_distance_nm=round(airway_dist_total, 1),
        fra_direct_distance_nm=round(fra_dct_dist_total, 1),
        fra_directness_index_pct=round(fra_pct, 1),
        ats_airway_adherence_pct=round(airway_pct, 1),
        distinct_airways_used=airways_used[:15],
        sample_snapped_fixes=fixes_used[:15],
        waypoints=snapped_wps,
    )


def compare_fra_evolution(
    period1: str = "201611",
    period2: str = "202510",
    sample_flights: int = 300,
) -> dict[str, Any]:
    """
    Compare Free Route Airspace (FRA) adoption and off-airway Direct (DCT) waypoint share
    between two Eurocontrol periods (e.g., Nov 2016 vs. Oct 2025).
    """
    results: dict[str, Any] = {}
    for pid in (period1, period2):
        ds = FlightDataset(period=pid)
        sql = f"""
            WITH sample_flts AS (
                SELECT ectrl_id
                FROM flights
                WHERE actual_distance_nm >= 300
                LIMIT {int(sample_flights)}
            ),
            enroute_pts AS (
                SELECT
                    t.ectrl_id,
                    round(t.latitude, 2) AS lat_grid,
                    round(t.longitude, 2) AS lon_grid
                FROM trajectory_actual t
                JOIN sample_flts s ON t.ectrl_id = s.ectrl_id
                WHERE t.flight_level >= 240
            ),
            route_grid AS (
                SELECT DISTINCT
                    round(Latitude, 2) AS lat_grid,
                    round(Longitude, 2) AS lon_grid
                FROM raw_routes
            )
            SELECT
                CAST(count(*) AS BIGINT) AS total_upper_waypoints,
                CAST(count(r.lat_grid) AS BIGINT) AS on_airway_waypoints,
                CAST(count(*) - count(r.lat_grid) AS BIGINT) AS fra_dct_waypoints,
                round((count(*) - count(r.lat_grid)) * 100.0 / nullif(count(*), 0), 1) AS fra_direct_pct
            FROM enroute_pts p
            LEFT JOIN route_grid r
              ON p.lat_grid = r.lat_grid AND p.lon_grid = r.lon_grid;
        """
        row = ds.cursor.execute(sql).fetchone()
        results[pid] = {
            "period": pid,
            "total_upper_waypoints": int(row[0] or 0) if row else 0,
            "on_airway_waypoints": int(row[1] or 0) if row else 0,
            "fra_dct_waypoints": int(row[2] or 0) if row else 0,
            "fra_direct_pct": float(row[3] or 0.0) if row else 0.0,
        }

    delta_pct = round(
        results[period2]["fra_direct_pct"] - results[period1]["fra_direct_pct"],
        1,
    )
    return {
        "period1": results[period1],
        "period2": results[period2],
        "fra_shift_percentage_points": delta_pct,
    }
