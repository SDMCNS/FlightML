"""
ICAO GANP & Eurocontrol PRU Standard Key Performance Indicator (KPI) Suite (Sprint 2).

Implements standardized air navigation performance metrics:
- KPI04 (KEP): Horizontal En-Route Inefficiency of the Filed Trajectory (outside 40 nm cylinders).
- KPI05 (KEA): Horizontal En-Route Inefficiency of the Actual Trajectory (outside 40 nm cylinders).
- Tactical Shortcut / Vectoring Delta (KEA - KEP): Isolates flight-planning detours from tactical ATC shortcuts/vectors.
- KPI08 (ASMA Additional Time): Arrival Sequencing and Metering Area transit time inside 40 nm / 100 nm arrival cylinders
  vs. 20th-percentile unimpeded transit baseline.
- KPI19 (Cruise Level-Capping & Lower/Upper FIR/UIR Split): Compares actual max cruise Flight Level against
  Requested FL and quantifies fuel penalty from cruising below optimal altitude.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.geo.geodesy import haversine_nm, haversine_vectorized_nm


@dataclass
class PRUFlightScorecard:
    ectrl_id: int
    period: str
    adep: str
    ades: str
    ac_operator: str
    ac_type: str
    great_circle_od_nm: float
    # KPI04 (KEP) & KPI05 (KEA) - En-Route (>40 nm from ADEP/ADES)
    enroute_achieved_dist_nm: float
    enroute_filed_dist_nm: float
    enroute_actual_dist_nm: float
    kpi04_kep_pct: float
    kpi05_kea_pct: float
    tactical_kea_minus_kep_pct: float
    tactical_distance_delta_nm: float
    enroute_routing_verdict: str
    # KPI08 (ASMA - Arrival Sequencing and Metering Area)
    asma_40nm_actual_min: float
    asma_40nm_unimpeded_min: float
    kpi08_asma_40nm_additional_min: float
    asma_100nm_actual_min: float
    arrival_tma_id: str | None
    arrival_tma_dwell_min: float | None
    # KPI19 (Vertical Cruise Efficiency & Level-Capping)
    requested_fl: float
    max_actual_fl: float
    median_cruise_fl: float
    level_cap_delta_fl: float
    is_level_capped: bool
    lower_fir_dwell_min: float
    upper_uir_dwell_min: float
    level_cap_excess_fuel_kg: float
    level_cap_excess_co2_kg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_clipped_enroute_distance(
    traj_df: pd.DataFrame,
    adep_lat: float,
    adep_lon: float,
    ades_lat: float,
    ades_lon: float,
    cylinder_nm: float = 40.0,
) -> tuple[float, float]:
    """
    Clip a trajectory to points outside `cylinder_nm` (40 nm) of ADEP and ADES.
    Returns (flown_enroute_distance_nm, achieved_enroute_distance_nm).
    """
    if len(traj_df) < 2:
        return 0.0, 0.0

    df = traj_df.sort_values("seq_num").reset_index(drop=True)
    lats = df["latitude"].to_numpy(dtype=float)
    lons = df["longitude"].to_numpy(dtype=float)

    d_adep = haversine_vectorized_nm(adep_lat, adep_lon, lats, lons)
    d_ades = haversine_vectorized_nm(ades_lat, ades_lon, lats, lons)

    mask = (d_adep >= cylinder_nm) & (d_ades >= cylinder_nm)
    enroute_pts = df[mask].reset_index(drop=True)

    if len(enroute_pts) < 2:
        # Short-haul flight or sparse points: use full trajectory minus terminal radii
        seg_dists = haversine_vectorized_nm(lats[:-1], lons[:-1], lats[1:], lons[1:])
        total_flown = float(np.sum(seg_dists))
        od_gc = haversine_nm(adep_lat, adep_lon, ades_lat, ades_lon)
        enroute_flown = max(1.0, total_flown - 2.0 * min(cylinder_nm, od_gc * 0.25))
        enroute_achieved = max(1.0, od_gc - 2.0 * min(cylinder_nm, od_gc * 0.25))
        return enroute_flown, min(enroute_flown, enroute_achieved)

    e_lats = enroute_pts["latitude"].to_numpy(dtype=float)
    e_lons = enroute_pts["longitude"].to_numpy(dtype=float)
    seg_dists = haversine_vectorized_nm(e_lats[:-1], e_lons[:-1], e_lats[1:], e_lons[1:])
    flown_enroute = float(np.sum(seg_dists))

    # Eurocontrol PRU Achieved Distance between entry point X (leaving 40nm ADEP) and exit point Y (entering 40nm ADES)
    entry_lat, entry_lon = float(e_lats[0]), float(e_lons[0])
    exit_lat, exit_lon = float(e_lats[-1]), float(e_lons[-1])
    achieved_enroute = haversine_nm(entry_lat, entry_lon, exit_lat, exit_lon)
    achieved_enroute = max(1.0, min(flown_enroute, achieved_enroute))
    return flown_enroute, achieved_enroute


def _compute_asma_transit_minutes(
    actual_df: pd.DataFrame,
    ades_lat: float,
    ades_lon: float,
    radius_nm: float = 40.0,
) -> float:
    """
    Compute actual transit duration (minutes) from first entry into the `radius_nm`
    arrival cylinder around ADES until touchdown (last point).
    """
    if len(actual_df) < 2:
        return 0.0

    df = actual_df.sort_values("seq_num").reset_index(drop=True)
    lats = df["latitude"].to_numpy(dtype=float)
    lons = df["longitude"].to_numpy(dtype=float)
    dists = haversine_vectorized_nm(ades_lat, ades_lon, lats, lons)

    inside_indices = np.where(dists <= radius_nm)[0]
    if len(inside_indices) == 0:
        return 0.0

    # Find entry into arrival cylinder in the second half of the flight (avoid origin if ADEP == ADES)
    half_idx = max(0, len(df) // 3)
    arr_indices = [idx for idx in inside_indices if idx >= half_idx]
    first_idx = arr_indices[0] if arr_indices else int(inside_indices[0])

    t_entry = pd.to_datetime(df.iloc[first_idx]["time_over"])
    t_land = pd.to_datetime(df.iloc[-1]["time_over"])
    dur_min = max(0.0, (t_land - t_entry).total_seconds() / 60.0)

    # Interpolate fractional segment if previous point was outside cylinder
    if first_idx > 0 and dists[first_idx - 1] > radius_nm:
        d_prev = float(dists[first_idx - 1])
        d_curr = float(dists[first_idx])
        if d_prev > d_curr:
            frac = (radius_nm - d_curr) / (d_prev - d_curr)
            t_prev = pd.to_datetime(df.iloc[first_idx - 1]["time_over"])
            seg_min = max(0.0, (t_entry - t_prev).total_seconds() / 60.0)
            dur_min += frac * seg_min

    return round(dur_min, 2)


def compute_flight_pru_kpis(dataset: FlightDataset, ectrl_id: int) -> PRUFlightScorecard | None:
    """
    Compute full ICAO GANP / Eurocontrol PRU Scorecard for a single flight:
    - KPI04 (KEP): Horizontal En-Route Inefficiency of Filed Trajectory (%)
    - KPI05 (KEA): Horizontal En-Route Inefficiency of Actual Trajectory (%)
    - KPI08 (ASMA): Arrival Sequencing & Metering Area Additional Time (min) inside 40 nm & 100 nm
    - KPI19 (Cruise Level-Capping): Requested FL vs. Actual Cruise FL + lower/upper FIR/UIR dwell split
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return None

    trajs = dataset.get_trajectory(ectrl_id)
    actual_df = trajs["actual"]
    planned_df = trajs["planned"]
    if actual_df.empty:
        return None

    adep = str(flight.get("adep") or "")
    ades = str(flight.get("ades") or "")
    ac_op = str(flight.get("ac_operator") or "")
    ac_type = str(flight.get("ac_type") or "B738")
    adep_lat = float(flight.get("adep_lat") or actual_df.iloc[0]["latitude"])
    adep_lon = float(flight.get("adep_lon") or actual_df.iloc[0]["longitude"])
    ades_lat = float(flight.get("ades_lat") or actual_df.iloc[-1]["latitude"])
    ades_lon = float(flight.get("ades_lon") or actual_df.iloc[-1]["longitude"])

    gc_od = haversine_nm(adep_lat, adep_lon, ades_lat, ades_lon)

    # 1. KPI04 (KEP) & KPI05 (KEA) outside 40 nm cylinders
    act_enroute_nm, act_achieved_nm = _compute_clipped_enroute_distance(
        actual_df, adep_lat, adep_lon, ades_lat, ades_lon, cylinder_nm=40.0
    )
    if not planned_df.empty and len(planned_df) >= 2:
        fld_enroute_nm, fld_achieved_nm = _compute_clipped_enroute_distance(
            planned_df, adep_lat, adep_lon, ades_lat, ades_lon, cylinder_nm=40.0
        )
    else:
        fld_enroute_nm, fld_achieved_nm = act_enroute_nm, act_achieved_nm

    ref_achieved = max(1.0, (act_achieved_nm + fld_achieved_nm) / 2.0)
    kep_pct = max(0.0, ((fld_enroute_nm - ref_achieved) / ref_achieved) * 100.0)
    kea_pct = max(0.0, ((act_enroute_nm - ref_achieved) / ref_achieved) * 100.0)
    tactical_pct_delta = kea_pct - kep_pct
    tactical_nm_delta = act_enroute_nm - fld_enroute_nm

    if tactical_nm_delta <= -5.0:
        verdict = "ATC_DIRECT_SHORTCUT"
    elif tactical_nm_delta >= 10.0:
        verdict = "TACTICAL_VECTORING_DETOUR"
    elif kep_pct >= 6.0:
        verdict = "SUBOPTIMAL_FILED_ROUTE"
    else:
        verdict = "HIGH_EFFICIENCY_ENROUTE"

    # 2. KPI08 (ASMA 40nm & 100nm)
    asma_40_actual = _compute_asma_transit_minutes(actual_df, ades_lat, ades_lon, radius_nm=40.0)
    asma_100_actual = _compute_asma_transit_minutes(actual_df, ades_lat, ades_lon, radius_nm=100.0)

    # Check TMA AUA dwell if available
    arr_tma_id = None
    arr_tma_dwell = None
    if dataset.has_aua_traversals():
        aua_df = dataset.get_flight_aua_traversals(ectrl_id)
        tma_rows = aua_df[aua_df["aua_type"] == "TMA"]
        if not tma_rows.empty:
            last_tma = tma_rows.iloc[-1]
            arr_tma_id = str(last_tma["aua_id"])
            arr_tma_dwell = round(float(last_tma["actual_dwell_min"] or 0.0), 1)
            if asma_40_actual <= 0.5 and arr_tma_dwell > 0:
                asma_40_actual = arr_tma_dwell

    # Unimpeded 40nm ASMA baseline (~9.5 min for 40nm at 250 kts IAS + approach deceleration, or 20th-percentile TMA dwell)
    unimpeded_asma_40 = 9.5
    if dataset.has_aua_traversals() and arr_tma_id:
        u_row = dataset.cursor.execute(
            """
            SELECT quantile_cont(epoch(exit_time - entry_time) / 60.0, 0.20)
            FROM flight_auas_actual
            WHERE aua_id = ? AND epoch(exit_time - entry_time) BETWEEN 120 AND 3600
            """,
            [arr_tma_id],
        ).fetchone()
        if u_row and u_row[0] is not None:
            unimpeded_asma_40 = max(6.0, min(14.0, float(u_row[0])))

    if asma_40_actual <= 0.1:
        asma_40_actual = unimpeded_asma_40
    asma_40_add = max(0.0, asma_40_actual - unimpeded_asma_40)

    # 3. KPI19 (Cruise Level-Capping & Lower/Upper FIR/UIR Split)
    req_fl = float(flight.get("requested_fl") or 0.0)
    max_act_fl = float(actual_df["flight_level"].max())
    high_pts = actual_df[actual_df["flight_level"] >= max(100.0, max_act_fl - 30.0)]
    median_cruise_fl = float(high_pts["flight_level"].median()) if not high_pts.empty else max_act_fl

    if req_fl <= 0:
        req_fl = float(planned_df["flight_level"].max()) if not planned_df.empty else max_act_fl

    level_cap_delta = max(0.0, req_fl - max_act_fl)
    is_capped = level_cap_delta >= 20.0  # Capped by >= 2,000 ft (20 FL)

    lower_fir_min = 0.0
    upper_uir_min = 0.0
    if dataset.has_fir_traversals():
        fir_df = dataset.get_flight_fir_traversals(ectrl_id)
        for _, f_row in fir_df.iterrows():
            fid = str(f_row["fir_id"])
            d_min = float(f_row["actual_dwell_min"] or 0.0)
            if "UIR" in fid or "UTA" in fid:
                upper_uir_min += d_min
            elif fid != "TAXI_OUT":
                lower_fir_min += d_min

    # Estimate fuel penalty from cruise level-capping (~1.5% extra cruise fuel per 2,000 ft below optimal FL)
    ac_perf = dataset.get_aircraft_performance(ac_type)
    wake = str(ac_perf.get("wake_category") or "M") if ac_perf else "M"
    base_cruise_ff_kg_hr = 6800.0 if wake == "H" else 11500.0 if wake == "J" else 1200.0 if wake == "L" else 2450.0
    cruise_hours = max(0.2, (act_enroute_nm / 440.0))
    penalty_fraction = (level_cap_delta / 20.0) * 0.015 if is_capped else 0.0
    cap_fuel_kg = base_cruise_ff_kg_hr * cruise_hours * penalty_fraction
    cap_co2_kg = cap_fuel_kg * 3.16

    return PRUFlightScorecard(
        ectrl_id=int(ectrl_id),
        period=dataset.period,
        adep=adep,
        ades=ades,
        ac_operator=ac_op,
        ac_type=ac_type,
        great_circle_od_nm=round(gc_od, 1),
        enroute_achieved_dist_nm=round(ref_achieved, 1),
        enroute_filed_dist_nm=round(fld_enroute_nm, 1),
        enroute_actual_dist_nm=round(act_enroute_nm, 1),
        kpi04_kep_pct=round(kep_pct, 2),
        kpi05_kea_pct=round(kea_pct, 2),
        tactical_kea_minus_kep_pct=round(tactical_pct_delta, 2),
        tactical_distance_delta_nm=round(tactical_nm_delta, 1),
        enroute_routing_verdict=verdict,
        asma_40nm_actual_min=round(asma_40_actual, 2),
        asma_40nm_unimpeded_min=round(unimpeded_asma_40, 2),
        kpi08_asma_40nm_additional_min=round(asma_40_add, 2),
        asma_100nm_actual_min=round(asma_100_actual, 2),
        arrival_tma_id=arr_tma_id,
        arrival_tma_dwell_min=arr_tma_dwell,
        requested_fl=round(req_fl, 0),
        max_actual_fl=round(max_act_fl, 0),
        median_cruise_fl=round(median_cruise_fl, 0),
        level_cap_delta_fl=round(level_cap_delta, 0),
        is_level_capped=bool(is_capped),
        lower_fir_dwell_min=round(lower_fir_min, 1),
        upper_uir_dwell_min=round(upper_uir_min, 1),
        level_cap_excess_fuel_kg=round(cap_fuel_kg, 1),
        level_cap_excess_co2_kg=round(cap_co2_kg, 1),
    )


def analyze_network_level_capping(
    dataset: FlightDataset,
    operator: str | None = None,
    min_distance_nm: float = 250.0,
    top_n: int = 15,
) -> dict[str, Any]:
    """
    Scan the network for KPI19 Cruise Flight-Level Capping (flights whose actual peak Flight Level
    was restricted >= 2,000 ft [20 FL] below their Requested FL).
    """
    where_clauses = [
        "f.requested_fl >= 280",
        "f.actual_distance_nm >= ?",
    ]
    params: list[Any] = [float(min_distance_nm)]

    if operator and operator.strip():
        where_clauses.append("f.ac_operator = ?")
        params.append(operator.strip().upper())

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        WITH flight_peaks AS (
            SELECT
                f.ectrl_id,
                f.adep,
                f.ades,
                f.ac_operator,
                f.ac_type,
                f.requested_fl,
                f.actual_distance_nm,
                max(t.flight_level) AS max_actual_fl
            FROM flights f
            JOIN trajectory_actual t ON f.ectrl_id = t.ectrl_id
            WHERE {where_sql}
            GROUP BY
                f.ectrl_id, f.adep, f.ades, f.ac_operator, f.ac_type,
                f.requested_fl, f.actual_distance_nm
        ),
        corridor_stats AS (
            SELECT
                adep || ' ➔ ' || ades AS corridor,
                adep,
                ades,
                CAST(count(*) AS BIGINT) AS total_flights,
                CAST(count(CASE WHEN (requested_fl - max_actual_fl) >= 20 THEN 1 END) AS BIGINT) AS capped_flights,
                round(
                    count(CASE WHEN (requested_fl - max_actual_fl) >= 20 THEN 1 END) * 100.0
                    / nullif(count(*), 0),
                    1
                ) AS level_capped_pct,
                round(avg(requested_fl), 0) AS avg_requested_fl,
                round(avg(max_actual_fl), 0) AS avg_actual_max_fl,
                round(avg(greatest(0, requested_fl - max_actual_fl)), 1) AS avg_fl_deficit
            FROM flight_peaks
            GROUP BY adep, ades
            HAVING count(*) >= 10
        )
        SELECT *
        FROM corridor_stats
        ORDER BY capped_flights DESC, level_capped_pct DESC
        LIMIT {int(top_n)};
    """
    corridors_df = dataset.cursor.execute(sql, params).df()

    summary_sql = f"""
        WITH flight_peaks AS (
            SELECT
                f.ectrl_id,
                f.requested_fl,
                max(t.flight_level) AS max_actual_fl
            FROM flights f
            JOIN trajectory_actual t ON f.ectrl_id = t.ectrl_id
            WHERE {where_sql}
            GROUP BY f.ectrl_id, f.requested_fl
        )
        SELECT
            CAST(count(*) AS BIGINT) AS analyzed_flights,
            CAST(count(CASE WHEN (requested_fl - max_actual_fl) >= 20 THEN 1 END) AS BIGINT) AS level_capped_flights,
            round(
                count(CASE WHEN (requested_fl - max_actual_fl) >= 20 THEN 1 END) * 100.0
                / nullif(count(*), 0),
                2
            ) AS level_capped_pct,
            round(avg(greatest(0, requested_fl - max_actual_fl)), 1) AS mean_fl_deficit
        FROM flight_peaks;
    """
    sum_row = dataset.cursor.execute(summary_sql, params).fetchone()

    return {
        "period": dataset.period,
        "operator_filter": operator,
        "min_distance_nm": min_distance_nm,
        "analyzed_flights": int(sum_row[0] or 0) if sum_row else 0,
        "level_capped_flights": int(sum_row[1] or 0) if sum_row else 0,
        "level_capped_pct": float(sum_row[2] or 0.0) if sum_row else 0.0,
        "mean_fl_deficit": float(sum_row[3] or 0.0) if sum_row else 0.0,
        "top_capped_corridors": corridors_df.to_dict(orient="records") if not corridors_df.empty else [],
    }
