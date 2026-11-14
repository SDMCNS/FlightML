"""
Multi-Level Key Performance Indicator (KPI) Hierarchical Analytics Suite.

Evaluates operational, environmental, and financial KPIs across four distinct aggregation tiers:
1. Flight Level: Micro 4D individual trajectory and flight diagnostics.
2. Fleet Level: Aggregated by Airline Operator (ac_operator) or Aircraft Type (ac_type).
3. Airport Level: Aggregated by Departure Aerodrome (ADEP) or Arrival Aerodrome (ADES).
4. Network Level: Airspace-wide macro totals, trends, and network efficiency benchmarks.

Supports temporal scoping across:
- Calendar date windows (start_date to end_date).
- Diurnal departure hours (hour_start to hour_end).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset


@dataclass
class KPIHierarchySummary:
    level: str
    group_by: str
    time_window: str
    hour_window: str
    total_flights: int
    avg_dep_delay_min: float
    avg_arr_delay_min: float
    pct_dep_delayed_15m: float
    pct_arr_delayed_15m: float
    pct_severe_arr_30m: float
    avg_actual_dist_nm: float
    avg_requested_fl: float
    total_detour_nm: float
    total_estimated_fuel_penalty_kg: float
    total_estimated_cost_eur: float
    records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_dataset_date_bounds(ds: FlightDataset) -> tuple[str, str]:
    """Retrieve the minimum and maximum calendar dates present in the active dataset period."""
    bounds = ds.query("""
        SELECT 
            MIN(CAST(COALESCE(actual_off_block_time, filed_off_block_time) AS DATE)) AS min_date,
            MAX(CAST(COALESCE(actual_off_block_time, filed_off_block_time) AS DATE)) AS max_date
        FROM flights
    """)
    if not bounds.empty and bounds.iloc[0]["min_date"] is not None:
        return str(bounds.iloc[0]["min_date"]), str(bounds.iloc[0]["max_date"])
    fallback_min = "2016-11-01" if ds.period == "201611" else "2025-10-01"
    fallback_max = "2016-11-30" if ds.period == "201611" else "2025-10-31"
    return fallback_min, fallback_max


def query_multi_level_kpis(
    ds: FlightDataset,
    level: str = "network",
    group_by: str = "operator",
    start_date: str | None = None,
    end_date: str | None = None,
    hour_start: int = 0,
    hour_end: int = 23,
    min_flights: int = 15,
    limit: int = 50,
) -> KPIHierarchySummary:
    """
    Execute optimized DuckDB aggregations across Network, Fleet, Airport, or Flight tiers,
    with flexible calendar date and diurnal hour filtering.
    """
    ds_min, ds_max = get_dataset_date_bounds(ds)
    effective_start = max(start_date, ds_min) if start_date else ds_min
    effective_end = min(end_date, ds_max) if end_date else ds_max
    if effective_start > effective_end:
        effective_start, effective_end = ds_min, ds_max

    h_min = max(0, min(23, hour_start))
    h_max = max(h_min, min(23, hour_end))

    # Base WHERE clauses
    time_col = "COALESCE(actual_off_block_time, filed_off_block_time)"
    where_parts = [
        f"CAST({time_col} AS DATE) >= '{effective_start}'",
        f"CAST({time_col} AS DATE) <= '{effective_end}'",
        f"EXTRACT(hour FROM {time_col}) BETWEEN {h_min} AND {h_max}",
    ]

    level_norm = level.lower().strip()
    grp_norm = group_by.lower().strip()

    if level_norm == "flight":
        where_sql = " AND ".join(where_parts)
        sql = f"""
            SELECT 
                ectrl_id,
                adep,
                ades,
                ac_operator,
                ac_type,
                CAST({time_col} AS TIMESTAMP) AS off_block_time,
                dep_delay_min,
                arr_delay_min,
                requested_fl,
                actual_distance_nm,
                actual_air_time_min,
                ROUND(GREATEST(0.0, actual_distance_nm - COALESCE(filed_air_time_min, actual_air_time_min) * 7.5), 1) AS detour_nm,
                ROUND(GREATEST(0.0, actual_distance_nm - COALESCE(filed_air_time_min, actual_air_time_min) * 7.5) * 5.4, 1) AS excess_fuel_kg,
                ROUND(GREATEST(0.0, actual_distance_nm - COALESCE(filed_air_time_min, actual_air_time_min) * 7.5) * 5.4 * 0.78, 2) AS excess_fuel_cost_eur
            FROM flights
            WHERE {where_sql}
            ORDER BY off_block_time DESC
            LIMIT {int(limit)};
        """
        df = ds.query(sql)
        recs = df.to_dict(orient="records") if not df.empty else []
        tot = len(df)
        avg_dep = round(float(df["dep_delay_min"].mean()), 1) if tot > 0 else 0.0
        avg_arr = round(float(df["arr_delay_min"].mean()), 1) if tot > 0 else 0.0
        pct_dep15 = round(float((df["dep_delay_min"] > 15).sum() * 100.0 / tot), 1) if tot > 0 else 0.0
        pct_arr15 = round(float((df["arr_delay_min"] > 15).sum() * 100.0 / tot), 1) if tot > 0 else 0.0
        pct_arr30 = round(float((df["arr_delay_min"] > 30).sum() * 100.0 / tot), 1) if tot > 0 else 0.0
        avg_dist = round(float(df["actual_distance_nm"].mean()), 1) if tot > 0 else 0.0
        avg_fl = round(float(df["requested_fl"].mean()), 0) if tot > 0 else 0.0
        tot_det = round(float(df["detour_nm"].sum()), 1) if tot > 0 else 0.0
        tot_fuel = round(float(df["excess_fuel_kg"].sum()), 1) if tot > 0 else 0.0
        tot_cost = round(float(df["excess_fuel_cost_eur"].sum()), 2) if tot > 0 else 0.0

        return KPIHierarchySummary(
            level="flight",
            group_by="flight_records",
            time_window=f"{effective_start} to {effective_end}",
            hour_window=f"{h_min:02d}:00 - {h_max:02d}:59 UTC",
            total_flights=tot,
            avg_dep_delay_min=avg_dep,
            avg_arr_delay_min=avg_arr,
            pct_dep_delayed_15m=pct_dep15,
            pct_arr_delayed_15m=pct_arr15,
            pct_severe_arr_30m=pct_arr30,
            avg_actual_dist_nm=avg_dist,
            avg_requested_fl=avg_fl,
            total_detour_nm=tot_det,
            total_estimated_fuel_penalty_kg=tot_fuel,
            total_estimated_cost_eur=tot_cost,
            records=recs,
        )

    # Fleet or Airport or Network
    if level_norm == "fleet":
        grp_col = "ac_operator" if grp_norm in ("operator", "airline", "carrier") else "ac_type"
        entity_name = "Airline Operator" if grp_col == "ac_operator" else "Aircraft Type"
        select_entity = f"{grp_col} AS entity_id"
        where_parts.append(f"{grp_col} IS NOT NULL")
        group_sql = f"GROUP BY {grp_col} HAVING COUNT(*) >= {int(min_flights)} ORDER BY total_flights DESC"
    elif level_norm == "airport":
        grp_col = "adep" if grp_norm in ("departure", "adep", "origin") else "ades"
        entity_name = "Departure Aerodrome (ADEP)" if grp_col == "adep" else "Arrival Aerodrome (ADES)"
        select_entity = f"{grp_col} AS entity_id"
        where_parts.append(f"{grp_col} IS NOT NULL")
        group_sql = f"GROUP BY {grp_col} HAVING COUNT(*) >= {int(min_flights)} ORDER BY total_flights DESC"
    else:
        # Network level
        entity_name = "Network Airspace System"
        select_entity = "'Network-Wide' AS entity_id"
        group_sql = ""

    where_sql = " AND ".join(where_parts)

    sql = f"""
        SELECT 
            {select_entity},
            COUNT(*) AS total_flights,
            ROUND(AVG(dep_delay_min), 1) AS avg_dep_delay_min,
            ROUND(AVG(arr_delay_min), 1) AS avg_arr_delay_min,
            ROUND(SUM(CASE WHEN dep_delay_min > 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_dep_delayed_15m,
            ROUND(SUM(CASE WHEN arr_delay_min > 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_arr_delayed_15m,
            ROUND(SUM(CASE WHEN arr_delay_min > 30 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_severe_arr_30m,
            ROUND(AVG(actual_distance_nm), 1) AS avg_actual_dist_nm,
            ROUND(AVG(requested_fl), 0) AS avg_requested_fl,
            ROUND(SUM(GREATEST(0.0, actual_distance_nm - COALESCE(filed_air_time_min, actual_air_time_min) * 7.5)), 0) AS total_detour_nm,
            ROUND(SUM(GREATEST(0.0, actual_distance_nm - COALESCE(filed_air_time_min, actual_air_time_min) * 7.5) * 5.4), 0) AS total_excess_fuel_kg,
            ROUND(SUM(GREATEST(0.0, actual_distance_nm - COALESCE(filed_air_time_min, actual_air_time_min) * 7.5) * 5.4 * 0.78), 2) AS total_excess_fuel_cost_eur
        FROM flights
        WHERE {where_sql}
        {group_sql}
        LIMIT {int(limit)};
    """

    df = ds.query(sql)
    recs = df.to_dict(orient="records") if not df.empty else []

    # High level rollups
    total_flts = int(df["total_flights"].sum()) if not df.empty else 0
    avg_dep = round(float(df["avg_dep_delay_min"].mean()), 1) if not df.empty else 0.0
    avg_arr = round(float(df["avg_arr_delay_min"].mean()), 1) if not df.empty else 0.0
    pct_dep15 = round(float(df["pct_dep_delayed_15m"].mean()), 1) if not df.empty else 0.0
    pct_arr15 = round(float(df["pct_arr_delayed_15m"].mean()), 1) if not df.empty else 0.0
    pct_arr30 = round(float(df["pct_severe_arr_30m"].mean()), 1) if not df.empty else 0.0
    avg_dist = round(float(df["avg_actual_dist_nm"].mean()), 1) if not df.empty else 0.0
    avg_fl = round(float(df["avg_requested_fl"].mean()), 0) if not df.empty else 0.0
    tot_detour = round(float(df["total_detour_nm"].sum()), 0) if not df.empty else 0.0
    tot_fuel = round(float(df["total_excess_fuel_kg"].sum()), 0) if not df.empty else 0.0
    tot_cost = round(float(df["total_excess_fuel_cost_eur"].sum()), 2) if not df.empty else 0.0

    return KPIHierarchySummary(
        level=level_norm,
        group_by=entity_name,
        time_window=f"{effective_start} to {effective_end}",
        hour_window=f"{h_min:02d}:00 - {h_max:02d}:59 UTC",
        total_flights=total_flts,
        avg_dep_delay_min=avg_dep,
        avg_arr_delay_min=avg_arr,
        pct_dep_delayed_15m=pct_dep15,
        pct_arr_delayed_15m=pct_arr15,
        pct_severe_arr_30m=pct_arr30,
        avg_actual_dist_nm=avg_dist,
        avg_requested_fl=avg_fl,
        total_detour_nm=tot_detour,
        total_estimated_fuel_penalty_kg=tot_fuel,
        total_estimated_cost_eur=tot_cost,
        records=recs,
    )
