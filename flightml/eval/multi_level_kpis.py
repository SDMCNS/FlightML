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
class FIRTrafficSegmentSummary:
    fir_id: str
    country_name: str
    unit_rate_eur: float
    time_window: str
    hour_window: str
    total_flights: int
    overflights_count: int
    arrivals_count: int
    departures_count: int
    internal_count: int
    overflights_pct: float
    arrivals_pct: float
    departures_pct: float
    internal_pct: float
    categories: list[dict[str, Any]]
    top_corridors: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def get_available_firs(ds: FlightDataset) -> list[dict[str, Any]]:
    """Retrieve list of significant European Flight Information Regions (FIRs) available in dataset."""
    firs: list[dict[str, Any]] = []
    if ds.has_fir_traversals():
        try:
            df = ds.query("""
                SELECT
                    CASE WHEN a.fir_id LIKE '%UIR' THEN REPLACE(a.fir_id, 'UIR', 'FIR') ELSE a.fir_id END AS fir_id,
                    COUNT(*) AS movements
                FROM flight_firs_actual a
                WHERE a.fir_id NOT LIKE '%TAXI%'
                GROUP BY fir_id
                ORDER BY movements DESC
                LIMIT 50;
            """)
            for _, r in df.iterrows():
                fid = str(r["fir_id"])
                rate_info = ds.get_crco_unit_rate(fid)
                c_name = str(rate_info.get("country_name") or rate_info.get("zone_name") or fid)
                u_eur = float(rate_info.get("unit_rate_eur") or 45.0)
                firs.append({
                    "fir_id": fid,
                    "country_name": c_name,
                    "unit_rate_eur": u_eur,
                    "movements": int(r["movements"]),
                    "label": f"{fid} — {c_name} (€{u_eur:.2f}/SU)",
                })
        except Exception:
            pass

    if not firs:
        # Fallback list of major Eurocontrol FIRs
        default_firs = [
            ("EGTTFIR", "United Kingdom (London)", 87.75),
            ("LFFFFIR", "France (Paris)", 79.58),
            ("EDGGFIR", "Germany (Langen)", 97.89),
            ("EDMMFIR", "Germany (Munich)", 97.89),
            ("LECMFIR", "Spain (Madrid)", 71.30),
            ("LECBFIR", "Spain (Barcelona)", 71.30),
            ("LIRRFIR", "Italy (Rome)", 73.71),
            ("LIMMFIR", "Italy (Milan)", 73.71),
            ("LOVVFIR", "Austria (Vienna)", 66.02),
            ("LSASFIR", "Switzerland (Zurich)", 168.95),
            ("EHAAFIR", "Netherlands (Amsterdam)", 136.38),
            ("EBBUFIR", "Belgium (Brussels)", 116.65),
            ("EISNFIR", "Ireland (Shannon)", 34.70),
            ("EKDKFIR", "Denmark (Copenhagen)", 91.68),
            ("ENORFIR", "Norway (Polaris)", 56.98),
            ("ESAAFIR", "Sweden (Sweden)", 90.34),
            ("EPWWFIR", "Poland (Warsaw)", 99.24),
            ("LKAAFIR", "Czech Republic (Prague)", 79.30),
            ("LHCCFIR", "Hungary (Budapest)", 39.36),
            ("LTBBFIR", "Türkiye (Istanbul)", 39.78),
            ("LGGGFIR", "Greece (Athens)", 22.39),
            ("LDZOFIR", "Croatia (Zagreb)", 39.19),
            ("LBSRFIR", "Bulgaria (Sofia)", 31.97),
            ("LRBBFIR", "Romania (Bucharest)", 50.25),
        ]
        for fid, c_name, u_eur in default_firs:
            rate_info = ds.get_crco_unit_rate(fid)
            rate_eur = float(rate_info.get("unit_rate_eur") or u_eur)
            firs.append({
                "fir_id": fid,
                "country_name": c_name,
                "unit_rate_eur": rate_eur,
                "movements": 0,
                "label": f"{fid} — {c_name} (€{rate_eur:.2f}/SU)",
            })

    return firs


def query_fir_traffic_segmentation(
    ds: FlightDataset,
    fir_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    hour_start: int = 0,
    hour_end: int = 23,
) -> FIRTrafficSegmentSummary:
    """
    Perform deep traffic segmentation for an FIR airspace:
    breaks out traffic volumes, punctuality, dwell times, and top city pairs
    into four distinct categories:
    1. Over-flights (flights crossing the FIR neither departing nor landing inside)
    2. Airport Arrivals (flights entering the FIR landing at an aerodrome inside)
    3. Airport Departures (flights departing from an aerodrome inside and climbing out)
    4. Internal / Domestic (flights both departing and arriving inside the FIR)
    """
    clean_fir = fir_id.strip().upper()
    rate_info = ds.get_crco_unit_rate(clean_fir)
    c_name = str(rate_info.get("country_name") or rate_info.get("zone_name") or clean_fir)
    u_eur = float(rate_info.get("unit_rate_eur") or 45.0)

    ds_min, ds_max = get_dataset_date_bounds(ds)
    eff_start = max(start_date, ds_min) if start_date else ds_min
    eff_end = min(end_date, ds_max) if end_date else ds_max
    if eff_start > eff_end:
        eff_start, eff_end = ds_min, ds_max

    h_min = max(0, min(23, hour_start))
    h_max = max(h_min, min(23, hour_end))

    time_col = "COALESCE(f.actual_off_block_time, f.filed_off_block_time)"
    date_filter = (
        f"CAST({time_col} AS DATE) >= '{eff_start}' AND "
        f"CAST({time_col} AS DATE) <= '{eff_end}' AND "
        f"EXTRACT(hour FROM {time_col}) BETWEEN {h_min} AND {h_max}"
    )

    pfx = clean_fir[:2] + "%"
    uir_id = clean_fir.replace("FIR", "UIR") if clean_fir.endswith("FIR") else clean_fir

    top_corridors: dict[str, list[dict[str, Any]]] = {
        "Overflight": [],
        "Arrival": [],
        "Departure": [],
        "Internal": [],
    }

    if ds.has_fir_traversals():
        sql = f"""
            WITH fir_flts AS (
                SELECT 
                    a.ectrl_id,
                    ROUND(AVG(epoch(a.exit_time - a.entry_time) / 60.0), 1) AS dwell_min
                FROM flight_firs_actual a
                WHERE a.fir_id IN ('{clean_fir}', '{uir_id}')
                GROUP BY a.ectrl_id
            )
            SELECT 
                CASE 
                    WHEN f.adep LIKE '{pfx}' AND f.ades LIKE '{pfx}' THEN 'Internal'
                    WHEN f.adep LIKE '{pfx}' AND f.ades NOT LIKE '{pfx}' THEN 'Departure'
                    WHEN f.adep NOT LIKE '{pfx}' AND f.ades LIKE '{pfx}' THEN 'Arrival'
                    ELSE 'Overflight'
                END AS traffic_type,
                COUNT(*) AS flight_count,
                ROUND(AVG(ff.dwell_min), 1) AS avg_dwell_min,
                ROUND(AVG(f.dep_delay_min), 1) AS avg_dep_delay_min,
                ROUND(AVG(f.arr_delay_min), 1) AS avg_arr_delay_min,
                ROUND(SUM(CASE WHEN f.dep_delay_min > 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_dep_delayed_15m,
                ROUND(SUM(CASE WHEN f.arr_delay_min > 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_arr_delayed_15m,
                ROUND(SUM(CASE WHEN f.arr_delay_min > 30 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_severe_arr_30m,
                ROUND(AVG(f.requested_fl), 0) AS avg_requested_fl
            FROM fir_flts ff
            JOIN flights f ON ff.ectrl_id = f.ectrl_id
            WHERE {date_filter}
            GROUP BY traffic_type
            ORDER BY flight_count DESC;
        """
        df = ds.query(sql)

        # Top city pair corridors for each category
        for cat, cond in [
            ("Overflight", f"f.adep NOT LIKE '{pfx}' AND f.ades NOT LIKE '{pfx}'"),
            ("Arrival", f"f.adep NOT LIKE '{pfx}' AND f.ades LIKE '{pfx}'"),
            ("Departure", f"f.adep LIKE '{pfx}' AND f.ades NOT LIKE '{pfx}'"),
            ("Internal", f"f.adep LIKE '{pfx}' AND f.ades LIKE '{pfx}'"),
        ]:
            c_sql = f"""
                WITH fir_flts AS (
                    SELECT a.ectrl_id
                    FROM flight_firs_actual a
                    WHERE a.fir_id IN ('{clean_fir}', '{uir_id}')
                    GROUP BY a.ectrl_id
                )
                SELECT f.adep || ' ➔ ' || f.ades AS city_pair, count(*) AS flt_cnt
                FROM fir_flts ff
                JOIN flights f ON ff.ectrl_id = f.ectrl_id
                WHERE {date_filter} AND {cond}
                GROUP BY city_pair
                ORDER BY flt_cnt DESC
                LIMIT 5;
            """
            c_df = ds.query(c_sql)
            if not c_df.empty:
                top_corridors[cat] = [
                    {"pair": str(r["city_pair"]), "count": int(r["flt_cnt"])}
                    for _, r in c_df.iterrows()
                ]

    else:
        # Fallback for periods without explicit sector traversal logs (e.g. 201611)
        sql = f"""
            SELECT 
                CASE 
                    WHEN f.adep LIKE '{pfx}' AND f.ades LIKE '{pfx}' THEN 'Internal'
                    WHEN f.adep LIKE '{pfx}' AND f.ades NOT LIKE '{pfx}' THEN 'Departure'
                    WHEN f.adep NOT LIKE '{pfx}' AND f.ades LIKE '{pfx}' THEN 'Arrival'
                    ELSE 'Overflight'
                END AS traffic_type,
                COUNT(*) AS flight_count,
                ROUND(AVG(f.actual_air_time_min * 0.35), 1) AS avg_dwell_min,
                ROUND(AVG(f.dep_delay_min), 1) AS avg_dep_delay_min,
                ROUND(AVG(f.arr_delay_min), 1) AS avg_arr_delay_min,
                ROUND(SUM(CASE WHEN f.dep_delay_min > 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_dep_delayed_15m,
                ROUND(SUM(CASE WHEN f.arr_delay_min > 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_arr_delayed_15m,
                ROUND(SUM(CASE WHEN f.arr_delay_min > 30 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_severe_arr_30m,
                ROUND(AVG(f.requested_fl), 0) AS avg_requested_fl
            FROM flights f
            WHERE {date_filter} AND (f.adep LIKE '{pfx}' OR f.ades LIKE '{pfx}')
            GROUP BY traffic_type
            ORDER BY flight_count DESC;
        """
        df = ds.query(sql)

        for cat, cond in [
            ("Arrival", f"f.adep NOT LIKE '{pfx}' AND f.ades LIKE '{pfx}'"),
            ("Departure", f"f.adep LIKE '{pfx}' AND f.ades NOT LIKE '{pfx}'"),
            ("Internal", f"f.adep LIKE '{pfx}' AND f.ades LIKE '{pfx}'"),
        ]:
            c_sql = f"""
                SELECT f.adep || ' ➔ ' || f.ades AS city_pair, count(*) AS flt_cnt
                FROM flights f
                WHERE {date_filter} AND {cond}
                GROUP BY city_pair
                ORDER BY flt_cnt DESC
                LIMIT 5;
            """
            c_df = ds.query(c_sql)
            if not c_df.empty:
                top_corridors[cat] = [
                    {"pair": str(r["city_pair"]), "count": int(r["flt_cnt"])}
                    for _, r in c_df.iterrows()
                ]

    tot_flts = int(df["flight_count"].sum()) if not df.empty else 0
    if not df.empty and tot_flts > 0:
        df["share_pct"] = (df["flight_count"] * 100.0 / tot_flts).round(1)
    else:
        df["share_pct"] = 0.0

    cats = df.to_dict(orient="records") if not df.empty else []

    by_type: dict[str, dict[str, Any]] = {r["traffic_type"]: r for r in cats}
    ov_c = int(by_type.get("Overflight", {}).get("flight_count", 0))
    ar_c = int(by_type.get("Arrival", {}).get("flight_count", 0))
    dp_c = int(by_type.get("Departure", {}).get("flight_count", 0))
    it_c = int(by_type.get("Internal", {}).get("flight_count", 0))

    ov_p = round(ov_c * 100.0 / max(1, tot_flts), 1)
    ar_p = round(ar_c * 100.0 / max(1, tot_flts), 1)
    dp_p = round(dp_c * 100.0 / max(1, tot_flts), 1)
    it_p = round(it_c * 100.0 / max(1, tot_flts), 1)

    return FIRTrafficSegmentSummary(
        fir_id=clean_fir,
        country_name=c_name,
        unit_rate_eur=u_eur,
        time_window=f"{eff_start} to {eff_end}",
        hour_window=f"{h_min:02d}:00 - {h_max:02d}:59 UTC",
        total_flights=tot_flts,
        overflights_count=ov_c,
        arrivals_count=ar_c,
        departures_count=dp_c,
        internal_count=it_c,
        overflights_pct=ov_p,
        arrivals_pct=ar_p,
        departures_pct=dp_p,
        internal_pct=it_p,
        categories=cats,
        top_corridors=top_corridors,
    )


def query_multi_level_kpis(
    ds: FlightDataset,
    level: str = "network",
    group_by: str = "operator",
    fir_filter: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    hour_start: int = 0,
    hour_end: int = 23,
    min_flights: int = 15,
    limit: int = 50,
) -> KPIHierarchySummary:
    """
    Execute optimized DuckDB aggregations across Network, Fleet, Airport, FIR, or Flight tiers,
    with flexible calendar date, diurnal hour, and FIR airspace filtering.
    """
    ds_min, ds_max = get_dataset_date_bounds(ds)
    effective_start = max(start_date, ds_min) if start_date else ds_min
    effective_end = min(end_date, ds_max) if end_date else ds_max
    if effective_start > effective_end:
        effective_start, effective_end = ds_min, ds_max

    h_min = max(0, min(23, hour_start))
    h_max = max(h_min, min(23, hour_end))

    time_col = "COALESCE(f.actual_off_block_time, f.filed_off_block_time)"
    where_parts = [
        f"CAST({time_col} AS DATE) >= '{effective_start}'",
        f"CAST({time_col} AS DATE) <= '{effective_end}'",
        f"EXTRACT(hour FROM {time_col}) BETWEEN {h_min} AND {h_max}",
    ]

    # FIR scoping filter across all tiers
    if fir_filter and fir_filter.strip() and fir_filter.upper() != "ALL":
        clean_fir = fir_filter.strip().upper()
        if ds.has_fir_traversals():
            clean_uir = clean_fir.replace("FIR", "UIR")
            where_parts.append(
                f"f.ectrl_id IN (SELECT ectrl_id FROM flight_firs_actual WHERE fir_id IN ('{clean_fir}', '{clean_uir}'))"
            )
        else:
            pfx = clean_fir[:2]
            where_parts.append(f"(f.adep LIKE '{pfx}%' OR f.ades LIKE '{pfx}%')")

    level_norm = level.lower().strip()
    grp_norm = group_by.lower().strip()

    if level_norm == "flight":
        where_sql = " AND ".join(where_parts)
        sql = f"""
            SELECT 
                f.ectrl_id,
                f.adep,
                f.ades,
                f.ac_operator,
                f.ac_type,
                CAST({time_col} AS TIMESTAMP) AS off_block_time,
                f.dep_delay_min,
                f.arr_delay_min,
                f.requested_fl,
                f.actual_distance_nm,
                f.actual_air_time_min,
                ROUND(GREATEST(0.0, f.actual_distance_nm - COALESCE(f.filed_air_time_min, f.actual_air_time_min) * 7.5), 1) AS detour_nm,
                ROUND(GREATEST(0.0, f.actual_distance_nm - COALESCE(f.filed_air_time_min, f.actual_air_time_min) * 7.5) * 5.4, 1) AS excess_fuel_kg,
                ROUND(GREATEST(0.0, f.actual_distance_nm - COALESCE(f.filed_air_time_min, f.actual_air_time_min) * 7.5) * 5.4 * 0.78, 2) AS excess_fuel_cost_eur
            FROM flights f
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

    # Fleet, Airport, FIR, or Network
    from_clause = "flights f"
    if level_norm == "fleet":
        grp_col = "f.ac_operator" if grp_norm in ("operator", "airline", "carrier") else "f.ac_type"
        entity_name = "Airline Operator" if "operator" in grp_col else "Aircraft Type"
        select_entity = f"{grp_col} AS entity_id"
        where_parts.append(f"{grp_col} IS NOT NULL")
        group_sql = f"GROUP BY {grp_col} HAVING COUNT(*) >= {int(min_flights)} ORDER BY total_flights DESC"
    elif level_norm == "airport":
        grp_col = "f.adep" if grp_norm in ("departure", "adep", "origin") else "f.ades"
        entity_name = "Departure Aerodrome (ADEP)" if "adep" in grp_col else "Arrival Aerodrome (ADES)"
        select_entity = f"{grp_col} AS entity_id"
        where_parts.append(f"{grp_col} IS NOT NULL")
        group_sql = f"GROUP BY {grp_col} HAVING COUNT(*) >= {int(min_flights)} ORDER BY total_flights DESC"
    elif level_norm == "fir":
        entity_name = "Flight Information Region (FIR)"
        if ds.has_fir_traversals():
            select_entity = "CASE WHEN a.fir_id LIKE '%UIR' THEN REPLACE(a.fir_id, 'UIR', 'FIR') ELSE a.fir_id END AS entity_id"
            from_clause = "flights f JOIN flight_firs_actual a ON f.ectrl_id = a.ectrl_id"
            where_parts.append("a.fir_id NOT LIKE '%TAXI%'")
            group_sql = f"GROUP BY entity_id HAVING COUNT(DISTINCT f.ectrl_id) >= {int(min_flights)} ORDER BY total_flights DESC"
        else:
            select_entity = "SUBSTR(f.adep, 1, 2) || 'XXFIR' AS entity_id"
            group_sql = f"GROUP BY entity_id HAVING COUNT(*) >= {int(min_flights)} ORDER BY total_flights DESC"
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
            ROUND(AVG(f.dep_delay_min), 1) AS avg_dep_delay_min,
            ROUND(AVG(f.arr_delay_min), 1) AS avg_arr_delay_min,
            ROUND(SUM(CASE WHEN f.dep_delay_min > 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_dep_delayed_15m,
            ROUND(SUM(CASE WHEN f.arr_delay_min > 15 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_arr_delayed_15m,
            ROUND(SUM(CASE WHEN f.arr_delay_min > 30 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_severe_arr_30m,
            ROUND(AVG(f.actual_distance_nm), 1) AS avg_actual_dist_nm,
            ROUND(AVG(f.requested_fl), 0) AS avg_requested_fl,
            ROUND(SUM(GREATEST(0.0, f.actual_distance_nm - COALESCE(f.filed_air_time_min, f.actual_air_time_min) * 7.5)), 0) AS total_detour_nm,
            ROUND(SUM(GREATEST(0.0, f.actual_distance_nm - COALESCE(f.filed_air_time_min, f.actual_air_time_min) * 7.5) * 5.4), 0) AS total_excess_fuel_kg,
            ROUND(SUM(GREATEST(0.0, f.actual_distance_nm - COALESCE(f.filed_air_time_min, f.actual_air_time_min) * 7.5) * 5.4 * 0.78), 2) AS total_excess_fuel_cost_eur
        FROM {from_clause}
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

