from __future__ import annotations

from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset


def calculate_fir_hourly_saturation(
    dataset: FlightDataset,
    fir_id: str | None = None,
    date_str: str | None = None,
    threshold_flights_per_hr: int = 40,
    top_n: int = 15,
) -> dict[str, Any]:
    """
    Analyze en-route FIR sector saturation and identify European airspace bottlenecks.
    Quantifies peak hourly movements, hours exceeding capacity threshold, and sector dwell profiles.
    """
    if not dataset.has_fir_traversals():
        return {
            "period": dataset.period,
            "has_data": False,
            "message": f"Period {dataset.period} does not have explicit FIR sector traversal logs (flight_firs_actual). Available in monthly archive periods (e.g. 202510).",
            "top_saturated_firs": [],
            "hourly_timeline": [],
        }

    where_clauses = ["fir_id NOT LIKE '%TAXI%'"]
    params: list[Any] = []

    if date_str and date_str.strip():
        where_clauses.append("CAST(entry_time AS DATE) = CAST(? AS DATE)")
        params.append(date_str.strip())

    where_sql = " AND ".join(where_clauses)

    # 1. Top saturated FIRs across the period
    top_firs_sql = f"""
        WITH hourly_counts AS (
            SELECT
                fir_id,
                date_trunc('hour', entry_time) AS hour_ts,
                count(*) AS flights_in_hour,
                avg(dwell_min) AS avg_dwell
            FROM flight_firs_actual
            WHERE {where_sql}
            GROUP BY fir_id, date_trunc('hour', entry_time)
        )
        SELECT
            fir_id,
            sum(flights_in_hour) AS total_traversals,
            max(flights_in_hour) AS peak_hourly_flights,
            round(avg(flights_in_hour), 1) AS avg_hourly_flights,
            round(avg(avg_dwell), 1) AS avg_dwell_min,
            count(CASE WHEN flights_in_hour >= {int(threshold_flights_per_hr)} THEN 1 END) AS hours_over_capacity,
            round(count(CASE WHEN flights_in_hour >= {int(threshold_flights_per_hr)} THEN 1 END) * 100.0 / nullif(count(*), 0), 1) AS saturation_pct
        FROM hourly_counts
        GROUP BY fir_id
        ORDER BY hours_over_capacity DESC, peak_hourly_flights DESC
        LIMIT {int(top_n)};
    """
    top_df = dataset.cursor.execute(top_firs_sql, params).df()

    # 2. Hourly timeline profile (overall or for a specific FIR)
    timeline_params = list(params)
    fir_filter = ""
    if fir_id and fir_id.strip():
        fir_filter = " AND fir_id = ?"
        timeline_params.append(fir_id.strip().upper())

    timeline_sql = f"""
        SELECT
            strftime(date_trunc('hour', entry_time), '%Y-%m-%d %H:00') AS hour_str,
            count(*) AS flight_count,
            round(avg(dwell_min), 1) AS avg_dwell_min
        FROM flight_firs_actual
        WHERE {where_sql} {fir_filter}
        GROUP BY date_trunc('hour', entry_time)
        ORDER BY date_trunc('hour', entry_time) ASC
        LIMIT 168; -- Up to 7 continuous days (24*7)
    """
    timeline_df = dataset.cursor.execute(timeline_sql, timeline_params).df()

    top_firs = top_df.to_dict(orient="records") if not top_df.empty else []
    timeline = timeline_df.to_dict(orient="records") if not timeline_df.empty else []

    return {
        "period": dataset.period,
        "has_data": True,
        "threshold_flights_per_hr": threshold_flights_per_hr,
        "selected_fir": fir_id,
        "selected_date": date_str,
        "top_saturated_firs": top_firs,
        "hourly_timeline": timeline,
    }


def calculate_flight_fir_dwell_inefficiency(dataset: FlightDataset, ectrl_id: int) -> dict[str, Any]:
    """
    Assess en-route flight dwell efficiency across all traversed airspace sectors.
    Detects airborne holding, speed vectoring, or direct-to shortcuts.
    """
    if not dataset.has_fir_traversals():
        return {
            "ectrl_id": ectrl_id,
            "period": dataset.period,
            "has_data": False,
            "message": f"Period {dataset.period} does not have explicit FIR sector traversal logs.",
            "sectors": [],
        }

    df = dataset.get_flight_fir_traversals(ectrl_id)
    if df.empty:
        return {
            "ectrl_id": ectrl_id,
            "period": dataset.period,
            "has_data": False,
            "message": f"No FIR traversal logs found for flight {ectrl_id}.",
            "sectors": [],
        }

    sectors = []
    total_actual_dwell = 0.0
    total_filed_dwell = 0.0
    holding_sectors = []
    shortcut_sectors = []

    for _, row in df.iterrows():
        act_dwell = float(row["actual_dwell_min"]) if pd.notna(row["actual_dwell_min"]) else 0.0
        fld_dwell = float(row["filed_dwell_min"]) if pd.notna(row["filed_dwell_min"]) else None
        dwell_diff = float(row["dwell_diff_min"]) if pd.notna(row["dwell_diff_min"]) else None

        total_actual_dwell += act_dwell
        if fld_dwell is not None:
            total_filed_dwell += fld_dwell

        status = "ON_PROFILE"
        if dwell_diff is not None:
            if dwell_diff >= 5.0:
                status = "HOLDING/VECTORING"
                holding_sectors.append(f"{row['fir_id']} (+{dwell_diff:.1f}m)")
            elif dwell_diff >= 2.0:
                status = "SLOWDOWN"
            elif dwell_diff <= -3.0:
                status = "DIRECT_SHORTCUT"
                shortcut_sectors.append(f"{row['fir_id']} ({dwell_diff:.1f}m)")

        sectors.append({
            "seq_num": int(row["seq_num"]),
            "fir_id": str(row["fir_id"]),
            "actual_entry": str(row["actual_entry"]),
            "actual_exit": str(row["actual_exit"]),
            "actual_dwell_min": act_dwell,
            "filed_entry": str(row["filed_entry"]) if pd.notna(row["filed_entry"]) else None,
            "filed_exit": str(row["filed_exit"]) if pd.notna(row["filed_exit"]) else None,
            "filed_dwell_min": fld_dwell,
            "dwell_diff_min": dwell_diff,
            "status": status,
        })

    net_diff = total_actual_dwell - total_filed_dwell if total_filed_dwell > 0 else 0.0
    if holding_sectors:
        verdict = f"En-route tactical intervention detected: holding/vectoring in {', '.join(holding_sectors)}. Net sector dwell penalty: {net_diff:+.1f} min."
    elif shortcut_sectors:
        verdict = f"Direct routing advantageous: shortcuts in {', '.join(shortcut_sectors)}. Net sector dwell saving: {net_diff:+.1f} min."
    else:
        verdict = f"Nominal en-route progression: actual dwell matches filed plan within normal corridor variance ({net_diff:+.1f} min)."

    return {
        "ectrl_id": ectrl_id,
        "period": dataset.period,
        "has_data": True,
        "total_sectors_traversed": len(sectors),
        "total_actual_dwell_min": round(total_actual_dwell, 1),
        "total_filed_dwell_min": round(total_filed_dwell, 1),
        "net_dwell_diff_min": round(net_diff, 1),
        "holding_sectors": holding_sectors,
        "shortcut_sectors": shortcut_sectors,
        "verdict": verdict,
        "sectors": sectors,
    }

