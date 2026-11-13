"""
ATC Unit Airspace (AUA: TMA, CTA, UTA) Saturation & Dwell Inefficiency Engine (Sprint 1).

Unlike geopolitical Flight Information Regions (FIRs), ATC Unit Airspaces (AUAs) represent
the operational Air Traffic Control centers, Approach units (TMAs), Control Areas (CTAs),
and Upper Control Areas (UTAs) that tactically manage aircraft separation.
"""

from __future__ import annotations

from typing import Any

from flightml.data.dataset import FlightDataset


def calculate_aua_hourly_saturation(
    dataset: FlightDataset,
    aua_type: str | None = None,
    aua_id: str | None = None,
    date_str: str | None = None,
    threshold_flights_per_hr: int = 30,
    top_n: int = 20,
) -> dict[str, Any]:
    """
    Analyze hourly traffic saturation across European ATC Unit Airspaces (TMA, CTA, UTA).
    Quantifies peak hourly movements, hours exceeding capacity threshold, and sector dwell profiles.
    """
    if not dataset.has_aua_traversals():
        return {
            "period": dataset.period,
            "has_data": False,
            "message": (
                f"Period {dataset.period} does not have explicit AUA traversal logs (flight_auas_actual). "
                "Available in monthly Eurocontrol archives (e.g. 202510)."
            ),
            "top_saturated_auas": [],
            "hourly_timeline": [],
        }

    where_clauses = ["aua_id NOT LIKE '%TAXI%'"]
    params: list[Any] = []

    if aua_type and aua_type.strip().upper() in ("TMA", "CTA", "UTA", "OTHER"):
        where_clauses.append("aua_type = ?")
        params.append(aua_type.strip().upper())

    if date_str and date_str.strip():
        where_clauses.append("CAST(entry_time AS DATE) = CAST(? AS DATE)")
        params.append(date_str.strip())

    where_sql = " AND ".join(where_clauses)

    top_auas_sql = f"""
        WITH hourly_counts AS (
            SELECT
                aua_id,
                aua_type,
                date_trunc('hour', entry_time) AS hour_ts,
                count(*) AS flights_in_hour,
                avg(epoch(exit_time - entry_time) / 60.0) AS avg_dwell
            FROM flight_auas_actual
            WHERE {where_sql}
            GROUP BY aua_id, aua_type, date_trunc('hour', entry_time)
        )
        SELECT
            aua_id,
            aua_type,
            CAST(sum(flights_in_hour) AS BIGINT) AS total_traversals,
            CAST(max(flights_in_hour) AS BIGINT) AS peak_hourly_flights,
            round(avg(flights_in_hour), 1) AS avg_hourly_flights,
            round(avg(avg_dwell), 1) AS avg_dwell_min,
            CAST(count(CASE WHEN flights_in_hour >= {int(threshold_flights_per_hr)} THEN 1 END) AS BIGINT) AS hours_over_capacity,
            round(
                count(CASE WHEN flights_in_hour >= {int(threshold_flights_per_hr)} THEN 1 END) * 100.0
                / nullif(count(*), 0),
                1
            ) AS saturation_pct
        FROM hourly_counts
        GROUP BY aua_id, aua_type
        ORDER BY hours_over_capacity DESC, peak_hourly_flights DESC
        LIMIT {int(top_n)};
    """
    top_df = dataset.cursor.execute(top_auas_sql, params).df()

    timeline_params = list(params)
    aua_filter = ""
    if aua_id and aua_id.strip():
        aua_filter = " AND aua_id = ?"
        timeline_params.append(aua_id.strip().upper())

    timeline_sql = f"""
        SELECT
            strftime(date_trunc('hour', entry_time), '%Y-%m-%d %H:00') AS hour_str,
            CAST(count(*) AS BIGINT) AS flight_count,
            round(avg(epoch(exit_time - entry_time) / 60.0), 1) AS avg_dwell_min
        FROM flight_auas_actual
        WHERE {where_sql} {aua_filter}
        GROUP BY date_trunc('hour', entry_time)
        ORDER BY date_trunc('hour', entry_time) ASC
        LIMIT 168;
    """
    timeline_df = dataset.cursor.execute(timeline_sql, timeline_params).df()

    return {
        "period": dataset.period,
        "has_data": True,
        "aua_type_filter": aua_type,
        "selected_aua": aua_id,
        "selected_date": date_str,
        "threshold_flights_per_hr": threshold_flights_per_hr,
        "top_saturated_auas": top_df.to_dict(orient="records") if not top_df.empty else [],
        "hourly_timeline": timeline_df.to_dict(orient="records") if not timeline_df.empty else [],
    }


def calculate_flight_aua_dwell_inefficiency(dataset: FlightDataset, ectrl_id: int) -> dict[str, Any]:
    """
    Assess sector-by-sector ATC Unit Airspace (TMA, CTA, UTA) dwell efficiency for a single flight.
    Isolates terminal arrival metering (TMA) vs. en-route control area (CTA/UTA) holding or shortcuts.
    """
    if not dataset.has_aua_traversals():
        return {
            "ectrl_id": ectrl_id,
            "period": dataset.period,
            "has_data": False,
            "message": f"Period {dataset.period} does not have explicit AUA sector traversal logs.",
            "sectors": [],
        }

    df = dataset.get_flight_aua_traversals(ectrl_id)
    if df.empty:
        return {
            "ectrl_id": ectrl_id,
            "period": dataset.period,
            "has_data": False,
            "message": f"No AUA traversal logs found for flight {ectrl_id}.",
            "sectors": [],
        }

    flight_info = dataset.get_flight(ectrl_id)
    sectors: list[dict[str, Any]] = []
    total_actual_min = 0.0
    total_filed_min = 0.0
    tma_dwell_delta_min = 0.0
    enroute_dwell_delta_min = 0.0
    worst_sector = None
    max_excess_min = 0.0

    for _, row in df.iterrows():
        act_d = float(row["actual_dwell_min"] or 0.0)
        fld_d = float(row["filed_dwell_min"]) if pd.notna(row["filed_dwell_min"]) else None
        diff_d = float(row["dwell_diff_min"]) if pd.notna(row["dwell_diff_min"]) else None
        atype = str(row["aua_type"])

        total_actual_min += act_d
        if fld_d is not None:
            total_filed_min += fld_d

        if diff_d is not None:
            if atype == "TMA":
                tma_dwell_delta_min += diff_d
            else:
                enroute_dwell_delta_min += diff_d

        status = "ON_PLAN"
        if diff_d is not None:
            if diff_d >= 5.0:
                status = "HOLDING_OR_VECTORING"
                if diff_d > max_excess_min:
                    max_excess_min = diff_d
                    worst_sector = str(row["aua_id"])
            elif diff_d <= -3.0:
                status = "DIRECT_SHORTCUT"
        else:
            status = "REROUTED_SECTOR"

        sectors.append(
            {
                "seq_num": int(row["seq_num"]),
                "aua_id": str(row["aua_id"]),
                "aua_type": atype,
                "actual_entry": str(row["actual_entry"]),
                "actual_exit": str(row["actual_exit"]),
                "actual_dwell_min": round(act_d, 1),
                "filed_dwell_min": round(fld_d, 1) if fld_d is not None else None,
                "dwell_diff_min": round(diff_d, 1) if diff_d is not None else None,
                "status": status,
            }
        )

    return {
        "ectrl_id": ectrl_id,
        "period": dataset.period,
        "has_data": True,
        "adep": flight_info.get("adep") if flight_info else None,
        "ades": flight_info.get("ades") if flight_info else None,
        "ac_operator": flight_info.get("ac_operator") if flight_info else None,
        "ac_type": flight_info.get("ac_type") if flight_info else None,
        "total_actual_aua_min": round(total_actual_min, 1),
        "total_filed_aua_min": round(total_filed_min, 1),
        "net_aua_dwell_delta_min": round(total_actual_min - total_filed_min, 1),
        "tma_dwell_delta_min": round(tma_dwell_delta_min, 1),
        "enroute_cta_uta_dwell_delta_min": round(enroute_dwell_delta_min, 1),
        "primary_bottleneck_aua": worst_sector,
        "max_sector_excess_min": round(max_excess_min, 1),
        "sectors": sectors,
    }


import pandas as pd  # noqa: E402

