"""
4D Airspace Co-Altitude Proximity Scanner & Sector Dynamic Complexity Engine (Sprint 5).

1. `scan_airspace_proximity_events`: Uses DuckDB spatial-temporal binning over `trajectory_actual`
   to detect aircraft pairs passing within `horiz_sep_nm` (default 5.0 nm) and `vert_sep_fl`
   (default 10 FL = 1,000 ft) within a 90-second temporal window above FL100.
2. `compute_sector_complexity_index`: Computes the Eurocontrol Dynamic Density / Airspace Complexity
   Index weighting vertical phase transitions (climbing/descending traffic) and track crossing entropy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from flightml.data.dataset import FlightDataset
from flightml.geo.geodesy import haversine_nm


@dataclass
class ProximityPairEvent:
    flight_1_id: int
    flight_2_id: int
    operator_1: str
    operator_2: str
    ac_type_1: str
    ac_type_2: str
    route_1: str
    route_2: str
    timestamp_utc: str
    latitude: float
    longitude: float
    fl_1: float
    fl_2: float
    vertical_sep_ft: float
    horizontal_sep_nm: float
    time_diff_sec: float
    severity: str  # 'LOSS_OF_SEPARATION_ALERT (<3nm/<1000ft)', 'HIGH_PROXIMITY (<5nm/<=1000ft)'


@dataclass
class ProximityReport:
    period: str
    date_str: str
    hour_utc: int
    horiz_threshold_nm: float
    vert_threshold_fl: float
    events_detected: int
    loss_of_sep_alerts: int
    events: list[ProximityPairEvent]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_core_date(dataset: FlightDataset, date_str: str | None = None) -> str:
    """Resolve a full-traffic day within the active period (avoiding pre-month UTC spillover dates)."""
    if date_str and date_str.strip():
        return date_str.strip()
    if len(dataset.period) == 6 and dataset.period.isdigit():
        candidate = f"{dataset.period[:4]}-{dataset.period[4:6]}-01"
        cnt = dataset.cursor.execute(
            "SELECT count(*) FROM flights WHERE CAST(filed_off_block_time AS DATE) = CAST(? AS DATE)",
            [candidate],
        ).fetchone()
        if cnt and cnt[0] and int(cnt[0]) > 100:
            return candidate
    row = dataset.cursor.execute(
        """
        SELECT CAST(filed_off_block_time AS DATE) AS dt, count(*) AS cnt
        FROM flights
        WHERE filed_off_block_time IS NOT NULL
        GROUP BY dt
        ORDER BY cnt DESC
        LIMIT 1
        """
    ).fetchone()
    return str(row[0]) if row and row[0] else ("2025-10-01" if dataset.period == "202510" else "2016-11-01")


def scan_airspace_proximity_events(
    dataset: FlightDataset,
    date_str: str | None = None,
    hour: int = 10,
    min_lat: float = 35.0,
    max_lat: float = 60.0,
    min_lon: float = -10.0,
    max_lon: float = 30.0,
    horiz_sep_nm: float = 5.0,
    vert_sep_fl: float = 10.0,
    limit: int = 25,
) -> ProximityReport:
    """
    Scan 4D trajectory points in a 1-hour European airspace window to detect co-altitude
    aircraft pairs passing within `horiz_sep_nm` (nm) and `vert_sep_fl` (Flight Levels, 10 FL = 1,000 ft).
    """
    date_str = _resolve_core_date(dataset, date_str)

    hr_int = max(0, min(23, int(hour)))
    deg_tol = max(0.08, horiz_sep_nm / 45.0)

    sql = f"""
        WITH window_pts AS (
            SELECT
                t.ectrl_id,
                t.seq_num,
                t.time_over,
                epoch(t.time_over) AS ts_sec,
                CAST(floor(epoch(t.time_over) / 120.0) AS BIGINT) AS time_bin,
                CAST(floor(t.latitude * 5.0) AS BIGINT) AS lat_bin,
                CAST(floor(t.longitude * 5.0) AS BIGINT) AS lon_bin,
                t.flight_level,
                t.latitude,
                t.longitude
            FROM trajectory_actual t
            WHERE CAST(t.time_over AS DATE) = CAST(? AS DATE)
              AND extract('hour' FROM t.time_over) = ?
              AND t.flight_level >= 150
              AND t.latitude BETWEEN ? AND ?
              AND t.longitude BETWEEN ? AND ?
        ),
        candidate_pairs AS (
            SELECT
                a.ectrl_id AS f1,
                b.ectrl_id AS f2,
                a.time_over AS t1,
                a.latitude AS lat1,
                a.longitude AS lon1,
                b.latitude AS lat2,
                b.longitude AS lon2,
                a.flight_level AS fl1,
                b.flight_level AS fl2,
                abs(a.flight_level - b.flight_level) * 100.0 AS vert_ft,
                abs(a.ts_sec - b.ts_sec) AS dt_sec
            FROM window_pts a
            JOIN window_pts b
              ON a.time_bin = b.time_bin
             AND a.lat_bin = b.lat_bin
             AND a.lon_bin = b.lon_bin
             AND a.ectrl_id < b.ectrl_id
            WHERE abs(a.ts_sec - b.ts_sec) <= 90
              AND abs(a.flight_level - b.flight_level) <= ?
              AND abs(a.latitude - b.latitude) <= ?
              AND abs(a.longitude - b.longitude) <= ?
        )
        SELECT
            cp.*,
            fl1.ac_operator AS op1,
            fl1.ac_type AS ac1,
            fl1.adep || '➔' || fl1.ades AS r1,
            fl2.ac_operator AS op2,
            fl2.ac_type AS ac2,
            fl2.adep || '➔' || fl2.ades AS r2
        FROM candidate_pairs cp
        JOIN flights fl1 ON cp.f1 = fl1.ectrl_id
        JOIN flights fl2 ON cp.f2 = fl2.ectrl_id
        LIMIT 400;
    """
    df = dataset.cursor.execute(
        sql,
        [date_str, hr_int, min_lat, max_lat, min_lon, max_lon, float(vert_sep_fl), deg_tol, deg_tol * 1.4],
    ).df()

    seen_pairs: set[tuple[int, int]] = set()
    events: list[ProximityPairEvent] = []
    los_alerts = 0

    for _, row in df.iterrows():
        pair_key = (int(row["f1"]), int(row["f2"]))
        if pair_key in seen_pairs:
            continue

        d_nm = haversine_nm(
            float(row["lat1"]),
            float(row["lon1"]),
            float(row["lat2"]),
            float(row["lon2"]),
        )
        if d_nm > horiz_sep_nm:
            continue

        seen_pairs.add(pair_key)
        vert_ft = float(row["vert_ft"])
        if d_nm < 3.0 and vert_ft < 1000.0:
            sev = "LOSS_OF_SEPARATION_ALERT (<3nm/<1000ft)"
            los_alerts += 1
        else:
            sev = "HIGH_PROXIMITY (<5nm/<=1000ft)"

        events.append(
            ProximityPairEvent(
                flight_1_id=pair_key[0],
                flight_2_id=pair_key[1],
                operator_1=str(row["op1"] or ""),
                operator_2=str(row["op2"] or ""),
                ac_type_1=str(row["ac1"] or ""),
                ac_type_2=str(row["ac2"] or ""),
                route_1=str(row["r1"] or ""),
                route_2=str(row["r2"] or ""),
                timestamp_utc=str(row["t1"]),
                latitude=round((float(row["lat1"]) + float(row["lat2"])) / 2.0, 4),
                longitude=round((float(row["lon1"]) + float(row["lon2"])) / 2.0, 4),
                fl_1=round(float(row["fl1"]), 0),
                fl_2=round(float(row["fl2"]), 0),
                vertical_sep_ft=round(vert_ft, 0),
                horizontal_sep_nm=round(d_nm, 2),
                time_diff_sec=round(float(row["dt_sec"]), 1),
                severity=sev,
            )
        )
        if len(events) >= int(limit):
            break

    events.sort(key=lambda e: (e.horizontal_sep_nm, e.vertical_sep_ft))

    return ProximityReport(
        period=dataset.period,
        date_str=date_str,
        hour_utc=hr_int,
        horiz_threshold_nm=horiz_sep_nm,
        vert_threshold_fl=vert_sep_fl,
        events_detected=len(events),
        loss_of_sep_alerts=los_alerts,
        events=events,
    )


def compute_sector_complexity_index(
    dataset: FlightDataset,
    date_str: str | None = None,
    hour: int = 10,
) -> dict[str, Any]:
    """
    Compute the Eurocontrol Dynamic Density / Airspace Complexity Index across 5°x5° European cells,
    combining aircraft count, vertical transition share (climbing/descending points vs level cruise),
    and altitude dispersion.
    """
    date_str = _resolve_core_date(dataset, date_str)

    sql = """
        WITH diffs AS (
            SELECT
                ectrl_id,
                floor(latitude / 5.0) * 5.0 AS cell_lat,
                floor(longitude / 5.0) * 5.0 AS cell_lon,
                flight_level,
                abs(flight_level - lag(flight_level) OVER (PARTITION BY ectrl_id ORDER BY seq_num)) AS dfl
            FROM trajectory_actual
            WHERE CAST(time_over AS DATE) = CAST(? AS DATE)
              AND extract('hour' FROM time_over) = ?
              AND flight_level >= 100
        )
        SELECT
            CAST(cell_lat AS INT) AS cell_lat,
            CAST(cell_lon AS INT) AS cell_lon,
            CAST(cell_lat AS INT) || '°N to ' || CAST(cell_lat + 5 AS INT) || '°N, ' ||
            CAST(cell_lon AS INT) || '°E to ' || CAST(cell_lon + 5 AS INT) || '°E' AS grid_sector,
            CAST(count(distinct ectrl_id) AS BIGINT) AS active_flights,
            CAST(count(*) AS BIGINT) AS waypoint_reports,
            round(
                count(CASE WHEN dfl >= 5 THEN 1 END) * 100.0 / nullif(count(*), 0),
                1
            ) AS vertical_transition_pct,
            round(stddev_pop(flight_level), 1) AS fl_dispersion,
            round(
                count(distinct ectrl_id) * (1.0 + count(CASE WHEN dfl >= 5 THEN 1 END) * 1.5 / nullif(count(*), 0)),
                1
            ) AS dynamic_complexity_score
        FROM diffs
        GROUP BY cell_lat, cell_lon
        HAVING count(distinct ectrl_id) >= 10
        ORDER BY dynamic_complexity_score DESC
        LIMIT 20;
    """
    df = dataset.cursor.execute(sql, [date_str, int(hour)]).df()
    return {
        "period": dataset.period,
        "date_str": date_str,
        "hour_utc": int(hour),
        "sectors": df.to_dict(orient="records") if not df.empty else [],
    }


def compute_hourly_grid_saturation(
    dataset: FlightDataset,
    hour_utc: int = 10,
    min_lat: float = 30.0,
    max_lat: float = 65.0,
    min_lon: float = -15.0,
    max_lon: float = 35.0,
    min_avg_flights: float = 3.0,
) -> dict[str, Any]:
    """
    Compute the average hourly traffic density, peak hourly flights, vertical climb/descent share,
    and dynamic complexity index for every 5°x5° geographic square across the dataset period
    for a selected hour of the day (`0` to `23` UTC).
    """
    hr = max(0, min(23, int(hour_utc)))
    sql = """
        WITH pts AS (
            SELECT
                CAST(time_over AS DATE) AS dt,
                ectrl_id,
                CAST(floor(latitude / 5.0) * 5 AS INT) AS cell_lat,
                CAST(floor(longitude / 5.0) * 5 AS INT) AS cell_lon,
                flight_level,
                abs(flight_level - lag(flight_level) OVER (PARTITION BY ectrl_id ORDER BY seq_num)) AS dfl
            FROM trajectory_actual
            WHERE extract('hour' FROM time_over) = ?
              AND flight_level >= 100
              AND latitude BETWEEN ? AND ?
              AND longitude BETWEEN ? AND ?
        ),
        daily_cell AS (
            SELECT
                cell_lat,
                cell_lon,
                dt,
                count(distinct ectrl_id) AS daily_flights,
                count(*) AS daily_pts,
                count(CASE WHEN dfl >= 5 THEN 1 END) AS vert_pts,
                avg(flight_level) AS mean_fl,
                stddev_pop(flight_level) AS std_fl
            FROM pts
            GROUP BY cell_lat, cell_lon, dt
        )
        SELECT
            cell_lat,
            cell_lon,
            CAST(cell_lat AS VARCHAR) || '°N to ' || CAST(cell_lat + 5 AS VARCHAR) || '°N, ' ||
            CAST(cell_lon AS VARCHAR) || '°E to ' || CAST(cell_lon + 5 AS VARCHAR) || '°E' AS grid_sector,
            CAST(count(distinct dt) AS INT) AS active_days,
            round(avg(daily_flights), 1) AS avg_flights_per_hour,
            CAST(max(daily_flights) AS INT) AS peak_flights_per_hour,
            CAST(sum(daily_flights) AS BIGINT) AS total_period_flights,
            round(sum(vert_pts) * 100.0 / nullif(sum(daily_pts), 0), 1) AS vertical_transition_pct,
            round(avg(mean_fl), 0) AS avg_flight_level,
            round(coalesce(avg(std_fl), 0.0), 1) AS fl_dispersion,
            round(
                avg(daily_flights) * (1.0 + (sum(vert_pts) * 1.5 / nullif(sum(daily_pts), 0))),
                1
            ) AS dynamic_complexity_score
        FROM daily_cell
        GROUP BY cell_lat, cell_lon
        HAVING avg(daily_flights) >= ?
        ORDER BY dynamic_complexity_score DESC;
    """
    df = dataset.cursor.execute(
        sql, [hr, min_lat, max_lat, min_lon, max_lon, float(min_avg_flights)]
    ).df()

    cells: list[dict[str, Any]] = []
    if not df.empty:
        max_comp = max(1.0, float(df["dynamic_complexity_score"].max()))
        for _, r in df.iterrows():
            comp = float(r["dynamic_complexity_score"])
            avg_f = float(r["avg_flights_per_hour"])
            ratio = comp / max_comp
            if ratio >= 0.65 or avg_f >= 180.0:
                status = "CRITICAL_SATURATION"
            elif ratio >= 0.40 or avg_f >= 100.0:
                status = "HIGH_DENSITY"
            elif ratio >= 0.20 or avg_f >= 45.0:
                status = "MODERATE_FLOW"
            else:
                status = "LOW_DENSITY"

            cells.append(
                {
                    "cell_lat": int(r["cell_lat"]),
                    "cell_lon": int(r["cell_lon"]),
                    "grid_sector": str(r["grid_sector"]),
                    "cell_label": str(r["grid_sector"]),
                    "active_days": int(r["active_days"]),
                    "avg_flights_per_hour": avg_f,
                    "peak_flights_per_hour": int(r["peak_flights_per_hour"]),
                    "total_period_flights": int(r["total_period_flights"]),
                    "vertical_transition_pct": float(r["vertical_transition_pct"] or 0.0),
                    "avg_flight_level": float(r["avg_flight_level"] or 320.0),
                    "fl_dispersion": float(r["fl_dispersion"] or 0.0),
                    "dynamic_complexity_score": comp,
                    "saturation_status": status,
                }
            )

    crit_count = sum(1 for c in cells if c["saturation_status"] == "CRITICAL_SATURATION")
    high_count = sum(1 for c in cells if c["saturation_status"] == "HIGH_DENSITY")
    max_avg = max((c["avg_flights_per_hour"] for c in cells), default=0.0)

    return {
        "period": dataset.period,
        "hour_utc": hr,
        "total_active_squares": len(cells),
        "critical_squares_count": crit_count,
        "high_density_squares_count": high_count,
        "max_avg_flights_per_hour": round(max_avg, 1),
        "cells": cells,
        "squares": cells,
    }

