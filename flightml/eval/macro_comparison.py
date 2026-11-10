from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import duckdb
import pandas as pd

from flightml.config import PARQUET_DIR

PERIODS_DIR = PARQUET_DIR / "periods"


@dataclass
class PeriodComparison:
    period1: str
    period2: str
    total_flights_p1: int
    total_flights_p2: int
    traffic_growth_pct: float
    avg_arr_delay_p1: float
    avg_arr_delay_p2: float
    delay_change_min: float
    top_growing_airlines: list[dict[str, Any]]
    fleet_modernization_shifts: list[dict[str, Any]]
    busiest_airports_comparison: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_baseline": self.period1,
            "period_comparison": self.period2,
            "total_flights_baseline": self.total_flights_p1,
            "total_flights_comparison": self.total_flights_p2,
            "traffic_growth_percentage": round(self.traffic_growth_pct, 1),
            "avg_arr_delay_baseline_min": round(self.avg_arr_delay_p1, 2),
            "avg_arr_delay_comparison_min": round(self.avg_arr_delay_p2, 2),
            "delay_change_min": round(self.delay_change_min, 2),
            "top_growing_airlines": self.top_growing_airlines,
            "fleet_modernization_shifts": self.fleet_modernization_shifts,
            "busiest_airports_comparison": self.busiest_airports_comparison,
        }


def _resolve_flights_parquet(period: str) -> str:
    if period in ("201611", "baseline"):
        p = PARQUET_DIR / "flights.parquet"
    else:
        p = PERIODS_DIR / period / "flights.parquet"
    return str(p).replace("\\", "/")


def compare_aviation_periods(
    period1: str = "201611", period2: str = "202510"
) -> PeriodComparison:
    """
    Perform deep macro comparative analytics between two Eurocontrol monthly archives
    (e.g. November 2016 vs October 2025).
    """
    con = duckdb.connect()
    p1_src = _resolve_flights_parquet(period1)
    p2_src = _resolve_flights_parquet(period2)

    # 1. Macro Volume & Delay Metrics
    stats_p1 = con.execute(f"""
        SELECT
            count(*) as total_flights,
            avg(datediff('minute', "FILED ARRIVAL TIME", "ACTUAL ARRIVAL TIME")) as avg_delay
        FROM '{p1_src}';
    """).fetchone()

    stats_p2 = con.execute(f"""
        SELECT
            count(*) as total_flights,
            avg(datediff('minute', "FILED ARRIVAL TIME", "ACTUAL ARRIVAL TIME")) as avg_delay
        FROM '{p2_src}';
    """).fetchone()

    f1, d1 = stats_p1[0], stats_p1[1] or 0.0
    f2, d2 = stats_p2[0], stats_p2[1] or 0.0
    growth = ((f2 - f1) / f1 * 100.0) if f1 > 0 else 0.0

    # 2. Airline Growth & Market Share Evolution
    airline_df = con.execute(f"""
        WITH t1 AS (
            SELECT "AC Operator" as op, count(*) as c1
            FROM '{p1_src}'
            WHERE "AC Operator" IS NOT NULL AND "AC Operator" != ''
            GROUP BY "AC Operator"
        ),
        t2 AS (
            SELECT "AC Operator" as op, count(*) as c2
            FROM '{p2_src}'
            WHERE "AC Operator" IS NOT NULL AND "AC Operator" != ''
            GROUP BY "AC Operator"
        )
        SELECT
            COALESCE(t2.op, t1.op) as airline,
            COALESCE(t1.c1, 0) as flights_p1,
            COALESCE(t2.c2, 0) as flights_p2,
            COALESCE(t2.c2, 0) - COALESCE(t1.c1, 0) as growth_flights,
            round((COALESCE(t2.c2, 0) - COALESCE(t1.c1, 0)) * 100.0 / NULLIF(t1.c1, 0), 1) as growth_pct
        FROM t2
        FULL OUTER JOIN t1 ON t2.op = t1.op
        ORDER BY flights_p2 DESC
        LIMIT 10;
    """).df()

    # 3. Fleet Modernization & Aircraft Type Shifts
    fleet_df = con.execute(f"""
        WITH f1 AS (
            SELECT "AC Type" as ac, count(*) as c1
            FROM '{p1_src}'
            WHERE "AC Type" IS NOT NULL AND "AC Type" != ''
            GROUP BY "AC Type"
        ),
        f2 AS (
            SELECT "AC Type" as ac, count(*) as c2
            FROM '{p2_src}'
            WHERE "AC Type" IS NOT NULL AND "AC Type" != ''
            GROUP BY "AC Type"
        )
        SELECT
            COALESCE(f2.ac, f1.ac) as ac_type,
            COALESCE(f1.c1, 0) as flights_p1,
            COALESCE(f2.c2, 0) as flights_p2,
            COALESCE(f2.c2, 0) - COALESCE(f1.c1, 0) as delta_flights
        FROM f2
        FULL OUTER JOIN f1 ON f2.ac = f1.ac
        ORDER BY flights_p2 DESC
        LIMIT 10;
    """).df()

    # 4. Top Hub Airport Movements Comparison
    hub_df = con.execute(f"""
        WITH a1 AS (
            SELECT ADEP as apt, count(*) as c1
            FROM '{p1_src}'
            GROUP BY ADEP
        ),
        a2 AS (
            SELECT ADEP as apt, count(*) as c2
            FROM '{p2_src}'
            GROUP BY ADEP
        )
        SELECT
            COALESCE(a2.apt, a1.apt) as airport,
            COALESCE(a1.c1, 0) as dep_p1,
            COALESCE(a2.c2, 0) as dep_p2,
            COALESCE(a2.c2, 0) - COALESCE(a1.c1, 0) as delta_movements
        FROM a2
        FULL OUTER JOIN a1 ON a2.apt = a1.apt
        ORDER BY dep_p2 DESC
        LIMIT 10;
    """).df()

    return PeriodComparison(
        period1=period1,
        period2=period2,
        total_flights_p1=f1,
        total_flights_p2=f2,
        traffic_growth_pct=growth,
        avg_arr_delay_p1=d1,
        avg_arr_delay_p2=d2,
        delay_change_min=d2 - d1,
        top_growing_airlines=airline_df.to_dict(orient="records"),
        fleet_modernization_shifts=fleet_df.to_dict(orient="records"),
        busiest_airports_comparison=hub_df.to_dict(orient="records"),
    )

