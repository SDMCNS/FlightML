from __future__ import annotations

import calendar
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import duckdb
import pandas as pd

from flightml.config import PARQUET_DIR
from flightml.data.archive_ingest import list_available_periods
from flightml.data.dataset import FlightDataset

PERIODS_DIR = PARQUET_DIR / "periods"

# Established ICAO type designators for new-generation fuel-efficient aircraft
NEWGEN_NARROWBODY = {"A20N", "A21N", "B38M", "B39M", "BCS1", "BCS3"}
NEWGEN_WIDEBODY = {"A359", "A35K", "B788", "B789", "B78X", "A339"}
ALL_NEWGEN_TYPES = NEWGEN_NARROWBODY | NEWGEN_WIDEBODY

# Equivalent legacy types for empirical fuel/CO2 comparisons
LEGACY_EQUIVALENTS: dict[str, str] = {
    "A20N": "A320",
    "A21N": "A321",
    "B38M": "B738",
    "B39M": "B739",
    "BCS1": "E190",
    "BCS3": "A319",
    "A359": "A333",
    "A35K": "B773",
    "B788": "B763",
    "B789": "B772",
    "B78X": "B772",
    "A339": "A333",
}

# Empirical average fuel burn reduction of new-gen vs legacy baseline (%)
NEWGEN_FUEL_SAVINGS_PCT: dict[str, float] = {
    "A20N": 16.0,
    "A21N": 18.0,
    "B38M": 15.0,
    "B39M": 15.0,
    "BCS1": 18.0,
    "BCS3": 20.0,
    "A359": 22.0,
    "A35K": 24.0,
    "B788": 20.0,
    "B789": 21.0,
    "B78X": 22.0,
    "A339": 14.0,
}


@dataclass
class PeriodMetadata:
    period_id: str
    year: int
    month: int
    days_in_month: int
    season_label: str
    is_leap_year: bool

    @classmethod
    def from_period_id(cls, period_id: str) -> PeriodMetadata:
        match = re.search(r"(\d{4})(\d{2})", period_id)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
        else:
            year, month = 2025, 10

        is_leap = calendar.isleap(year)
        days = calendar.monthrange(year, month)[1]

        if period_id == "201611":
            label = "Historical Baseline"
        elif month in (1, 2, 12):
            label = "Winter Trough"
        elif month in (6, 7, 8):
            label = "Summer Peak"
        else:
            label = "Autumn Shoulder"

        return cls(
            period_id=period_id,
            year=year,
            month=month,
            days_in_month=days,
            season_label=label,
            is_leap_year=is_leap,
        )


@dataclass
class PeriodMacroMetrics:
    period_id: str
    metadata: PeriodMetadata
    total_flights: int
    daily_flights: float
    avg_dep_delay_min: float
    avg_arr_delay_min: float
    otp_15_pct: float
    avg_distance_nm: float
    avg_air_time_min: float
    newgen_flights: int
    newgen_share_pct: float
    top_airlines: list[dict[str, Any]]
    top_aircraft_types: list[dict[str, Any]]
    top_airports: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metadata"] = asdict(self.metadata)
        return d


@dataclass
class YearOverYearDelta:
    period_baseline: str
    period_target: str
    season_label: str
    total_flights_baseline: int
    total_flights_target: int
    total_flights_delta: int
    total_flights_growth_pct: float
    daily_flights_baseline: float
    daily_flights_target: float
    daily_flights_delta: float
    daily_flights_growth_pct: float
    avg_arr_delay_baseline_min: float
    avg_arr_delay_target_min: float
    arr_delay_delta_min: float
    otp_15_baseline_pct: float
    otp_15_target_pct: float
    otp_15_delta_pct: float
    avg_distance_delta_nm: float
    avg_air_time_delta_min: float
    newgen_share_baseline_pct: float
    newgen_share_target_pct: float
    newgen_share_delta_pct: float
    top_growing_airlines: list[dict[str, Any]]
    top_aircraft_type_shifts: list[dict[str, Any]]
    top_city_pairs_evolution: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnnualSeasonalDecomposition:
    year: int
    winter_period: str
    summer_period: str
    autumn_period: str
    winter_daily_flights: float
    summer_daily_flights: float
    autumn_daily_flights: float
    summer_surge_pct: float
    autumn_taper_pct: float
    winter_avg_delay_min: float
    summer_avg_delay_min: float
    autumn_avg_delay_min: float
    summer_delay_surge_pct: float
    delay_demand_elasticity: float
    top_surging_airports: list[dict[str, Any]]
    top_surging_firs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LongitudinalEngine:
    """
    Core engine for longitudinal, multi-period comparative analytics and seasonal decomposition
    across Eurocontrol flight archives.
    """

    def __init__(self, periods: list[str] | None = None) -> None:
        self.available_periods = periods or list_available_periods()
        self._metrics_cache: dict[str, PeriodMacroMetrics] = {}

    def _resolve_parquet_path(self, period_id: str) -> str:
        if period_id in ("201611", "baseline"):
            p = PARQUET_DIR / "flights.parquet"
        else:
            p = PERIODS_DIR / period_id / "flights.parquet"
        return str(p).replace("\\", "/")

    def _resolve_fir_actual_path(self, period_id: str) -> str | None:
        p = PERIODS_DIR / period_id / "flight_firs_actual.parquet"
        if p.exists():
            return str(p).replace("\\", "/")
        return None

    def _register_flights_view(
        self, con: duckdb.DuckDBPyConnection, period_id: str, view_name: str
    ) -> None:
        """Register a normalized flights view in a DuckDB connection for cross-period analysis."""
        src_path = self._resolve_parquet_path(period_id)
        con.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                '{period_id}' AS period_id,
                "ECTRL ID" AS ectrl_id,
                ADEP AS adep,
                ADES AS ades,
                "AC Type" AS ac_type,
                "AC Operator" AS ac_operator,
                "AC Registration" AS ac_reg,
                "Actual Distance Flown (nm)" AS actual_distance_nm,
                datediff('minute', "FILED OFF BLOCK TIME", "ACTUAL OFF BLOCK TIME") AS dep_delay_min,
                datediff('minute', "FILED ARRIVAL TIME", "ACTUAL ARRIVAL TIME") AS arr_delay_min,
                datediff('minute', "ACTUAL OFF BLOCK TIME", "ACTUAL ARRIVAL TIME") AS actual_air_time_min
            FROM '{src_path}';
        """)

    def _register_fir_view(
        self, con: duckdb.DuckDBPyConnection, period_id: str, view_name: str
    ) -> bool:
        """Register a normalized FIR traversals view in DuckDB if present."""
        fir_path = self._resolve_fir_actual_path(period_id)
        if not fir_path:
            return False
        con.execute(f"""
            CREATE OR REPLACE VIEW {view_name} AS
            SELECT
                '{period_id}' AS period_id,
                "ECTRL ID" AS ectrl_id,
                "Sequence Number" AS seq_num,
                "FIR ID" AS fir_id,
                "Entry Time" AS entry_time,
                "Exit Time" AS exit_time
            FROM '{fir_path}';
        """)
        return True

    def get_period_metrics(self, period_id: str) -> PeriodMacroMetrics:
        """Calculate and return macro network metrics for a specific Eurocontrol period."""
        if period_id in self._metrics_cache:
            return self._metrics_cache[period_id]

        ds = FlightDataset(period=period_id)
        meta = PeriodMetadata.from_period_id(period_id)

        # 1. Macro Volume, Delay, Distance, and Newgen Share
        newgen_str = "'" + "','".join(ALL_NEWGEN_TYPES) + "'"
        macro_sql = f"""
            SELECT
                count(*) AS total_flights,
                COALESCE(avg(dep_delay_min), 0.0) AS avg_dep_delay,
                COALESCE(avg(arr_delay_min), 0.0) AS avg_arr_delay,
                COALESCE(avg(actual_distance_nm), 0.0) AS avg_dist_nm,
                COALESCE(avg(actual_air_time_min), 0.0) AS avg_air_time_min,
                COALESCE(sum(CASE WHEN arr_delay_min <= 15 THEN 1 ELSE 0 END) * 100.0 / NULLIF(count(*), 0), 0.0) AS otp_15,
                COALESCE(sum(CASE WHEN ac_type IN ({newgen_str}) THEN 1 ELSE 0 END), 0) AS newgen_flights,
                COALESCE(sum(CASE WHEN ac_type IN ({newgen_str}) THEN 1 ELSE 0 END) * 100.0 / NULLIF(count(*), 0), 0.0) AS newgen_share_pct
            FROM flights;
        """
        macro_row = ds.query(macro_sql).iloc[0]

        total_flights = int(macro_row["total_flights"])
        daily_flights = total_flights / meta.days_in_month if meta.days_in_month > 0 else 0.0

        # 2. Top Airlines
        airline_sql = """
            SELECT
                ac_operator AS airline,
                count(*) AS flights,
                round(count(*) * 100.0 / (SELECT count(*) FROM flights), 2) AS share_pct
            FROM flights
            WHERE ac_operator IS NOT NULL AND ac_operator != ''
            GROUP BY ac_operator
            ORDER BY flights DESC
            LIMIT 10;
        """
        top_airlines = ds.query(airline_sql).to_dict(orient="records")

        # 3. Top Aircraft Types
        ac_sql = f"""
            SELECT
                ac_type,
                count(*) AS flights,
                round(count(*) * 100.0 / (SELECT count(*) FROM flights), 2) AS share_pct,
                CASE WHEN ac_type IN ({newgen_str}) THEN true ELSE false END AS is_newgen
            FROM flights
            WHERE ac_type IS NOT NULL AND ac_type != ''
            GROUP BY ac_type
            ORDER BY flights DESC
            LIMIT 10;
        """
        top_ac_types = ds.query(ac_sql).to_dict(orient="records")

        # 4. Top European Airports
        apt_sql = """
            WITH deps AS (
                SELECT adep AS icao, count(*) AS dep_count
                FROM flights
                WHERE adep IS NOT NULL AND adep != ''
                GROUP BY adep
            ),
            arrs AS (
                SELECT ades AS icao, count(*) AS arr_count
                FROM flights
                WHERE ades IS NOT NULL AND ades != ''
                GROUP BY ades
            )
            SELECT
                COALESCE(deps.icao, arrs.icao) AS icao,
                COALESCE(deps.dep_count, 0) AS departures,
                COALESCE(arrs.arr_count, 0) AS arrivals,
                COALESCE(deps.dep_count, 0) + COALESCE(arrs.arr_count, 0) AS total_movements
            FROM deps
            FULL OUTER JOIN arrs ON deps.icao = arrs.icao
            ORDER BY total_movements DESC
            LIMIT 10;
        """
        top_apts = ds.query(apt_sql).to_dict(orient="records")

        metrics = PeriodMacroMetrics(
            period_id=period_id,
            metadata=meta,
            total_flights=total_flights,
            daily_flights=round(daily_flights, 1),
            avg_dep_delay_min=round(float(macro_row["avg_dep_delay"]), 2),
            avg_arr_delay_min=round(float(macro_row["avg_arr_delay"]), 2),
            otp_15_pct=round(float(macro_row["otp_15"]), 2),
            avg_distance_nm=round(float(macro_row["avg_dist_nm"]), 1),
            avg_air_time_min=round(float(macro_row["avg_air_time_min"]), 1),
            newgen_flights=int(macro_row["newgen_flights"]),
            newgen_share_pct=round(float(macro_row["newgen_share_pct"]), 2),
            top_airlines=top_airlines,
            top_aircraft_types=top_ac_types,
            top_airports=top_apts,
        )

        self._metrics_cache[period_id] = metrics
        return metrics

    def get_all_period_metrics(self) -> list[PeriodMacroMetrics]:
        """Return macro metrics across all available periods."""
        return [self.get_period_metrics(p) for p in self.available_periods]

    def get_timeline_dataframe(self) -> pd.DataFrame:
        """Return a structured tabular timeline dataframe across all available periods."""
        metrics_list = self.get_all_period_metrics()
        rows = []
        for m in metrics_list:
            rows.append({
                "period_id": m.period_id,
                "year": m.metadata.year,
                "month": m.metadata.month,
                "season": m.metadata.season_label,
                "days": m.metadata.days_in_month,
                "total_flights": m.total_flights,
                "daily_flights": m.daily_flights,
                "avg_arr_delay_min": m.avg_arr_delay_min,
                "avg_dep_delay_min": m.avg_dep_delay_min,
                "otp_15_pct": m.otp_15_pct,
                "avg_distance_nm": m.avg_distance_nm,
                "avg_air_time_min": m.avg_air_time_min,
                "newgen_share_pct": m.newgen_share_pct,
            })
        return pd.DataFrame(rows).sort_values("period_id").reset_index(drop=True)

    def compare_yoy(
        self, period_baseline: str, period_target: str
    ) -> YearOverYearDelta:
        """
        Conduct a controlled Year-Over-Year (YoY) comparison between two periods
        (e.g. 202410 vs 202510 or 202406 vs 202506).
        """
        m_base = self.get_period_metrics(period_baseline)
        m_targ = self.get_period_metrics(period_target)

        # Volume & Rate Deltas
        f_delta = m_targ.total_flights - m_base.total_flights
        f_growth = (f_delta / m_base.total_flights * 100.0) if m_base.total_flights > 0 else 0.0

        daily_delta = m_targ.daily_flights - m_base.daily_flights
        daily_growth = (daily_delta / m_base.daily_flights * 100.0) if m_base.daily_flights > 0 else 0.0

        # Performance Deltas
        arr_delay_delta = m_targ.avg_arr_delay_min - m_base.avg_arr_delay_min
        otp_delta = m_targ.otp_15_pct - m_base.otp_15_pct
        dist_delta = m_targ.avg_distance_nm - m_base.avg_distance_nm
        time_delta = m_targ.avg_air_time_min - m_base.avg_air_time_min
        newgen_delta = m_targ.newgen_share_pct - m_base.newgen_share_pct

        # Top Growing Airlines & Aircraft Shifts
        con = duckdb.connect()
        self._register_flights_view(con, period_baseline, "flights_base")
        self._register_flights_view(con, period_target, "flights_targ")

        airline_growth_df = con.execute("""
            WITH b AS (
                SELECT ac_operator AS op, count(*) AS c_base
                FROM flights_base
                WHERE ac_operator IS NOT NULL AND ac_operator != ''
                GROUP BY ac_operator
            ),
            t AS (
                SELECT ac_operator AS op, count(*) AS c_targ
                FROM flights_targ
                WHERE ac_operator IS NOT NULL AND ac_operator != ''
                GROUP BY ac_operator
            )
            SELECT
                t.op AS airline,
                b.c_base AS flights_baseline,
                t.c_targ AS flights_target,
                t.c_targ - COALESCE(b.c_base, 0) AS flights_delta,
                round((t.c_targ - COALESCE(b.c_base, 0)) * 100.0 / NULLIF(b.c_base, 0), 1) AS growth_pct
            FROM t
            LEFT JOIN b ON t.op = b.op
            WHERE b.c_base > 500
            ORDER BY flights_delta DESC
            LIMIT 10;
        """).df()
        top_growing_airlines = airline_growth_df.to_dict(orient="records")

        # Top Aircraft Type Shifts
        ac_shift_df = con.execute("""
            WITH b AS (
                SELECT ac_type, count(*) AS c_base
                FROM flights_base
                WHERE ac_type IS NOT NULL AND ac_type != ''
                GROUP BY ac_type
            ),
            t AS (
                SELECT ac_type, count(*) AS c_targ
                FROM flights_targ
                WHERE ac_type IS NOT NULL AND ac_type != ''
                GROUP BY ac_type
            )
            SELECT
                COALESCE(t.ac_type, b.ac_type) AS ac_type,
                COALESCE(b.c_base, 0) AS flights_baseline,
                COALESCE(t.c_targ, 0) AS flights_target,
                COALESCE(t.c_targ, 0) - COALESCE(b.c_base, 0) AS delta_flights,
                round((COALESCE(t.c_targ, 0) - COALESCE(b.c_base, 0)) * 100.0 / NULLIF(b.c_base, 0), 1) AS growth_pct
            FROM t
            FULL OUTER JOIN b ON t.ac_type = b.ac_type
            WHERE COALESCE(b.c_base, 0) > 1000 OR COALESCE(t.c_targ, 0) > 1000
            ORDER BY delta_flights DESC
            LIMIT 10;
        """).df()
        top_ac_shifts = ac_shift_df.to_dict(orient="records")

        # Top City Pairs Evolution (Efficiency, Volume, New-gen adoption)
        newgen_str = "'" + "','".join(ALL_NEWGEN_TYPES) + "'"
        city_pair_df = con.execute(f"""
            WITH b AS (
                SELECT
                    adep || '-' || ades AS route,
                    count(*) AS c_base,
                    avg(actual_distance_nm) AS dist_base,
                    sum(CASE WHEN ac_type IN ({newgen_str}) THEN 1 ELSE 0 END) * 100.0 / count(*) AS newgen_pct_base
                FROM flights_base
                WHERE adep IS NOT NULL AND ades IS NOT NULL
                GROUP BY adep, ades
            ),
            t AS (
                SELECT
                    adep || '-' || ades AS route,
                    count(*) AS c_targ,
                    avg(actual_distance_nm) AS dist_targ,
                    sum(CASE WHEN ac_type IN ({newgen_str}) THEN 1 ELSE 0 END) * 100.0 / count(*) AS newgen_pct_targ
                FROM flights_targ
                WHERE adep IS NOT NULL AND ades IS NOT NULL
                GROUP BY adep, ades
            )
            SELECT
                t.route,
                b.c_base AS volume_base,
                t.c_targ AS volume_targ,
                t.c_targ - b.c_base AS volume_delta,
                round(b.dist_base, 1) AS mean_dist_base_nm,
                round(t.dist_targ, 1) AS mean_dist_targ_nm,
                round(t.dist_targ - b.dist_base, 1) AS dist_delta_nm,
                round(b.newgen_pct_base, 1) AS newgen_share_base_pct,
                round(t.newgen_pct_targ, 1) AS newgen_share_targ_pct,
                round(t.newgen_pct_targ - b.newgen_pct_base, 1) AS newgen_share_delta_pct
            FROM t
            JOIN b ON t.route = b.route
            WHERE b.c_base >= 200
            ORDER BY t.c_targ DESC
            LIMIT 15;
        """).df()
        top_city_pairs = city_pair_df.to_dict(orient="records")

        return YearOverYearDelta(
            period_baseline=period_baseline,
            period_target=period_target,
            season_label=m_targ.metadata.season_label,
            total_flights_baseline=m_base.total_flights,
            total_flights_target=m_targ.total_flights,
            total_flights_delta=f_delta,
            total_flights_growth_pct=round(f_growth, 2),
            daily_flights_baseline=m_base.daily_flights,
            daily_flights_target=m_targ.daily_flights,
            daily_flights_delta=round(daily_delta, 1),
            daily_flights_growth_pct=round(daily_growth, 2),
            avg_arr_delay_baseline_min=m_base.avg_arr_delay_min,
            avg_arr_delay_target_min=m_targ.avg_arr_delay_min,
            arr_delay_delta_min=round(arr_delay_delta, 2),
            otp_15_baseline_pct=m_base.otp_15_pct,
            otp_15_target_pct=m_targ.otp_15_pct,
            otp_15_delta_pct=round(otp_delta, 2),
            avg_distance_delta_nm=round(dist_delta, 2),
            avg_air_time_delta_min=round(time_delta, 2),
            newgen_share_baseline_pct=m_base.newgen_share_pct,
            newgen_share_target_pct=m_targ.newgen_share_pct,
            newgen_share_delta_pct=round(newgen_delta, 2),
            top_growing_airlines=top_growing_airlines,
            top_aircraft_type_shifts=top_ac_shifts,
            top_city_pairs_evolution=top_city_pairs,
        )

    def decompose_seasonality(self, year: int) -> AnnualSeasonalDecomposition:
        """
        Perform three-season annual decomposition (Winter Trough, Summer Peak, Autumn Shoulder)
        for a designated year (e.g. 2024 or 2025).
        """
        p_winter = f"{year}02"
        p_summer = f"{year}06"
        p_autumn = f"{year}10"

        m_win = self.get_period_metrics(p_winter)
        m_sum = self.get_period_metrics(p_summer)
        m_aut = self.get_period_metrics(p_autumn)

        # Seasonal Surge & Taper
        surge_pct = (
            ((m_sum.daily_flights - m_win.daily_flights) / m_win.daily_flights * 100.0)
            if m_win.daily_flights > 0
            else 0.0
        )
        taper_pct = (
            ((m_aut.daily_flights - m_sum.daily_flights) / m_sum.daily_flights * 100.0)
            if m_sum.daily_flights > 0
            else 0.0
        )

        # Delay Surge & Elasticity
        delay_surge_pct = (
            ((m_sum.avg_arr_delay_min - m_win.avg_arr_delay_min) / m_win.avg_arr_delay_min * 100.0)
            if m_win.avg_arr_delay_min > 0
            else 0.0
        )
        elasticity = (delay_surge_pct / surge_pct) if surge_pct > 0 else 0.0

        # Top Surging Leisure Airports (Summer vs Winter)
        con = duckdb.connect()
        self._register_flights_view(con, p_winter, "flights_win")
        self._register_flights_view(con, p_summer, "flights_sum")

        apt_surge_df = con.execute("""
            WITH win AS (
                SELECT adep AS icao, count(*) AS win_cnt
                FROM flights_win
                WHERE adep IS NOT NULL AND adep != ''
                GROUP BY adep
            ),
            sum AS (
                SELECT adep AS icao, count(*) AS sum_cnt
                FROM flights_sum
                WHERE adep IS NOT NULL AND adep != ''
                GROUP BY adep
            )
            SELECT
                sum.icao,
                win.win_cnt AS winter_flights,
                sum.sum_cnt AS summer_flights,
                sum.sum_cnt - win.win_cnt AS volume_increase,
                round((sum.sum_cnt - win.win_cnt) * 100.0 / win.win_cnt, 1) AS surge_pct
            FROM sum
            JOIN win ON sum.icao = win.icao
            WHERE win.win_cnt > 500
            ORDER BY surge_pct DESC
            LIMIT 10;
        """).df()
        top_surging_apts = apt_surge_df.to_dict(orient="records")

        # Top Surging FIRs (Summer vs Winter)
        has_win_fir = self._register_fir_view(con, p_winter, "fir_win")
        has_sum_fir = self._register_fir_view(con, p_summer, "fir_sum")

        top_surging_firs = []
        if has_win_fir and has_sum_fir:
            fir_surge_df = con.execute("""
                WITH win AS (
                    SELECT fir_id, count(*) AS win_cnt
                    FROM fir_win
                    GROUP BY fir_id
                ),
                sum AS (
                    SELECT fir_id, count(*) AS sum_cnt
                    FROM fir_sum
                    GROUP BY fir_id
                )
                SELECT
                    sum.fir_id,
                    win.win_cnt AS winter_traversals,
                    sum.sum_cnt AS summer_traversals,
                    sum.sum_cnt - win.win_cnt AS volume_increase,
                    round((sum.sum_cnt - win.win_cnt) * 100.0 / win.win_cnt, 1) AS surge_pct
                FROM sum
                JOIN win ON sum.fir_id = win.fir_id
                WHERE win.win_cnt > 15000
                ORDER BY surge_pct DESC
                LIMIT 10;
            """).df()
            top_surging_firs = fir_surge_df.to_dict(orient="records")

        return AnnualSeasonalDecomposition(
            year=year,
            winter_period=p_winter,
            summer_period=p_summer,
            autumn_period=p_autumn,
            winter_daily_flights=m_win.daily_flights,
            summer_daily_flights=m_sum.daily_flights,
            autumn_daily_flights=m_aut.daily_flights,
            summer_surge_pct=round(surge_pct, 1),
            autumn_taper_pct=round(taper_pct, 1),
            winter_avg_delay_min=m_win.avg_arr_delay_min,
            summer_avg_delay_min=m_sum.avg_arr_delay_min,
            autumn_avg_delay_min=m_aut.avg_arr_delay_min,
            summer_delay_surge_pct=round(delay_surge_pct, 1),
            delay_demand_elasticity=round(elasticity, 2),
            top_surging_airports=top_surging_apts,
            top_surging_firs=top_surging_firs,
        )

    def get_fleet_evolution_timeline(self) -> pd.DataFrame:
        """
        Return longitudinal fleet evolution metrics across all available periods,
        tracking adoption rates of neo/MAX/new-gen aircraft and estimated fuel/CO2 benefits.
        """
        con = duckdb.connect()
        rows = []

        for p in self.available_periods:
            self._register_flights_view(con, p, f"f_{p}")
            meta = PeriodMetadata.from_period_id(p)
            newgen_str = "'" + "','".join(ALL_NEWGEN_TYPES) + "'"

            df = con.execute(f"""
                SELECT
                    count(*) AS total,
                    sum(CASE WHEN ac_type IN ('A20N', 'A21N') THEN 1 ELSE 0 END) AS airbus_neo,
                    sum(CASE WHEN ac_type IN ('B38M', 'B39M') THEN 1 ELSE 0 END) AS boeing_max,
                    sum(CASE WHEN ac_type IN ('BCS1', 'BCS3') THEN 1 ELSE 0 END) AS a220,
                    sum(CASE WHEN ac_type IN ('A359', 'A35K') THEN 1 ELSE 0 END) AS a350,
                    sum(CASE WHEN ac_type IN ('B788', 'B789', 'B78X') THEN 1 ELSE 0 END) AS b787,
                    sum(CASE WHEN ac_type IN ({newgen_str}) THEN 1 ELSE 0 END) AS total_newgen,
                    sum(CASE WHEN ac_type IN ('A319', 'A320', 'A321', 'B737', 'B738', 'B739') THEN 1 ELSE 0 END) AS classic_narrowbody,
                    sum(CASE WHEN ac_type IN ('A332', 'A333', 'B772', 'B773', 'B744', 'A343') THEN 1 ELSE 0 END) AS classic_widebody
                FROM f_{p}
                WHERE ac_type IS NOT NULL AND ac_type != '';
            """).df()

            tot = int(df.iloc[0]["total"])
            newgen = int(df.iloc[0]["total_newgen"])
            newgen_pct = (newgen * 100.0 / tot) if tot > 0 else 0.0

            rows.append({
                "period_id": p,
                "year": meta.year,
                "month": meta.month,
                "season": meta.season_label,
                "total_flights": tot,
                "airbus_neo_flights": int(df.iloc[0]["airbus_neo"]),
                "boeing_max_flights": int(df.iloc[0]["boeing_max"]),
                "a220_flights": int(df.iloc[0]["a220"]),
                "a350_flights": int(df.iloc[0]["a350"]),
                "b787_flights": int(df.iloc[0]["b787"]),
                "total_newgen_flights": newgen,
                "newgen_share_pct": round(newgen_pct, 2),
                "classic_narrowbody_flights": int(df.iloc[0]["classic_narrowbody"]),
                "classic_widebody_flights": int(df.iloc[0]["classic_widebody"]),
            })

        return pd.DataFrame(rows).sort_values("period_id").reset_index(drop=True)
