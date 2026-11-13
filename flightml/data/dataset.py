from __future__ import annotations

from pathlib import Path
from typing import Any
import duckdb
import pandas as pd

from flightml.config import DATA_DIR, PARQUET_DIR

PERIODS_PARQUET_DIR = PARQUET_DIR / "periods"


class FlightDataset:
    """
    High-performance, columnar analytics and query engine for the FlightML dataset.
    Supports multi-period federation across Eurocontrol monthly archives (e.g. 201611, 202510).
    """

    def __init__(
        self,
        period: str = "201611",
        parquet_dir: Path | str | None = None,
        data_dir: Path | str | None = None,
        db_path: str = ":memory:",
    ) -> None:
        self.period = str(period)
        self.parquet_dir = Path(parquet_dir or PARQUET_DIR)
        self.data_dir = Path(data_dir or DATA_DIR)
        self.periods_dir = PERIODS_PARQUET_DIR
        self.con = duckdb.connect(db_path)
        self._init_views()

    def _resolve_universal_source(self, parquet_name: str, csv_name: str) -> str:
        pq_path = self.parquet_dir / parquet_name
        if pq_path.exists():
            return f"'{str(pq_path).replace(chr(92), '/')}'"
        csv_path = self.data_dir / csv_name
        return f"read_csv_auto('{str(csv_path).replace(chr(92), '/')}', header=true, ignore_errors=true)"

    def _resolve_period_source(self, table_name: str, fallback_parquet: str, fallback_csv: str) -> str:
        if self.period not in ("201611", "baseline"):
            period_pq = self.periods_dir / self.period / f"{table_name}.parquet"
            if period_pq.exists():
                return f"'{str(period_pq).replace(chr(92), '/')}'"

        return self._resolve_universal_source(fallback_parquet, fallback_csv)

    def _init_views(self) -> None:
        """Register logical tables and views in DuckDB."""
        airports_src = self._resolve_universal_source("airports.parquet", "airports.csv")
        runways_src = self._resolve_universal_source("runways.parquet", "runways.csv")
        freqs_src = self._resolve_universal_source("airport_frequencies.parquet", "airport-frequencies.csv")
        navaids_src = self._resolve_universal_source("navaids.parquet", "navaids.csv")

        fir_src = self._resolve_period_source("fir", "fir.parquet", "FIR.csv")
        routes_src = self._resolve_period_source("routes", "routes.parquet", "Routes.csv")
        flights_src = self._resolve_period_source("flights", "flights.parquet", "Flights.csv")

        self.con.execute(f"CREATE OR REPLACE VIEW raw_airports AS SELECT * FROM {airports_src};")
        self.con.execute(f"CREATE OR REPLACE VIEW raw_runways AS SELECT * FROM {runways_src};")
        self.con.execute(f"CREATE OR REPLACE VIEW raw_frequencies AS SELECT * FROM {freqs_src};")
        self.con.execute(f"CREATE OR REPLACE VIEW raw_navaids AS SELECT * FROM {navaids_src};")
        self.con.execute(f"CREATE OR REPLACE VIEW raw_fir AS SELECT * FROM {fir_src};")
        self.con.execute(f"CREATE OR REPLACE VIEW raw_routes AS SELECT * FROM {routes_src};")
        self.con.execute(f"CREATE OR REPLACE VIEW raw_flights AS SELECT * FROM {flights_src};")

        # Standardized views with clean identifiers and computed delay metrics
        self.con.execute(f"""
            CREATE OR REPLACE VIEW flights AS
            SELECT
                '{self.period}' AS period_id,
                "ECTRL ID" AS ectrl_id,
                ADEP AS adep,
                "ADEP Latitude" AS adep_lat,
                "ADEP Longitude" AS adep_lon,
                ADES AS ades,
                "ADES Latitude" AS ades_lat,
                "ADES Longitude" AS ades_lon,
                "FILED OFF BLOCK TIME" AS filed_off_block_time,
                "FILED ARRIVAL TIME" AS filed_arrival_time,
                "ACTUAL OFF BLOCK TIME" AS actual_off_block_time,
                "ACTUAL ARRIVAL TIME" AS actual_arrival_time,
                "AC Type" AS ac_type,
                "AC Operator" AS ac_operator,
                "AC Registration" AS ac_reg,
                "ICAO Flight Type" AS flight_type,
                "STATFOR Market Segment" AS market_segment,
                "Requested FL" AS requested_fl,
                "Actual Distance Flown (nm)" AS actual_distance_nm,
                datediff('minute', "FILED OFF BLOCK TIME", "ACTUAL OFF BLOCK TIME") AS dep_delay_min,
                datediff('minute', "FILED ARRIVAL TIME", "ACTUAL ARRIVAL TIME") AS arr_delay_min,
                datediff('minute', "ACTUAL OFF BLOCK TIME", "ACTUAL ARRIVAL TIME") AS actual_air_time_min,
                datediff('minute', "FILED OFF BLOCK TIME", "FILED ARRIVAL TIME") AS filed_air_time_min
            FROM raw_flights;
        """)

        # Trajectories views
        actual_traj_src = self._resolve_period_source(
            "flights_actual", "flights_actual.parquet", "flights_anonymized.csv"
        )
        planned_traj_src = self._resolve_period_source(
            "flights_planned", "flights_planned.parquet", "planned_flights_anonymized.csv"
        )

        self.con.execute(f"""
            CREATE OR REPLACE VIEW trajectory_actual AS
            SELECT
                '{self.period}' AS period_id,
                "ECTRL ID" AS ectrl_id,
                "Sequence Number" AS seq_num,
                "Time Over" AS time_over,
                "Flight Level" AS flight_level,
                Latitude AS latitude,
                Longitude AS longitude
            FROM {actual_traj_src};
        """)

        self.con.execute(f"""
            CREATE OR REPLACE VIEW trajectory_planned AS
            SELECT
                '{self.period}' AS period_id,
                "ECTRL ID" AS ectrl_id,
                "Sequence Number" AS seq_num,
                "Time Over" AS time_over,
                "Flight Level" AS flight_level,
                Latitude AS latitude,
                Longitude AS longitude
            FROM {planned_traj_src};
        """)

        # Optional en-route FIR sector traversal views (if ingested from monthly Eurocontrol archive)
        fir_act_pq = self.periods_dir / self.period / "flight_firs_actual.parquet"
        if fir_act_pq.exists():
            fir_act_src = f"'{str(fir_act_pq).replace(chr(92), '/')}'"
            self.con.execute(f"""
                CREATE OR REPLACE VIEW flight_firs_actual AS
                SELECT
                    '{self.period}' AS period_id,
                    "ECTRL ID" AS ectrl_id,
                    "Sequence Number" AS seq_num,
                    "FIR ID" AS fir_id,
                    "Entry Time" AS entry_time,
                    "Exit Time" AS exit_time,
                    datediff('minute', "Entry Time", "Exit Time") AS dwell_min
                FROM {fir_act_src};
            """)

        fir_fld_pq = self.periods_dir / self.period / "flight_firs_filed.parquet"
        if fir_fld_pq.exists():
            fir_fld_src = f"'{str(fir_fld_pq).replace(chr(92), '/')}'"
            self.con.execute(f"""
                CREATE OR REPLACE VIEW flight_firs_filed AS
                SELECT
                    '{self.period}' AS period_id,
                    "ECTRL ID" AS ectrl_id,
                    "Sequence Number" AS seq_num,
                    "FIR ID" AS fir_id,
                    "Entry Time" AS entry_time,
                    "Exit Time" AS exit_time,
                    datediff('minute', "Entry Time", "Exit Time") AS dwell_min
                FROM {fir_fld_src};
            """)

        # Optional ATC Unit Airspace (AUA: TMA, CTA, UTA) traversal views
        aua_act_pq = self.periods_dir / self.period / "flight_auas_actual.parquet"
        if aua_act_pq.exists():
            aua_act_src = f"'{str(aua_act_pq).replace(chr(92), '/')}'"
            self.con.execute(f"""
                CREATE OR REPLACE VIEW flight_auas_actual AS
                SELECT
                    '{self.period}' AS period_id,
                    "ECTRL ID" AS ectrl_id,
                    "Sequence Number" AS seq_num,
                    "AUA ID" AS aua_id,
                    CASE
                        WHEN "AUA ID" LIKE '%TMA%' THEN 'TMA'
                        WHEN "AUA ID" LIKE '%UTA%' OR "AUA ID" LIKE '%UIR%' THEN 'UTA'
                        WHEN "AUA ID" LIKE '%CTA%' THEN 'CTA'
                        ELSE 'OTHER'
                    END AS aua_type,
                    "Entry Time" AS entry_time,
                    "Exit Time" AS exit_time,
                    datediff('minute', "Entry Time", "Exit Time") AS dwell_min
                FROM {aua_act_src};
            """)

        aua_fld_pq = self.periods_dir / self.period / "flight_auas_filed.parquet"
        if aua_fld_pq.exists():
            aua_fld_src = f"'{str(aua_fld_pq).replace(chr(92), '/')}'"
            self.con.execute(f"""
                CREATE OR REPLACE VIEW flight_auas_filed AS
                SELECT
                    '{self.period}' AS period_id,
                    "ECTRL ID" AS ectrl_id,
                    "Sequence Number" AS seq_num,
                    "AUA ID" AS aua_id,
                    CASE
                        WHEN "AUA ID" LIKE '%TMA%' THEN 'TMA'
                        WHEN "AUA ID" LIKE '%UTA%' OR "AUA ID" LIKE '%UIR%' THEN 'UTA'
                        WHEN "AUA ID" LIKE '%CTA%' THEN 'CTA'
                        ELSE 'OTHER'
                    END AS aua_type,
                    "Entry Time" AS entry_time,
                    "Exit Time" AS exit_time,
                    datediff('minute', "Entry Time", "Exit Time") AS dwell_min
                FROM {aua_fld_src};
            """)

        # External Reference Views (fix.dat, nav.dat, OpenAP aircraft/dragpolar/wrap/engines, ICAO EDB, CRCO unit rates)
        ext_tables = [
            "nav_fixes",
            "nav_aids_xplane",
            "openap_aircraft",
            "openap_wrap",
            "openap_engines",
            "icao_edb_gaseous",
            "icao_edb_nvpm",
            "crco_unit_rates",
        ]
        if not all((self.parquet_dir / f"{t}.parquet").exists() for t in ext_tables):
            from flightml.data.external_sources import compile_all_external_sources

            compile_all_external_sources(parquet_dir=self.parquet_dir, force=False)

        for tbl in ext_tables:
            pq_file = self.parquet_dir / f"{tbl}.parquet"
            if pq_file.exists():
                pq_src = f"'{str(pq_file).replace(chr(92), '/')}'"
                self.con.execute(f"CREATE OR REPLACE VIEW {tbl} AS SELECT * FROM {pq_src};")

    @property
    def cursor(self) -> duckdb.DuckDBPyConnection:
        """Return an independent, thread-safe cursor for concurrent query execution."""
        return self.con.cursor()

    @property
    def total_flights(self) -> int:
        """Return total flight count in the current dataset period."""
        return int(self.cursor.execute("SELECT COUNT(*) FROM flights").fetchone()[0])

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Execute arbitrary SQL and return a Pandas DataFrame using a thread-isolated cursor."""
        cur = self.cursor
        if params:
            return cur.execute(sql, params).df()
        return cur.execute(sql).df()

    def get_summary(self) -> dict[str, Any]:
        """Return dataset high-level summary metrics."""
        cur = self.cursor
        flight_stats = cur.execute("""
            SELECT
                count(*) AS total_flights,
                count(distinct ac_operator) AS total_operators,
                count(distinct adep) AS unique_dep_airports,
                count(distinct ades) AS unique_arr_airports,
                min(actual_off_block_time) AS min_time,
                max(actual_arrival_time) AS max_time,
                avg(dep_delay_min) AS avg_dep_delay,
                avg(arr_delay_min) AS avg_arr_delay
            FROM flights
        """).fetchone()

        apt_count = cur.execute("SELECT count(*) FROM raw_airports").fetchone()[0]
        runway_count = cur.execute("SELECT count(*) FROM raw_runways").fetchone()[0]
        fir_count = cur.execute("SELECT count(distinct \"Airspace ID\") FROM raw_fir").fetchone()[0]

        return {
            "period": self.period,
            "total_flights": flight_stats[0],
            "total_operators": flight_stats[1],
            "unique_dep_airports": flight_stats[2],
            "unique_arr_airports": flight_stats[3],
            "date_range": f"{flight_stats[4]} to {flight_stats[5]}",
            "avg_dep_delay_min": round(flight_stats[6] or 0.0, 2),
            "avg_arr_delay_min": round(flight_stats[7] or 0.0, 2),
            "airports_catalog": apt_count,
            "runways_catalog": runway_count,
            "fir_airspaces": fir_count,
        }

    def get_flight(self, ectrl_id: int) -> dict[str, Any] | None:
        """Fetch details for a specific flight using a thread-isolated cursor."""
        df = self.cursor.execute(
            "SELECT * FROM flights WHERE ectrl_id = ?", [int(ectrl_id)]
        ).df()
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_trajectory(self, ectrl_id: int) -> dict[str, pd.DataFrame]:
        """Fetch paired planned and actual trajectories for an ECTRL ID."""
        cur = self.cursor
        actual_df = cur.execute(
            "SELECT * FROM trajectory_actual WHERE ectrl_id = ? ORDER BY seq_num ASC",
            [int(ectrl_id)],
        ).df()
        planned_df = cur.execute(
            "SELECT * FROM trajectory_planned WHERE ectrl_id = ? ORDER BY seq_num ASC",
            [int(ectrl_id)],
        ).df()
        return {"actual": actual_df, "planned": planned_df}

    def get_top_airports(self, limit: int = 15) -> pd.DataFrame:
        """Fetch the busiest departure/arrival airports by flight count."""
        return self.cursor.execute(
            """
            SELECT
                a.ident,
                a.name,
                a.municipality,
                a.iso_country,
                count(*) AS departure_count,
                round(avg(f.dep_delay_min), 2) AS avg_dep_delay_min
            FROM flights f
            JOIN raw_airports a ON f.adep = a.ident
            GROUP BY a.ident, a.name, a.municipality, a.iso_country
            ORDER BY departure_count DESC
            LIMIT ?
            """,
            [limit],
        ).df()

    def get_top_delayed_routes(self, min_flights: int = 50, limit: int = 15) -> pd.DataFrame:
        """Identify routes with highest average arrival delays."""
        return self.cursor.execute(
            """
            SELECT
                adep,
                ades,
                count(*) as flight_count,
                round(avg(dep_delay_min), 2) as avg_dep_delay,
                round(avg(arr_delay_min), 2) as avg_arr_delay,
                round(avg(actual_distance_nm), 1) as avg_distance_nm
            FROM flights
            GROUP BY adep, ades
            HAVING count(*) >= ?
            ORDER BY avg_arr_delay DESC
            LIMIT ?
            """,
            [min_flights, limit],
        ).df()

    def search_flights(
        self,
        adep: str | None = None,
        ades: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        time_of_day: str | None = None,
        hour_min: int | None = None,
        hour_max: int | None = None,
        min_distance_nm: float | None = None,
        max_distance_nm: float | None = None,
        min_fl: float | None = None,
        max_fl: float | None = None,
        ac_operator: str | None = None,
        ac_type: str | None = None,
        ac_reg: str | None = None,
        flight_type: str | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """
        Multi-attribute flight search engine filtering by city pair, dates, time of day,
        distance, cruising flight level, operator, aircraft type, and tail number.
        """
        conditions: list[str] = ["actual_off_block_time IS NOT NULL"]
        params: list[Any] = []

        if adep and adep.strip():
            conditions.append("adep = ?")
            params.append(adep.strip().upper())

        if ades and ades.strip():
            conditions.append("ades = ?")
            params.append(ades.strip().upper())

        if date_from and date_from.strip():
            conditions.append("actual_off_block_time >= ?")
            params.append(f"{date_from.strip()} 00:00:00")

        if date_to and date_to.strip():
            conditions.append("actual_off_block_time <= ?")
            params.append(f"{date_to.strip()} 23:59:59")

        # Time of day filtering
        if time_of_day and time_of_day != "All":
            tod_lower = time_of_day.lower()
            if "morning" in tod_lower:
                conditions.append("EXTRACT(HOUR FROM actual_off_block_time) BETWEEN 6 AND 11")
            elif "afternoon" in tod_lower:
                conditions.append("EXTRACT(HOUR FROM actual_off_block_time) BETWEEN 12 AND 17")
            elif "evening" in tod_lower:
                conditions.append("EXTRACT(HOUR FROM actual_off_block_time) BETWEEN 18 AND 23")
            elif "night" in tod_lower:
                conditions.append("EXTRACT(HOUR FROM actual_off_block_time) BETWEEN 0 AND 5")

        if hour_min is not None and hour_max is not None:
            conditions.append("EXTRACT(HOUR FROM actual_off_block_time) BETWEEN ? AND ?")
            params.extend([int(hour_min), int(hour_max)])

        if min_distance_nm is not None and min_distance_nm > 0:
            conditions.append("actual_distance_nm >= ?")
            params.append(float(min_distance_nm))

        if max_distance_nm is not None and max_distance_nm > 0:
            conditions.append("actual_distance_nm <= ?")
            params.append(float(max_distance_nm))

        if min_fl is not None and min_fl > 0:
            conditions.append("requested_fl >= ?")
            params.append(float(min_fl))

        if max_fl is not None and max_fl > 0:
            conditions.append("requested_fl <= ?")
            params.append(float(max_fl))

        if ac_operator and ac_operator.strip():
            conditions.append("ac_operator = ?")
            params.append(ac_operator.strip().upper())

        if ac_type and ac_type.strip():
            conditions.append("ac_type = ?")
            params.append(ac_type.strip().upper())

        if ac_reg and ac_reg.strip():
            conditions.append("ac_reg LIKE ?")
            params.append(f"%{ac_reg.strip().upper()}%")

        if flight_type and flight_type.strip():
            conditions.append("flight_type = ?")
            params.append(flight_type.strip().upper())

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT 
                ectrl_id,
                adep,
                ades,
                actual_off_block_time,
                actual_arrival_time,
                ac_operator,
                ac_type,
                ac_reg,
                requested_fl,
                actual_distance_nm,
                dep_delay_min,
                arr_delay_min,
                actual_air_time_min,
                flight_type,
                market_segment
            FROM flights
            WHERE {where_clause}
            ORDER BY actual_off_block_time ASC
            LIMIT {int(limit)};
        """
        df = self.cursor.execute(sql, params).df()
        if not df.empty:
            df["label"] = df.apply(self.format_flight_label, axis=1)
        return df

    @staticmethod
    def format_flight_label(row: dict[str, Any] | pd.Series) -> str:
        """
        Format flight record into clean, human-readable label:
        'ADEP ➔ ADES (YYYY-MM-DD HH:MM) [Operator Type Tail] ... Flight ID: 123456'
        """
        adep = str(row.get("adep") or "???")
        ades = str(row.get("ades") or "???")
        t_raw = str(row.get("actual_off_block_time") or "")
        t_str = t_raw[:16] if len(t_raw) >= 16 else t_raw
        op = str(row.get("ac_operator") or "").strip()
        ac = str(row.get("ac_type") or "").strip()
        reg = str(row.get("ac_reg") or "").strip()
        reg_clean = f" {reg}" if reg and reg.lower() != "nan" and reg != "" else ""
        meta = f"[{op} {ac}{reg_clean}]".strip()
        fid = row.get("ectrl_id")
        return f"{adep} ➔ {ades} ({t_str}) {meta} ... Flight ID: {fid}"

    def has_fir_traversals(self) -> bool:
        """Check if en-route FIR sector traversal logs exist for this period."""
        try:
            res = self.cursor.execute("SELECT 1 FROM flight_firs_actual LIMIT 1").fetchone()
            return res is not None
        except Exception:
            return False

    def get_flight_fir_traversals(self, ectrl_id: int) -> pd.DataFrame:
        """
        Fetch sequenced sector traversals for an individual flight,
        comparing actual dwell vs filed planned dwell in each airspace.
        """
        if not self.has_fir_traversals():
            return pd.DataFrame()

        sql = """
            SELECT
                a.seq_num,
                a.fir_id,
                a.entry_time AS actual_entry,
                a.exit_time AS actual_exit,
                round(epoch(a.exit_time - a.entry_time) / 60.0, 1) AS actual_dwell_min,
                f.entry_time AS filed_entry,
                f.exit_time AS filed_exit,
                round(epoch(f.exit_time - f.entry_time) / 60.0, 1) AS filed_dwell_min,
                round((epoch(a.exit_time - a.entry_time) - epoch(f.exit_time - f.entry_time)) / 60.0, 1) AS dwell_diff_min
            FROM flight_firs_actual a
            LEFT JOIN flight_firs_filed f
              ON a.ectrl_id = f.ectrl_id AND a.fir_id = f.fir_id
            WHERE a.ectrl_id = ? AND a.fir_id NOT LIKE '%TAXI%'
            ORDER BY a.seq_num ASC;
        """
        return self.cursor.execute(sql, [int(ectrl_id)]).df()

    def get_fir_boundaries(self, fir_ids: list[str]) -> dict[str, list[tuple[float, float]]]:
        """
        Fetch sequenced polygon boundary coordinates (lat, lon) for a list of FIR airspace IDs.
        """
        if not fir_ids:
            return {}
        clean_ids = [str(f).strip().upper() for f in fir_ids if f and "TAXI" not in str(f).upper()]
        if not clean_ids:
            return {}
        placeholders = ", ".join(["?"] * len(clean_ids))
        sql = f"""
            SELECT "Airspace ID" as fir_id, "Latitude" as lat, "Longitude" as lon
            FROM raw_fir
            WHERE "Airspace ID" IN ({placeholders})
            ORDER BY "Airspace ID", "Sequence Number" ASC;
        """
        try:
            df = self.cursor.execute(sql, clean_ids).df()
            res: dict[str, list[tuple[float, float]]] = {}
            for fid, group in df.groupby("fir_id"):
                coords = list(zip(group["lat"], group["lon"]))
                if len(coords) >= 3:
                    res[str(fid)] = coords
            return res
        except Exception:
            return {}

    def has_aua_traversals(self) -> bool:
        """Check whether the active period includes ATC Unit Airspace (AUA) traversal tables."""
        try:
            self.cursor.execute("SELECT 1 FROM flight_auas_actual LIMIT 1")
            return True
        except Exception:
            return False

    def get_flight_aua_traversals(self, ectrl_id: int) -> pd.DataFrame:
        """
        Fetch sequenced ATC Unit Airspace (TMA, CTA, UTA) traversals for an individual flight,
        comparing actual dwell vs filed planned dwell.
        """
        if not self.has_aua_traversals():
            return pd.DataFrame()

        sql = """
            SELECT
                a.seq_num,
                a.aua_id,
                a.aua_type,
                a.entry_time AS actual_entry,
                a.exit_time AS actual_exit,
                round(epoch(a.exit_time - a.entry_time) / 60.0, 1) AS actual_dwell_min,
                f.entry_time AS filed_entry,
                f.exit_time AS filed_exit,
                round(epoch(f.exit_time - f.entry_time) / 60.0, 1) AS filed_dwell_min,
                round((epoch(a.exit_time - a.entry_time) - epoch(f.exit_time - f.entry_time)) / 60.0, 1) AS dwell_diff_min
            FROM flight_auas_actual a
            LEFT JOIN flight_auas_filed f
              ON a.ectrl_id = f.ectrl_id AND a.aua_id = f.aua_id
            WHERE a.ectrl_id = ?
            ORDER BY a.seq_num ASC;
        """
        return self.cursor.execute(sql, [int(ectrl_id)]).df()

    def find_nearest_waypoints(
        self,
        lat: float,
        lon: float,
        radius_nm: float = 15.0,
        limit: int = 10,
        include_fixes: bool = True,
        include_navaids: bool = True,
    ) -> pd.DataFrame:
        """
        Find the closest global 5-letter RNAV waypoints (`fix.dat`) and/or radio navaids (`nav.dat`)
        within `radius_nm` nautical miles of `(lat, lon)`.
        """
        deg_margin = (radius_nm / 60.0) * 1.5
        lon_margin = deg_margin / max(0.2, abs(__import__("math").cos(__import__("math").radians(lat))))

        subqueries: list[str] = []
        params: list[Any] = []

        if include_fixes:
            subqueries.append("""
                SELECT
                    fix_ident AS ident,
                    'FIX' AS kind,
                    fix_ident AS name,
                    latitude,
                    longitude,
                    NULL AS frequency_mhz
                FROM nav_fixes
                WHERE latitude BETWEEN ? AND ?
                  AND longitude BETWEEN ? AND ?
            """)
            params.extend([lat - deg_margin, lat + deg_margin, lon - lon_margin, lon + lon_margin])

        if include_navaids:
            subqueries.append("""
                SELECT
                    ident,
                    nav_type AS kind,
                    name,
                    latitude,
                    longitude,
                    frequency_mhz
                FROM nav_aids_xplane
                WHERE latitude BETWEEN ? AND ?
                  AND longitude BETWEEN ? AND ?
            """)
            params.extend([lat - deg_margin, lat + deg_margin, lon - lon_margin, lon + lon_margin])

        if not subqueries:
            return pd.DataFrame()

        union_sql = " UNION ALL ".join(subqueries)
        sql = f"""
            WITH candidates AS (
                {union_sql}
            ),
            scored AS (
                SELECT
                    ident,
                    kind,
                    name,
                    latitude,
                    longitude,
                    frequency_mhz,
                    round(
                        3440.065 * 2 * asin(sqrt(
                            pow(sin(radians(latitude - ?) / 2), 2) +
                            cos(radians(?)) * cos(radians(latitude)) *
                            pow(sin(radians(longitude - ?) / 2), 2)
                        )),
                        2
                    ) AS distance_nm
                FROM candidates
            )
            SELECT * FROM scored
            WHERE distance_nm <= ?
            ORDER BY distance_nm ASC
            LIMIT ?;
        """
        params.extend([lat, lat, lon, radius_nm, int(limit)])
        return self.cursor.execute(sql, params).df()

    def get_aircraft_performance(self, ac_type: str) -> dict[str, Any] | None:
        """
        Retrieve OpenAP aircraft performance, MTOW, CRCO weight factor, drag polar, and fuel polynomial
        parameters for a 4-letter ICAO aircraft type (including synonyms).
        """
        code = ac_type.strip().upper()
        df = self.cursor.execute("SELECT * FROM openap_aircraft WHERE ac_type = ?", [code]).df()
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_crco_unit_rate(self, fir_or_zone: str) -> dict[str, Any] | None:
        """
        Resolve a 2-letter CRCO zone code (e.g. 'EG', 'ED', 'LF', 'LT') or an ICAO FIR ID
        (e.g. 'EGTTFIR', 'EDGGFIR', 'LPPOFIR') to its Eurocontrol CRCO unit rate in EUR.
        """
        clean = fir_or_zone.strip().upper()
        # Special Atlantic / Canarias zone overrides in Eurocontrol CRCO
        if clean.startswith("LPPO"):
            zone = "AZ"
        elif clean.startswith("GCCC"):
            zone = "GC"
        else:
            zone = clean[:2]

        df = self.cursor.execute("SELECT * FROM crco_unit_rates WHERE zone_code = ?", [zone]).df()
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_engine_emissions(self, engine_query: str) -> dict[str, Any]:
        """
        Query ICAO EDB v32B gaseous (NOx, CO, HC) and nvPM (soot mass & particle number)
        emission indices for a matching engine designation.
        """
        q = f"%{engine_query.strip()}%"
        gas_df = self.cursor.execute(
            "SELECT * FROM icao_edb_gaseous WHERE engine_name ILIKE ? ORDER BY rated_thrust_kn DESC LIMIT 5",
            [q],
        ).df()
        nvpm_df = self.cursor.execute(
            "SELECT * FROM icao_edb_nvpm WHERE engine_name ILIKE ? ORDER BY rated_thrust_kn DESC LIMIT 5",
            [q],
        ).df()
        return {
            "query": engine_query,
            "gaseous_matches": gas_df.to_dict(orient="records"),
            "nvpm_matches": nvpm_df.to_dict(orient="records"),
        }

    def get_external_data_summary(self) -> dict[str, int]:
        """Return row counts for all registered external reference views in DuckDB."""
        views = [
            "nav_fixes",
            "nav_aids_xplane",
            "openap_aircraft",
            "openap_wrap",
            "openap_engines",
            "icao_edb_gaseous",
            "icao_edb_nvpm",
            "crco_unit_rates",
        ]
        cur = self.cursor
        counts: dict[str, int] = {}
        for v in views:
            try:
                counts[v] = int(cur.execute(f"SELECT COUNT(*) FROM {v}").fetchone()[0])
            except Exception:
                counts[v] = 0
        return counts

    @classmethod
    def list_available_periods(cls, parquet_dir: Path | str | None = None) -> list[str]:
        """Scan for all available ingested Eurocontrol periods."""
        pq_dir = Path(parquet_dir or PARQUET_DIR)
        periods = ["201611"]
        periods_sub = pq_dir / "periods"
        if periods_sub.exists():
            for p in sorted(periods_sub.iterdir()):
                if p.is_dir() and (p / "flights.parquet").exists():
                    if p.name not in periods:
                        periods.append(p.name)
        return sorted(periods)
