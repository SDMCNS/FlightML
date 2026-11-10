from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import duckdb

from flightml.config import DATA_DIR


@dataclass
class RouteChangeSummary:
    cycle1: str
    cycle2: str
    total_routes_c1: int
    total_routes_c2: int
    added_routes_count: int
    deleted_routes_count: int
    modified_routes_count: int
    sample_added_routes: list[str]
    sample_deleted_routes: list[str]


@dataclass
class FIRChangeSummary:
    cycle1: str
    cycle2: str
    total_firs_c1: int
    total_firs_c2: int
    added_firs_count: int
    deleted_firs_count: int
    modified_boundaries_count: int
    sample_modified_firs: list[str]


@dataclass
class AIRACDiffReport:
    cycle1: str
    cycle2: str
    route_changes: RouteChangeSummary
    fir_changes: FIRChangeSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison": f"AIRAC {self.cycle1} ➔ AIRAC {self.cycle2}",
            "airway_routes": {
                "cycle1_total": self.route_changes.total_routes_c1,
                "cycle2_total": self.route_changes.total_routes_c2,
                "added_airways": self.route_changes.added_routes_count,
                "deleted_airways": self.route_changes.deleted_routes_count,
                "modified_geometry_airways": self.route_changes.modified_routes_count,
                "sample_added": self.route_changes.sample_added_routes,
                "sample_deleted": self.route_changes.sample_deleted_routes,
            },
            "fir_airspaces": {
                "cycle1_total": self.fir_changes.total_firs_c1,
                "cycle2_total": self.fir_changes.total_firs_c2,
                "added_firs": self.fir_changes.added_firs_count,
                "deleted_firs": self.fir_changes.deleted_firs_count,
                "modified_boundary_sectors": self.fir_changes.modified_boundaries_count,
                "sample_modified_firs": self.fir_changes.sample_modified_firs,
            },
        }


def _find_airac_file(base_dir: Path, prefix: str, cycle: str) -> Path | None:
    """Locate route or FIR file for a given AIRAC cycle (e.g. Route_2509.csv.gz)."""
    # Check directly under base_dir and all subdirectories
    candidates = list(base_dir.rglob(f"{prefix}*_{cycle}*")) + list(base_dir.rglob(f"{prefix}*{cycle}*"))
    for c in candidates:
        if c.is_file() and (".csv" in c.name or ".parquet" in c.name):
            return c
    return None


def compare_airac_cycles(
    cycle1: str = "2509",
    cycle2: str = "2510",
    data_dir: Path | str | None = None,
) -> AIRACDiffReport:
    """
    Perform deep structural diffing between two AIRAC cycles.
    Tracks additions, deletions, and geometry revisions across airways and FIR airspaces.
    """
    base = Path(data_dir or DATA_DIR)
    con = duckdb.connect()

    # Locate Route files
    r1_file = _find_airac_file(base, "Route", cycle1)
    r2_file = _find_airac_file(base, "Route", cycle2)

    # Locate FIR files
    f1_file = _find_airac_file(base, "FIR", cycle1)
    f2_file = _find_airac_file(base, "FIR", cycle2)

    if not r1_file or not r2_file:
        raise FileNotFoundError(f"Could not locate Route files for cycles {cycle1} and {cycle2}")
    if not f1_file or not f2_file:
        raise FileNotFoundError(f"Could not locate FIR files for cycles {cycle1} and {cycle2}")

    r1_str = str(r1_file).replace("\\", "/")
    r2_str = str(r2_file).replace("\\", "/")
    f1_str = str(f1_file).replace("\\", "/")
    f2_str = str(f2_file).replace("\\", "/")

    # 1. Compare Routes
    routes_diff = con.execute(f"""
        WITH r1 AS (
            SELECT "Route ID" as rid, count(*) as pts1
            FROM read_csv_auto('{r1_str}', header=true, ignore_errors=true)
            GROUP BY "Route ID"
        ),
        r2 AS (
            SELECT "Route ID" as rid, count(*) as pts2
            FROM read_csv_auto('{r2_str}', header=true, ignore_errors=true)
            GROUP BY "Route ID"
        )
        SELECT 
            (SELECT count(*) FROM r1) as total_r1,
            (SELECT count(*) FROM r2) as total_r2,
            (SELECT count(*) FROM r2 WHERE rid NOT IN (SELECT rid FROM r1)) as added_count,
            (SELECT count(*) FROM r1 WHERE rid NOT IN (SELECT rid FROM r2)) as deleted_count,
            (SELECT count(*) FROM r1 JOIN r2 ON r1.rid = r2.rid WHERE r1.pts1 != r2.pts2) as modified_count
    """).fetchone()

    sample_added_r = con.execute(f"""
        SELECT "Route ID" FROM read_csv_auto('{r2_str}')
        WHERE "Route ID" NOT IN (SELECT "Route ID" FROM read_csv_auto('{r1_str}'))
        GROUP BY "Route ID" LIMIT 5
    """).df()["Route ID"].tolist()

    sample_del_r = con.execute(f"""
        SELECT "Route ID" FROM read_csv_auto('{r1_str}')
        WHERE "Route ID" NOT IN (SELECT "Route ID" FROM read_csv_auto('{r2_str}'))
        GROUP BY "Route ID" LIMIT 5
    """).df()["Route ID"].tolist()

    route_summary = RouteChangeSummary(
        cycle1=cycle1,
        cycle2=cycle2,
        total_routes_c1=routes_diff[0],
        total_routes_c2=routes_diff[1],
        added_routes_count=routes_diff[2],
        deleted_routes_count=routes_diff[3],
        modified_routes_count=routes_diff[4],
        sample_added_routes=sample_added_r,
        sample_deleted_routes=sample_del_r,
    )

    # 2. Compare FIRs
    fir_diff = con.execute(f"""
        WITH f1 AS (
            SELECT "Airspace ID" as fid, count(*) as pts1
            FROM read_csv_auto('{f1_str}', header=true, ignore_errors=true)
            GROUP BY "Airspace ID"
        ),
        f2 AS (
            SELECT "Airspace ID" as fid, count(*) as pts2
            FROM read_csv_auto('{f2_str}', header=true, ignore_errors=true)
            GROUP BY "Airspace ID"
        )
        SELECT
            (SELECT count(*) FROM f1) as total_f1,
            (SELECT count(*) FROM f2) as total_f2,
            (SELECT count(*) FROM f2 WHERE fid NOT IN (SELECT fid FROM f1)) as added_count,
            (SELECT count(*) FROM f1 WHERE fid NOT IN (SELECT fid FROM f2)) as deleted_count,
            (SELECT count(*) FROM f1 JOIN f2 ON f1.fid = f2.fid WHERE f1.pts1 != f2.pts2) as modified_count
    """).fetchone()

    sample_mod_fir = con.execute(f"""
        WITH f1 AS (
            SELECT "Airspace ID" as fid, count(*) as pts1
            FROM read_csv_auto('{f1_str}')
            GROUP BY "Airspace ID"
        ),
        f2 AS (
            SELECT "Airspace ID" as fid, count(*) as pts2
            FROM read_csv_auto('{f2_str}')
            GROUP BY "Airspace ID"
        )
        SELECT f1.fid FROM f1 JOIN f2 ON f1.fid = f2.fid WHERE f1.pts1 != f2.pts2 LIMIT 6
    """).df()["fid"].tolist()

    fir_summary = FIRChangeSummary(
        cycle1=cycle1,
        cycle2=cycle2,
        total_firs_c1=fir_diff[0],
        total_firs_c2=fir_diff[1],
        added_firs_count=fir_diff[2],
        deleted_firs_count=fir_diff[3],
        modified_boundaries_count=fir_diff[4],
        sample_modified_firs=sample_mod_fir,
    )

    return AIRACDiffReport(
        cycle1=cycle1,
        cycle2=cycle2,
        route_changes=route_summary,
        fir_changes=fir_summary,
    )

