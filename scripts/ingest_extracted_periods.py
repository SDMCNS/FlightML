"""
Ingest all extracted Eurocontrol monthly archive datasets into partitioned Zstandard Parquet tables.
Ensures a complete contiguous 12-month Eurocontrol operational cycle (202411 -> 202510) and full-year capability.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
import duckdb
from rich.console import Console
from rich.table import Table

console = Console()
DATA_DIR = Path("c:/Projects/data/FlightML/data")
EXTRACTED_DIR = DATA_DIR / "extracted"
PERIODS_DIR = DATA_DIR / "parquet" / "periods"


def classify_files(period_dir: Path) -> dict[str, Path]:
    classified: dict[str, Path] = {}
    all_files = list(period_dir.glob("*.csv*"))
    if not all_files:
        all_files = list(period_dir.rglob("*.csv*"))

    period_id = period_dir.name
    airac_cycle = period_id[2:] if len(period_id) == 6 else ""

    for f in all_files:
        name = f.name.lower()
        if name.startswith("flights_") or name == "flights.csv":
            classified["flights"] = f
        elif "flight_points_actual" in name or "flights_anonymized" in name:
            classified["flights_actual"] = f
        elif "flight_points_filed" in name or "planned_flights" in name:
            classified["flights_planned"] = f
        elif "flight_firs_actual" in name:
            classified["flight_firs_actual"] = f
        elif "flight_firs_filed" in name:
            classified["flight_firs_filed"] = f
        elif "flight_auas_actual" in name:
            classified["flight_auas_actual"] = f
        elif "flight_auas_filed" in name:
            classified["flight_auas_filed"] = f
        elif name.startswith("fir_") or name == "fir.csv":
            if airac_cycle and airac_cycle in name:
                classified["fir"] = f
            elif "fir" not in classified:
                classified["fir"] = f
        elif name.startswith("route_") or name == "routes.csv":
            if airac_cycle and airac_cycle in name:
                classified["routes"] = f
            elif "routes" not in classified:
                classified["routes"] = f

    return classified


def ingest_period(period_dir: Path, con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    period_id = period_dir.name
    out_dir = PERIODS_DIR / period_id
    out_dir.mkdir(parents=True, exist_ok=True)

    classified = classify_files(period_dir)
    timings = {}

    for table_name, src_file in classified.items():
        dest_pq = out_dir / f"{table_name}.parquet"
        if dest_pq.exists() and dest_pq.stat().st_size > 1024:
            continue

        src_str = str(src_file).replace("\\", "/")
        dest_str = str(dest_pq).replace("\\", "/")
        t0 = time.time()
        query = f"""
            COPY (
                SELECT * FROM read_csv_auto('{src_str}', header=true, ignore_errors=true)
            ) TO '{dest_str}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        con.execute(query)
        dt = time.time() - t0
        timings[table_name] = dt
        sz_mb = dest_pq.stat().st_size / 1e6
        console.print(f"  ✓ {period_id}/{table_name:20s}: {sz_mb:6.1f} MB in {dt:5.2f}s")

    return timings


def main():
    PERIODS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    period_dirs = sorted([p for p in EXTRACTED_DIR.iterdir() if p.is_dir() and re.match(r"^\d{6}$", p.name)])
    console.print(f"[bold cyan]🚀 Scanning {len(period_dirs)} extracted archive directories...[/bold cyan]")

    total_start = time.time()
    results = {}

    for p_dir in period_dirs:
        period_id = p_dir.name
        console.print(f"\n[bold yellow]Processing Period {period_id}...[/bold yellow]")
        timings = ingest_period(p_dir, con)
        results[period_id] = timings

    console.print("\n" + "=" * 60)
    console.print(f"[bold green]✨ All periods processed in {time.time() - total_start:.1f}s![/bold green]")

    # Print summary table of available parquet periods
    table = Table(title="📦 Available Parquet Archive Periods (Eurocontrol)", header_style="bold green")
    table.add_column("Period", style="cyan")
    table.add_column("Flights (Rows)", justify="right", style="green")
    table.add_column("FIR Crossings (Rows)", justify="right", style="yellow")
    table.add_column("Tables Present", style="white")

    for p_dir in sorted(PERIODS_DIR.iterdir()):
        if not p_dir.is_dir():
            continue
        pid = p_dir.name
        fl_pq = p_dir / "flights.parquet"
        fir_pq = p_dir / "flight_firs_actual.parquet"
        tables = [t.stem for t in p_dir.glob("*.parquet")]

        fl_count = con.execute(f"SELECT count(*) FROM '{str(fl_pq).replace(chr(92), '/')}'").fetchone()[0] if fl_pq.exists() else 0
        fir_count = con.execute(f"SELECT count(*) FROM '{str(fir_pq).replace(chr(92), '/')}'").fetchone()[0] if fir_pq.exists() else 0

        table.add_row(
            pid,
            f"{fl_count:,}",
            f"{fir_count:,}",
            f"{len(tables)} tables ({', '.join(sorted(tables))})",
        )

    console.print(table)


if __name__ == "__main__":
    main()
