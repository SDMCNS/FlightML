from __future__ import annotations

import re
import time
import zipfile
from pathlib import Path
from typing import Any
import duckdb
from rich.console import Console

from flightml.config import DATA_DIR, PARQUET_DIR

console = Console()
PERIODS_PARQUET_DIR = PARQUET_DIR / "periods"
PERIODS_PARQUET_DIR.mkdir(parents=True, exist_ok=True)


class EurocontrolArchiveIngestor:
    """
    Intelligent ingestion engine for monthly Eurocontrol archives (e.g. 202510.zip, 201611.zip).
    Extracts, inspects, classifies varying filenames, and converts to partitioned Apache Parquet.
    """

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir or DATA_DIR)
        self.extracted_dir = self.data_dir / "extracted"
        self.periods_dir = PERIODS_PARQUET_DIR

    def extract_zip(self, zip_path: Path) -> Path:
        """Extract a monthly Eurocontrol zip archive to extracted/{period_id}."""
        match = re.search(r"(\d{6})", zip_path.name)
        period_id = match.group(1) if match else zip_path.stem

        target_dir = self.extracted_dir / period_id
        target_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        console.print(f"[cyan]Extracting[/cyan] {zip_path.name} to {target_dir.name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.extracted_dir)
        console.print(f"[green]✓ Extracted[/green] in {time.time() - t0:.2f}s")
        return target_dir

    def classify_files(self, period_dir: Path) -> dict[str, Path]:
        """
        Classify files inside a period directory based on regex filename patterns and schemas.
        """
        classified: dict[str, Path] = {}
        all_files = list(period_dir.glob("*.csv*"))

        # In case files were extracted inside a subfolder with same name
        if not all_files:
            all_files = list(period_dir.rglob("*.csv*"))

        # Determine active AIRAC cycle from directory name (e.g. 202510 -> 2510)
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
            elif name.startswith("fir_") or name == "fir.csv":
                # Prefer active cycle if available (e.g. FIR_2510 over FIR_2509)
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

    def convert_period_to_parquet(
        self, period_id: str, classified_files: dict[str, Path]
    ) -> dict[str, Path]:
        """
        Stream classified CSV/CSV.GZ files directly into partitioned Apache Parquet via DuckDB.
        """
        out_dir = self.periods_dir / period_id
        out_dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()

        result_paths: dict[str, Path] = {}

        for table_name, src_file in classified_files.items():
            dest_file = out_dir / f"{table_name}.parquet"
            src_str = str(src_file).replace("\\", "/")
            dest_str = str(dest_file).replace("\\", "/")

            t0 = time.time()
            console.print(f"[cyan]Ingesting[/cyan] {table_name} ({src_file.name}) -> {dest_file.name}...")

            query = f"""
                COPY (
                    SELECT * FROM read_csv_auto('{src_str}', header=true, ignore_errors=true)
                ) TO '{dest_str}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
            try:
                con.execute(query)
                size_mb = dest_file.stat().st_size / (1024 * 1024)
                console.print(f"[green]✓ Completed[/green] in {time.time() - t0:.2f}s ({size_mb:.2f} MB)")
                result_paths[table_name] = dest_file
            except Exception as e:
                console.print(f"[red]Failed to convert {src_file.name}: {e}[/red]")

        return result_paths

    def ingest_zip(self, zip_path: Path) -> dict[str, Path]:
        """End-to-end ingestion of a Eurocontrol monthly zip file."""
        match = re.search(r"(\d{6})", zip_path.name)
        period_id = match.group(1) if match else zip_path.stem

        period_dir = self.extract_zip(zip_path)
        classified = self.classify_files(period_dir)
        console.print(f"[bold cyan]Classified {len(classified)} core tables for period {period_id}:[/bold cyan]")
        for k, v in classified.items():
            console.print(f"  • {k:20s} -> {v.name}")

        return self.convert_period_to_parquet(period_id, classified)


def list_available_periods() -> list[str]:
    """List all ingested historical periods available in parquet."""
    periods = []
    # Check baseline 201611
    if (PARQUET_DIR / "flights.parquet").exists():
        periods.append("201611")

    # Check periods folder
    if PERIODS_PARQUET_DIR.exists():
        for p in PERIODS_PARQUET_DIR.iterdir():
            if p.is_dir() and (p / "flights.parquet").exists():
                periods.append(p.name)

    if not periods:
        periods = ["201611"]

    return sorted(list(set(periods)))

