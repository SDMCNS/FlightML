"""
Batch Ingestion Script for Eurocontrol Archives
Converts monthly archives (202402, 202406, 202410, 202502, 202506)
into partitioned, Zstandard-compressed Apache Parquet tables.
"""

from __future__ import annotations

import re
import sys
import time
import zipfile
from pathlib import Path
import duckdb

DATA_DIR = Path("c:/Projects/data/FlightML/data")
PERIODS_DIR = DATA_DIR / "parquet" / "periods"
EXTRACT_TEMP = DATA_DIR / "temp_extract"


def classify_archive_members(namelist: list[str], airac_cycle: str) -> dict[str, str]:
    classified = {}
    for n in namelist:
        name = Path(n).name.lower()
        if name.startswith("flights_") or name == "flights.csv" or name.startswith("flights."):
            classified["flights"] = n
        elif "flight_points_actual" in name or "flights_anonymized" in name:
            classified["flights_actual"] = n
        elif "flight_points_filed" in name or "planned_flights" in name:
            classified["flights_planned"] = n
        elif "flight_firs_actual" in name:
            classified["flight_firs_actual"] = n
        elif "flight_firs_filed" in name:
            classified["flight_firs_filed"] = n
        elif "flight_auas_actual" in name:
            classified["flight_auas_actual"] = n
        elif "flight_auas_filed" in name:
            classified["flight_auas_filed"] = n
        elif name.startswith("fir_") or name == "fir.csv":
            if airac_cycle and airac_cycle in name:
                classified["fir"] = n
            elif "fir" not in classified:
                classified["fir"] = n
        elif name.startswith("route_") or name == "routes.csv":
            if airac_cycle and airac_cycle in name:
                classified["routes"] = n
            elif "routes" not in classified:
                classified["routes"] = n
    return classified


def ingest_archive(zip_path: Path) -> dict[str, Path]:
    period_id = zip_path.stem
    airac = period_id[2:] if len(period_id) == 6 else ""
    out_dir = PERIODS_DIR / period_id
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = EXTRACT_TEMP / period_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==========================================")
    print(f"🚀 Processing Archive: {zip_path.name} (Period: {period_id})")
    print(f"==========================================")

    t_start = time.time()
    with zipfile.ZipFile(zip_path, "r") as archive:
        namelist = archive.namelist()
        classified = classify_archive_members(namelist, airac)
        print(f"Classified {len(classified)} tables: {list(classified.keys())}")

        con = duckdb.connect()

        for table_name, member_name in classified.items():
            dest_pq = out_dir / f"{table_name}.parquet"
            if dest_pq.exists() and dest_pq.stat().st_size > 1000:
                print(f"  ⏭️ {table_name:20s} already exists ({dest_pq.stat().st_size / 1e6:.1f} MB), skipping.")
                continue

            t0 = time.time()
            # Extract single member to temp
            archive.extract(member_name, temp_dir)
            extracted_file = temp_dir / member_name
            ext_mb = extracted_file.stat().st_size / 1e6

            # Stream into Parquet via DuckDB
            src_str = str(extracted_file).replace("\\", "/")
            dest_str = str(dest_pq).replace("\\", "/")
            query = f"""
                COPY (
                    SELECT * FROM read_csv_auto('{src_str}', header=true, ignore_errors=true)
                ) TO '{dest_str}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
            con.execute(query)
            pq_mb = dest_pq.stat().st_size / 1e6
            dt = time.time() - t0
            print(f"  ✓ {table_name:20s}: {ext_mb:6.1f} MB -> {pq_mb:6.1f} MB in {dt:5.2f}s")

            # Remove temp file to keep disk space free
            try:
                extracted_file.unlink()
            except Exception:
                pass

    # Clean up empty temp dir
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    total_time = time.time() - t_start
    print(f"✨ Period {period_id} completed in {total_time:.1f}s")


def main():
    archives = [
        DATA_DIR / "202402.zip",
        DATA_DIR / "202406.zip",
        DATA_DIR / "202410.zip",
        DATA_DIR / "202502.zip",
        DATA_DIR / "202506.zip",
    ]

    EXTRACT_TEMP.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    for arch in archives:
        if not arch.exists():
            print(f"⚠️ Archive not found: {arch}")
            continue
        ingest_archive(arch)

    print("\n==========================================")
    print(f"🎉 All archives ingested successfully in {time.time() - t_all:.1f}s!")
    print(f"==========================================")

    # Clean up top temp dir
    try:
        import shutil
        shutil.rmtree(EXTRACT_TEMP, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
