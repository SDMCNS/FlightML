import time
from pathlib import Path
import duckdb
from rich.console import Console

from flightml.config import DATA_DIR, PARQUET_DIR

console = Console()

FILES_TO_CONVERT = [
    ("airports.csv", "airports.parquet"),
    ("runways.csv", "runways.parquet"),
    ("airport-frequencies.csv", "airport_frequencies.parquet"),
    ("navaids.csv", "navaids.parquet"),
    ("FIR.csv", "fir.parquet"),
    ("Routes.csv", "routes.parquet"),
    ("Flights.csv", "flights.parquet"),
]

LARGE_TRAJECTORY_FILES = [
    ("flights_anonymized.csv", "flights_actual.parquet"),
    ("planned_flights_anonymized.csv", "flights_planned.parquet"),
]


def convert_csv_to_parquet(
    source_csv: Path,
    target_parquet: Path,
    con: duckdb.DuckDBPyConnection | None = None,
) -> None:
    if con is None:
        con = duckdb.connect()

    target_parquet.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    console.print(f"[cyan]Converting[/cyan] {source_csv.name} -> {target_parquet.name}...")

    # Normalize backslashes for DuckDB SQL
    src_str = str(source_csv).replace("\\", "/")
    dst_str = str(target_parquet).replace("\\", "/")

    query = f"""
        COPY (
            SELECT * FROM read_csv_auto('{src_str}', header=true, ignore_errors=true)
        ) TO '{dst_str}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)
    elapsed = time.time() - t0
    size_mb = target_parquet.stat().st_size / (1024 * 1024)
    console.print(f"[green]✓ Done[/green] in {elapsed:.2f}s ({size_mb:.2f} MB)")


def convert_all(include_trajectories: bool = False) -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    for csv_name, parquet_name in FILES_TO_CONVERT:
        src = DATA_DIR / csv_name
        dst = PARQUET_DIR / parquet_name
        if src.exists():
            convert_csv_to_parquet(src, dst, con)

    if include_trajectories:
        for csv_name, parquet_name in LARGE_TRAJECTORY_FILES:
            src = DATA_DIR / csv_name
            dst = PARQUET_DIR / parquet_name
            if src.exists():
                convert_csv_to_parquet(src, dst, con)


if __name__ == "__main__":
    convert_all(include_trajectories=False)

