"""
Synthetic Demo Dataset Generator for FlightML.

Creates an anonymized, production-grade synthetic dataset of 1,000,000 flights (or custom count)
derived from empirical Eurocontrol DDR2 flight archives to showcase the full FlightML platform:
1. Real flight trajectories (lat/lon, flight level, sequence number) preserved for authentic routing.
2. Aircraft types randomized within physically authentic like-for-like swap categories.
3. Operator identifiers substituted with realistic synthetic airline codes.
4. Aircraft tail registrations deterministically pseudonymized (e.g., SYNTH-00042) to preserve
   multi-leg rotation traceability, turnaround buffers, and tail-swap recovery optimization.
5. Flight schedules and waypoint timestamps shifted by controlled random offsets while guaranteed
   to remain on the same calendar operating day.
6. Exported as partitioned Apache Parquet tables directly usable by FlightDataset(period='demo_1m').
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

DATA_DIR = Path("c:/Projects/data/FlightML/data")
PERIODS_DIR = DATA_DIR / "parquet" / "periods"

# Physically sound "like-for-like" aircraft swap clusters
AIRCRAFT_SWAP_POOLS: dict[str, list[str]] = {
    # Narrowbody mainline jets (Standard 150-189 pax)
    "NARROWBODY_STD": ["A320", "B738", "A20N", "B38M", "A319", "B737"],
    # Narrowbody mainline jets (High-capacity / Stretched 200-240 pax)
    "NARROWBODY_LONG": ["A321", "B739", "A21N", "B39M", "B752"],
    # Regional mainline / CSeries / E-Jets (Heavy 100-140 pax)
    "REGIONAL_JET_HEAVY": ["A220", "BCS1", "BCS3", "E190", "E195", "E290", "E295", "CRJ9", "CRJX"],
    # Small regional feeder jets (50-88 pax)
    "REGIONAL_JET_LIGHT": ["E170", "E175", "CRJ1", "CRJ2", "CRJ7", "ERJ145", "E135", "E145"],
    # Regional turboprops (30-78 pax)
    "REGIONAL_PROP": ["AT72", "AT76", "DH8D", "AT45", "AT46", "DH8C", "SF34", "B190"],
    # Widebody twin standard (mid-haul / long-haul 250-330 pax)
    "WIDEBODY_STD": ["A332", "A333", "A339", "B788", "B789", "B78X", "B763", "B764"],
    # Widebody heavy twin (ultra long-haul 350-420 pax)
    "WIDEBODY_HEAVY": ["A359", "A35K", "B772", "B77W", "B77L", "A343"],
    # Very large quad (high-density / flagship 450-600 pax)
    "QUAD_HEAVY": ["A388", "B744", "B748", "A346"],
    # Business & light general aviation
    "BIZJET": ["C56X", "C680", "CL30", "CL60", "GL5T", "GLF5", "GLF6", "FA7X", "E55P", "PC12", "BE20", "LJ45"],
}

# Reverse lookup: ac_type -> pool_name
AC_TO_POOL: dict[str, str] = {}
for pool_name, ac_list in AIRCRAFT_SWAP_POOLS.items():
    for ac in ac_list:
        AC_TO_POOL[ac] = pool_name

# Fictional / synthetic 3-letter airline ICAO codes for demo environments
SYNTHETIC_AIRLINE_CODES: list[str] = [
    "AEX", "AUR", "NOV", "SKY", "VEX", "SOL", "MER", "ECL", "ZEN", "NEX",
    "VAL", "PAC", "TRN", "STR", "BLU", "ALX", "FLX", "SVI", "JET", "CIR",
    "KNT", "PRM", "ORB", "VNT", "CRN", "AST", "PLX", "ARG", "VTA", "BLZ",
    "AER", "QNT", "AVN", "TAL", "WST", "GLO", "RHO", "CYG", "LYR", "PEG",
    "HYP", "DRA", "VAP", "SMR", "BZN", "ELT", "CAP", "ATX", "NEB", "LUM",
    "SYN01", "SYN02", "SYN03", "SYN04", "SYN05", "SYN06", "SYN07", "SYN08",
]


def swap_aircraft_type_like_for_like(original_ac: str | None, rng: random.Random) -> str:
    """Randomize an aircraft type within its certified aerodynamic and operational peer group."""
    if not original_ac or pd.isna(original_ac):
        return "A320"
    ac_clean = str(original_ac).strip().upper()
    pool_name = AC_TO_POOL.get(ac_clean)
    if not pool_name:
        # Heuristic fallback matching for variations
        if ac_clean.startswith("A32") or ac_clean.startswith("B73"):
            pool_name = "NARROWBODY_STD"
        elif ac_clean.startswith("A33") or ac_clean.startswith("B78") or ac_clean.startswith("B76"):
            pool_name = "WIDEBODY_STD"
        elif ac_clean.startswith("A35") or ac_clean.startswith("B77"):
            pool_name = "WIDEBODY_HEAVY"
        elif ac_clean.startswith("E1") or ac_clean.startswith("CRJ"):
            pool_name = "REGIONAL_JET_HEAVY"
        elif ac_clean.startswith("AT") or ac_clean.startswith("DH"):
            pool_name = "REGIONAL_PROP"
        else:
            return ac_clean

    candidate_pool = AIRCRAFT_SWAP_POOLS[pool_name]
    return rng.choice(candidate_pool)


def compute_same_day_time_shift(
    off_block_ts: pd.Timestamp | None,
    rng: random.Random,
    max_shift_minutes: int = 45,
) -> int:
    """
    Compute a random time shift (in seconds) that is guaranteed to keep the flight
    departure on the exact same calendar operating date.
    """
    if off_block_ts is None or pd.isna(off_block_ts):
        return rng.randint(-max_shift_minutes, max_shift_minutes) * 60

    dt = pd.to_datetime(off_block_ts)
    # Boundaries of the calendar day
    sec_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
    sec_until_midnight = 86399 - sec_since_midnight

    max_backward_sec = -min(sec_since_midnight, max_shift_minutes * 60)
    max_forward_sec = min(sec_until_midnight, max_shift_minutes * 60)

    if max_backward_sec >= max_forward_sec:
        return 0

    return rng.randint(int(max_backward_sec), int(max_forward_sec))


def generate_synthetic_demo_dataset(
    target_flights: int = 1_000_000,
    source_periods: list[str] | None = None,
    period_name: str = "demo_1m",
    tail_prefix: str = "SYNTH-",
    max_time_shift_minutes: int = 45,
    seed: int = 42,
    periods_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """
    Generate the complete synthetic demo dataset directly into the FlightML parquet archive structure.
    """
    p_dir = Path(periods_dir or PERIODS_DIR)
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = p_dir / period_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    np.random.seed(seed)

    if source_periods is None:
        source_periods = ["202510", "202506", "202410"]

    console.print(
        Panel.fit(
            f"[bold cyan]🚀 FlightML Synthetic Demo Dataset Generator[/bold cyan]\n"
            f"[dim]Target: {target_flights:,} flights | Output Period: '{period_name}'[/dim]\n"
            f"[yellow]Source Archives: {', '.join(source_periods)}[/yellow]\n"
            f"[green]Tail Prefix: '{tail_prefix}XXXXX' | Time Shift: ±{max_time_shift_minutes}m (same-day guaranteed)[/green]",
            border_style="cyan",
        )
    )

    con = duckdb.connect()

    # 1. Discover available flights across source periods
    available_period_counts: dict[str, int] = {}
    for p in source_periods:
        p_path = p_dir / p / "flights.parquet"
        if p_path.exists():
            cnt = con.execute(f"SELECT count(*) FROM '{str(p_path).replace(chr(92), '/')}'").fetchone()[0]
            available_period_counts[p] = cnt

    total_avail = sum(available_period_counts.values())
    if total_avail == 0:
        raise FileNotFoundError(f"No valid source flight tables found in {p_dir} for periods {source_periods}")

    console.print(f"Discovered [bold green]{total_avail:,}[/bold green] candidate flights across {len(available_period_counts)} periods.")

    # 2. Sample flights from each period proportionally or sequentially to reach target_flights
    flights_sampled_by_period: dict[str, int] = {}
    remaining_needed = target_flights
    for p, cnt in available_period_counts.items():
        take = min(cnt, remaining_needed)
        flights_sampled_by_period[p] = take
        remaining_needed -= take
        if remaining_needed <= 0:
            break

    # 3. Build flight-level transformation mapping
    console.print("\n[cyan]Step 1/4:[/cyan] Extracting flight metadata & generating deterministic anonymization maps...")
    flight_maps: list[pd.DataFrame] = []
    tail_mapping: dict[str, str] = {}
    operator_mapping: dict[str, str] = {}
    tail_counter = 1
    next_synthetic_id = 900000001

    t0_map = time.time()
    for p, n_take in flights_sampled_by_period.items():
        p_fl_path = str(p_dir / p / "flights.parquet").replace("\\", "/")
        f_sub = con.execute(f"""
            SELECT
                "ECTRL ID" AS old_ectrl_id,
                "ACTUAL OFF BLOCK TIME" AS aobt,
                "AC Type" AS ac_type,
                "AC Operator" AS ac_operator,
                "AC Registration" AS ac_reg
            FROM '{p_fl_path}'
            LIMIT {n_take}
        """).df()

        # Generate deterministic synthetic tails
        new_tails = []
        for reg in f_sub["ac_reg"]:
            if pd.isna(reg) or not str(reg).strip():
                new_tails.append("")
            else:
                r_str = str(reg).strip().upper()
                if r_str not in tail_mapping:
                    tail_mapping[r_str] = f"{tail_prefix}{tail_counter:05d}"
                    tail_counter += 1
                new_tails.append(tail_mapping[r_str])

        # Generate deterministic synthetic operators
        new_ops = []
        for op in f_sub["ac_operator"]:
            if pd.isna(op) or not str(op).strip():
                new_ops.append("AEX")
            else:
                op_str = str(op).strip().upper()
                if op_str not in operator_mapping:
                    c_idx = (len(operator_mapping)) % len(SYNTHETIC_AIRLINE_CODES)
                    operator_mapping[op_str] = SYNTHETIC_AIRLINE_CODES[c_idx]
                new_ops.append(operator_mapping[op_str])

        # Generate like-for-like aircraft types
        new_types = [swap_aircraft_type_like_for_like(ac, rng) for ac in f_sub["ac_type"]]

        # Generate same-day time shifts
        shifts = [compute_same_day_time_shift(t, rng, max_time_shift_minutes) for t in f_sub["aobt"]]

        # Generate unique sequential synthetic ECTRL IDs
        n_rows = len(f_sub)
        new_ids = list(range(next_synthetic_id, next_synthetic_id + n_rows))
        next_synthetic_id += n_rows

        m_df = pd.DataFrame({
            "source_period": p,
            "old_ectrl_id": f_sub["old_ectrl_id"],
            "new_ectrl_id": new_ids,
            "time_shift_sec": shifts,
            "new_ac_type": new_types,
            "new_operator": new_ops,
            "new_registration": new_tails,
        })
        flight_maps.append(m_df)

    full_map_df = pd.concat(flight_maps, ignore_index=True)
    con.register("flight_map_all", full_map_df)

    console.print(
        f"✓ Created mapping for [bold cyan]{len(full_map_df):,}[/bold cyan] flights in {time.time() - t0_map:.2f}s "
        f"({len(tail_mapping):,} unique traceable tails, {len(operator_mapping)} operators pseudonymized)."
    )

    # 4. Stream and generate flights.parquet
    console.print("\n[cyan]Step 2/4:[/cyan] Streaming transformed flights into [bold]flights.parquet[/bold]...")
    t0_fl = time.time()
    flights_queries = []
    for p, n_take in flights_sampled_by_period.items():
        p_fl_path = str(p_dir / p / "flights.parquet").replace("\\", "/")
        q = f"""
            SELECT
                m.new_ectrl_id AS "ECTRL ID",
                f.ADEP,
                f."ADEP Latitude",
                f."ADEP Longitude",
                f.ADES,
                f."ADES Latitude",
                f."ADES Longitude",
                f."FILED OFF BLOCK TIME" + (m.time_shift_sec * INTERVAL '1 second') AS "FILED OFF BLOCK TIME",
                f."FILED ARRIVAL TIME" + (m.time_shift_sec * INTERVAL '1 second') AS "FILED ARRIVAL TIME",
                f."ACTUAL OFF BLOCK TIME" + (m.time_shift_sec * INTERVAL '1 second') AS "ACTUAL OFF BLOCK TIME",
                f."ACTUAL ARRIVAL TIME" + (m.time_shift_sec * INTERVAL '1 second') AS "ACTUAL ARRIVAL TIME",
                m.new_ac_type AS "AC Type",
                m.new_operator AS "AC Operator",
                m.new_registration AS "AC Registration",
                f."ICAO Flight Type",
                f."STATFOR Market Segment",
                f."Requested FL",
                f."Actual Distance Flown (nm)"
            FROM '{p_fl_path}' f
            JOIN flight_map_all m
              ON f."ECTRL ID" = m.old_ectrl_id AND m.source_period = '{p}'
        """
        flights_queries.append(q)

    union_flights_sql = " UNION ALL ".join(flights_queries)
    out_flights_pq = str(out_dir / "flights.parquet").replace("\\", "/")
    con.execute(f"COPY ({union_flights_sql}) TO '{out_flights_pq}' (FORMAT PARQUET, COMPRESSION ZSTD);")
    fl_size_mb = (out_dir / "flights.parquet").stat().st_size / (1024 * 1024)
    console.print(f"✓ Generated [bold green]flights.parquet[/bold green] ({len(full_map_df):,} rows, {fl_size_mb:.1f} MB) in {time.time() - t0_fl:.2f}s.")

    # 5. Stream and generate flights_actual.parquet
    console.print("\n[cyan]Step 3/4:[/cyan] Streaming radar trajectories into [bold]flights_actual.parquet[/bold]...")
    t0_act = time.time()
    act_queries = []
    for p in flights_sampled_by_period:
        p_act_path = p_dir / p / "flights_actual.parquet"
        if p_act_path.exists():
            p_act_str = str(p_act_path).replace("\\", "/")
            q = f"""
                SELECT
                    m.new_ectrl_id AS "ECTRL ID",
                    t."Sequence Number",
                    t."Time Over" + (m.time_shift_sec * INTERVAL '1 second') AS "Time Over",
                    t."Flight Level",
                    t.Latitude,
                    t.Longitude
                FROM '{p_act_str}' t
                JOIN flight_map_all m
                  ON t."ECTRL ID" = m.old_ectrl_id AND m.source_period = '{p}'
            """
            act_queries.append(q)

    out_act_pq = str(out_dir / "flights_actual.parquet").replace("\\", "/")
    con.execute(f"COPY ({' UNION ALL '.join(act_queries)}) TO '{out_act_pq}' (FORMAT PARQUET, COMPRESSION ZSTD);")
    act_rows = con.execute(f"SELECT count(*) FROM '{out_act_pq}'").fetchone()[0]
    act_size_mb = (out_dir / "flights_actual.parquet").stat().st_size / (1024 * 1024)
    console.print(f"✓ Generated [bold green]flights_actual.parquet[/bold green] ({act_rows:,} waypoints, {act_size_mb:.1f} MB) in {time.time() - t0_act:.2f}s.")

    # 6. Stream and generate flights_planned.parquet
    console.print("\n[cyan]Step 4/4:[/cyan] Streaming filed plans into [bold]flights_planned.parquet[/bold]...")
    t0_pln = time.time()
    pln_queries = []
    for p in flights_sampled_by_period:
        p_pln_path = p_dir / p / "flights_planned.parquet"
        if p_pln_path.exists():
            p_pln_str = str(p_pln_path).replace("\\", "/")
            q = f"""
                SELECT
                    m.new_ectrl_id AS "ECTRL ID",
                    t."Sequence Number",
                    t."Time Over" + (m.time_shift_sec * INTERVAL '1 second') AS "Time Over",
                    t."Flight Level",
                    t.Latitude,
                    t.Longitude
                FROM '{p_pln_str}' t
                JOIN flight_map_all m
                  ON t."ECTRL ID" = m.old_ectrl_id AND m.source_period = '{p}'
            """
            pln_queries.append(q)

    out_pln_pq = str(out_dir / "flights_planned.parquet").replace("\\", "/")
    con.execute(f"COPY ({' UNION ALL '.join(pln_queries)}) TO '{out_pln_pq}' (FORMAT PARQUET, COMPRESSION ZSTD);")
    pln_rows = con.execute(f"SELECT count(*) FROM '{out_pln_pq}'").fetchone()[0]
    pln_size_mb = (out_dir / "flights_planned.parquet").stat().st_size / (1024 * 1024)
    console.print(f"✓ Generated [bold green]flights_planned.parquet[/bold green] ({pln_rows:,} waypoints, {pln_size_mb:.1f} MB) in {time.time() - t0_pln:.2f}s.")

    # 7. Optional FIR traversals and reference copy
    primary_src = source_periods[0]
    for aux_name in ["flight_firs_actual", "flight_firs_filed"]:
        aux_src = p_dir / primary_src / f"{aux_name}.parquet"
        if aux_src.exists():
            try:
                aux_dst = str(out_dir / f"{aux_name}.parquet").replace("\\", "/")
                con.execute(f"""
                    COPY (
                        SELECT
                            m.new_ectrl_id AS "ECTRL ID",
                            t."Sequence Number",
                            t."FIR ID",
                            t."Entry Time" + (m.time_shift_sec * INTERVAL '1 second') AS "Entry Time",
                            t."Exit Time" + (m.time_shift_sec * INTERVAL '1 second') AS "Exit Time"
                        FROM '{str(aux_src).replace(chr(92), '/')}' t
                        JOIN flight_map_all m
                          ON t."ECTRL ID" = m.old_ectrl_id AND m.source_period = '{primary_src}'
                    ) TO '{aux_dst}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)
            except Exception as e:
                console.print(f"[dim]Note: Skipping {aux_name} ({e})[/dim]")

    # Copy static reference tables (fir.parquet, routes.parquet)
    for ref_table in ["fir.parquet", "routes.parquet"]:
        src_ref = p_dir / primary_src / ref_table
        dst_ref = out_dir / ref_table
        if src_ref.exists() and not dst_ref.exists():
            shutil.copy2(src_ref, dst_ref)

    # 8. Verification and Inspection
    sample_preview = con.execute(f"""
        SELECT
            "ECTRL ID" AS fid,
            ADEP || '➔' || ADES AS route,
            "AC Operator" AS op,
            "AC Type" AS ac,
            "AC Registration" AS tail,
            "ACTUAL OFF BLOCK TIME" AS off_block,
            "ACTUAL ARRIVAL TIME" AS arr
        FROM '{out_flights_pq}'
        LIMIT 5
    """).df()

    preview_table = Table(title="Synthetic Demo Flights Preview (Sample Records)", header_style="bold cyan")
    preview_table.add_column("Flight ID", style="bold white")
    preview_table.add_column("Route", style="yellow")
    preview_table.add_column("Operator", style="cyan")
    preview_table.add_column("A/C Type", style="green")
    preview_table.add_column("Tail Number", style="magenta")
    preview_table.add_column("Off-Block (Shifted)", style="white")
    preview_table.add_column("Arrival (Shifted)", style="white")

    for _, r in sample_preview.iterrows():
        preview_table.add_row(
            str(r["fid"]),
            str(r["route"]),
            str(r["op"]),
            str(r["ac"]),
            str(r["tail"]),
            str(r["off_block"]),
            str(r["arr"]),
        )
    console.print(preview_table)

    console.print(
        Panel.fit(
            f"[bold green]✓ Demo Dataset Generation Complete![/bold green]\n"
            f"[white]Output Directory:[/white] [cyan]{out_dir}[/cyan]\n"
            f"[white]Use in FlightML:[/white] [yellow]FlightDataset(period='{period_name}')[/yellow]\n"
            f"[white]CLI Usage:[/white] [dim]flightml flight-kpis --period {period_name}[/dim]\n"
            f"[white]Dashboard Usage:[/white] [dim]Select '{period_name}' in Sidebar Period Dropdown[/dim]",
            border_style="green",
        )
    )

    return out_dir
