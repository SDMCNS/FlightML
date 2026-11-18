"""
Synthetic Demo Dataset Generator CLI for FlightML.

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
import sys
from pathlib import Path

# Add project root to sys.path if invoked standalone
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from flightml.data.synthetic_generator import (
    AIRCRAFT_SWAP_POOLS,
    AC_TO_POOL,
    SYNTHETIC_AIRLINE_CODES,
    swap_aircraft_type_like_for_like,
    compute_same_day_time_shift,
    generate_synthetic_demo_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic 1M flight demo dataset for FlightML.")
    parser.add_argument("--count", "-n", type=int, default=1_000_000, help="Number of flights to generate (default: 1,000,000)")
    parser.add_argument("--periods", "-p", type=str, default="202510,202506", help="Comma-separated source periods")
    parser.add_argument("--name", type=str, default="demo_1m", help="Target period name (default: demo_1m)")
    parser.add_argument("--tail-prefix", type=str, default="SYNTH-", help="Prefix for synthetic tail numbers (default: SYNTH-)")
    parser.add_argument("--max-shift", type=int, default=45, help="Max time shift in minutes (default: 45)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    period_list = [p.strip() for p in args.periods.split(",") if p.strip()]

    generate_synthetic_demo_dataset(
        target_flights=args.count,
        source_periods=period_list,
        period_name=args.name,
        tail_prefix=args.tail_prefix,
        max_time_shift_minutes=args.max_shift,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
