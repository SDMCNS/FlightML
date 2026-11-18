import random
from pathlib import Path
import duckdb
import pandas as pd
import pytest

from flightml.data.synthetic_generator import (
    AIRCRAFT_SWAP_POOLS,
    AC_TO_POOL,
    SYNTHETIC_AIRLINE_CODES,
    swap_aircraft_type_like_for_like,
    compute_same_day_time_shift,
    generate_synthetic_demo_dataset,
)
from flightml.data.dataset import FlightDataset


def test_swap_aircraft_type_like_for_like():
    rng = random.Random(42)

    # Narrowbody standard swap
    for _ in range(20):
        swapped = swap_aircraft_type_like_for_like("A320", rng)
        assert swapped in AIRCRAFT_SWAP_POOLS["NARROWBODY_STD"]

    # Widebody heavy swap
    for _ in range(20):
        swapped = swap_aircraft_type_like_for_like("A359", rng)
        assert swapped in AIRCRAFT_SWAP_POOLS["WIDEBODY_HEAVY"]

    # Regional turboprop swap
    for _ in range(20):
        swapped = swap_aircraft_type_like_for_like("AT72", rng)
        assert swapped in AIRCRAFT_SWAP_POOLS["REGIONAL_PROP"]

    # Quad heavy swap
    for _ in range(20):
        swapped = swap_aircraft_type_like_for_like("A388", rng)
        assert swapped in AIRCRAFT_SWAP_POOLS["QUAD_HEAVY"]

    # Heuristic prefix fallback for unmapped prefix
    swapped_prefix = swap_aircraft_type_like_for_like("B738X", rng)
    assert swapped_prefix in AIRCRAFT_SWAP_POOLS["NARROWBODY_STD"]


def test_compute_same_day_time_shift():
    rng = random.Random(123)

    # 1. Early morning flight: 00:10:00 (cannot shift backward more than 10 minutes)
    ts_morning = pd.Timestamp("2025-10-15 00:10:00")
    for _ in range(50):
        shift_sec = compute_same_day_time_shift(ts_morning, rng, max_shift_minutes=45)
        shifted_ts = ts_morning + pd.Timedelta(seconds=shift_sec)
        assert shifted_ts.date() == ts_morning.date()
        assert shift_sec >= -600  # -10 minutes in seconds

    # 2. Late night flight: 23:50:00 (cannot shift forward more than 9m59s)
    ts_night = pd.Timestamp("2025-10-15 23:50:00")
    for _ in range(50):
        shift_sec = compute_same_day_time_shift(ts_night, rng, max_shift_minutes=45)
        shifted_ts = ts_night + pd.Timedelta(seconds=shift_sec)
        assert shifted_ts.date() == ts_night.date()
        assert shift_sec <= 599  # until midnight

    # 3. Midday flight: 14:00:00 (can shift full +/- 45 minutes)
    ts_midday = pd.Timestamp("2025-10-15 14:00:00")
    shifts = [compute_same_day_time_shift(ts_midday, rng, max_shift_minutes=45) for _ in range(100)]
    assert min(shifts) < 0
    assert max(shifts) > 0
    for s in shifts:
        assert abs(s) <= 45 * 60
        assert (ts_midday + pd.Timedelta(seconds=s)).date() == ts_midday.date()


def test_synthetic_demo_generation_end_to_end(tmp_path: Path):
    """
    Test end-to-end generation using a small slice of real source data,
    verifying Parquet schemas, DuckDB compatibility, tail traceability,
    and same-day departure invariant.
    """
    source_periods_dir = Path("c:/Projects/data/FlightML/data/parquet/periods")
    if not (source_periods_dir / "202510" / "flights.parquet").exists():
        pytest.skip("Source period 202510 not available on disk")

    out_period = "pytest_synth_period"
    target_count = 20

    out_dir = generate_synthetic_demo_dataset(
        target_flights=target_count,
        source_periods=["202510"],
        period_name=out_period,
        tail_prefix="SYNTH-",
        max_time_shift_minutes=30,
        seed=999,
        output_dir=tmp_path / out_period,
    )

    assert (out_dir / "flights.parquet").exists()
    assert (out_dir / "flights_actual.parquet").exists()
    assert (out_dir / "flights_planned.parquet").exists()

    con = duckdb.connect()
    fl_df = con.execute(f"SELECT * FROM '{(out_dir / 'flights.parquet').as_posix()}'").df()
    assert len(fl_df) == target_count

    # Check synthetic tail formatting
    valid_tails = fl_df["AC Registration"].dropna()
    for t in valid_tails:
        if t:
            assert t.startswith("SYNTH-")

    # Check synthetic operators
    valid_ops = fl_df["AC Operator"].dropna()
    for op in valid_ops:
        assert op in SYNTHETIC_AIRLINE_CODES

    # Verify trajectory linkage
    act_df = con.execute(f"SELECT * FROM '{(out_dir / 'flights_actual.parquet').as_posix()}'").df()
    assert len(act_df) > 0
    unique_act_ids = set(act_df["ECTRL ID"].unique())
    flight_ids = set(fl_df["ECTRL ID"].unique())
    assert unique_act_ids.issubset(flight_ids)


def test_tail_mapping_determinism_and_traceability():
    """
    Verify that multiple flight legs operated by the same physical aircraft registration
    are deterministically assigned the exact same synthetic tail identifier,
    preserving rotation traceability, tail-swap recovery, and turnaround buffers.
    """
    tail_mapping = {}
    tail_prefix = "SYNTH-"
    tail_counter = 1
    input_tails = ["D-AIXS", "OE-LBA", "D-AIXS", "G-EUPA", "OE-LBA", "D-AIXS"]
    assigned_tails = []
    for reg in input_tails:
        r_str = reg.strip().upper()
        if r_str not in tail_mapping:
            tail_mapping[r_str] = f"{tail_prefix}{tail_counter:05d}"
            tail_counter += 1
        assigned_tails.append(tail_mapping[r_str])

    assert assigned_tails[0] == assigned_tails[2] == assigned_tails[5]  # D-AIXS
    assert assigned_tails[1] == assigned_tails[4]  # OE-LBA
    assert assigned_tails[0] != assigned_tails[1]
    assert assigned_tails[0] == "SYNTH-00001"
    assert assigned_tails[1] == "SYNTH-00002"
    assert assigned_tails[3] == "SYNTH-00003"


def test_operator_mapping_determinism():
    """
    Verify that operator identifiers are deterministically substituted with
    consistent synthetic airline codes across all flights.
    """
    operator_mapping = {}
    input_ops = ["DLH", "BAW", "DLH", "AFR", "BAW"]
    assigned_ops = []
    for op in input_ops:
        op_str = op.strip().upper()
        if op_str not in operator_mapping:
            c_idx = (len(operator_mapping)) % len(SYNTHETIC_AIRLINE_CODES)
            operator_mapping[op_str] = SYNTHETIC_AIRLINE_CODES[c_idx]
        assigned_ops.append(operator_mapping[op_str])

    assert assigned_ops[0] == assigned_ops[2]
    assert assigned_ops[1] == assigned_ops[4]
    assert assigned_ops[0] != assigned_ops[1]
    assert assigned_ops[0] == SYNTHETIC_AIRLINE_CODES[0]
    assert assigned_ops[1] == SYNTHETIC_AIRLINE_CODES[1]
    assert assigned_ops[3] == SYNTHETIC_AIRLINE_CODES[2]

