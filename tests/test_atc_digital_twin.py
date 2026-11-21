from pathlib import Path
import pandas as pd
import pytest
from typer.testing import CliRunner

from flightml.cli import app
from flightml.data.dataset import FlightDataset
from flightml.sim.atc_digital_twin import (
    AirportOperationalState,
    analyze_airport_weather_and_capacity,
    get_airport_fids_board,
    get_or_compute_ideal_flight,
    reconstruct_ideal_flight,
    resolve_optimal_cruise_fl,
)
from flightml.viz.map_builder import create_trajectory_map

runner = CliRunner()


@pytest.fixture(scope="module")
def dataset():
    return FlightDataset(period="202510")


def test_airport_weather_and_capacity_rules(dataset):
    """
    Verify the two fundamental ATC Digital Twin axioms:
    1. movements < 20 / hr -> unconstrained capacity (no weather lookup needed).
    2. movements >= 20 / hr and delayed flights > 5 / hr -> weather-impacted.
    3. movements >= 20 / hr and delayed flights <= 5 / hr -> non-weather procedural inefficiency.
    """
    states = analyze_airport_weather_and_capacity(dataset, period="202510")
    assert len(states) > 0

    found_unconstrained = False
    found_weather = False
    found_non_weather_busy = False

    for st in states.values():
        if st.movements < 20:
            assert st.is_unconstrained is True
            assert st.is_weather_impacted is False
            found_unconstrained = True
        elif st.movements >= 20 and st.delayed_flights > 5:
            assert st.is_unconstrained is False
            assert st.is_weather_impacted is True
            assert st.effective_capacity_rate == st.movements
            found_weather = True
        elif st.movements >= 20 and st.delayed_flights <= 5:
            assert st.is_unconstrained is False
            assert st.is_weather_impacted is False
            found_non_weather_busy = True

    assert found_unconstrained, "Should have airports with <20 movements/hr"
    assert found_weather, "Should have weather-impacted airport hours with >5 delayed flights"
    assert found_non_weather_busy, "Should have non-weather busy airport hours"


def test_resolve_optimal_cruise_fl():
    """Verify optimal cruising flight level rules based on stage distance, aircraft type, and bearing."""
    # Eastbound (Odd FL)
    fl_east = resolve_optimal_cruise_fl(ac_type="A320", distance_nm=600.0, bearing_deg=90.0)
    assert fl_east % 20 != 0 or fl_east % 10 == 0
    assert fl_east in [350, 370, 390]

    # Westbound (Even FL)
    fl_west = resolve_optimal_cruise_fl(ac_type="B738", distance_nm=600.0, bearing_deg=270.0)
    assert fl_west in [360, 380, 400]

    # Short stage length
    fl_short = resolve_optimal_cruise_fl(ac_type="A320", distance_nm=100.0, bearing_deg=90.0)
    assert fl_short <= 200


def test_reconstruct_ideal_flight(dataset):
    """Verify reconstruction of 4D ideal trajectory, holding elimination, and fuel savings."""
    flight_id = int(dataset.query("SELECT ectrl_id FROM flights LIMIT 1").iloc[0]["ectrl_id"])
    metrics, ideal_df = get_or_compute_ideal_flight(dataset, flight_id)

    assert metrics.ectrl_id == flight_id
    assert metrics.ideal_distance_nm <= metrics.actual_distance_nm + 0.1
    assert metrics.holding_duration_ideal_min == 0.0
    assert metrics.fuel_saved_kg >= 0.0
    assert metrics.co2_saved_kg >= 0.0
    assert metrics.total_economic_savings_eur >= 0.0

    # 4D Trajectory DataFrame checks
    assert len(ideal_df) >= 15
    for col in ["period_id", "ectrl_id", "seq_num", "time_over", "flight_level", "latitude", "longitude"]:
        assert col in ideal_df.columns
    assert ideal_df.iloc[0]["flight_level"] == 0
    assert ideal_df.iloc[-1]["flight_level"] == 0
    assert ideal_df["flight_level"].max() == metrics.optimal_cruise_fl


def test_mandatory_sid_and_star_preservation(dataset):
    """
    Verify that the ATC Digital Twin preserves mandatory departure SIDs and arrival STARs:
    1. Initial trajectory points follow the departure SID procedure up to the SID exit fix.
    2. Final trajectory points follow the arrival STAR procedure from the STAR entry fix to the runway.
    3. Only the intermediate en-route segment between SID exit and STAR entry is reconstructed.
    """
    from flightml.sim.atc_digital_twin import extract_mandatory_sid_and_star

    flight_id = int(dataset.query("SELECT ectrl_id FROM flights LIMIT 1").iloc[0]["ectrl_id"])
    flt = dataset.get_flight(flight_id)
    trajs = dataset.get_trajectory(flight_id)
    ref_df = trajs["actual"] if not trajs["actual"].empty else trajs["planned"]

    adep_lat = float(flt.get("adep_lat") or 50.0)
    adep_lon = float(flt.get("adep_lon") or 10.0)
    ades_lat = float(flt.get("ades_lat") or 52.0)
    ades_lon = float(flt.get("ades_lon") or 13.0)
    fl_dist = float(flt.get("flt_plan_distance_flown_nm", 500.0) or 500.0)

    sid_points, star_points, sid_exit, star_entry, sid_dist, star_dist = extract_mandatory_sid_and_star(
        ref_df, adep_lat, adep_lon, ades_lat, ades_lon, fl_dist, 360
    )

    # Must extract non-empty SID and STAR legs
    assert len(sid_points) >= 2
    assert len(star_points) >= 2
    assert sid_dist > 0.0
    assert star_dist > 0.0

    metrics, ideal_df = get_or_compute_ideal_flight(dataset, flight_id)

    # First SID waypoint must match the first ideal waypoint (departure airport/threshold)
    assert ideal_df.iloc[0]["flight_level"] == 0
    assert abs(ideal_df.iloc[0]["latitude"] - sid_points[0]["latitude"]) < 1e-4
    assert abs(ideal_df.iloc[0]["longitude"] - sid_points[0]["longitude"]) < 1e-4

    # Last STAR waypoint must match the last ideal waypoint (arrival airport/threshold)
    assert ideal_df.iloc[-1]["flight_level"] == 0
    assert abs(ideal_df.iloc[-1]["latitude"] - star_points[-1]["latitude"]) < 1e-4
    assert abs(ideal_df.iloc[-1]["longitude"] - star_points[-1]["longitude"]) < 1e-4

    # The SID exit point must be represented at the transition to en-route
    sid_exit_pt = sid_points[-1]
    matching_sid_exit = ideal_df.iloc[len(sid_points) - 1]
    assert abs(matching_sid_exit["latitude"] - sid_exit_pt["latitude"]) < 1e-4
    assert abs(matching_sid_exit["longitude"] - sid_exit_pt["longitude"]) < 1e-4

    # The STAR entry point must match the start of the terminal arrival segment
    star_entry_pt = star_points[0]
    matching_star_entry = ideal_df.iloc[-len(star_points)]
    assert abs(matching_star_entry["latitude"] - star_entry_pt["latitude"]) < 1e-4
    assert abs(matching_star_entry["longitude"] - star_entry_pt["longitude"]) < 1e-4

    # Intermediate waypoints strictly at optimal cruise FL
    enroute_slice = ideal_df.iloc[len(sid_points) : -len(star_points)]
    assert len(enroute_slice) > 0
    for _, row in enroute_slice.iterrows():
        assert row["flight_level"] == metrics.optimal_cruise_fl


def test_create_trajectory_map_with_ideal(dataset):
    """Verify Leaflet map builder incorporates ideal trajectory layer."""
    flight_id = int(dataset.query("SELECT ectrl_id FROM flights LIMIT 1").iloc[0]["ectrl_id"])
    flt = dataset.get_flight(flight_id)
    trajs = dataset.get_trajectory(flight_id)
    _, ideal_df = get_or_compute_ideal_flight(dataset, flight_id)

    m = create_trajectory_map(
        flight_info=flt,
        actual_df=trajs["actual"],
        planned_df=trajs["planned"],
        ideal_df=ideal_df,
    )
    assert m is not None
    html_repr = m.get_root().render()
    assert "ATC Digital Twin: Ideal Trajectory" in html_repr
    assert "#10b981" in html_repr  # Emerald green color


def test_cli_atc_twin_subset():
    """Verify CLI command `flightml atc-twin` executes cleanly on subset."""
    res = runner.invoke(app, ["atc-twin", "--period", "202510", "--max-flights", "3"])
    assert res.exit_code == 0
    assert "FlightML ATC Digital Twin: 4D Trajectory Rebuild" in res.stdout
    assert "ATC Digital Twin Execution Complete" in res.stdout


def test_airport_fids_board(dataset):
    """Verify airport-level arrivals and departures FIDS board generation."""
    # Find a major airport and date with flights in 202510
    row = dataset.query("SELECT adep, CAST(actual_off_block_time AS DATE) as d FROM flights WHERE actual_off_block_time IS NOT NULL LIMIT 1").iloc[0]
    apt = row["adep"]
    d_str = str(row["d"])

    fids = get_airport_fids_board(dataset, airport_icao=apt, target_date=d_str, period="202510", max_flights=10)
    assert "departures" in fids
    assert "arrivals" in fids

    dep_df = fids["departures"]
    assert not dep_df.empty
    expected_cols = [
        "ectrl_id", "flight", "operator", "ac_type", "destination",
        "planned_time", "actual_time", "ideal_time",
        "actual_delay_min", "ideal_delay_min", "delay_saved_min",
        "fuel_saved_kg", "co2_saved_kg", "economic_savings_eur",
        "status", "weather_restricted"
    ]
    for col in expected_cols:
        assert col in dep_df.columns

    # Verify times and delays
    sample_flight = dep_df.iloc[0]
    assert sample_flight["destination"] is not None
    assert sample_flight["actual_delay_min"] is not None
    # For flights with delays, ideal delay should be <= actual delay
    delayed_flights = dep_df[dep_df["actual_delay_min"] > 0]
    for _, r in delayed_flights.iterrows():
        assert r["ideal_delay_min"] <= r["actual_delay_min"] + 0.1
