from pathlib import Path
import pandas as pd
import pytest
from typer.testing import CliRunner

from flightml.cli import app
from flightml.data.dataset import FlightDataset
from flightml.sim.atc_digital_twin import (
    AirportOperationalState,
    analyze_airport_weather_and_capacity,
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
    assert len(ideal_df) == 21
    for col in ["period_id", "ectrl_id", "seq_num", "time_over", "flight_level", "latitude", "longitude"]:
        assert col in ideal_df.columns
    assert ideal_df.iloc[0]["flight_level"] == 0
    assert ideal_df.iloc[-1]["flight_level"] == 0
    assert ideal_df["flight_level"].max() == metrics.optimal_cruise_fl


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
