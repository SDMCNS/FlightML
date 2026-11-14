import pytest
from flightml.data.dataset import FlightDataset
from flightml.sim.gdp_optimizer import optimize_ground_delay_program, GDPResult


@pytest.fixture(scope="module")
def dataset():
    return FlightDataset(period="202510")


def test_gdp_optimizer_with_custom_date(dataset):
    # Test specific regulated date
    target_date = "2025-10-18"
    res = optimize_ground_delay_program(
        dataset,
        target_airport="EGLL",
        program_date=target_date,
        start_hour=8,
        end_hour=14,
        degraded_capacity_rate_per_hr=25,
    )
    assert isinstance(res, GDPResult)
    assert res.airport == "EGLL"
    assert res.program_start.startswith(target_date)
    assert res.degraded_capacity_rate_per_hr == 25
    assert res.total_flights_regulated > 0
    assert res.total_fuel_saved_kg > 0.0
    assert res.total_fuel_cost_saved_eur > 0.0
    assert len(res.slot_assignments) > 0

    s = res.slot_assignments[0]
    assert s.ades == "EGLL"
    assert s.assigned_slot_time >= s.original_arrival_time


def test_gdp_optimizer_auto_date(dataset):
    res = optimize_ground_delay_program(
        dataset,
        target_airport="LFPG",
        start_hour=6,
        end_hour=12,
        degraded_capacity_rate_per_hr=30,
    )
    assert isinstance(res, GDPResult)
    assert res.airport == "LFPG"
    assert res.total_flights_regulated > 0
