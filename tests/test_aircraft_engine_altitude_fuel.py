"""
Unit tests for Aircraft & Engine Combination Performance, Ideal Cruising Altitude,
and Multi-Dimensional Vertical Fuel Optimization.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from flightml.api.app import app
from flightml.data.dataset import FlightDataset
from flightml.eval.aircraft_engine_perf import (
    AircraftEnginePerformance,
    CruiseAltitudeEfficiency,
    compute_altitude_burn_penalty,
    compute_cruise_altitude_efficiency,
    resolve_aircraft_engine_perf,
)
from flightml.eval.fuel_carbon import (
    EnvironmentalImpact,
    compute_flight_fuel_and_emissions,
)

client = TestClient(app)


@pytest.fixture(scope="module")
def dataset() -> FlightDataset:
    return FlightDataset(period="201611")


def test_resolve_aircraft_engine_perf_types(dataset: FlightDataset):
    # 1. A320 narrowbody with default engine
    a320_perf = resolve_aircraft_engine_perf(dataset, "A320")
    assert a320_perf.ac_type == "A320"
    assert len(a320_perf.engine_options) >= 2
    assert a320_perf.bypass_ratio > 4.0
    assert a320_perf.pressure_ratio > 20.0
    assert 340.0 <= a320_perf.optimal_cruise_fl <= 390.0
    assert a320_perf.nominal_fuel_flow_kg_hr > 1800.0

    # 2. A320 with explicit engine override (V2527-A5)
    a320_v25 = resolve_aircraft_engine_perf(dataset, "A320", engine_override="V2527-A5")
    assert a320_v25.selected_engine == "V2527-A5"
    assert a320_v25.is_engine_override is True

    # 3. B77W widebody
    b77w_perf = resolve_aircraft_engine_perf(dataset, "B77W")
    assert b77w_perf.ac_type == "B77W"
    assert "GE90" in b77w_perf.selected_engine
    assert b77w_perf.pressure_ratio > 35.0
    assert b77w_perf.nominal_fuel_flow_kg_hr > 5000.0

    # 4. Turboprop (AT72)
    at72_perf = resolve_aircraft_engine_perf(dataset, "AT72")
    assert at72_perf.ac_type == "AT72"
    assert at72_perf.engine_type_class == "Turboprop"
    assert at72_perf.optimal_cruise_fl == 200.0
    assert at72_perf.operational_ceiling_fl == 250.0
    assert at72_perf.nominal_fuel_flow_kg_hr < 800.0


def test_altitude_burn_penalty_physics():
    opt_fl = 360.0
    pr = 30.0

    # At optimal altitude: 0% penalty
    assert compute_altitude_burn_penalty(360.0, opt_fl, pr) == 0.0

    # 2,000 ft below optimal (FL340 vs FL360) -> ~3% penalty
    pen_low = compute_altitude_burn_penalty(340.0, opt_fl, pr)
    assert 0.025 <= pen_low <= 0.040

    # 5,000 ft below optimal (FL310 vs FL360) -> ~7.5% penalty
    pen_vlow = compute_altitude_burn_penalty(310.0, opt_fl, pr)
    assert pen_vlow > pen_low
    assert 0.065 <= pen_vlow <= 0.100

    # Above optimal (FL380 vs FL360) -> induced drag penalty
    pen_high = compute_altitude_burn_penalty(380.0, opt_fl, pr)
    assert pen_high > 0.030


def test_flight_cruise_altitude_efficiency(dataset: FlightDataset):
    flight_id = 202007026
    eff = compute_cruise_altitude_efficiency(dataset, flight_id, fuel_price_per_kg=0.78)

    assert eff.ectrl_id == flight_id
    assert eff.optimal_cruise_fl > 0.0
    assert eff.requested_fl > 0.0
    assert eff.actual_cruise_fl > 0.0
    assert eff.cruise_hours > 0.0
    assert eff.base_cruise_fuel_kg > 0.0

    # Decomposition
    assert hasattr(eff, "planning_excess_fuel_kg")
    assert hasattr(eff, "atc_tactical_excess_fuel_kg")
    assert hasattr(eff, "potential_fuel_savings_kg")
    assert hasattr(eff, "potential_fuel_savings_eur")
    assert eff.potential_fuel_savings_pct >= 0.0


def test_compute_flight_fuel_and_emissions_multi_dimensional(dataset: FlightDataset):
    flight_id = 202007026
    impact = compute_flight_fuel_and_emissions(dataset, flight_id)

    # Core existing metrics preserved
    assert impact.ectrl_id == flight_id
    assert impact.actual_distance_nm > 0.0
    assert impact.total_excess_fuel_kg > 0.0
    assert impact.estimated_excess_fuel_cost_eur > 0.0
    assert impact.excess_co2_kg == pytest.approx(impact.total_excess_fuel_kg * 3.16, rel=1e-3)

    # Extended multi-dimensional engine & altitude metrics
    assert impact.optimal_cruise_fl > 0.0
    assert impact.requested_fl > 0.0
    assert impact.actual_cruise_fl > 0.0
    assert impact.selected_engine != ""
    assert isinstance(impact.engine_options, list)
    assert impact.excess_altitude_fuel_kg >= 0.0

    # Total excess fuel reflects holding + routing + altitude
    sum_components = (
        impact.excess_holding_fuel_kg
        + impact.excess_routing_fuel_kg
        + impact.excess_altitude_fuel_kg
    )
    assert impact.total_excess_fuel_kg == pytest.approx(sum_components, abs=0.1)

    # Test dictionary export
    d = impact.to_dict()
    assert "optimal_cruise_fl" in d
    assert "requested_fl" in d
    assert "actual_cruise_fl" in d
    assert "excess_altitude_fuel_kg" in d
    assert "potential_fuel_savings_kg" in d
    assert "potential_fuel_savings_eur" in d


def test_api_fuel_with_engine_override():
    flight_id = 202007026
    # 1. Standard fuel endpoint
    r1 = client.get(f"/api/v1/flights/{flight_id}/fuel")
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["optimal_cruise_fl"] > 0
    assert "selected_engine" in data1
    assert "excess_altitude_fuel_kg" in data1

    # 2. Fuel endpoint with engine override
    r2 = client.get(f"/api/v1/flights/{flight_id}/fuel?engine_override=PW4056")
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["selected_engine"] == "PW4056"
    assert data2["is_engine_override"] is True

    # 3. Engines metadata endpoint
    r3 = client.get(f"/api/v1/flights/{flight_id}/engines")
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3["ac_type"] == "B763"
    assert len(data3["engine_options"]) >= 1
    assert data3["optimal_cruise_fl"] > 0

