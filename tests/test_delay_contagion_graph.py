import pytest
from flightml.models.delay_contagion_graph import (
    DelayContagionGraphEngine,
    AirframeDailyRotation,
    DelaySuperSpreaderFlight,
    DisruptionSimulationResult,
)
from flightml.data.dataset import FlightDataset


@pytest.fixture(scope="module")
def dataset():
    return FlightDataset("202510")


@pytest.fixture(scope="module")
def engine(dataset):
    return DelayContagionGraphEngine(dataset)


def test_reconstruct_airframe_rotations(engine):
    rotations = engine.reconstruct_airframe_rotations(
        date_str="2025-10-15",
        min_legs=3,
        limit_rotations=10,
    )
    assert len(rotations) > 0
    r0 = rotations[0]
    assert isinstance(r0, AirframeDailyRotation)
    assert r0.total_legs >= 3
    assert len(r0.legs) == r0.total_legs
    assert r0.contagion_multiplier >= 0.0

    d = r0.to_dict()
    assert d["ac_reg"] == r0.ac_reg
    assert len(d["legs"]) == r0.total_legs


def test_identify_super_spreaders(engine):
    spreaders = engine.identify_super_spreaders(
        date_str="2025-10-15",
        min_primary_delay_min=15.0,
        top_n=5,
    )
    assert isinstance(spreaders, list)
    if spreaders:
        s0 = spreaders[0]
        assert isinstance(s0, DelaySuperSpreaderFlight)
        assert s0.primary_delay_min >= 15.0
        assert s0.infected_downstream_legs >= 1
        assert s0.downstream_delay_hours > 0
        assert s0.downstream_economic_cost_eur > 0
        assert len(s0.infected_airports) > 0

        d = s0.to_dict()
        assert d["ectrl_id"] == s0.ectrl_id
        assert "downstream_economic_cost_eur" in d


def test_simulate_network_shock_airport(engine):
    shock = engine.simulate_network_shock(
        disruption_target="EGLL",
        target_type="AIRPORT",
        duration_hours=2.0,
        capacity_cut_pct=50.0,
        date_str="2025-10-15",
    )
    assert isinstance(shock, DisruptionSimulationResult)
    assert shock.target_id == "EGLL"
    assert shock.target_type == "AIRPORT"
    assert shock.directly_delayed_flights > 0
    assert shock.reactionary_infected_flights > shock.directly_delayed_flights
    assert shock.total_impacted_flights == shock.directly_delayed_flights + shock.reactionary_infected_flights
    assert shock.total_delay_hours_incurred > 0
    assert shock.total_economic_delay_cost_eur > 0
    assert shock.estimated_passenger_misconnections > 0
    assert len(shock.affected_downstream_airports) > 0

    d = shock.to_dict()
    assert d["target_id"] == "EGLL"
    assert "total_economic_delay_cost_eur" in d


def test_simulate_network_shock_fir(engine):
    shock = engine.simulate_network_shock(
        disruption_target="EDUUUIR",
        target_type="FIR",
        duration_hours=1.5,
        capacity_cut_pct=40.0,
        date_str="2025-10-15",
    )
    assert isinstance(shock, DisruptionSimulationResult)
    assert shock.target_id == "EDUUUIR"
    assert shock.target_type == "FIR"
    assert shock.total_impacted_flights > 0
