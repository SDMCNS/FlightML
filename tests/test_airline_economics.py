import pytest
from flightml.eval.airline_economics import (
    AirlineEconomicsEngine,
    DirectOperatingCostProfile,
    CorridorAvoidanceSummary,
    HarmonizedPolicySimulation,
)
from flightml.data.dataset import FlightDataset


@pytest.fixture(scope="module")
def dataset():
    return FlightDataset("202510")


@pytest.fixture(scope="module")
def engine(dataset):
    return AirlineEconomicsEngine(dataset)


def test_evaluate_flight_doc(engine, dataset):
    sample_fids = dataset.query(
        "SELECT ectrl_id FROM flights WHERE actual_distance_nm > 400 LIMIT 1"
    )["ectrl_id"].tolist()
    assert len(sample_fids) > 0
    fid = sample_fids[0]

    doc = engine.evaluate_flight_doc(fid)
    assert isinstance(doc, DirectOperatingCostProfile)
    assert doc.ectrl_id == fid
    assert doc.distance_nm > 400.0
    assert doc.fuel_cost_eur > 0
    assert doc.time_crew_cost_eur > 0
    assert doc.crco_navigation_charges_eur > 0
    assert doc.total_doc_eur == round(
        doc.fuel_cost_eur + doc.time_crew_cost_eur + doc.crco_navigation_charges_eur, 2
    )
    assert 99.0 <= (doc.fuel_cost_share_pct + doc.time_cost_share_pct + doc.crco_cost_share_pct) <= 101.0
    assert doc.avoidance_incentive_ratio > 0

    d = doc.to_dict()
    assert d["ectrl_id"] == fid
    assert "total_doc_eur" in d


def test_scan_network_route_charge_avoidance(engine):
    avoid_list = engine.scan_network_route_charge_avoidance(
        min_corridor_flights=50,
        detour_distance_threshold_nm=15.0,
        fee_savings_threshold_eur=25.0,
        top_n=5,
    )
    assert isinstance(avoid_list, list)
    if avoid_list:
        c0 = avoid_list[0]
        assert isinstance(c0, CorridorAvoidanceSummary)
        assert c0.mean_extra_distance_nm >= 15.0
        assert c0.mean_crco_savings_eur >= 25.0
        assert c0.mean_extra_co2_kg > 0
        d = c0.to_dict()
        assert "route" in d
        assert "mean_extra_distance_nm" in d


def test_simulate_harmonized_eurocontrol_rate(engine):
    sim = engine.simulate_harmonized_eurocontrol_rate(
        harmonized_rate_eur=72.50,
        sample_size=20,
    )
    assert isinstance(sim, HarmonizedPolicySimulation)
    assert sim.period == engine.period
    assert sim.harmonized_unit_rate_eur == 72.50
    assert sim.total_flights_simulated == 20
    assert sim.current_total_crco_charges_eur > 0
    assert sim.simulated_harmonized_crco_charges_eur > 0
    assert sim.estimated_annual_fuel_saved_tonnes > 0
    assert sim.estimated_annual_co2_saved_tonnes > 0
    assert sim.estimated_annual_airline_cost_saved_eur > 0
    assert len(sim.airspace_traffic_shifts) > 0

    d = sim.to_dict()
    assert d["period"] == engine.period
    assert "estimated_annual_co2_saved_tonnes" in d
