from __future__ import annotations

import numpy as np
import pytest

from flightml.data.dataset import FlightDataset
from flightml.models.flight2vec import Flight2VecEngine, OperationalTwinMatch
from flightml.models.pinn_mass import PINNMassInverter, MassInversionResult
from flightml.models.contrail_surrogate import ContrailNeuralSurrogate, ContrailSurrogateOutput
from flightml.models.st_gnn import SpatioTemporalDelayEngine, STGNNPropagationResult
from flightml.sim.airspace_closure import AirspaceClosureSimulator, FIRClosureSimulationResult
from flightml.sim.marl_atfm import MultiAgentATFMSimulator, MARLGamingSimulationResult


@pytest.fixture(scope="module")
def dataset() -> FlightDataset:
    return FlightDataset("202510")


@pytest.fixture(scope="module")
def sample_flight_id(dataset: FlightDataset) -> int:
    row = dataset.cursor.execute(
        "SELECT ectrl_id FROM flights WHERE actual_distance_nm >= 300 LIMIT 1"
    ).fetchone()
    assert row is not None
    return int(row[0])


def test_flight2vec_embedding_and_twins(dataset: FlightDataset, sample_flight_id: int):
    f2v = Flight2VecEngine(dataset)
    emb = f2v.embed_flight(sample_flight_id)
    assert emb is not None
    assert emb.shape == (128,)
    # Embedding is L2-normalized onto unit hypersphere S^127
    norm = float(np.linalg.norm(emb))
    assert abs(norm - 1.0) < 1e-3

    # Operational twins retrieval
    twins = f2v.find_operational_twins(sample_flight_id, search_sample_size=30, top_k=3)
    assert isinstance(twins, list)
    if twins:
        t0 = twins[0]
        assert isinstance(t0, OperationalTwinMatch)
        assert -1.0 <= t0.cosine_similarity <= 1.001
        assert 0.0 <= t0.similarity_pct <= 100.0
        d = t0.to_dict()
        assert d["ectrl_id"] == t0.ectrl_id

    # 2D PCA projection
    pca_df = f2v.project_latent_space_2d(sample_size=10)
    assert not pca_df.empty
    assert "PCA_Dim_1" in pca_df.columns
    assert "PCA_Dim_2" in pca_df.columns
    assert "Operator" in pca_df.columns


def test_pinn_mass_inversion(dataset: FlightDataset, sample_flight_id: int):
    inverter = PINNMassInverter(dataset)
    res = inverter.invert_flight_mass(sample_flight_id)
    assert res is not None
    assert isinstance(res, MassInversionResult)
    assert res.oew_kg > 0
    assert res.mtow_kg >= res.oew_kg
    assert res.oew_kg <= res.estimated_takeoff_weight_kg <= res.mtow_kg * 1.05
    assert res.estimated_payload_kg >= 0.0
    assert 20.0 <= res.estimated_load_factor_pct <= 100.0
    assert res.confidence_score > 0.0
    assert res.physics_residual_error >= 0.0

    d = res.to_dict()
    assert d["ectrl_id"] == sample_flight_id
    assert d["estimated_takeoff_weight_kg"] == res.estimated_takeoff_weight_kg


def test_contrail_neural_surrogate():
    surrogate = ContrailNeuralSurrogate()

    # Case 1: Persistent ISSR warming condition (cold and humid above FL350)
    cold_pt = surrogate.evaluate_atmospheric_point(
        temperature_c=-54.0,
        rh_water_pct=70.0,
        rh_ice_pct=125.0,
        flight_level=370.0,
        bypass_ratio=11.5,
    )
    assert isinstance(cold_pt, ContrailSurrogateOutput)
    assert cold_pt.schmidt_appleman_satisfied is True
    assert cold_pt.is_persistent_issr is True
    assert cold_pt.optical_depth > 0.0
    assert cold_pt.instantaneous_rf_mw_m2 > 0.0
    assert cold_pt.climate_forcing_severity in ("MODERATE_WARMING", "SEVERE_CLIMATE_FORCING")
    assert cold_pt.inference_latency_us > 0.0
    assert cold_pt.inference_latency_us < 50_000.0  # Well under 50ms (typically < 100us)

    # Case 2: Warm and dry clear-air condition (no contrails possible)
    warm_pt = surrogate.evaluate_atmospheric_point(
        temperature_c=-25.0,
        rh_water_pct=30.0,
        rh_ice_pct=45.0,
        flight_level=180.0,
        bypass_ratio=8.0,
    )
    assert warm_pt.schmidt_appleman_satisfied is False
    assert warm_pt.is_persistent_issr is False
    assert warm_pt.optical_depth == 0.0
    assert warm_pt.instantaneous_rf_mw_m2 == 0.0
    assert warm_pt.climate_forcing_severity == "NONE_CLEAR_AIR"

    d = cold_pt.to_dict()
    assert d["temperature_c"] == -54.0


def test_st_gnn_delay_propagation(dataset: FlightDataset):
    engine = SpatioTemporalDelayEngine(dataset)
    res = engine.simulate_delay_propagation(shock_origin="EGLL", injected_delay_min=60.0)

    assert isinstance(res, STGNNPropagationResult)
    assert res.shock_origin_node == "EGLL"
    assert res.injected_shock_delay_min == 60.0
    assert res.lookahead_hours == [0, 1, 2, 4, 6]
    assert res.total_network_reactionary_delay_hours > 0.0
    assert len(res.top_vulnerable_nodes) > 0
    assert "EGLL" in res.node_delay_matrix
    assert len(res.node_delay_matrix["EGLL"]) == 5

    # Check top vulnerable node
    top_node = res.top_vulnerable_nodes[0]
    assert top_node.vulnerability_index >= 0.0
    assert top_node.risk_level in ("CRITICAL_SHOCK_ABSORBER", "ELEVATED_RIPPLE_RISK", "NOMINAL_ABSORPTION")
    assert res.shock_datetime != ""
    assert res.total_flights_impacted >= 0
    assert isinstance(res.sample_impacted_flights, list)
    assert isinstance(res.sample_turnaround_rotations, list)
    assert res.total_freed_slots_count >= 0
    assert res.unserved_rotations_count >= 0
    assert isinstance(res.downstream_airports_impacted, list)
    assert res.backlog_acceleration_pct >= 0.0

    d = res.to_dict()
    assert d["shock_origin_node"] == "EGLL"
    assert len(d["top_vulnerable_nodes"]) == len(res.top_vulnerable_nodes)
    assert "total_freed_slots_count" in d
    assert "unserved_rotations_count" in d
    assert "backlog_acceleration_pct" in d

    # Test with explicit shock date and time and duration
    valid_date = res.shock_datetime.split()[0]
    res_custom = engine.simulate_delay_propagation(
        shock_origin="EGLL",
        injected_delay_min=60.0,
        shock_date=valid_date,
        shock_time_hour=10,
        shock_duration_hours=3.0,
    )
    assert f"{valid_date} 10:00 UTC" in res_custom.shock_datetime
    assert res_custom.shock_duration_hours == 3.0
    assert res_custom.shock_end_time != ""
    assert len(res_custom.sample_impacted_flights) > 0
    assert len(res_custom.sample_turnaround_rotations) >= 0


def test_airspace_closure_simulator(dataset: FlightDataset):
    sim = AirspaceClosureSimulator(dataset)
    res = sim.simulate_fir_closure(
        target_fir="LSASFIR",
        closure_start_hour=10,
        closure_duration_hours=3.0,
    )

    assert isinstance(res, FIRClosureSimulationResult)
    assert res.target_fir == "LSASFIR"
    assert res.closure_duration_hours == 3.0
    assert res.total_flights_impacted > 0
    assert res.rerouted_flights_count > 0
    assert res.total_extra_distance_flown_nm > 0.0
    assert res.total_extra_fuel_burned_tonnes > 0.0
    assert res.total_airline_economic_loss_eur > 0.0
    assert res.post_reopening_recovery_hours > 0.0
    assert res.unserved_return_flights_count >= 0
    assert res.total_freed_slots_count >= 0
    assert isinstance(res.downstream_airports_impacted, list)
    assert res.backlog_acceleration_pct >= 0.0
    assert isinstance(res.sample_tail_rotations, list)
    assert len(res.recovery_curve) > 0
    c0 = res.recovery_curve[0]
    assert "time_utc" in c0
    assert "backlog_flights" in c0
    assert "naive_backlog_flights" in c0
    assert "sector_surge_capacity" in c0
    assert "Hours_Post_Reopening" in c0
    assert "Remaining_Backlog_Flights" in c0
    assert "Airspace_Throughput_Pct" in c0
    assert "Status" in c0
    assert len(res.sample_rerouted_flights) > 0

    rf0 = res.sample_rerouted_flights[0]
    assert rf0.extra_distance_nm > 0.0
    assert rf0.extra_cost_eur > 0.0
    assert rf0.avoidance_fir_diverted_to in ("LFFFFIR", "EDUUUIR", "LIRRFIR", "ADJACENT_FIR_1", "ADJACENT_FIR_2")

    d = res.to_dict()
    assert d["target_fir"] == "LSASFIR"
    assert len(d["sample_rerouted_flights"]) == len(res.sample_rerouted_flights)
    assert "unserved_return_flights_count" in d
    assert "total_freed_slots_count" in d
    assert "backlog_acceleration_pct" in d

    # Test with explicit date_str
    res_date = sim.simulate_fir_closure(
        target_fir="LSASFIR",
        closure_start_hour=10,
        closure_duration_hours=3.0,
        date_str=res.closure_date,
    )
    assert res_date.closure_date == res.closure_date
    assert res_date.total_flights_impacted > 0


def test_marl_atfm_simulator(dataset: FlightDataset):
    marl = MultiAgentATFMSimulator(dataset)
    res = marl.run_marl_simulation(regulated_airport="EGLL", capacity_cut_pct=40.0)

    assert isinstance(res, MARLGamingSimulationResult)
    assert res.regulated_airport == "EGLL"
    assert res.declared_capacity_cut_pct == 40.0
    assert res.total_flights_competing > 0
    assert "NASH" in res.market_equilibrium_type
    assert res.cooperative_vs_competitive_efficiency_gap_pct > 0.0
    assert res.total_network_delay_saved_min > 0.0
    assert res.total_net_economic_benefit_eur > 0.0
    assert len(res.airline_strategies) == 4
    assert res.regulation_date != ""
    assert res.regulation_start_time != ""
    assert res.regulation_duration_hours == 3.0
    assert isinstance(res.sample_swapped_flights, list)

    # Verify heterogeneous archetype strategies and complete balance sheet fields
    for strat in res.airline_strategies:
        assert strat.flights_regulated > 0
        assert strat.dominant_strategy != ""
        assert strat.net_economic_payoff_eur >= 0.0
        assert strat.baseline_cost_no_gaming_eur > 0.0
        assert strat.post_gaming_cost_eur >= 0.0
        assert abs(strat.net_economic_payoff_eur - (strat.baseline_cost_no_gaming_eur - strat.post_gaming_cost_eur)) < 1.0

    # Verify blow-by-blow decision log and infilled assumptions glossary
    assert len(res.decision_log) >= 30  # Comprehensive stream of all tactical flight actions (50+ steps)
    assert len(res.assumptions_glossary) >= 5
    dl0 = res.decision_log[0]
    assert dl0.step_number == 1
    assert dl0.action_type == "REGULATION_ISSUED"
    assert dl0.timestamp_rel == "T+00:00"

    # Mathematical tie-out: tactical actions match aggregate network delay saved and economic payoff
    tactical_entries = res.decision_log[2:-1]
    assert len(tactical_entries) == sum(
        s.slots_swapped_count + s.tactical_speedups_count + s.tactical_cancellations_count
        for s in res.airline_strategies
    )
    delay_mitigated_sum = sum(-e.delay_impact_min for e in tactical_entries)
    benefit_eur_sum = sum(e.economic_benefit_eur for e in tactical_entries)
    assert abs(delay_mitigated_sum - res.total_network_delay_saved_min) < 1e-4
    assert abs(benefit_eur_sum - res.total_net_economic_benefit_eur) < 1e-4

    d = res.to_dict()
    assert d["regulated_airport"] == "EGLL"
    assert len(d["airline_strategies"]) == 4
    assert len(d["decision_log"]) == len(res.decision_log)
    assert len(d["assumptions_glossary"]) == len(res.assumptions_glossary)

    # Test with explicit date, start hour, and duration
    valid_date = res.regulation_date
    res_custom = marl.run_marl_simulation(
        regulated_airport="EGLL",
        capacity_cut_pct=40.0,
        date_str=valid_date,
        start_time_hour=10,
        duration_hours=3.0,
    )
    assert res_custom.regulation_date == valid_date
    assert res_custom.regulation_duration_hours == 3.0
    assert res_custom.total_flights_competing > 0


def test_flight2vec_prevectorization_cache(dataset: FlightDataset, sample_flight_id: int):
    f2v = Flight2VecEngine(dataset)
    # Pre-vectorize a small test slice (e.g. 100 flights)
    count = f2v.build_vector_cache(limit=100)
    assert count == 100
    assert f2v.is_cache_available() is True
    assert f2v.get_cached_count() >= 100

    # Retrieve twins using the pre-vectorized cache
    cached_twins = f2v.find_operational_twins(sample_flight_id, top_k=3, use_cache=True)
    assert isinstance(cached_twins, list)
    if cached_twins:
        t = cached_twins[0]
        assert -1.0 <= t.cosine_similarity <= 1.001
        assert 0.0 <= t.similarity_pct <= 100.0


def test_flight2vec_commercial_jet_excludes_helicopters(dataset: FlightDataset):
    """
    Regression test: verify that a commercial narrowbody jet (e.g. A320) matches with
    commercial jet/transport twins and strictly excludes helicopters (e.g. S92).
    """
    f2v = Flight2VecEngine(dataset)
    target_id = 292396738
    target_flight = dataset.get_flight(target_id)
    if not target_flight:
        # Fallback to any A320 if target id not in dataset
        row = dataset.cursor.execute("SELECT ectrl_id FROM flights WHERE ac_type = 'A320' LIMIT 1").fetchone()
        if not row:
            pytest.skip("No A320 flights found in dataset")
        target_id = int(row[0])

    twins = f2v.find_operational_twins(target_id, top_k=5, use_cache=False)
    assert len(twins) > 0
    helicopter_types = {"S92", "S76", "A139", "AW139", "EC35", "EC45", "H145", "H160"}
    for twin in twins:
        # Cosine similarity for genuine commercial twins should be high
        assert twin.cosine_similarity > 0.80
        assert twin.ac_type.upper() not in helicopter_types
        assert twin.explanation != ""


