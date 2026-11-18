"""
Unit tests for the new aviation analytics engines:
1. Airport Operations & Runway Metar Extrapolation (Use Case 1)
2. ICAO nvPM Soot Microphysics & Ice Nucleation Scaling (Use Case 2)
3. OpenAP WRAP Kinematics & Envelope Analysis (Use Case 3)
4. DME-DME RNAV Backup Coverage & Decommissioning Recommender (Use Case 5)
5. Longitudinal Vertical Capping & Airway FRA Dividend (Use Cases 6 & 7)
"""

from __future__ import annotations

import math
import pytest
from flightml.data.dataset import FlightDataset
import flightml.math as fmath
from flightml.eval.airport_ops import get_airport_operations_report, assess_runways_with_wind
from flightml.eval.openap_wrap_engine import analyze_flight_wrap_envelope
from flightml.eval.dme_rnav_coverage import get_dme_rnav_coverage_report
from flightml.eval.longitudinal_airspace_modernization import (
    get_vertical_capping_analysis,
    get_fixed_airways_vs_fra_analysis,
)


@pytest.fixture(scope="module")
def dataset() -> FlightDataset:
    return FlightDataset()


def test_use_case_1_airport_operations_and_crosswinds(dataset: FlightDataset):
    """Verify airport traffic extraction and runway crosswind/active configuration modeling."""
    rep = get_airport_operations_report(dataset, "EGLL", period="201611")
    assert rep.airport_ident == "EGLL"
    assert "Heathrow" in rep.name
    assert rep.elevation_ft > 0
    assert len(rep.runways) >= 2
    assert "Recommended Flow" in rep.recommended_active_config

    # Check runway crosswind / headwind calculations
    rwy = rep.runways[0]
    assert rwy.le_crosswind_kts >= 0.0
    assert rwy.le_lda_ft > 0
    assert rwy.preferred_end in (rwy.le_ident, rwy.he_ident)

    # Test westerly wind (250° at 16 kts): 27L/27R should be preferred over 09L/09R
    pref_ends = [r.preferred_end for r in rep.runways]
    assert "27R" in pref_ends or "27L" in pref_ends


def test_use_case_2_soot_microphysics_and_optical_depth():
    """Verify ICAO nvPM soot number scaling on contrail optical depth."""
    # Nominal baseline: 1.0e15 part/kg -> tau = 0.30
    tau_nominal = fmath.soot_number_to_optical_depth(1.0e15, nominal_ei_num=1.0e15, nominal_optical_depth=0.30)
    assert math.isclose(tau_nominal, 0.30, abs_tol=1e-3)

    # Modern lean-burn combustor: 1.0e14 part/kg (-90% soot) -> tau significantly lower
    tau_lean = fmath.soot_number_to_optical_depth(1.0e14, nominal_ei_num=1.0e15, nominal_optical_depth=0.30)
    assert 0.05 < tau_lean < 0.10
    assert tau_lean < tau_nominal

    # Initial ice crystal number density in wake
    n_ice = fmath.initial_ice_crystal_concentration(
        ei_num_per_kg=1.74e14,
        fuel_flow_kg_s=0.75,
        wingspan_m=35.8,
        true_airspeed_mps=230.0,
    )
    assert n_ice > 1.0e8


def test_use_case_3_openap_wrap_kinematics(dataset: FlightDataset):
    """Verify observed flight trajectory benchmarking against OpenAP WRAP envelope."""
    rep = analyze_flight_wrap_envelope(dataset, ectrl_id=202037446, period="201611")
    assert rep.ectrl_id == 202037446
    assert rep.ac_type in ("A319", "A320")
    assert rep.overall_normalcy_score_pct >= 50.0
    assert len(rep.metrics) >= 15

    # Check constant Mach crossover altitude metric
    crossover_metrics = [m for m in rep.metrics if "crossover" in m.description.lower()]
    assert len(crossover_metrics) > 0
    m = crossover_metrics[0]
    assert m.min_val <= m.opt_val <= m.max_val
    assert m.percentile_pct >= 0.0


def test_use_case_5_dme_dme_rnav_coverage_and_rationalization(dataset: FlightDataset):
    """Verify 10x10 NM DME-DME RNAV coverage and greedy rationalization deactivation suggestions."""
    rep = get_dme_rnav_coverage_report(dataset, flight_level=300)
    assert rep.flight_level == 300
    assert rep.coverage_percentage >= 95.0
    assert rep.redundant_cells_percentage >= 80.0
    assert rep.total_dmes_count > 50
    assert len(rep.suggested_deactivations) > 0

    # Every suggested deactivation must maintain high network coverage
    first_sug = rep.suggested_deactivations[0]
    assert first_sug["coverage_impact_pct"] <= 1.0
    assert first_sug["remaining_network_coverage_pct"] >= 95.0


def test_use_cases_6_and_7_longitudinal_capping_and_fra(dataset: FlightDataset):
    """Verify vertical FL capping penalties and fixed airways vs FRA dividend."""
    # Use Case 6: Vertical Capping
    cap = get_vertical_capping_analysis(dataset, sample_size=500)
    assert cap["capping_rate_pct"] >= 0.0
    assert cap["mean_deficit_fl"] >= 0.0
    assert cap["total_excess_co2_tonnes"] >= 0.0
    assert len(cap["by_market_segment"]) > 0

    # Use Case 7: Fixed Airways vs FRA
    fra = get_fixed_airways_vs_fra_analysis(dataset, sample_size=500)
    assert fra["mean_distance_saved_pct"] > 0.0
    assert fra["total_distance_saved_nm"] > 0.0
    assert fra["total_fuel_saved_tonnes"] > 0.0
    assert fra["total_co2_avoided_tonnes"] > 0.0
    assert len(fra["top_beneficiary_routes"]) > 0
