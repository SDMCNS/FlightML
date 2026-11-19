import os
from pathlib import Path
import pytest
from typer.testing import CliRunner

from flightml.data.dataset import FlightDataset
from flightml.eval.radar_permutations import (
    simulate_radar_permutations,
    generate_radar_resilience_html_report,
    create_radar_resilience_map,
    RadarFIRResilienceAnalysis,
    RADAR_ANNUAL_COST_SAVINGS_EUR,
)
from flightml.eval.radar_ecac_pipeline import (
    get_ranked_radar_firs,
    run_ecac_radar_pipeline,
    RADAR_OUTCOMES_JSONL_PATH,
)
from flightml.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def dataset():
    return FlightDataset(period="202510")


def test_get_ranked_radar_firs(dataset):
    """Verify that only FIRs containing disclosed radar stations are included, ranked by traffic."""
    ranked = get_ranked_radar_firs(dataset, period="202510")
    assert len(ranked) > 0

    ranked_fir_ids = [r["fir_id"] for r in ranked]

    # Every FIR in the list MUST contain at least 1 disclosed radar station
    for r in ranked:
        assert r["radar_count"] > 0, f"FIR {r['fir_id']} has 0 radars but was included"
        assert r["traffic_moves"] > 0

    # Verify descending traffic order
    traffic_counts = [r["traffic_moves"] for r in ranked]
    assert traffic_counts == sorted(traffic_counts, reverse=True)

    # Disclosed radar FIRs must be present
    assert "EDUUUIR" in ranked_fir_ids
    assert "LECMFIR" in ranked_fir_ids
    assert "EDWWFIR" in ranked_fir_ids
    assert "EGTTFIR" in ranked_fir_ids

    # Undisclosed radar FIRs must be strictly excluded
    assert "LSASUIR" not in ranked_fir_ids  # Switzerland has no disclosed radars in dataset
    assert "EBBRFIR" not in ranked_fir_ids  # Belgium has no disclosed radars in dataset
    assert "ESMMFIR" not in ranked_fir_ids  # Sweden has no disclosed radars in dataset


def test_simulate_radar_permutations_eduu(dataset):
    """Verify radar permutation simulation on EDUUUIR with VIEW_1 standard."""
    analysis = simulate_radar_permutations(
        dataset=dataset,
        fir_id="EDUUUIR",
        period="202510",
        target_standard="VIEW_1",
        max_permutations=5,
    )
    assert isinstance(analysis, RadarFIRResilienceAnalysis)
    assert analysis.fir_id == "EDUUUIR"
    assert analysis.target_standard == "VIEW_1"
    assert analysis.baseline_station_count > 0
    assert len(analysis.permutations) >= 3

    p0 = analysis.permutations[0]
    assert p0.category == "BASELINE"
    assert p0.active_count == analysis.baseline_station_count
    assert p0.disabled_count == 0
    assert p0.windows["24h"].incremental_delay_cost_eur == 0.0
    assert p0.windows["24h"].net_economic_impact_eur == 0.0

    # Verify optimal permutation exists and provides positive net savings (€85k/yr per decommissioned radar head)
    opt = analysis.optimal_permutation
    assert opt is not None
    assert opt.active_count < analysis.baseline_station_count
    assert opt.windows["24h"].net_economic_impact_eur < 0.0  # Net savings
    assert opt.windows["1h"].net_economic_impact_eur < 0.0
    assert opt.windows["6h"].net_economic_impact_eur < 0.0

    # Verify air traffic micro-simulation across 100% census traffic
    assert analysis.traffic_simulation is not None
    assert analysis.traffic_simulation.total_flights_simulated > 100_000
    assert analysis.traffic_simulation.directly_impacted_flights >= 0
    assert analysis.traffic_simulation.separation_standard_nm in (5.0, 10.0, 20.0)

    # Verify Radar Station Utilization table
    assert analysis.station_utilization is not None
    assert len(analysis.station_utilization) >= analysis.baseline_station_count
    top_u = analysis.station_utilization[0]
    assert top_u.daily_flights_using > 50.0
    assert top_u.hourly_flights_using > 2.0
    assert top_u.utilization_pct > 0.0

    # Verify 0.5% Planned Outages and Maintenance Reliability Model
    assert analysis.maintenance_impact is not None
    assert analysis.maintenance_impact.outage_rate_pct == 0.5
    assert analysis.maintenance_impact.total_cells > 0
    assert analysis.maintenance_impact.annual_cell_hours_reduced_rnav >= 0.0


def test_cross_border_radar_cascading(dataset):
    """Verify that external active radars from neighboring FIRs cascade coverage into the analyzed FIR."""
    # Run EDUUUIR with dummy active neighbor radars from EDWWFIR
    analysis = simulate_radar_permutations(
        dataset=dataset,
        fir_id="EDUUUIR",
        period="202510",
        target_standard="VIEW_1",
        max_permutations=3,
        external_active_station_ids=["BREMEN", "DEISTER", "HAMBURG"],
    )
    assert hasattr(analysis, "cascaded_adjacent_stations")
    # Verify adjacent stations are identified within 250 NM line-of-sight
    assert len(analysis.cascaded_adjacent_stations) > 0
    casc_ids = [c["id"] for c in analysis.cascaded_adjacent_stations]
    assert any(cid in ("BREMEN", "DEISTER", "HAMBURG") for cid in casc_ids)


def test_generate_radar_resilience_html_report(dataset, tmp_path):
    """Verify standalone radar HTML report generation with Leaflet maps and non-radar procedural appendices."""
    analysis = simulate_radar_permutations(
        dataset=dataset,
        fir_id="EDUUUIR",
        period="202510",
        target_standard="VIEW_1",
        max_permutations=4,
    )
    out_html = tmp_path / "test_radar_resilience_report.html"
    res_path = generate_radar_resilience_html_report(analysis, output_path=out_html, dataset=dataset)

    assert res_path.exists()
    assert res_path.stat().st_size > 40_000

    html_content = res_path.read_text(encoding="utf-8")
    assert "Radar Network Resilience" in html_content
    assert "EDUUUIR" in html_content
    assert "<iframe srcdoc=" in html_content
    assert "Procedural &amp; Non-Radar Longitudinal Separation Expansion" in html_content
    assert "esri.com" in html_content or "arcgisonline.com" in html_content
    assert "cartocdn.com" not in html_content

    # Verify worst-case 1-day SPOF outage traffic impact section placed after Maintenance section
    assert "Worst-Case Single Point of Failure: 1-Day Outage Traffic Impact" in html_content
    maint_pos = html_content.index("Maintenance &amp; Planned Outage Reliability Assessment (0.5% Radar Outage Rate)")
    spof_pos = html_content.index("Worst-Case Single Point of Failure: 1-Day Outage Traffic Impact")
    assert maint_pos < spof_pos, "Worst-case SPOF 1-day outage section must appear after Maintenance section"
    assert "Acute 24-Hour SPOF Outage (Worst-Case)" in html_content
    assert "Worst-Case Single Point of Failure (SPOF) 1-Day Outage Formulation" in html_content
    if analysis.critical_stations:
        assert analysis.critical_stations[0]["id"] in html_content


def test_cli_radar_resilience_valid_and_invalid():
    """Verify CLI radar-resilience command succeeds for disclosed FIR and blocks undisclosed FIR."""
    # Valid disclosed FIR
    res_valid = runner.invoke(app, ["radar-resilience", "--fir", "EDUUUIR", "--max-permutations", "3"])
    assert res_valid.exit_code == 0
    assert "Optimal Rationalization Posture" in res_valid.stdout
    assert "Evaluated Permutations" in res_valid.stdout

    # Undisclosed FIR (e.g. LSASUIR)
    res_invalid = runner.invoke(app, ["radar-resilience", "--fir", "LSASUIR", "--max-permutations", "3"])
    assert res_invalid.exit_code == 0
    assert "Undisclosed Radar Airspace" in res_invalid.stdout
    assert "Airspace Disclosed Radar Restriction" in res_invalid.stdout


def test_cli_radar_ecac_pipeline():
    """Verify CLI radar-ecac-pipeline command executes correctly with max-firs constraint."""
    res = runner.invoke(app, ["radar-ecac-pipeline", "--period", "202510", "--standard", "VIEW_1", "--max-firs", "2", "--max-permutations", "2"])
    assert res.exit_code == 0
    assert "Pan-European Radar Surveillance Resilience Summary" in res.stdout
    assert "ECAC Radar Resilience Pipeline Execution Complete" in res.stdout
    Path("data/reports/radar_resilience/ecac_radar_summary_report_202510_VIEW_1.html").unlink(missing_ok=True)


def test_joint_cns_surveillance_compensation(dataset, tmp_path):
    """Verify Joint CNS Navigation-Surveillance Coupling: Radar compensation for degraded RNAV and absence of curfew declarations."""
    analysis = simulate_radar_permutations(
        dataset=dataset,
        fir_id="EDUUUIR",
        period="202510",
        target_standard="VIEW_1",
        max_permutations=3,
    )
    assert analysis.traffic_simulation is not None
    assert analysis.traffic_simulation.curfew_breaches_count == 0

    assert analysis.cns_joint_summary is not None
    cns_sum = analysis.cns_joint_summary
    assert "regimes" in cns_sum
    regimes = cns_sum["regimes"]
    assert "optimal_cns" in regimes
    assert "surveillance_compensated" in regimes
    assert "autonomous_pbn" in regimes
    assert "compound_deficit" in regimes

    assert regimes["optimal_cns"]["cell_count"] > 0
    assert "compensating_radar_rankings" in cns_sum
    assert len(cns_sum["compensating_radar_rankings"]) > 0

    out_html = tmp_path / "test_cns_report.html"
    rep_path = generate_radar_resilience_html_report(analysis, output_path=out_html, dataset=dataset)
    html_text = rep_path.read_text(encoding="utf-8")
    assert "Joint CNS Integration: Radar Surveillance Compensation for Navaid Degradations" in html_text
    assert "Surveillance Compensated" in html_text
    assert "curfew breaches" not in html_text.lower()


def test_simulate_radar_permutations_view3_fl50_fl10(dataset):
    """Verify candidate configuration targeting FL50 coverage as opposed to FL200, with terminal FL10 at significant airports."""
    # When target_standard is omitted, it should default to VIEW_3 (FL50 + FL10 hubs)
    analysis = simulate_radar_permutations(
        dataset=dataset,
        fir_id="EDUUUIR",
        period="202510",
        max_permutations=4,
    )
    assert analysis.target_standard == "VIEW_3"
    assert analysis.optimal_permutation is not None

    # Verify candidates include FL50 low-level infill & terminal FL10 candidate
    cand_ids = [p.permutation_id for p in analysis.permutations]
    assert any("VIEW3" in cid or "FL50" in cid for cid in cand_ids)

    # Verify coverage achieved at FL50
    opt = analysis.optimal_permutation
    assert opt.coverage_coverable_pct > 80.0
    assert opt.terminal_airports_total > 0
    # Optimal choice under VIEW_3 must not pick P-CORE-VIEW1 which violates FL50/FL10
    assert opt.permutation_id != "P-CORE-VIEW1"
    assert opt.degraded_airports_count == 0

    # Verify 1-day outage economic penalty vs 1-hour peak shock ratio
    assert analysis.critical_stations is not None
    for cs in analysis.critical_stations:
        day_cost = cs["outage_daily_delay_cost_eur"]
        hr_cost = cs["outage_hourly_delay_cost_eur"]
        if hr_cost > 0:
            assert day_cost <= 24.0 * hr_cost, f"Daily shock {day_cost} exceeds 24 * hourly shock {hr_cost}"
            assert day_cost <= 20.0 * hr_cost, f"Daily shock {day_cost} should reflect diurnal off-peak curve <= 20x hourly"

