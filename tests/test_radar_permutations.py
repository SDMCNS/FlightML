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
    res = runner.invoke(app, ["radar-ecac-pipeline", "--max-firs", "2", "--max-permutations", "2"])
    assert res.exit_code == 0
    assert "Pan-European Radar Surveillance Resilience Summary" in res.stdout
    assert "ECAC Radar Resilience Pipeline Execution Complete" in res.stdout
