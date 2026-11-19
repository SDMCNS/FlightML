import os
from pathlib import Path
import pytest
from typer.testing import CliRunner

from flightml.data.dataset import FlightDataset
from flightml.eval.dme_permutations import (
    simulate_dme_permutations,
    generate_dme_resilience_html_report,
    create_dme_resilience_map,
    DMEFIRResilienceAnalysis,
)
from flightml.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def dataset():
    return FlightDataset(period="202510")


def test_simulate_dme_permutations_egtt(dataset):
    """Verify DME permutation simulation on EGTTFIR with RNAV_5 standard."""
    analysis = simulate_dme_permutations(
        dataset=dataset,
        fir_id="EGTTFIR",
        period="202510",
        target_standard="RNAV_5",
        max_permutations=6,
    )
    assert isinstance(analysis, DMEFIRResilienceAnalysis)
    assert analysis.fir_id == "EGTTFIR"
    assert analysis.target_standard == "RNAV_5"
    assert analysis.baseline_station_count > 0
    assert len(analysis.permutations) >= 4

    p0 = analysis.permutations[0]
    assert p0.category == "BASELINE"
    assert p0.active_count == analysis.baseline_station_count
    assert p0.disabled_count == 0
    assert p0.windows["24h"].incremental_delay_cost_eur == 0.0
    assert p0.windows["24h"].net_economic_impact_eur == 0.0

    # Verify optimal permutation exists and provides positive net savings
    opt = analysis.optimal_permutation
    assert opt is not None
    assert opt.active_count < analysis.baseline_station_count
    assert opt.windows["24h"].net_economic_impact_eur < 0.0  # Net savings (negative cost)
    assert opt.windows["1h"].net_economic_impact_eur < 0.0
    assert opt.windows["6h"].net_economic_impact_eur < 0.0


def test_generate_dme_resilience_html_report(dataset, tmp_path):
    """Verify standalone HTML report generation with Leaflet OpenStreetMap tiles & appendix."""
    analysis = simulate_dme_permutations(
        dataset=dataset,
        fir_id="EGTTFIR",
        period="202510",
        target_standard="RNAV_5",
        max_permutations=4,
    )
    out_html = tmp_path / "test_resilience_report.html"
    res_path = generate_dme_resilience_html_report(analysis, output_path=out_html, dataset=dataset)

    assert res_path.exists()
    assert res_path.stat().st_size > 50_000

    html_content = res_path.read_text(encoding="utf-8")
    assert "DME Network Resilience &amp; Navigation Rationalization" in html_content
    assert "EGTTFIR" in html_content
    assert "<iframe srcdoc=" in html_content
    assert "Line-of-Sight &amp; DME-DME GDOP Geometry Formulation" in html_content
    assert "Eurocontrol Standard Inputs for Economic Cost-Benefit Analyses" in html_content


def test_cli_dme_resilience_command(tmp_path):
    """Verify CLI command `flightml dme-resilience` execution."""
    result = runner.invoke(
        app,
        [
            "dme-resilience",
            "--fir", "EGTTFIR",
            "--period", "202510",
            "--standard", "RNAV_5",
            "--max-permutations", "5",
            "--output-dir", str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "FlightML DME Resilience & Navigation Permutations Engine" in result.output
    assert "Optimal Rationalization Posture" in result.output
    assert "Evaluated Permutations" in result.output
    assert "Standalone HTML Report Successfully Generated" in result.output
