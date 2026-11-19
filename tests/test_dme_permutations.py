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

    # Verify self-reinforcing ML experience recording and diagnostics
    assert analysis.ml_diagnostics is not None
    assert analysis.ml_diagnostics["total_samples_learned"] > 0
    assert "EGTTFIR" in analysis.ml_diagnostics["unique_firs_evaluated"]

    # Verify air traffic simulation and rotational knock-on modeling across ALL flights
    assert analysis.traffic_simulation is not None
    assert analysis.traffic_simulation.total_flights_simulated > 100_000
    assert analysis.traffic_simulation.directly_impacted_flights >= 0

    # Verify DME Station Utilization table
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
    assert analysis.maintenance_impact.annual_cell_hours_reduced_rnav > 0.0
    assert analysis.maintenance_impact.annual_outage_cost_eur > 0.0


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
    assert "Air Traffic Micro-Simulation: Direct Separation" in html_content
    assert "Rotational Delay Knock-on &amp; Night Curfew / Cancellation Mechanics" in html_content
    assert "cartocdn.com" not in html_content
    assert "Standardized European Assessment Altitude References" in html_content
    assert "FL50" in html_content and "FL95" in html_content and "FL350" in html_content
    assert "Strict High-Power DME En-Route Filtering" in html_content
    assert "Self-Reinforcing Multi-FIR Machine Learning Engine" in html_content
    assert "Line-of-Sight &amp; DME-DME GDOP Geometry Formulation" in html_content
    assert "Eurocontrol Standard Inputs for Economic Cost-Benefit Analyses" in html_content
    assert "DME Station Utilization &amp; Flight Journey Exposure" in html_content
    assert "Maintenance &amp; Planned Outage Reliability Assessment (0.5% Station Outage Rate)" in html_content
    assert "Navaid Selection Likelihood &amp; Flight Journey DME Utilization" in html_content
    assert "Planned Outages, Maintenance Combinatorics &amp; Double-Counting Avoidance" in html_content
    assert "ATC Operational Restrictions &amp; Tactical Intervention Economics" in html_content


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
    assert "Self-Reinforcing Multi-FIR Machine Learning Status" in result.output
    assert "Air Traffic Micro-Simulation: Separation Delays" in result.output
    assert "Standalone HTML Report Successfully Generated" in result.output


def test_cross_border_dme_cascading(dataset, tmp_path):
    """Verify that active DMEs from a busier FIR cascade outward to compensate coverage in adjacent FIR."""
    from flightml.eval.ecac_station_rationalization import load_ecac_stations, get_fir_geometry

    # Simulate active core German DMEs (EDUUUIR)
    geom_de = get_fir_geometry(dataset, "EDUUUIR")
    poly_de, _, bbox_de = geom_de
    stas_de, _ = load_ecac_stations(dataset, station_type="DME", fir_poly=poly_de, fir_bbox=bbox_de, include_adjacent=False)
    active_de_ids = {s.id for s in stas_de[:15]}

    # Evaluate Austria (LOVVFIR) with active German stations cascading across the border
    analysis = simulate_dme_permutations(
        dataset=dataset,
        fir_id="LOVVFIR",
        period="202510",
        target_standard="RNAV_5",
        max_permutations=4,
        external_active_station_ids=active_de_ids,
        enable_cross_border_cascading=True,
    )
    assert analysis.cascaded_adjacent_stations is not None
    assert len(analysis.cascaded_adjacent_stations) > 0
    cascaded_ids = {cs["id"] for cs in analysis.cascaded_adjacent_stations}
    assert cascaded_ids.issubset(active_de_ids)

    # Verify report includes the cross-border cascading section and appendix
    out_html = tmp_path / "test_cascaded_report.html"
    rep_path = generate_dme_resilience_html_report(analysis, output_path=out_html, dataset=dataset)
    html_text = rep_path.read_text(encoding="utf-8")
    assert "Cross-Border Navaid Cascading &amp; Adjacent FIR Compensation" in html_text
    assert "Cross-Border Active Navaid Cascading &amp; Regional Rationalization" in html_text
    assert "Cascaded Active Neighbor DMEs (Cross-Border Infill)" in html_text
