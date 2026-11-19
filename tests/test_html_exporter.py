from __future__ import annotations

import pytest

from flightml.data.dataset import FlightDataset
from flightml.viz.html_exporter import (
    REPORT_PAGE_KEYS,
    REPORT_PAGE_TITLES,
    generate_static_html_report,
)


@pytest.fixture(scope="module")
def dataset() -> FlightDataset:
    return FlightDataset("202402")


def test_report_page_keys_and_titles_consistency():
    """Verify all defined page keys have corresponding human-readable titles."""
    for key in REPORT_PAGE_KEYS:
        assert key in REPORT_PAGE_TITLES, f"Key {key} missing from REPORT_PAGE_TITLES"
        assert len(REPORT_PAGE_TITLES[key]) > 0


def test_fra_boundaries_html_export(dataset: FlightDataset):
    """Verify that FRA boundaries static HTML export contains figures, tables, and KPIs (not just header/footer)."""
    html = generate_static_html_report(
        dataset,
        page_key="fra_boundaries",
        fra_min_flights=500,
        fra_annual_growth=0.02,
        fra_realization=0.85,
    )
    assert len(html) > 50000, f"HTML output unexpectedly small: {len(html)} bytes"
    assert "Free Route Airspace (FRA) Boundary Efficiency Analysis" in html
    assert "Prime Candidates for Immediate Cross-Border FRA Implementation" in html
    assert "Long-Term Compounding Benefit Horizons" in html
    assert "Top 25 FIR Boundary Inefficiency League Table" in html
    assert "data:image/png;base64" in html
    assert "report-header" in html
    assert "report-footer" in html


def test_macro_reference_html_export(dataset: FlightDataset):
    """Verify that macro_reference page key produces valid non-empty content."""
    html = generate_static_html_report(dataset, page_key="macro_reference")
    assert len(html) > 20000
    assert "Macro Period Evolution" in html
    assert "ICAO GANP / Eurocontrol PRU" in html
    assert "data:image/png;base64" in html


def test_system_summary_includes_fra(dataset: FlightDataset):
    """Verify system_summary master report bundles airspace overview, FRA, and macro evolution."""
    html = generate_static_html_report(dataset, page_key="system_summary")
    assert len(html) > 100000
    assert "5°×5° Airspace Saturation" in html
    assert "Free Route Airspace (FRA)" in html
    assert "Macro Period Evolution" in html


def test_unknown_page_key_fallback_is_not_empty(dataset: FlightDataset):
    """Verify that passing an unknown or custom page_key gracefully generates non-empty fallback content."""
    html = generate_static_html_report(dataset, page_key="unknown_custom_view")
    assert len(html) > 30000
    assert "Operational Report Overview" in html
    assert "Free Route Airspace (FRA)" in html
    assert "report-footer" in html


def test_comprehensive_flight_html_report_generation():
    """Verify that generate_flight_comprehensive_html_report produces a complete 4D report with working map tiles and math appendix."""
    from flightml.viz.html_exporter import generate_flight_comprehensive_html_report

    ds = FlightDataset("202510")
    flt_df = ds.query("SELECT ectrl_id FROM flights WHERE actual_distance_nm > 500 LIMIT 1")
    target_id = int(flt_df.iloc[0]["ectrl_id"]) if not flt_df.empty else 292222833

    report_html = generate_flight_comprehensive_html_report(ds, flight_id=target_id)
    assert len(report_html) > 100000, f"Comprehensive report unexpectedly small: {len(report_html)} bytes"

    # 1. Map Tiles & Leaflet Iframe Container Verification
    assert "<iframe" in report_html, "Missing isolated iframe map containers"
    assert "srcdoc=" in report_html, "Missing srcdoc for isolated Leaflet execution"
    assert "openstreetmap.fr" in report_html, "Missing Humanitarian OpenStreetMap tile layer"

    # 2. Section & View Logical Ordering Verification
    assert "Executive Flight Summary &amp; Operational Telemetry" in report_html
    assert "View 1: 4D Trajectory, Radar Tracking &amp; Flight Geometry" in report_html
    assert "View 2: Active Radio Navigation Aids &amp; RNP / RNAV Corridor Capability" in report_html
    assert "View 3: Vertical Profile Efficiency: Continuous Climb (CCO) &amp; Continuous Descent (CDO)" in report_html
    assert "View 4: Multi-Factor Delay Attribution &amp; Runway Braking Safety Diagnostics" in report_html
    assert "View 5: Airport Surface Movement, Taxi-Out &amp; ICAO LTO Emissions" in report_html
    assert "View 6: Eurocontrol CRCO Route Charges &amp; Airspace Economics (€ EUR)" in report_html
    assert "View 7: Upper-Air Winds &amp; Equivalent Still Air Distance (ESAD)" in report_html
    assert "View 8: Persistent Contrail Cirrus &amp; Climate Radiative Forcing" in report_html
    assert "View 9: Alternative Eco-Optimal Trajectory: Multi-Objective Trade-Off &amp; Feasibility" in report_html
    assert "Appendix: Comprehensive Mathematical &amp; Atmospheric Physics Methodology" in report_html

    # 3. Mathematical Appendix Content Verification
    assert "Haversine Formula" in report_html
    assert "Equivalent Still Air Distance (ESAD)" in report_html
    assert "Schmidt-Appleman Criterion" in report_html
    assert "Murphy &amp; Koop (2005)" in report_html
    assert "18.5 kg CO₂e / nm" in report_html


def test_single_flight_page_key_delegation():
    """Verify that calling generate_static_html_report with page_key='single_flight' returns the comprehensive report."""
    ds = FlightDataset("202510")
    html = generate_static_html_report(ds, page_key="single_flight", flight_id=292222833)
    assert len(html) > 100000
    assert "<iframe" in html
    assert "openstreetmap.fr" in html
    assert "Appendix: Comprehensive Mathematical &amp; Atmospheric Physics Methodology" in html


def test_ecac_rationalization_html_exports(dataset: FlightDataset):
    """
    Verify full HTML exports for ECAC DME and Radar rationalization subtabs:
    - Preserves interactive Leaflet OpenStreetMap tiles in an isolated iframe srcdoc.
    - Excludes the deprecated 'Minimum Active Equipment Target' column from the RNAV/RNP table.
    - Includes KPIs, executive charts, gap-filler/infill tables, and cost/power engineering breakdowns.
    """
    # 1. DME Rationalization HTML export
    html_dme = generate_static_html_report(
        dataset,
        page_key="ecac_dme_rationalization",
        dme_fir="EGTTFIR",
        dme_view="RNAV_5",
        dme_fl=200,
    )
    assert len(html_dme) > 50000, f"DME HTML export unexpectedly small: {len(html_dme)} bytes"
    assert "<iframe" in html_dme, "Missing Leaflet map iframe container in DME export"
    assert "srcdoc=" in html_dme, "Missing srcdoc for preserving interactive map tiles"
    assert "DME-DME RNAV" in html_dme
    assert "Minimum Active Equipment Target" not in html_dme, "Found deprecated column in RNAV/RNP description table!"
    assert "Navigation Performance Standards (RNAV / RNP) Criteria" in html_dme
    assert "Eurocontrol &amp; ICAO Cost and Power Savings" in html_dme

    # 2. Radar Rationalization HTML export
    html_radar = generate_static_html_report(
        dataset,
        page_key="ecac_radar_rationalization",
        radar_fir="EGTTFIR",
        radar_view="VIEW_1",
        radar_fl=200,
    )
    assert len(html_radar) > 50000, f"Radar HTML export unexpectedly small: {len(html_radar)} bytes"
    assert "<iframe" in html_radar, "Missing Leaflet map iframe container in Radar export"
    assert "srcdoc=" in html_radar, "Missing srcdoc for preserving interactive map tiles"
    assert "Radar" in html_radar
    assert "Eurocontrol &amp; ICAO Cost and Power Savings" in html_radar

