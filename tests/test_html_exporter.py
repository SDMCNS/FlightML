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
