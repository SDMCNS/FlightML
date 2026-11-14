import pytest
from flightml.models.airspace_forecaster import (
    AirspaceForecaster,
    SpatialCellDensity,
    SectorCapacityBreach,
    AirspaceForecastReport,
)


@pytest.fixture(scope="module")
def forecaster():
    return AirspaceForecaster()


def test_compute_variable_grid_densities_1deg(forecaster):
    cells = forecaster.compute_variable_grid_densities(
        period_baseline="202406",
        period_target="202506",
        grid_size_deg=1.0,
        hour_utc=11,
        projected_growth_pct=3.5,
        min_flights_filter=10,
    )
    assert len(cells) > 0
    top_cell = cells[0]
    assert isinstance(top_cell, SpatialCellDensity)
    assert top_cell.grid_size_deg == 1.0
    assert top_cell.projected_flights > top_cell.flights_target
    assert top_cell.complexity_score > 0
    assert 30.0 <= top_cell.cell_lat <= 68.0
    assert -15.0 <= top_cell.cell_lon <= 38.0


def test_compute_variable_grid_densities_half_deg(forecaster):
    cells = forecaster.compute_variable_grid_densities(
        period_baseline="202406",
        period_target="202506",
        grid_size_deg=0.5,
        hour_utc=11,
        projected_growth_pct=4.0,
        min_flights_filter=5,
    )
    assert len(cells) > 0
    assert cells[0].grid_size_deg == 0.5


def test_project_sector_growth(forecaster):
    breaches = forecaster.project_sector_growth(
        baseline_period="202406",
        target_period="202506",
        scenario_growth_pct=3.5,
    )
    assert len(breaches) > 0
    b0 = breaches[0]
    assert isinstance(b0, SectorCapacityBreach)
    assert b0.projected_peak_hourly > 0
    assert b0.nominal_capacity_hourly > 0
    assert b0.projected_saturation_pct > 0
    assert b0.breach_level in ("NOMINAL", "ELEVATED", "SEVERE", "BREACH")
    assert len(b0.mitigation_recommendation) > 0


def test_generate_comprehensive_forecast(forecaster):
    report = forecaster.generate_comprehensive_forecast(
        forecast_year=2026,
        season="summer",
        growth_scenario_pct=3.5,
        grid_size_deg=1.0,
    )
    assert isinstance(report, AirspaceForecastReport)
    assert "Summer 2026" in report.forecast_label
    assert report.total_sectors_analyzed > 50
    assert report.total_sectors_analyzed == (
        report.nominal_sectors_count
        + report.elevated_sectors_count
        + report.severe_sectors_count
        + report.breached_sectors_count
    )
    assert 0.0 <= report.network_atfm_risk_index <= 10.0
    assert len(report.top_capacity_breaches) > 0
    assert len(report.top_grid_hotspots) > 0

    d = report.to_dict()
    assert "top_capacity_breaches" in d
    assert "top_grid_hotspots" in d
    assert d["forecast_horizon_year"] == 2026
