"""
Unit & Integration Tests for Free Route Airspace (FRA) FIR Boundary Inefficiency Engine.
"""

from __future__ import annotations

import json
from typer.testing import CliRunner

from flightml.cli import app
from flightml.data.dataset import FlightDataset
from flightml.geo.fra_boundary_engine import (
    analyze_fir_boundary_inefficiencies,
    compare_fra_boundary_evolution,
    get_boundary_detail,
    get_worst_flights_for_boundary,
    calculate_fra_long_term_forecast,
    interpolate_great_circle,
    FRABoundaryReport,
)
from flightml.viz.map_builder import create_fra_boundary_map

runner = CliRunner()


def test_analyze_fir_boundary_inefficiencies_structure():
    """Verify deterministic boundary metrics calculation on archive period 202510."""
    ds = FlightDataset(period="202510")
    report = analyze_fir_boundary_inefficiencies(
        dataset=ds,
        min_flights=2000,
        limit=20,
        cross_border_only=True,
    )

    assert isinstance(report, FRABoundaryReport)
    assert report.total_boundaries_analyzed > 0
    assert report.cross_border_boundaries_count > 0
    assert report.total_network_excess_fuel_tonnes > 0.0
    assert report.total_network_excess_cost_eur > 0.0

    # Verify descending priority ranking
    scores = [b.fra_priority_score for b in report.boundaries]
    assert scores == sorted(scores, reverse=True)

    # Check metric fields on top boundary
    top_b = report.boundaries[0]
    assert top_b.priority_rank == 1
    assert "<->" in top_b.boundary_pair
    assert top_b.total_crossings >= 2000
    assert top_b.directness_index_pct > 80.0
    assert top_b.mean_detour_nm > 0.0
    assert top_b.excess_cost_eur > 0.0
    assert top_b.fra_classification in [
        "🚨 PRIME_CANDIDATE_IMMEDIATE_FRA",
        "⚠️ MEDIUM_PRIORITY_FRA",
        "🟢 LIKELY_CROSS_BORDER_FRA",
        "ℹ️ DOMESTIC_OR_LOW_IMPACT",
    ]


def test_fra_classification_categories():
    """Verify that both prime candidates and likely FRA implementations are identified."""
    ds = FlightDataset(period="202510")
    report = analyze_fir_boundary_inefficiencies(
        dataset=ds,
        min_flights=2000,
        limit=100,
        cross_border_only=True,
    )

    classifications = {b.fra_classification for b in report.boundaries}
    assert "🚨 PRIME_CANDIDATE_IMMEDIATE_FRA" in classifications
    assert "🟢 LIKELY_CROSS_BORDER_FRA" in classifications

    # Check that a known FRA boundary is correctly identified
    boundary_map = {b.boundary_pair: b for b in report.boundaries}
    known_pairs = [
        "LBSRFIR <-> LRBBFIR",
        "LDZOFIR <-> LQSBUIR",
        "LHCCFIR <-> LRBBFIR",
        "LJLAFIR <-> LOVVFIR",
        "EKDKFIR <-> ESAAFIR",
    ]
    found_known = any(pair in boundary_map for pair in known_pairs)
    assert found_known, f"At least one known FRA boundary must be present in {list(boundary_map.keys())[:10]}"


def test_compare_fra_boundary_evolution():
    """Verify 2024 vs 2025 multi-period longitudinal comparison."""
    comp = compare_fra_boundary_evolution(
        period1="202410",
        period2="202510",
        min_flights=2000,
        top_n=10,
    )

    assert comp["period1"] == "202410"
    assert comp["period2"] == "202510"
    assert comp["total_boundaries_compared"] > 0
    assert len(comp["comparison_table"]) > 0

    first = comp["comparison_table"][0]
    assert "boundary_pair" in first
    assert "volume_2024" in first
    assert "volume_2025" in first
    assert "volume_growth_pct" in first
    assert "directness_delta_pp" in first
    assert "trend" in first


def test_get_boundary_detail():
    """Verify single-boundary detailed diagnostic."""
    ds = FlightDataset(period="202510")
    detail = get_boundary_detail(ds, "EGTTUIR <-> LFFFUIR")

    assert detail["boundary_pair"] == "EGTTUIR <-> LFFFUIR"
    assert "United Kingdom" in detail["country_a"] or "United Kingdom" in detail["country_b"]
    assert "France" in detail["country_a"] or "France" in detail["country_b"]
    assert len(detail["hourly_distribution"]) > 0
    assert len(detail["top_operators"]) > 0


def test_cli_fra_boundaries_json():
    """Verify CLI command output with --json flag."""
    res = runner.invoke(app, ["fra-boundaries", "--limit", "3", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert "period" in data
    assert "total_boundaries_analyzed" in data
    assert "boundaries" in data
    assert len(data["boundaries"]) == 3
    assert data["boundaries"][0]["priority_rank"] == 1


def test_get_worst_flights_and_map():
    """Verify querying worst detoured flights and generating the Leaflet comparison map."""
    ds = FlightDataset(period="202510")
    worst = get_worst_flights_for_boundary(ds, "EGTTUIR <-> LFFFUIR", limit=5, include_trajectories=True)

    assert len(worst) > 0
    f0 = worst[0]
    assert "ectrl_id" in f0
    assert "adep" in f0
    assert "ades" in f0
    assert "actual_distance_nm" in f0
    assert "great_circle_nm" in f0
    assert "detour_nm" in f0
    assert "excess_fuel_kg" in f0
    assert "excess_cost_eur" in f0
    assert "fra_direct_path" in f0
    assert len(f0["fra_direct_path"]) > 0

    # Verify descending detour
    detours = [f["detour_nm"] for f in worst]
    assert detours == sorted(detours, reverse=True)

    # Verify map generation
    polys = ds.get_fir_boundaries(["EGTTUIR", "LFFFUIR"])
    m = create_fra_boundary_map(
        boundary_pair="EGTTUIR <-> LFFFUIR",
        worst_flights=worst,
        fir_polygons=polys,
    )
    import folium
    assert isinstance(m, folium.Map)


def test_calculate_fra_long_term_forecast():
    """Verify multi-horizon compound growth benefit forecasting (1, 5, and 20 years),
    including seasonal synthesis (Spring, Summer, Autumn) and 3 fuel price growth pathways."""
    fc = calculate_fra_long_term_forecast(
        year_baseline="2025",
        use_seasonal_synthesis=True,
        annual_growth_rate=0.02,
        realization_rate=0.85,
        fuel_growth_optimistic=0.015,
        fuel_growth_median=0.035,
        fuel_growth_shock=0.065,
        horizons=[1, 5, 20],
        scope="prime_only",
    )

    assert fc["scope"] == "prime_only"
    assert fc["total_boundaries_evaluated"] > 0
    assert fc["annual_growth_rate"] == 0.02
    assert fc["realization_rate"] == 0.85

    # 1. Verify Seasonal Synthesis (Spring vs Summer Peak vs Autumn)
    sb = fc["seasonal_breakdown"]
    assert "spring" in sb and "summer" in sb and "autumn" in sb and "annual_blended" in sb
    assert sb["spring"]["seasonal_crossings"] > 0
    assert sb["summer"]["seasonal_crossings"] > 0
    assert sb["autumn"]["seasonal_crossings"] > 0
    # Summer peak volume surges over spring shoulder
    assert sb["summer"]["seasonal_crossings"] > sb["spring"]["seasonal_crossings"]
    assert sb["annual_blended"]["annual_crossings"] == (
        sb["spring"]["seasonal_crossings"] + sb["summer"]["seasonal_crossings"] + sb["autumn"]["seasonal_crossings"]
    )

    # 2. Verify Horizons (1, 5, 20 Years) and Compounding Growth
    h = fc["horizons"]
    assert 1 in h and 5 in h and 20 in h
    assert h[20]["cumulative_crossings"] > h[5]["cumulative_crossings"] > h[1]["cumulative_crossings"]
    assert h[20]["cumulative_fuel_saved_tonnes"] > h[5]["cumulative_fuel_saved_tonnes"] > h[1]["cumulative_fuel_saved_tonnes"]

    # 3. Verify Fuel Price Growth Pathways (Shock > Median > Optimistic > Constant)
    h20 = h[20]
    assert h20["savings_shock_eur"] > h20["savings_median_eur"] > h20["savings_optimistic_eur"] > h20["savings_constant_eur"]
    assert h20["fuel_price_shock_eur"] > h20["fuel_price_median_eur"] > h20["fuel_price_optimistic_eur"] > h20["fuel_price_constant_eur"]

    # 4. Verify 20-Year Progression Series
    assert len(fc["yearly_progression"]) == 20
    assert fc["yearly_progression"][-1]["cum_savings_median_eur"] == h[20]["savings_median_eur"]
    assert fc["yearly_progression"][-1]["cum_savings_shock_eur"] == h[20]["savings_shock_eur"]

    # 5. Verify Individual Boundary Contributions
    assert len(fc["boundary_contributions"]) > 0
    b0 = fc["boundary_contributions"][0]
    assert b0["savings_20yr_shock_eur"] > b0["savings_20yr_median_eur"] > b0["savings_20yr_optimistic_eur"]
    assert b0["spring_crossings"] > 0
    assert b0["summer_crossings"] > 0


def test_cli_fra_worst_flights_json():
    """Verify CLI fra-worst-flights command with --json."""
    res = runner.invoke(app, ["fra-worst-flights", "EGTTUIR <-> LFFFUIR", "--limit", "2", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert isinstance(data, list)
    assert len(data) == 2
    assert "ectrl_id" in data[0]
    assert "detour_nm" in data[0]


def test_cli_fra_forecast_json():
    """Verify CLI fra-forecast command with --json."""
    res = runner.invoke(app, ["fra-forecast", "--growth", "2.0", "--scope", "prime_only", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert "horizons" in data
    assert "1" in data["horizons"]
    assert "5" in data["horizons"]
    assert "20" in data["horizons"]
    assert data["horizons"]["20"]["cumulative_cost_saved_eur"] > 0.0
