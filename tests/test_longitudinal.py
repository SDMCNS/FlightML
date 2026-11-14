import pytest
from flightml.eval.longitudinal import (
    LongitudinalEngine,
    PeriodMetadata,
    PeriodMacroMetrics,
    YearOverYearDelta,
    AnnualSeasonalDecomposition,
)


@pytest.fixture(scope="module")
def engine():
    return LongitudinalEngine()


def test_period_metadata():
    m_baseline = PeriodMetadata.from_period_id("201611")
    assert m_baseline.year == 2016
    assert m_baseline.month == 11
    assert m_baseline.season_label == "Historical Baseline"
    assert m_baseline.days_in_month == 30

    m_feb = PeriodMetadata.from_period_id("202402")
    assert m_feb.year == 2024
    assert m_feb.month == 2
    assert m_feb.is_leap_year is True
    assert m_feb.days_in_month == 29
    assert m_feb.season_label == "Winter Trough"

    m_jun = PeriodMetadata.from_period_id("202506")
    assert m_jun.season_label == "Summer Peak"
    assert m_jun.days_in_month == 30

    m_oct = PeriodMetadata.from_period_id("202510")
    assert m_oct.season_label == "Autumn Shoulder"
    assert m_oct.days_in_month == 31


def test_get_period_metrics(engine):
    metrics = engine.get_period_metrics("202510")
    assert isinstance(metrics, PeriodMacroMetrics)
    assert metrics.total_flights > 900_000
    assert metrics.daily_flights > 28_000
    assert metrics.avg_arr_delay_min > 0
    assert metrics.newgen_share_pct > 25.0
    assert len(metrics.top_airlines) > 0
    assert len(metrics.top_aircraft_types) > 0
    assert len(metrics.top_airports) > 0

    d = metrics.to_dict()
    assert "metadata" in d
    assert d["period_id"] == "202510"


def test_get_timeline_dataframe(engine):
    tl = engine.get_timeline_dataframe()
    assert len(tl) >= 7
    assert "period_id" in tl.columns
    assert "daily_flights" in tl.columns
    assert "newgen_share_pct" in tl.columns
    assert "201611" in tl["period_id"].values
    assert "202510" in tl["period_id"].values


def test_compare_yoy_october(engine):
    yoy = engine.compare_yoy("202410", "202510")
    assert isinstance(yoy, YearOverYearDelta)
    assert yoy.total_flights_delta > 0
    assert yoy.total_flights_growth_pct > 0
    assert yoy.newgen_share_delta_pct > 0  # Fleet modernization increased
    assert len(yoy.top_growing_airlines) > 0
    assert len(yoy.top_aircraft_type_shifts) > 0
    assert len(yoy.top_city_pairs_evolution) > 0

    d = yoy.to_dict()
    assert d["period_baseline"] == "202410"
    assert d["period_target"] == "202510"


def test_compare_yoy_summer(engine):
    yoy = engine.compare_yoy("202406", "202506")
    assert yoy.total_flights_delta > 0
    assert yoy.daily_flights_growth_pct > 0
    assert yoy.season_label == "Summer Peak"


def test_decompose_seasonality_2024(engine):
    decomp = engine.decompose_seasonality(2024)
    assert isinstance(decomp, AnnualSeasonalDecomposition)
    assert decomp.year == 2024
    assert decomp.summer_surge_pct > 35.0  # Significant surge from Feb to Jun
    assert decomp.summer_delay_surge_pct > 100.0  # Delays surge in summer
    assert decomp.delay_demand_elasticity > 1.0  # Super-linear delay sensitivity
    assert len(decomp.top_surging_airports) > 0
    assert len(decomp.top_surging_firs) > 0

    d = decomp.to_dict()
    assert d["year"] == 2024
    assert "top_surging_airports" in d


def test_get_fleet_evolution_timeline(engine):
    fleet_df = engine.get_fleet_evolution_timeline()
    assert len(fleet_df) >= 7
    assert "airbus_neo_flights" in fleet_df.columns
    assert "boeing_max_flights" in fleet_df.columns
    assert "a220_flights" in fleet_df.columns
    assert "newgen_share_pct" in fleet_df.columns

    # 2016 should have near-zero newgen narrowbodies compared to 2025
    p2016 = fleet_df[fleet_df["period_id"] == "201611"].iloc[0]
    p2025 = fleet_df[fleet_df["period_id"] == "202510"].iloc[0]
    assert p2016["airbus_neo_flights"] == 0
    assert p2025["airbus_neo_flights"] > 100_000
    assert p2025["newgen_share_pct"] > p2016["newgen_share_pct"]


def test_yoy_capable_periods_and_cohort_filter(engine):
    yoy_periods = engine.get_yoy_capable_periods()
    assert "201611" not in yoy_periods  # Isolated anchor without matching month
    assert "202402" in yoy_periods
    assert "202502" in yoy_periods
    assert "202406" in yoy_periods
    assert "202506" in yoy_periods
    assert "202410" in yoy_periods
    assert "202510" in yoy_periods

    pairs = engine.get_yoy_matching_pairs()
    assert len(pairs) >= 3
    pair_labels = [p["label"] for p in pairs]
    assert any("202402 ➔ 202502" in l for l in pair_labels)
    assert any("202406 ➔ 202506" in l for l in pair_labels)
    assert any("202410 ➔ 202510" in l for l in pair_labels)

    # Filtered timeline cohort
    tl_yoy = engine.get_timeline_dataframe(cohort="yoy")
    assert len(tl_yoy) == 6
    assert "201611" not in tl_yoy["period_id"].values
    assert all(tl_yoy["is_yoy_cohort"])

    # Baseline-only cohort
    tl_base = engine.get_timeline_dataframe(cohort="baseline")
    assert "201611" in tl_base["period_id"].values
    assert not any(tl_base["is_yoy_cohort"])
