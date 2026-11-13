from flightml.eval.jet_fuel_prices import (
    get_weekly_fuel_price,
    get_all_weekly_fuel_prices_df,
    JET_A_KG_PER_GALLON,
)
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
from flightml.data.dataset import FlightDataset


def test_weekly_fuel_price_lookups():
    # November 2016 week 44
    p1 = get_weekly_fuel_price("2016-11-02 08:30:00")
    assert p1.period_id == "201611"
    assert p1.week_number == 44
    assert p1.price_eur_per_gallon == 1.265
    assert p1.price_eur_per_kg == round(1.265 / JET_A_KG_PER_GALLON, 4)

    # October 2025 week 42
    p2 = get_weekly_fuel_price("2025-10-15 14:10:00")
    assert p2.period_id == "202510"
    assert p2.week_number == 42
    assert p2.price_eur_per_gallon == 2.021
    assert p2.price_eur_per_kg == round(2.021 / JET_A_KG_PER_GALLON, 4)


def test_all_weekly_fuel_prices_df():
    df = get_all_weekly_fuel_prices_df()
    assert len(df) == 10
    assert "201611" in df["period_id"].values
    assert "202510" in df["period_id"].values
    assert (df["price_eur_per_metric_ton"] > 300).all()


def test_dynamic_fuel_emissions_calculation():
    ds = FlightDataset("201611")
    impact = compute_flight_fuel_and_emissions(ds, 202007026)
    assert impact.ectrl_id == 202007026
    # Flight departed 2016-11-01 -> Week 44 (€1.265/gal -> €0.4157/kg)
    assert impact.fuel_price_eur_per_gallon == 1.265
    assert impact.fuel_price_eur_per_kg == 0.4157
    assert "2016-W44" in impact.fuel_price_week
    assert impact.estimated_excess_fuel_cost_eur > 0

    # Test flight 202007029 (KORD -> LFPG on 2016-11-01)
    impact29 = compute_flight_fuel_and_emissions(ds, 202007029)
    assert impact29.ectrl_id == 202007029
    assert impact29.fuel_price_eur_per_gallon == 1.265
    assert impact29.fuel_price_eur_per_kg == 0.4157
    assert hasattr(impact29, "fuel_price_eur_per_gallon")


