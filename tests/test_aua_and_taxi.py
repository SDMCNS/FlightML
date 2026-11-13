"""
Unit & Integration Tests for Sprint 1:
- ATC Unit Airspace (AUA: TMA, CTA, UTA) Hourly Saturation & Flight Dwell Inefficiency
- Airport Surface Movement (TAXI_OUT) / ICAO KPI02 Additional Taxi-Out Time (ATOT) & LTO Idle Emissions
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from flightml.api.app import app
from flightml.data.dataset import FlightDataset
from flightml.eval.surface_taxi import (
    analyze_airport_taxi_out_performance,
    analyze_flight_taxi_out,
)
from flightml.features.aua_saturation import (
    calculate_aua_hourly_saturation,
    calculate_flight_aua_dwell_inefficiency,
)


def test_aua_hourly_saturation_tma_and_cta():
    ds = FlightDataset(period="202510")
    assert ds.has_aua_traversals() is True

    res_tma = calculate_aua_hourly_saturation(ds, aua_type="TMA", threshold_flights_per_hr=25, top_n=5)
    assert res_tma["has_data"] is True
    assert len(res_tma["top_saturated_auas"]) > 0
    for r in res_tma["top_saturated_auas"]:
        assert r["aua_type"] == "TMA"
        assert r["peak_hourly_flights"] > 0

    res_cta = calculate_aua_hourly_saturation(ds, aua_type="CTA", threshold_flights_per_hr=30, top_n=5)
    assert res_cta["has_data"] is True
    assert len(res_cta["top_saturated_auas"]) > 0
    for r in res_cta["top_saturated_auas"]:
        assert r["aua_type"] == "CTA"


def test_flight_aua_dwell_inefficiency():
    ds = FlightDataset(period="202510")
    # Pick a sample flight from flight_auas_actual
    row = ds.cursor.execute("SELECT ectrl_id FROM flight_auas_actual LIMIT 1").fetchone()
    assert row is not None
    ectrl_id = int(row[0])

    rep = calculate_flight_aua_dwell_inefficiency(ds, ectrl_id=ectrl_id)
    assert rep["has_data"] is True
    assert rep["ectrl_id"] == ectrl_id
    assert len(rep["sectors"]) > 0
    assert "tma_dwell_delta_min" in rep
    assert "enroute_cta_uta_dwell_delta_min" in rep


def test_airport_taxi_out_performance():
    ds = FlightDataset(period="202510")
    res = analyze_airport_taxi_out_performance(ds, min_flights=100, top_n=8)
    assert res["has_data"] is True
    assert len(res["airports"]) > 0
    top = res["airports"][0]
    assert top["departures"] >= 100
    assert top["unimpeded_utot_min"] > 0
    assert top["avg_actual_taxi_min"] >= top["unimpeded_utot_min"]
    assert top["avg_additional_atot_min"] >= 0
    assert top["excess_taxi_co2_kg"] >= 0


def test_single_flight_taxi_out_report():
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute(
        "SELECT ectrl_id FROM flight_firs_actual WHERE fir_id = 'TAXI_OUT' LIMIT 1"
    ).fetchone()
    assert row is not None
    ectrl_id = int(row[0])

    rep = analyze_flight_taxi_out(ds, ectrl_id=ectrl_id)
    assert rep is not None
    assert rep.ectrl_id == ectrl_id
    assert rep.actual_taxi_out_min > 0
    assert rep.unimpeded_utot_min > 0
    assert rep.additional_atot_min >= 0
    assert rep.idle_fuel_rate_kg_min > 0
    assert rep.total_taxi_fuel_kg > 0


def test_sprint1_fastapi_endpoints():
    client = TestClient(app)

    r1 = client.get("/api/v1/aua/saturation?period=202510&aua_type=TMA&top_n=3")
    assert r1.status_code == 200
    assert r1.json()["has_data"] is True

    r2 = client.get("/api/v1/surface/taxi-out?period=202510&min_flights=100&top_n=3")
    assert r2.status_code == 200
    assert len(r2.json()["airports"]) > 0

