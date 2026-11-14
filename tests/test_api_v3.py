import pytest
from starlette.testclient import TestClient
from flightml.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_api_v3_longitudinal_timeline(client):
    response = client.get("/api/v3/longitudinal/timeline")
    assert response.status_code == 200
    data = response.json()
    assert "timeline" in data
    assert data["periods_count"] >= 7
    assert any(p["period_id"] == "202510" for p in data["timeline"])


def test_api_v3_longitudinal_compare(client):
    response = client.get("/api/v3/longitudinal/compare?baseline=202410&target=202510")
    assert response.status_code == 200
    data = response.json()
    assert data["period_baseline"] == "202410"
    assert data["period_target"] == "202510"
    assert data["total_flights_growth_pct"] > 0
    assert "top_aircraft_type_shifts" in data


def test_api_v3_longitudinal_seasonality(client):
    response = client.get("/api/v3/longitudinal/seasonality?year=2024")
    assert response.status_code == 200
    data = response.json()
    assert data["year"] == 2024
    assert data["summer_surge_pct"] > 30.0
    assert data["delay_demand_elasticity"] > 1.0


def test_api_v3_predict_airspace(client):
    response = client.get("/api/v3/predict/airspace?year=2026&season=summer&growth=3.5&grid_size=1.0")
    assert response.status_code == 200
    data = response.json()
    assert "Summer 2026" in data["forecast_label"]
    assert len(data["top_capacity_breaches"]) > 0
    assert len(data["top_grid_hotspots"]) > 0


def test_api_v3_flight_doc(client):
    # Test valid flight ID from 202510
    response = client.get("/api/v3/economics/flight-doc/291262109?period=202510")
    assert response.status_code == 200
    data = response.json()
    assert data["ectrl_id"] == 291262109
    assert data["fuel_cost_eur"] > 0
    assert data["time_crew_cost_eur"] > 0
    assert data["crco_navigation_charges_eur"] > 0
    assert data["total_doc_eur"] > 0


def test_api_v3_simulate_harmonization(client):
    response = client.get("/api/v3/economics/simulate-harmonization?period=202510&rate=72.50&sample=15")
    assert response.status_code == 200
    data = response.json()
    assert data["period"] == "202510"
    assert data["harmonized_unit_rate_eur"] == 72.50
    assert data["estimated_annual_fuel_saved_tonnes"] > 0


def test_api_v3_delay_spreaders(client):
    response = client.get("/api/v3/delay/spreaders?period=202510&date=2025-10-15&min_delay=20.0&top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert data["period"] == "202510"
    assert "super_spreaders" in data


def test_api_v3_network_shock(client):
    response = client.get("/api/v3/sim/network-shock?target=EGLL&target_type=AIRPORT&duration=2.0&capacity_cut=50.0&period=202510")
    assert response.status_code == 200
    data = response.json()
    assert data["target_id"] == "EGLL"
    assert data["total_impacted_flights"] > 0
    assert data["total_economic_delay_cost_eur"] > 0
