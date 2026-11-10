from starlette.testclient import TestClient
from flightml.api.app import app

client = TestClient(app)


def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["service"] == "FlightML API"


def test_summary():
    res = client.get("/api/v1/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_flights"] > 600000


def test_top_airports():
    res = client.get("/api/v1/airports/top?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 5
    assert data[0]["ident"] == "EGLL"


def test_predict_delay():
    payload = {
        "adep": "EGLL",
        "ades": "KORD",
        "ac_operator": "BAW",
        "market_segment": "Traditional Scheduled",
        "requested_fl": 360,
        "actual_distance_nm": 3500,
        "dep_hour": 14,
        "dep_dayofweek": 3,
        "filed_air_time_min": 480,
    }
    res = client.post("/api/v1/predict/delay", json=payload)
    assert res.status_code == 200
    assert "predicted_arrival_delay_min" in res.json()


def test_periods_endpoints():
    res = client.get("/api/v1/periods")
    assert res.status_code == 200
    assert "available_periods" in res.json()
    assert len(res.json()["available_periods"]) >= 1

    # Test period comparison
    res = client.get("/api/v1/periods/compare?p1=201611&p2=202510")
    assert res.status_code == 200
    data = res.json()
    assert data["total_flights_baseline"] > 600000
    assert data["total_flights_comparison"] > 900000
    assert data["traffic_growth_percentage"] > 30


def test_cdo_analysis_endpoint():
    res = client.get("/api/v1/flights/202007026/cdo")
    assert res.status_code == 200
    data = res.json()
    assert data["ectrl_id"] == 202007026
    assert "cdo_compliance_score_pct" in data
    assert "inefficiency_events_count" in data


def test_twins_endpoint():
    res = client.get("/api/v1/twins/compare?adep=RKSI&ades=EGLL&ac_type=B77W")
    assert res.status_code == 200
    data = res.json()
    assert data["route"] == "RKSI ➔ EGLL"
    assert data["disparity_deltas"]["excess_fuel_burned_kg"] > 0


def test_airac_diff_endpoint():
    res = client.get("/api/v1/airac/diff?c1=2509&c2=2510")
    assert res.status_code == 200
    data = res.json()
    assert "airway_routes" in data
    assert "fir_airspaces" in data
    assert data["airway_routes"]["added_airways"] > 0


def test_weather_impact_endpoint():
    res = client.get("/api/v1/weather/impact?start_date=2016-11-01&end_date=2016-11-03")
    assert res.status_code == 200
    data = res.json()
    assert "correlations" in data
    assert "daily_trends" in data
    assert len(data["daily_trends"]) == 3


def test_search_flights_endpoint():
    res = client.get("/api/v1/flights/search?adep=EGLL&ades=KORD&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 5
    assert "label" in data[0]
    assert "EGLL" in data[0]["label"]


def test_recurring_series_endpoint():
    res = client.get("/api/v1/flights/recurring?adep=EGLL&ades=KORD&operator=BAW")
    assert res.status_code == 200
    data = res.json()
    assert data["route"] == "EGLL ➔ KORD"
    assert "diurnal_time_of_day_breakdown" in data
    assert data["total_flights"] > 10


def test_fir_saturation_endpoint():
    res = client.get("/api/v1/fir/saturation?period=202510&threshold=40&top_n=5")
    assert res.status_code == 200
    data = res.json()
    assert data["has_data"] is True
    assert len(data["top_saturated_firs"]) <= 5
    assert data["top_saturated_firs"][0]["peak_hourly_flights"] > 0


def test_flight_fir_dwell_endpoint():
    res = client.get("/api/v1/flights/291262109/fir-dwell?period=202510")
    assert res.status_code == 200
    data = res.json()
    assert data["has_data"] is True
    assert data["ectrl_id"] == 291262109
    assert len(data["sectors"]) > 0
    assert "verdict" in data


def test_fuel_prices_endpoints():
    res = client.get("/api/v1/fuel/prices")
    assert res.status_code == 200
    prices = res.json()
    assert len(prices) >= 10
    assert prices[0]["price_usd_per_gallon"] > 1.0
    assert prices[0]["price_usd_per_kg"] > 0.3

    # Filtered by period
    res_p = client.get("/api/v1/fuel/prices?period=201611")
    assert res_p.status_code == 200
    assert len(res_p.json()) == 5

    # Flight fuel impact endpoint
    res_f = client.get("/api/v1/flights/202007026/fuel?period=201611")
    assert res_f.status_code == 200
    f_data = res_f.json()
    assert f_data["ectrl_id"] == 202007026
    assert f_data["fuel_price_usd_per_gallon"] == 1.368
    assert f_data["fuel_price_usd_per_kg"] == 0.4496
    assert "2016-W44" in f_data["fuel_price_week"]



