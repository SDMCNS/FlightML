"""
Unit & Integration Tests for Sprint 6:
- Upper-Air Physics (`UpperAirClient`) & Persistent Contrail ISSR Avoidance (`contrail_climate.py`)
- Prescriptive Hub Tail-Swap Recovery Optimizer (`tail_swap_optimizer.py`)
- Heterogeneous Spatio-Temporal GNN (`HeterogeneousAirspaceGNN`)
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from flightml.api.app import app
from flightml.data.dataset import FlightDataset
from flightml.eval.contrail_climate import analyze_flight_wind_and_contrails
from flightml.models.gnn_delay import HeterogeneousAirspaceGNN
from flightml.sim.tail_swap_optimizer import optimize_hub_tail_swaps
from flightml.weather.upper_air_client import UpperAirClient


def test_upper_air_and_contrail_climate():
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute(
        "SELECT ectrl_id FROM flights WHERE actual_distance_nm >= 500 LIMIT 1"
    ).fetchone()
    assert row is not None
    fid = int(row[0])

    ua_client = UpperAirClient(use_live_api=False)
    pt = ua_client.get_upper_air_state(51.5, -0.4, 350.0, "2025-10-01 12:00:00")
    assert pt.pressure_hpa > 150.0
    assert pt.wind_speed_kts > 0.0
    assert pt.rh_ice_pct >= pt.rh_water_pct

    rep = analyze_flight_wind_and_contrails(ds, ectrl_id=fid, upper_air_client=ua_client)
    assert rep is not None
    assert rep.ectrl_id == fid
    assert rep.cruise_distance_nm > 0.0
    assert rep.avg_groundspeed_kts > 200.0
    assert rep.total_climate_co2e_multiplier >= 1.0
    assert rep.ground_distance_nm > 0.0
    assert rep.esad_nm > 0.0
    assert isinstance(rep.wind_distance_benefit_nm, float)
    assert isinstance(rep.wind_fuel_impact_kg, float)
    assert isinstance(rep.wind_fuel_impact_eur, float)
    assert rep.wind_efficiency_verdict != ""


def test_tail_swap_optimizer_and_hetero_gnn():
    ds = FlightDataset(period="201611")
    swap_res = optimize_hub_tail_swaps(ds, operator="RYR", date_str="2016-11-01")
    assert swap_res.operator == "RYR"
    assert swap_res.candidate_rotations_analyzed >= 0
    assert swap_res.total_delay_minutes_saved >= 0.0

    gnn = HeterogeneousAirspaceGNN(top_n_airports=8)
    gnn_res = gnn.train_and_evaluate(ds, epochs=5)
    assert gnn_res["num_airport_nodes"] == 8
    assert len(gnn_res["next_day_hub_delay_forecasts_min"]) == 8

    client = TestClient(app)
    r_ts = client.get("/api/v1/sim/tail-swaps?operator=RYR&period=201611&date=2016-11-01")
    assert r_ts.status_code == 200
    assert r_ts.json()["operator"] == "RYR"

