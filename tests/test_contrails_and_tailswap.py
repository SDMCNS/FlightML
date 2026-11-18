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
from flightml.viz.map_builder import create_contrail_horizontal_map, create_trajectory_map
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
    assert isinstance(rep.annotated_profile, list)
    assert len(rep.annotated_profile) > 0
    p0 = rep.annotated_profile[0]
    assert "flight_level" in p0
    assert "cum_dist_nm" in p0
    assert "is_issr_contrail" in p0

    # Test Map Visualizations with Contrail Elements
    fl_info = ds.get_flight(fid) or {}
    trajs = ds.get_trajectory(fid)
    act_df = trajs["actual"]
    plan_df = trajs["planned"]

    c_map = create_contrail_horizontal_map(fl_info, act_df, rep.issr_segments)
    assert c_map is not None

    traj_map = create_trajectory_map(fl_info, act_df, plan_df, contrail_segments=rep.issr_segments)
    assert traj_map is not None


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


def test_eco_optimal_minimal_contrail_trajectory():
    import json
    from typer.testing import CliRunner
    from flightml.cli import app as cli_app
    from flightml.eval.eco_route_optimizer import optimize_eco_minimal_contrail_trajectory
    from flightml.mcp.server import create_mcp_server
    from flightml.viz.map_builder import create_contrail_horizontal_map

    ds = FlightDataset(period="202510")
    row = ds.cursor.execute(
        "SELECT ectrl_id FROM flights WHERE actual_distance_nm >= 500 LIMIT 1"
    ).fetchone()
    assert row is not None
    fid = int(row[0])

    ua_client = UpperAirClient(use_live_api=False)
    eco_rep = optimize_eco_minimal_contrail_trajectory(ds, ectrl_id=fid, upper_air_client=ua_client)
    assert eco_rep is not None
    assert eco_rep.ectrl_id == fid
    assert eco_rep.operational_ceiling_fl >= 300.0
    assert eco_rep.eco_fuel_kg > 0.0
    assert eco_rep.eco_co2_kg > 0.0
    assert eco_rep.eco_nox_kg > 0.0
    assert isinstance(eco_rep.net_climate_reduction_pct, float)
    assert len(eco_rep.eco_waypoints) > 0
    assert eco_rep.is_within_aircraft_envelope is True

    # Verify aircraft capabilities & ceiling constraint
    max_fl = max(w.flight_level for w in eco_rep.eco_waypoints)
    assert max_fl <= eco_rep.operational_ceiling_fl

    # Verify horizontal map with eco_route_waypoints
    fl_info = ds.get_flight(fid) or {}
    trajs = ds.get_trajectory(fid)
    act_df = trajs["actual"]
    m = create_contrail_horizontal_map(
        fl_info,
        act_df,
        contrail_segments=None,
        eco_route_waypoints=eco_rep.eco_waypoints,
    )
    assert m is not None

    # Test CLI Command
    runner = CliRunner()
    res = runner.invoke(cli_app, ["eco-route", str(fid), "--period", "202510", "--json"])
    assert res.exit_code == 0
    cli_data = json.loads(res.stdout)
    assert cli_data["ectrl_id"] == fid
    assert "eco_fuel_kg" in cli_data
    assert "net_climate_reduction_pct" in cli_data

    # Test MCP Server Tool
    from flightml.mcp.server import create_mcp_server
    import asyncio
    mcp_server = create_mcp_server()
    tools = asyncio.run(mcp_server.list_tools())
    tool_names = [t.name for t in tools]
    assert "optimize_eco_flight_trajectory" in tool_names

