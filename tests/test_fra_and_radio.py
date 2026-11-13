"""
Unit & Integration Tests for Sprint 3:
- Free Route Airspace (FRA), ATS Airway Snapping (`raw_routes`), and 5-Letter RNAV Fix Snapping (`nav_fixes`)
- ATC Controller-Pilot VHF Radio Frequency Workload & Communications Occupancy (`raw_frequencies` & `nav_aids_xplane`)
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from flightml.api.app import app
from flightml.data.dataset import FlightDataset
from flightml.features.radio_workload import estimate_airport_and_sector_radio_workload
from flightml.geo.fra_analyzer import (
    analyze_flight_fra_adherence,
    compare_fra_evolution,
)
from flightml.geo.rnp_navaid_analyzer import analyze_flight_navaids_and_rnp
from flightml.viz.map_builder import create_trajectory_map


def test_analyze_flight_fra_adherence():
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute(
        "SELECT ectrl_id FROM flights WHERE actual_distance_nm BETWEEN 400 AND 1200 LIMIT 1"
    ).fetchone()
    assert row is not None
    ectrl_id = int(row[0])

    rep = analyze_flight_fra_adherence(ds, ectrl_id=ectrl_id, snap_tolerance_nm=3.0)
    assert rep is not None
    assert rep.ectrl_id == ectrl_id
    assert rep.enroute_distance_nm > 0.0
    assert 0.0 <= rep.fra_directness_index_pct <= 100.0
    assert abs((rep.fra_directness_index_pct + rep.ats_airway_adherence_pct) - 100.0) <= 0.5
    assert len(rep.waypoints) > 0


def test_compare_fra_evolution():
    evo = compare_fra_evolution(period1="201611", period2="202510", sample_flights=80)
    assert "period1" in evo
    assert "period2" in evo
    assert evo["period1"]["total_upper_waypoints"] > 0
    assert evo["period2"]["total_upper_waypoints"] > 0


def test_airport_radio_workload_and_fastapi():
    ds = FlightDataset(period="202510")
    res = estimate_airport_and_sector_radio_workload(ds, airport_icao="EGLL")
    assert res["airport_icao"] == "EGLL"
    assert res["total_vhf_channels_cataloged"] > 0
    assert res["peak_hourly_movements"] > 0
    assert len(res["positions"]) == 3

    client = TestClient(app)
    r_rad = client.get("/api/v1/airports/EGLL/radio-workload?period=202510")
    assert r_rad.status_code == 200
    assert r_rad.json()["airport_icao"] == "EGLL"


def test_analyze_flight_navaids_and_rnp():
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute(
        "SELECT ectrl_id FROM flights WHERE actual_distance_nm BETWEEN 300 AND 1200 LIMIT 1"
    ).fetchone()
    assert row is not None
    ectrl_id = int(row[0])

    rep = analyze_flight_navaids_and_rnp(ds, ectrl_id=ectrl_id)
    assert rep is not None
    assert rep.ectrl_id == ectrl_id
    assert rep.total_active_navaids > 0
    assert len(rep.profile_points) > 0
    assert all(p.anticipated_rnp_nm in (0.1, 0.3, 1.0, 2.0, 5.0, 10.0) for p in rep.profile_points)
    assert all(len(p.availability_reason) > 10 for p in rep.profile_points)

    flight = ds.get_flight(ectrl_id)
    trajs = ds.get_trajectory(ectrl_id)
    m = create_trajectory_map(flight, trajs["actual"], trajs["planned"], active_navaids=rep.active_navaids)
    html = m._repr_html_()
    assert "Active Route Navaids" in html

    client = TestClient(app)
    r_nav = client.get(f"/api/v1/flights/{ectrl_id}/navaids-rnp?period=202510")
    assert r_nav.status_code == 200
    assert r_nav.json()["total_active_navaids"] > 0


