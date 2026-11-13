"""
Unit & Integration Tests for Sprint 2:
- ICAO GANP & Eurocontrol PRU Standard KPI Suite (KPI04 KEP, KPI05 KEA, KPI08 ASMA, KPI19 Cruise Level-Capping)
- Continuous Climb Operations (KPI17 CCO) & Unified Vertical Flight Efficiency Scorecard
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from flightml.api.app import app
from flightml.data.dataset import FlightDataset
from flightml.eval.cdo_analyzer import (
    analyze_cco_climb,
    analyze_vertical_profile_efficiency,
)
from flightml.eval.pru_kpis import (
    analyze_network_level_capping,
    compute_flight_pru_kpis,
)


def test_compute_flight_pru_kpis():
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute(
        "SELECT ectrl_id FROM flights WHERE actual_distance_nm >= 400 LIMIT 1"
    ).fetchone()
    assert row is not None
    ectrl_id = int(row[0])

    sc = compute_flight_pru_kpis(ds, ectrl_id=ectrl_id)
    assert sc is not None
    assert sc.ectrl_id == ectrl_id
    assert sc.great_circle_od_nm > 0
    assert sc.enroute_achieved_dist_nm > 0
    assert sc.kpi04_kep_pct >= 0.0
    assert sc.kpi05_kea_pct >= 0.0
    assert sc.asma_40nm_actual_min > 0.0
    assert sc.max_actual_fl > 0.0


def test_analyze_cco_climb_and_vertical_profile():
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute(
        "SELECT ectrl_id FROM flights WHERE actual_distance_nm >= 400 LIMIT 1"
    ).fetchone()
    assert row is not None
    ectrl_id = int(row[0])

    cco = analyze_cco_climb(ds, ectrl_id=ectrl_id)
    assert cco is not None
    assert 0.0 <= cco.cco_compliance_score <= 100.0
    assert cco.total_climb_duration_min > 0.0

    vert = analyze_vertical_profile_efficiency(ds, ectrl_id=ectrl_id)
    assert vert is not None
    assert vert["ectrl_id"] == ectrl_id
    assert "cco_climb" in vert
    assert "cruise_level_capping" in vert
    assert "cdo_descent" in vert


def test_network_level_capping_and_fastapi():
    ds = FlightDataset(period="201611")
    res = analyze_network_level_capping(ds, min_distance_nm=300.0, top_n=5)
    assert res["analyzed_flights"] > 0
    assert 0.0 <= res["level_capped_pct"] <= 100.0

    client = TestClient(app)
    row = ds.cursor.execute(
        "SELECT ectrl_id FROM flights WHERE actual_distance_nm >= 400 LIMIT 1"
    ).fetchone()
    fid = int(row[0])

    r1 = client.get(f"/api/v1/flights/{fid}/pru-kpis?period=201611")
    assert r1.status_code == 200
    assert r1.json()["ectrl_id"] == fid

    r2 = client.get(f"/api/v1/flights/{fid}/cco?period=201611")
    assert r2.status_code == 200
    assert r2.json()["ectrl_id"] == fid

