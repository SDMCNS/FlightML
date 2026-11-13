"""
Unit & Integration Tests for Sprint 5:
- 4D Co-Altitude Airspace Proximity Scanner & Sector Dynamic Complexity (`conflict_scanner.py`)
- Arrival Runway Braking Action, Density Altitude & Overrun Safety Margin (`runway_safety.py`)
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from flightml.api.app import app
from flightml.data.dataset import FlightDataset
from flightml.geo.conflict_scanner import (
    compute_sector_complexity_index,
    scan_airspace_proximity_events,
)
from flightml.weather.runway_safety import assess_flight_runway_safety


def test_scan_airspace_proximity_and_complexity():
    ds = FlightDataset(period="202510")
    rep = scan_airspace_proximity_events(ds, date_str="2025-10-01", hour=10, horiz_sep_nm=6.0, limit=10)
    assert rep.period == "202510"
    assert rep.events_detected >= 0
    for ev in rep.events:
        assert ev.horizontal_sep_nm <= 6.0
        assert ev.vertical_sep_ft <= 1000.0

    comp = compute_sector_complexity_index(ds, date_str="2025-10-01", hour=10)
    assert comp["period"] == "202510"
    assert len(comp["sectors"]) > 0


def test_assess_flight_runway_safety_and_fastapi():
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute("SELECT ectrl_id FROM flights WHERE ades = 'EGLL' LIMIT 1").fetchone()
    if not row:
        row = ds.cursor.execute("SELECT ectrl_id FROM flights LIMIT 1").fetchone()
    assert row is not None
    fid = int(row[0])

    rep = assess_flight_runway_safety(ds, metar_client=None, ectrl_id=fid, fetch_live_metar=False)
    assert rep is not None
    assert rep.ectrl_id == fid
    assert rep.available_lda_m > 500.0
    assert rep.factored_required_landing_dist_m > 300.0
    assert rep.lda_utilization_pct > 0.0

    client = TestClient(app)
    r1 = client.get(f"/api/v1/flights/{fid}/runway-safety?period=202510")
    assert r1.status_code == 200
    assert r1.json()["ectrl_id"] == fid

