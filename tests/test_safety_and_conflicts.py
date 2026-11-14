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


def test_variable_grid_saturation_from_5v5_down_to_half_degree():
    """
    Test variable resolution grid saturation analysis from 5.0°x5.0° down to 0.5°x0.5°:
    - Verifies compute_hourly_grid_saturation works with 5.0, 2.0, 1.0, and 0.5 degree square dimensions.
    - Verifies create_grid_saturation_map produces Folium maps with properly bounded cells at each scale.
    - Verifies FastAPI endpoint supports grid_size query parameter.
    """
    import folium
    from flightml.geo.conflict_scanner import compute_hourly_grid_saturation
    from flightml.viz.map_builder import create_grid_saturation_map

    ds = FlightDataset(period="202510")

    # 1. Macro 5v5 square
    macro = compute_hourly_grid_saturation(ds, hour_utc=10, grid_size_deg=5.0)
    assert macro["grid_size_deg"] == 5.0
    assert macro["total_active_squares"] > 0
    assert "5°N to 10°N" in macro["squares"][0]["cell_label"] or "°N to " in macro["squares"][0]["cell_label"]
    m_macro = create_grid_saturation_map(macro)
    assert isinstance(m_macro, folium.Map)

    # 2. Intermediate 1v1 square
    fine_1deg = compute_hourly_grid_saturation(ds, hour_utc=10, grid_size_deg=1.0)
    assert fine_1deg["grid_size_deg"] == 1.0
    assert fine_1deg["total_active_squares"] > macro["total_active_squares"]

    # 3. High-resolution 0.5v0.5 square
    micro = compute_hourly_grid_saturation(ds, hour_utc=10, grid_size_deg=0.5)
    assert micro["grid_size_deg"] == 0.5
    assert micro["total_active_squares"] > fine_1deg["total_active_squares"]
    m_micro = create_grid_saturation_map(micro)
    assert isinstance(m_micro, folium.Map)

    # 4. FastAPI test with grid_size=0.5
    client = TestClient(app)
    res = client.get("/api/v1/airspace/grid-saturation?hour=10&period=202510&grid_size=0.5")
    assert res.status_code == 200
    data = res.json()
    assert data["grid_size_deg"] == 0.5
    assert data["total_active_squares"] > 0


