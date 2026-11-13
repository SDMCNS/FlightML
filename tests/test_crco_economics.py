"""
Unit & Integration Tests for Sprint 4:
- Eurocontrol CRCO En-Route Navigation Charges (`compute_flight_crco_charges`)
- Corridor Route-Charge Avoidance vs. Fuel Burn Trade-Off (`detect_route_charge_avoidance`)
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from flightml.api.app import app
from flightml.data.dataset import FlightDataset
from flightml.eval.crco_economics import (
    compute_flight_crco_charges,
    detect_route_charge_avoidance,
)


def test_compute_flight_crco_charges():
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute(
        "SELECT ectrl_id FROM flights WHERE actual_distance_nm >= 300 LIMIT 1"
    ).fetchone()
    assert row is not None
    ectrl_id = int(row[0])

    rep = compute_flight_crco_charges(ds, ectrl_id=ectrl_id)
    assert rep is not None
    assert rep.ectrl_id == ectrl_id
    assert rep.mtow_tons > 0.0
    assert rep.crco_weight_factor > 0.0
    assert rep.total_service_units > 0.0
    assert rep.actual_crco_charge_eur > 0.0
    assert rep.actual_crco_charge_usd > 0.0
    assert rep.total_direct_route_cost_usd > rep.actual_crco_charge_usd
    assert len(rep.fir_breakdown) > 0


def test_detect_route_charge_avoidance_and_fastapi():
    ds = FlightDataset(period="202510")
    pair = ds.cursor.execute(
        """
        SELECT adep, ades
        FROM flights
        WHERE actual_distance_nm >= 300
        GROUP BY adep, ades
        HAVING count(*) >= 5
        LIMIT 1
        """
    ).fetchone()
    assert pair is not None
    adep, ades = str(pair[0]), str(pair[1])

    res = detect_route_charge_avoidance(ds, adep=adep, ades=ades, limit=8)
    assert res["analyzed_flights"] >= 5
    assert len(res["flights"]) >= 5

    client = TestClient(app)
    fid = int(res["flights"][0]["ectrl_id"])
    r1 = client.get(f"/api/v1/flights/{fid}/crco-cost?period=202510")
    assert r1.status_code == 200
    assert r1.json()["ectrl_id"] == fid

