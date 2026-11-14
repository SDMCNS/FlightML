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


def test_crco_transatlantic_trajectory_traversal_and_map():
    """
    Test transatlantic flight 202007029 (KORD -> LFPG) in period 201611:
    - Traverses full sequence of 7 FIRs (not merely 2 naive departure/arrival FIRs).
    - Unlisted / non-CRCO oceanic airspaces default strictly to €45.00 EUR/SU.
    - Official CRCO rates apply for Shanwick/UK, Shannon/Ireland, London/UK, and Paris/France.
    - create_flight_fir_charge_map generates an interactive Folium map with FIR polygons.
    """
    import folium
    from flightml.viz.map_builder import create_flight_fir_charge_map

    ds = FlightDataset(period="201611")
    rep = compute_flight_crco_charges(ds, ectrl_id=202007029)
    assert rep is not None
    assert rep.ectrl_id == 202007029
    assert rep.adep == "KORD"
    assert rep.ades == "LFPG"

    # Must have all 7 traversed FIRs
    assert len(rep.fir_breakdown) == 7
    fir_ids = [b.fir_id for b in rep.fir_breakdown]
    assert fir_ids == ["KKKKFIR", "CCCCFIR", "CZZZFIR", "EGGXFIR", "EISNFIR", "EGTTFIR", "LFFFFIR"]

    # Check fallback rates (€45.00 EUR for KKKK, CCCC, CZZZ)
    rate_map = {b.fir_id: b.unit_rate_eur for b in rep.fir_breakdown}
    assert rate_map["KKKKFIR"] == 45.0
    assert rate_map["CCCCFIR"] == 45.0
    assert rate_map["CZZZFIR"] == 45.0
    assert rate_map["EGGXFIR"] == 87.75
    assert rate_map["EISNFIR"] == 34.70
    assert rate_map["EGTTFIR"] == 87.75
    assert rate_map["LFFFFIR"] == 79.58

    # Sum of segment distances matches actual flight distance
    total_seg_nm = sum(b.estimated_distance_nm for b in rep.fir_breakdown)
    assert abs(total_seg_nm - rep.actual_distance_nm) < 1.0

    # Total navigation fee computed
    assert rep.actual_crco_charge_eur > 6000.0

    # Interactive FIR underlay Folium map
    fir_map = create_flight_fir_charge_map(ds, 202007029, crco_report=rep)
    assert isinstance(fir_map, folium.Map)


