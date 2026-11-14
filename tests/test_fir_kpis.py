"""
Unit & Integration Tests for FIR Multi-Level KPIs and Traffic Segmentation:
- get_available_firs
- query_fir_traffic_segmentation (Overflights, Arrivals, Departures, Internal)
- query_multi_level_kpis with FIR tier and FIR scoping filter
"""

from __future__ import annotations

from flightml.data.dataset import FlightDataset
from flightml.eval.multi_level_kpis import (
    get_available_firs,
    query_fir_traffic_segmentation,
    query_multi_level_kpis,
)


def test_get_available_firs():
    ds25 = FlightDataset(period="202510")
    firs25 = get_available_firs(ds25)
    assert len(firs25) > 0
    assert any("EGTTFIR" in f["fir_id"] for f in firs25)
    assert any("LFFFFIR" in f["fir_id"] for f in firs25)

    ds16 = FlightDataset(period="201611")
    firs16 = get_available_firs(ds16)
    assert len(firs16) > 0


def test_query_fir_traffic_segmentation_202510():
    ds = FlightDataset(period="202510")
    seg = query_fir_traffic_segmentation(ds, fir_id="EGTTFIR")
    assert seg.fir_id == "EGTTFIR"
    assert "United Kingdom" in seg.country_name or "EG" in seg.country_name
    assert seg.unit_rate_eur == 87.75
    assert seg.total_flights > 1000

    # Overflights, Arrivals, Departures, Internal are all present
    assert seg.overflights_count > 0
    assert seg.arrivals_count > 0
    assert seg.departures_count > 0
    assert seg.internal_count > 0

    total_sum = seg.overflights_count + seg.arrivals_count + seg.departures_count + seg.internal_count
    assert total_sum == seg.total_flights

    pct_sum = seg.overflights_pct + seg.arrivals_pct + seg.departures_pct + seg.internal_pct
    assert 99.0 <= pct_sum <= 101.0

    # Top corridors populated
    assert len(seg.top_corridors["Overflight"]) > 0
    assert len(seg.top_corridors["Arrival"]) > 0
    assert len(seg.top_corridors["Departure"]) > 0


def test_query_fir_traffic_segmentation_201611():
    ds = FlightDataset(period="201611")
    seg = query_fir_traffic_segmentation(ds, fir_id="EGTTFIR")
    assert seg.fir_id == "EGTTFIR"
    assert seg.arrivals_count > 0
    assert seg.departures_count > 0
    assert seg.internal_count > 0
    assert seg.total_flights > 0


def test_query_multi_level_kpis_fir_and_filter():
    ds = FlightDataset(period="202510")

    # FIR tier grouping
    kpi_fir = query_multi_level_kpis(ds, level="fir", limit=10)
    assert kpi_fir.level == "fir"
    assert len(kpi_fir.records) > 0
    assert kpi_fir.records[0]["total_flights"] > 0

    # Fleet tier scoped to EGTTFIR
    kpi_fleet_fir = query_multi_level_kpis(ds, level="fleet", fir_filter="EGTTFIR", limit=5)
    assert kpi_fleet_fir.level == "fleet"
    assert len(kpi_fleet_fir.records) > 0
