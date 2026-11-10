from flightml.data.dataset import FlightDataset
from flightml.features.fir_saturation import (
    calculate_fir_hourly_saturation,
    calculate_flight_fir_dwell_inefficiency,
)


def test_fir_hourly_saturation():
    ds = FlightDataset(period="202510")
    assert ds.has_fir_traversals() is True

    res = calculate_fir_hourly_saturation(ds, threshold_flights_per_hr=50, top_n=5)
    assert res["has_data"] is True
    assert len(res["top_saturated_firs"]) == 5
    first_fir = res["top_saturated_firs"][0]
    assert "fir_id" in first_fir
    assert first_fir["peak_hourly_flights"] > 100
    assert first_fir["saturation_pct"] > 50.0


def test_flight_fir_dwell_inefficiency():
    ds = FlightDataset(period="202510")
    flight_id = 291262109

    res = calculate_flight_fir_dwell_inefficiency(ds, ectrl_id=flight_id)
    assert res["has_data"] is True
    assert res["ectrl_id"] == flight_id
    assert res["total_sectors_traversed"] == 10
    assert len(res["sectors"]) == 10
    assert res["total_actual_dwell_min"] > 100
    assert "verdict" in res
    assert len(res["shortcut_sectors"]) > 0


def test_fir_saturation_fallback_graceful():
    ds = FlightDataset(period="201611")
    assert ds.has_fir_traversals() is False

    res = calculate_fir_hourly_saturation(ds)
    assert res["has_data"] is False
    assert "Available in monthly archive" in res["message"]

    res_dwell = calculate_flight_fir_dwell_inefficiency(ds, ectrl_id=202007026)
    assert res_dwell["has_data"] is False

