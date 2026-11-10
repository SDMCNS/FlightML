import pytest
from flightml.geo.geodesy import haversine_distance_nm, initial_bearing_deg
from flightml.geo.fir_engine import FIREngine


def test_haversine_distance():
    # London Heathrow (51.4775, -0.4614) to Paris Charles de Gaulle (49.0097, 2.5479)
    dist = haversine_distance_nm(51.4775, -0.4614, 49.0097, 2.5479)
    assert 180.0 < dist < 200.0


def test_initial_bearing():
    # Bearing due north
    bearing = initial_bearing_deg(0.0, 0.0, 10.0, 0.0)
    assert round(bearing, 1) == 0.0

    # Bearing due east along equator
    bearing_east = initial_bearing_deg(0.0, 0.0, 0.0, 10.0)
    assert round(bearing_east, 1) == 90.0


def test_fir_engine_lookup():
    engine = FIREngine()
    london_firs = engine.locate_fir(51.5, -0.1, 350.0)
    assert "EGTTUIR" in london_firs

