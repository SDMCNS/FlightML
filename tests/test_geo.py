import math
import pytest
from flightml.geo.geodesy import (
    haversine_distance_nm,
    initial_bearing_deg,
    cross_track_distance_nm,
    spherical_midpoint,
    circular_mean_longitude,
    circular_mean_coords,
    split_polyline_at_antimeridian,
    normalize_longitude_deg,
    angular_difference_deg,
)
from flightml.geo.fir_engine import FIREngine
from flightml.geo.fra_boundary_engine import interpolate_great_circle
from flightml.viz.map_builder import create_safe_polyline


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


def test_antimeridian_distance_and_azimuth():
    """Verify haversine and forward azimuth calculate correctly across the 180° Date Line."""
    # Two points 2 degrees of longitude apart across the antimeridian at 60°N
    # 2° * cos(60°) * 60 nm/deg = 60.0 nm
    dist = haversine_distance_nm(60.0, 179.0, 60.0, -179.0)
    assert 59.0 < dist < 61.0

    # Heading East across the Date Line
    bearing_east = initial_bearing_deg(60.0, 179.0, 60.0, -179.0)
    assert 88.0 < bearing_east < 92.0

    # Heading West across the Date Line
    bearing_west = initial_bearing_deg(60.0, -179.0, 60.0, 179.0)
    assert 268.0 < bearing_west < 272.0


def test_angular_difference_and_normalization():
    """Verify shortest angular distance across the Date Line."""
    # +179° to -179° is +2° eastward
    assert round(angular_difference_deg(179.0, -179.0), 1) == 2.0
    # -179° to +179° is -2° westward
    assert round(angular_difference_deg(-179.0, 179.0), 1) == -2.0
    # Equator normalization
    assert normalize_longitude_deg(190.0) == -170.0
    assert normalize_longitude_deg(-190.0) == 170.0


def test_spherical_midpoint_antimeridian_and_polar():
    """Verify spherical midpoint calculation across the Date Line and pole."""
    # Antimeridian midpoint: (60°N, 179°E) and (60°N, 179°W)
    mid_lat, mid_lon = spherical_midpoint(60.0, 179.0, 60.0, -179.0)
    assert 59.9 < mid_lat < 60.5
    assert abs(abs(mid_lon) - 180.0) < 0.01  # Must be ±180.0°, NOT 0.0° (Greenwich)

    # Polar midpoint: (80°N, 0°E) and (80°N, 180°E) passes over the North Pole
    p_lat, p_lon = spherical_midpoint(80.0, 0.0, 80.0, 180.0)
    assert p_lat >= 89.9  # Exactly at or near 90.0°N


def test_circular_mean_longitude():
    """Verify circular mean of longitudes for trans-Pacific / Date Line trajectory."""
    # Trajectory crossing Date Line: [175, 178, -179, -176]
    lons = [175.0, 178.0, 180.0, -179.0, -176.0]
    mean_lon = circular_mean_longitude(lons)
    assert abs(abs(mean_lon) - 180.0) < 5.0  # Must be ~180° / -180°, NOT 0.0°

    # Circular mean coords
    c_lat, c_lon = circular_mean_coords([[60.0, 175.0], [61.0, 179.0], [61.0, -179.0]])
    assert 60.0 <= c_lat <= 61.5
    assert abs(abs(c_lon) - 180.0) < 5.0


def test_split_polyline_at_antimeridian():
    """Verify splitting polyline into clean multi-polyline segments at ±180°."""
    # 1. Antimeridian crossing East -> West
    pts_ew = [[60.0, 175.0], [61.0, 179.0], [61.5, -179.0], [62.0, -175.0]]
    segs_ew = split_polyline_at_antimeridian(pts_ew)
    assert len(segs_ew) == 2
    assert segs_ew[0][-1][1] == 180.0  # Ends at +180.0°
    assert segs_ew[1][0][1] == -180.0  # Starts at -180.0°
    assert abs(segs_ew[0][-1][0] - segs_ew[1][0][0]) < 1e-4  # Continuous crossing latitude

    # 2. Antimeridian crossing West -> East
    pts_we = [[62.0, -175.0], [61.5, -179.0], [61.0, 179.0], [60.0, 175.0]]
    segs_we = split_polyline_at_antimeridian(pts_we)
    assert len(segs_we) == 2
    assert segs_we[0][-1][1] == -180.0  # Ends at -180.0°
    assert segs_we[1][0][1] == 180.0   # Starts at +180.0°

    # 3. Polar crossing over North Pole
    pts_polar = [[75.0, 10.0], [84.0, 10.0], [84.0, -170.0], [75.0, -170.0]]
    segs_polar = split_polyline_at_antimeridian(pts_polar, max_lat=85.0)
    assert len(segs_polar) == 2
    assert segs_polar[0][-1][0] == 85.0  # Reaches top of Web Mercator map
    assert segs_polar[1][0][0] == 85.0   # Resumes from top on other meridian


def test_cross_track_distance_asin_clamping():
    """Verify cross_track_distance_nm handles extreme inputs without math domain errors."""
    # Collinear points
    d_zero = cross_track_distance_nm(50.0, 0.0, 40.0, 0.0, 60.0, 0.0)
    assert abs(d_zero) < 0.01

    # Right angle deviation
    d_perp = cross_track_distance_nm(0.0, 10.0, 0.0, 0.0, 10.0, 0.0)
    assert d_perp > 0.0


def test_interpolate_great_circle_clamping():
    """Verify great circle interpolation never produces out-of-bounds latitudes."""
    pts = interpolate_great_circle(70.0, 10.0, 70.0, -170.0, num_points=9)
    assert len(pts) == 9
    for lat, lon in pts:
        assert -85.0 <= lat <= 85.0
        assert -180.0 <= lon <= 180.0


def test_map_builder_safe_polyline():
    """Verify create_safe_polyline handles antimeridian and polar coords in Folium."""
    pts = [[60.0, 175.0], [61.0, 179.0], [61.5, -179.0], [62.0, -175.0]]
    pl = create_safe_polyline(pts, color="purple", weight=3)
    assert pl is not None
    # Locations should be a list of 2 sub-segments (MultiPolyLine)
    assert len(pl.locations) == 2
