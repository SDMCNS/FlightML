"""
FlightML Mathematical Geodesy & Spherical Trigonometry Module.

Provides high-precision geodesic formulas:
- Haversine great-circle distance (NM and KM)
- Initial bearing (forward azimuth) and final bearing
- Cross-track distance (XTD) and along-track distance (ATD)
- Spherical midpoint and Great Circle spherical linear interpolation (Slerp)
- Antimeridian (180° meridian) crossing detection and polyline splitting
- Directional and circular mean longitude / coordinate clustering
"""

from __future__ import annotations

import math
from typing import Sequence

EARTH_RADIUS_NM: float = 3440.065
EARTH_RADIUS_KM: float = 6371.0088


def normalize_longitude_deg(lon: float) -> float:
    """
    Normalize longitude to [-180.0, 180.0].
    Preserves +180.0 if explicitly approaching from the east.
    """
    wrapped = (float(lon) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0 and lon > 0:
        return 180.0
    return wrapped


def angular_difference_deg(lon1: float, lon2: float) -> float:
    """
    Calculate the shortest signed angular difference (lon2 - lon1) in degrees,
    wrapped to [-180.0, 180.0]. Positive indicates eastward displacement.
    """
    return ((float(lon2) - float(lon1) + 180.0) % 360.0) - 180.0


def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth in nautical miles.
    Numerically stable for small and antipodal distances, and antimeridian crossing.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(max(0.0, a)), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_NM * c


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(max(0.0, a)), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial forward bearing (forward azimuth) from point 1 to point 2 in degrees (0-360).
    Properly handles international date line / antimeridian traversal.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def final_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate final bearing arriving at point 2 from point 1 in degrees (0-360).
    Equal to initial bearing from point 2 to point 1 + 180°.
    """
    b = initial_bearing_deg(lat2, lon2, lat1, lon1)
    return (b + 180.0) % 360.0


def cross_track_distance_nm(
    p_lat: float, p_lon: float, start_lat: float, start_lon: float, end_lat: float, end_lon: float
) -> float:
    """
    Calculate spherical cross-track distance (XTD) from point P to great-circle path (start -> end).
    Sign convention: positive if point P is to the right of the path, negative if to the left.
    """
    d13 = haversine_distance_nm(start_lat, start_lon, p_lat, p_lon)
    if d13 < 1e-6:
        return 0.0

    delta13 = d13 / EARTH_RADIUS_NM
    theta13 = math.radians(initial_bearing_deg(start_lat, start_lon, p_lat, p_lon))
    theta12 = math.radians(initial_bearing_deg(start_lat, start_lon, end_lat, end_lon))

    xtd_rad = math.asin(max(-1.0, min(1.0, math.sin(delta13) * math.sin(theta13 - theta12))))
    return xtd_rad * EARTH_RADIUS_NM


def along_track_distance_nm(
    p_lat: float, p_lon: float, start_lat: float, start_lon: float, end_lat: float, end_lon: float
) -> float:
    """
    Calculate along-track distance (ATD) from start point along path towards end point to projection of P.
    """
    d13 = haversine_distance_nm(start_lat, start_lon, p_lat, p_lon)
    xtd = cross_track_distance_nm(p_lat, p_lon, start_lat, start_lon, end_lat, end_lon)

    delta13 = d13 / EARTH_RADIUS_NM
    delta_xtd = xtd / EARTH_RADIUS_NM

    cos_atd = math.cos(delta13) / max(1e-12, math.cos(delta_xtd))
    atd_rad = math.acos(max(-1.0, min(1.0, cos_atd)))
    return atd_rad * EARTH_RADIUS_NM


def spherical_midpoint(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> tuple[float, float]:
    """
    Calculate the exact spherical midpoint along the great-circle path between two coordinates.
    Returns (mid_lat, mid_lon) in degrees. Correctly spans the antimeridian.
    """
    phi1, lambda1 = math.radians(lat1), math.radians(lon1)
    phi2, lambda2 = math.radians(lat2), math.radians(lon2)

    d_lambda = lambda2 - lambda1
    bx = math.cos(phi2) * math.cos(d_lambda)
    by = math.cos(phi2) * math.sin(d_lambda)

    phi3 = math.atan2(
        math.sin(phi1) + math.sin(phi2),
        math.sqrt((math.cos(phi1) + bx) ** 2 + by ** 2)
    )
    lambda3 = lambda1 + math.atan2(by, math.cos(phi1) + bx)

    mid_lat = math.degrees(phi3)
    mid_lon = (math.degrees(lambda3) + 540.0) % 360.0 - 180.0
    return mid_lat, mid_lon


def slerp_interpolation(
    lat1: float, lon1: float, lat2: float, lon2: float, fraction: float
) -> tuple[float, float]:
    """
    Spherical Linear Interpolation (Slerp) along the great-circle route between two points.
    `fraction` = 0.0 returns start point, 1.0 returns end point.
    """
    f = max(0.0, min(1.0, float(fraction)))
    if f == 0.0:
        return lat1, lon1
    if f == 1.0:
        return lat2, lon2

    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)

    # Convert spherical to 3D Cartesian unit vectors
    v1 = (math.cos(phi1) * math.cos(lam1), math.cos(phi1) * math.sin(lam1), math.sin(phi1))
    v2 = (math.cos(phi2) * math.cos(lam2), math.cos(phi2) * math.sin(lam2), math.sin(phi2))

    dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]))
    omega = math.acos(dot)

    if omega < 1e-6:
        return lat1, lon1

    sin_omega = math.sin(omega)
    scale1 = math.sin((1.0 - f) * omega) / sin_omega
    scale2 = math.sin(f * omega) / sin_omega

    vx = scale1 * v1[0] + scale2 * v2[0]
    vy = scale1 * v1[1] + scale2 * v2[1]
    vz = scale1 * v1[2] + scale2 * v2[2]

    interp_lat = math.degrees(math.atan2(vz, math.sqrt(vx * vx + vy * vy)))
    interp_lon = math.degrees(math.atan2(vy, vx))
    return interp_lat, normalize_longitude_deg(interp_lon)


def circular_mean_longitude(lons: Sequence[float]) -> float:
    """
    Compute circular mean of a sequence of longitudes using vector phase averaging.
    Prevents artificial wrapping distortions when points cluster near ±180° (Antimeridian).
    """
    if not lons:
        return 0.0
    sin_sum = sum(math.sin(math.radians(lon)) for lon in lons)
    cos_sum = sum(math.cos(math.radians(lon)) for lon in lons)
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(sin_sum, cos_sum))


def circular_mean_coords(coords: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """
    Compute the centroid (mean_lat, mean_lon) of spherical coordinates with robust
    antimeridian circular averaging for longitudes and standard linear mean for latitudes.
    """
    if not coords:
        return 0.0, 0.0
    mean_lat = sum(c[0] for c in coords) / len(coords)
    mean_lon = circular_mean_longitude([c[1] for c in coords])
    return mean_lat, mean_lon


def split_polyline_at_antimeridian(
    coords: Sequence[tuple[float, float]],
    threshold_deg: float = 180.0
) -> list[list[tuple[float, float]]]:
    """
    Splits a polyline of (lat, lon) coordinates into multiple segment parts whenever
    a consecutive longitude jump exceeds `threshold_deg`. Prevents map-spanning horizontal streaks.
    """
    if not coords:
        return []

    segments: list[list[tuple[float, float]]] = []
    current_segment: list[tuple[float, float]] = [coords[0]]

    for i in range(1, len(coords)):
        lat1, lon1 = coords[i - 1]
        lat2, lon2 = coords[i]

        if abs(lon2 - lon1) > threshold_deg:
            segments.append(current_segment)
            current_segment = [coords[i]]
        else:
            current_segment.append(coords[i])

    if current_segment:
        segments.append(current_segment)

    return segments
