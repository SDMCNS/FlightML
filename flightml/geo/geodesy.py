from __future__ import annotations

import math

EARTH_RADIUS_NM = 3440.065
EARTH_RADIUS_KM = 6371.0


def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points in nautical miles.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_NM * c


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial bearing (forward azimuth) from point 1 to point 2 in degrees (0-360).
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def cross_track_distance_nm(
    p_lat: float, p_lon: float, start_lat: float, start_lon: float, end_lat: float, end_lon: float
) -> float:
    """
    Compute cross-track distance (perpendicular deviation) from point P
    to the great-circle path from start to end in nautical miles.
    """
    d13 = haversine_distance_nm(start_lat, start_lon, p_lat, p_lon) / EARTH_RADIUS_NM
    theta13 = math.radians(initial_bearing_deg(start_lat, start_lon, p_lat, p_lon))
    theta12 = math.radians(initial_bearing_deg(start_lat, start_lon, end_lat, end_lon))

    d_xt = math.asin(math.sin(d13) * math.sin(theta13 - theta12))
    return d_xt * EARTH_RADIUS_NM


haversine_nm = haversine_distance_nm


def haversine_vectorized_nm(lat1, lon1, lat2, lon2):
    """Vectorized NumPy great-circle distance in nautical miles."""
    import numpy as np

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return EARTH_RADIUS_NM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


