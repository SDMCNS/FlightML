from __future__ import annotations

import math
from typing import Sequence

EARTH_RADIUS_NM = 3440.065
EARTH_RADIUS_KM = 6371.0


def normalize_longitude_deg(lon: float) -> float:
    """Normalize longitude to [-180.0, 180.0]."""
    wrapped = (float(lon) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0 and lon > 0:
        return 180.0
    return wrapped


def angular_difference_deg(lon1: float, lon2: float) -> float:
    """
    Calculate the shortest signed angular difference (lon2 - lon1) in degrees,
    wrapped to [-180.0, 180.0].
    """
    return ((float(lon2) - float(lon1) + 180.0) % 360.0) - 180.0


def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points in nautical miles.
    Mathematically robust to antimeridian and polar traversal.
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


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial bearing (forward azimuth) from point 1 to point 2 in degrees (0-360).
    Robust to antimeridian crossing.
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
    Clamps internal sine values to [-1.0, 1.0] to prevent math domain errors.
    """
    d13 = haversine_distance_nm(start_lat, start_lon, p_lat, p_lon) / EARTH_RADIUS_NM
    theta13 = math.radians(initial_bearing_deg(start_lat, start_lon, p_lat, p_lon))
    theta12 = math.radians(initial_bearing_deg(start_lat, start_lon, end_lat, end_lon))

    sin_term = max(-1.0, min(1.0, math.sin(d13) * math.sin(theta13 - theta12)))
    d_xt = math.asin(sin_term)
    return d_xt * EARTH_RADIUS_NM


def spherical_midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """
    Compute the exact spherical great-circle midpoint between two coordinates.
    Handles traversal across the International Date Line / Antimeridian and polar regions,
    preventing midpoints from collapsing to 0.0° (Prime Meridian).
    """
    phi1, lambda1 = math.radians(lat1), math.radians(lon1)
    phi2, lambda2 = math.radians(lat2), math.radians(lon2)
    dlambda = math.radians(angular_difference_deg(lon1, lon2))

    bx = math.cos(phi2) * math.cos(dlambda)
    by = math.cos(phi2) * math.sin(dlambda)

    phi_m = math.atan2(
        math.sin(phi1) + math.sin(phi2),
        math.sqrt((math.cos(phi1) + bx) ** 2 + by ** 2),
    )
    lambda_m = lambda1 + math.atan2(by, math.cos(phi1) + bx)

    mid_lat = math.degrees(phi_m)
    mid_lon = normalize_longitude_deg(math.degrees(lambda_m))
    return round(mid_lat, 5), round(mid_lon, 5)


def spherical_intermediate_point(
    lat1: float, lon1: float, lat2: float, lon2: float, fraction: float
) -> tuple[float, float]:
    """
    Calculate an intermediate coordinate at fractional distance `fraction` [0.0, 1.0]
    along the great-circle path from point 1 to point 2.
    """
    if fraction <= 0.0:
        return round(lat1, 5), round(normalize_longitude_deg(lon1), 5)
    if fraction >= 1.0:
        return round(lat2, 5), round(normalize_longitude_deg(lon2), 5)

    phi1, lambda1 = math.radians(lat1), math.radians(lon1)
    phi2, lambda2 = math.radians(lat2), math.radians(lon2)

    # Angular distance
    sin_dphi2 = math.sin((phi2 - phi1) / 2.0)
    sin_dlam2 = math.sin(math.radians(angular_difference_deg(lon1, lon2)) / 2.0)
    a = sin_dphi2 * sin_dphi2 + math.cos(phi1) * math.cos(phi2) * sin_dlam2 * sin_dlam2
    delta = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))

    if delta < 1e-9:
        return round(lat1, 5), round(normalize_longitude_deg(lon1), 5)

    sin_delta = math.sin(delta)
    f1 = math.sin((1.0 - fraction) * delta) / sin_delta
    f2 = math.sin(fraction * delta) / sin_delta

    x = f1 * math.cos(phi1) * math.cos(lambda1) + f2 * math.cos(phi2) * math.cos(lambda2)
    y = f1 * math.cos(phi1) * math.sin(lambda1) + f2 * math.cos(phi2) * math.sin(lambda2)
    z = f1 * math.sin(phi1) + f2 * math.sin(phi2)

    int_lat = math.atan2(z, math.sqrt(x * x + y * y))
    int_lon = math.atan2(y, x)

    return round(math.degrees(int_lat), 5), round(normalize_longitude_deg(math.degrees(int_lon)), 5)


def circular_mean_longitude(lons: Sequence[float]) -> float:
    """
    Calculate circular vector mean longitude for a series of coordinates.
    Guarantees that trajectories crossing the 180°/-180° International Date Line
    properly center around ±180° rather than collapsing to 0° (Greenwich).
    """
    clean_lons = [float(x) for x in lons if math.isfinite(float(x))]
    if not clean_lons:
        return 0.0
    s = sum(math.sin(math.radians(x)) for x in clean_lons)
    c = sum(math.cos(math.radians(x)) for x in clean_lons)
    if abs(s) < 1e-9 and abs(c) < 1e-9:
        return float(clean_lons[0])
    mean_deg = math.degrees(math.atan2(s, c))
    return normalize_longitude_deg(mean_deg)


def circular_mean_coords(coords: Sequence[Sequence[float]]) -> tuple[float, float]:
    """
    Calculate the (mean_latitude, circular_mean_longitude) for a list of [lat, lon] points.
    """
    if not coords:
        return 50.0, 10.0
    lats = [float(c[0]) for c in coords if len(c) >= 2 and math.isfinite(float(c[0]))]
    lons = [float(c[1]) for c in coords if len(c) >= 2 and math.isfinite(float(c[1]))]
    if not lats:
        return 50.0, 10.0
    c_lat = sum(lats) / len(lats)
    c_lon = circular_mean_longitude(lons)
    return round(c_lat, 5), round(c_lon, 5)


def split_polyline_at_antimeridian(
    coords: Sequence[Sequence[float]], max_lat: float = 85.0
) -> list[list[list[float]]]:
    """
    Split a list of [latitude, longitude] trajectory coordinates into continuous
    sub-segments whenever the path traverses the International Date Line (antimeridian)
    or crosses polar convergence regions.

    When rendering in Leaflet / Folium:
    - Normal polyline: Leaflet draws an erroneous horizontal stripe across the entire globe
      from +179.9° to -179.9°.
    - Multi-polyline (list of segments): Leaflet renders each segment up to the edge of the
      map at ±180.0° without connecting the gap across the world.
    - Clamps polar latitudes to [-max_lat, max_lat] to maintain Web Mercator stability.
    """
    if not coords:
        return []

    segments: list[list[list[float]]] = []
    current_segment: list[list[float]] = []

    for i in range(len(coords)):
        pt = coords[i]
        if len(pt) < 2:
            continue
        lat = float(pt[0])
        lon = float(pt[1])
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue

        lat_clamped = max(-max_lat, min(max_lat, lat))
        norm_lon = normalize_longitude_deg(lon)

        if not current_segment:
            current_segment.append([lat_clamped, norm_lon])
            continue

        prev_lat, prev_lon = current_segment[-1]
        d_lon = abs(norm_lon - prev_lon)

        # Detect antimeridian crossing (|Δλ| > 180°) or polar meridian convergence
        is_antimeridian = d_lon > 180.0
        is_polar_flip = (prev_lat >= 80.0 or lat_clamped >= 80.0) and d_lon > 90.0

        if is_antimeridian:
            # Interpolate exact latitude at 180°/-180° boundary
            if prev_lon > 0 and norm_lon < 0:
                # East to West (e.g. +179° -> -179°)
                d1 = 180.0 - prev_lon
                d2 = norm_lon + 180.0
                t = d1 / (d1 + d2) if (d1 + d2) > 0 else 0.5
                mid_lat = round(prev_lat + t * (lat_clamped - prev_lat), 5)
                mid_lat = max(-max_lat, min(max_lat, mid_lat))

                current_segment.append([mid_lat, 180.0])
                segments.append(current_segment)
                current_segment = [[mid_lat, -180.0], [lat_clamped, norm_lon]]
            else:
                # West to East (e.g. -179° -> +179°)
                d1 = prev_lon + 180.0
                d2 = 180.0 - norm_lon
                t = d1 / (d1 + d2) if (d1 + d2) > 0 else 0.5
                mid_lat = round(prev_lat + t * (lat_clamped - prev_lat), 5)
                mid_lat = max(-max_lat, min(max_lat, mid_lat))

                current_segment.append([mid_lat, -180.0])
                segments.append(current_segment)
                current_segment = [[mid_lat, 180.0], [lat_clamped, norm_lon]]
        elif is_polar_flip:
            # Polar flight: terminates at top of Web Mercator map and resumes on other meridian
            current_segment.append([max_lat, prev_lon])
            segments.append(current_segment)
            current_segment = [[max_lat, norm_lon], [lat_clamped, norm_lon]]
        else:
            current_segment.append([lat_clamped, norm_lon])

    if current_segment:
        segments.append(current_segment)

    return segments


haversine_nm = haversine_distance_nm


def haversine_vectorized_nm(lat1, lon1, lat2, lon2):
    """Vectorized NumPy great-circle distance in nautical miles."""
    import numpy as np

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return EARTH_RADIUS_NM * 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, a)), np.sqrt(np.maximum(0.0, 1.0 - a)))
