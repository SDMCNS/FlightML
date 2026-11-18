"""
FlightML Aerodynamics & Upper-Air Kinematics Module.

Provides exact vector wind calculations, airspeed conversions, and flight mechanics:
- Mach to True Airspeed (TAS) conversion
- Wind triangle vector resolution (Heading, Drift angle, Groundspeed)
- Longitudinal headwind/tailwind component and lateral crosswind component
- Equivalent Still Air Distance (ESAD) air-mile integration
- Coordinated turn radius, rate of turn, and aerodynamic load factor
"""

from __future__ import annotations

import math
from flightml.math.atmosphere import (
    GAMMA_AIR,
    R_GAS_AIR,
    MPS_TO_KTS,
    KTS_TO_MPS,
    G_GRAVITY,
)


def mach_to_tas_kts(mach: float, temperature_k: float) -> float:
    """
    Calculate True Airspeed (TAS) in knots from Mach number and ambient temperature in Kelvin:
    TAS = Mach * sqrt(gamma * R * T).
    """
    m = max(0.0, float(mach))
    t = max(150.0, float(temperature_k))
    speed_of_sound_mps = math.sqrt(GAMMA_AIR * R_GAS_AIR * t)
    return m * speed_of_sound_mps * MPS_TO_KTS


def tas_to_mach(tas_kts: float, temperature_k: float) -> float:
    """Calculate Mach number from True Airspeed (TAS) in knots and ambient temperature in Kelvin."""
    v = max(0.0, float(tas_kts))
    t = max(150.0, float(temperature_k))
    speed_of_sound_kts = math.sqrt(GAMMA_AIR * R_GAS_AIR * t) * MPS_TO_KTS
    return v / max(1e-6, speed_of_sound_kts)


def headwind_tailwind_component_kts(
    track_deg: float,
    wind_from_deg: float,
    wind_speed_kts: float,
) -> float:
    """
    Calculate along-track wind component in knots.
    Positive value indicates a TAILWIND (assisting flight).
    Negative value indicates a HEADWIND (opposing flight).
    Formula: V_wind * cos(wind_from - track - 180°)
    """
    angle_rad = math.radians(wind_from_deg - track_deg - 180.0)
    return float(wind_speed_kts) * math.cos(angle_rad)


def crosswind_component_kts(
    heading_or_runway_deg: float,
    wind_from_deg: float,
    wind_speed_kts: float,
) -> float:
    """
    Calculate absolute lateral crosswind component perpendicular to runway or aircraft track in knots.
    Formula: V_wind * |sin(wind_from - track)|
    """
    angle_rad = math.radians(wind_from_deg - heading_or_runway_deg)
    return abs(float(wind_speed_kts) * math.sin(angle_rad))


def wind_triangle_decomposition(
    track_deg: float,
    tas_kts: float,
    wind_from_deg: float,
    wind_speed_kts: float,
) -> tuple[float, float, float]:
    """
    Resolve the complete aerodynamic wind triangle.
    Given desired track, true airspeed, and wind vector:
    Returns:
        (heading_deg, groundspeed_kts, drift_angle_deg)
    """
    track_rad = math.radians(track_deg)
    # Wind vector blowing TO:
    wind_to_rad = math.radians((wind_from_deg + 180.0) % 360.0)
    w_speed = float(wind_speed_kts)
    tas = float(tas_kts)

    # Wind components along and across desired track
    w_parallel = w_speed * math.cos(wind_to_rad - track_rad)
    w_cross = w_speed * math.sin(wind_to_rad - track_rad)

    # Wind correction angle (drift correction)
    sin_wca = - w_cross / max(1e-6, tas)
    sin_wca = max(-1.0, min(1.0, sin_wca))
    wca_rad = math.asin(sin_wca)
    wca_deg = math.degrees(wca_rad)

    heading_deg = (track_deg + wca_deg + 360.0) % 360.0
    groundspeed_kts = tas * math.cos(wca_rad) + w_parallel
    drift_angle_deg = -wca_deg

    return heading_deg, max(0.0, groundspeed_kts), drift_angle_deg


def equivalent_still_air_distance_nm(
    ground_dist_nm: float,
    tas_kts: float,
    groundspeed_kts: float,
) -> float:
    """
    Calculate Equivalent Still Air Distance (ESAD) in nautical miles:
    ESAD = (TAS / Groundspeed) * GroundDistance
    - In tailwinds (GS > TAS): ESAD < GroundDistance (less fuel/air-miles required).
    - In headwinds (GS < TAS): ESAD > GroundDistance (more fuel/air-miles required).
    """
    gs = max(10.0, float(groundspeed_kts))
    tas = max(10.0, float(tas_kts))
    d = max(0.0, float(ground_dist_nm))
    return (tas / gs) * d


def turn_radius_nm(tas_kts: float, bank_angle_deg: float) -> float:
    """
    Calculate steady-state coordinated turn radius in nautical miles:
    R = V² / (g * tan(phi)).
    """
    tas_mps = float(tas_kts) * KTS_TO_MPS
    phi_rad = math.radians(min(80.0, max(1.0, abs(bank_angle_deg))))
    radius_m = (tas_mps ** 2) / (G_GRAVITY * math.tan(phi_rad))
    return radius_m / 1852.0  # meters to nautical miles


def load_factor_g(bank_angle_deg: float) -> float:
    """
    Calculate coordinated turn aerodynamic load factor (g's):
    n = 1 / cos(phi).
    At 60° bank angle, n = 2.0g.
    """
    phi_rad = math.radians(min(85.0, max(0.0, abs(bank_angle_deg))))
    return 1.0 / math.cos(phi_rad)


def rate_of_turn_deg_per_s(tas_kts: float, bank_angle_deg: float) -> float:
    """
    Calculate standard coordinated rate of turn in degrees per second:
    omega = (g * tan(phi)) / V.
    Standard rate turn (Rate 1) is 3°/s.
    """
    tas_mps = float(tas_kts) * KTS_TO_MPS
    if tas_mps < 1.0:
        return 0.0
    phi_rad = math.radians(min(80.0, max(1.0, abs(bank_angle_deg))))
    omega_rad_s = (G_GRAVITY * math.tan(phi_rad)) / tas_mps
    return math.degrees(omega_rad_s)
