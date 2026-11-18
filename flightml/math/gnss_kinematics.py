"""
FlightML GNSS Kinematics, Electronic Warfare & Spoofing Detection Module.

Provides mathematical kinematics and physics-based anomaly screening for:
- Teleportation groundspeed jumps exceeding commercial aircraft envelopes (> 750 kts)
- Vertical climb/descent velocity spikes (> 6,000 fpm)
- Sharp angular jitter vs. Terminal 50 nm radar vectoring zone exclusion
- Composite spoofing and jamming confirmation scoring (Threshold >= 75.0)
"""

from __future__ import annotations

import math
from typing import Sequence


def teleportation_groundspeed_kts(distance_nm: float, delta_time_s: float) -> float:
    """
    Calculate kinematic groundspeed across a telemetry segment:
    V (kts) = distance_nm / (delta_time_s / 3600.0).
    For delta_time <= 0, returns 0.0.
    """
    dt = float(delta_time_s)
    if dt <= 0.0:
        return 0.0
    return float(distance_nm) / (dt / 3600.0)


def vertical_rate_fpm(delta_altitude_ft: float, delta_time_s: float) -> float:
    """
    Calculate vertical climb or descent rate in feet per minute (fpm):
    V_z = abs(delta_altitude_ft) / (delta_time_s / 60.0).
    """
    dt = float(delta_time_s)
    if dt <= 0.0:
        return 0.0
    return abs(float(delta_altitude_ft)) / (dt / 60.0)


def track_jitter_rate_deg_per_s(angular_change_deg: float, delta_time_s: float) -> float:
    """
    Calculate track turning angular rate in degrees per second:
    omega = abs(angular_change_deg) / delta_time_s.
    """
    dt = float(delta_time_s)
    if dt <= 0.0:
        return 0.0
    return abs(float(angular_change_deg)) / dt


def is_terminal_vectoring_zone(
    dist_to_adep_nm: float,
    dist_to_ades_nm: float,
    terminal_radius_nm: float = 50.0,
) -> bool:
    """
    Determine whether aircraft is within the 50 nm terminal control zone of ADEP or ADES.
    Inside this 50 nm radius, sharp turns and heading changes are routine ATC radar vectors
    and MUST NOT trigger GNSS jamming/spoofing markers. Only unphysical vertical rates
    or altitude glitches indicate true telemetry disruption in this envelope.
    """
    r = float(terminal_radius_nm)
    return float(dist_to_adep_nm) <= r or float(dist_to_ades_nm) <= r


def gnss_composite_spoofing_score(
    teleport_count: int,
    vertical_spike_count: int,
    altitude_glitch_count: int,
    track_jitter_count: int,
    in_hotspot: bool = False,
) -> float:
    """
    Calculate the composite GNSS spoofing & jamming severity score (0.0 to 100.0):
    Weights:
    - Speed Teleportation (> 750 kts): 35 points each
    - Vertical Rate Spike (> 6,000 fpm): 30 points each
    - Impossible Altitude Glitch (> FL600 or < 0): 25 points each
    - Track Jitter (en-route outside 50nm): 10 points each
    - Geopolitical EW Hotspot Corridor: +15 bonus base points
    """
    score = 0.0
    score += min(70.0, teleport_count * 35.0)
    score += min(60.0, vertical_spike_count * 30.0)
    score += min(50.0, altitude_glitch_count * 25.0)
    score += min(20.0, track_jitter_count * 10.0)

    if in_hotspot and (teleport_count > 0 or vertical_spike_count > 0 or altitude_glitch_count > 0):
        score += 15.0

    return min(100.0, score)


def is_confirmed_spoofing(
    score: float,
    has_teleport: bool,
    has_vertical_spike: bool,
    has_altitude_glitch: bool,
    confirmation_threshold: float = 75.0,
) -> bool:
    """
    Strict confirmation rule:
    Score must be >= 75.0 AND flight must exhibit at least one unphysical kinematic anomaly
    (teleportation speed, impossible vertical spike, or negative/excessive altitude).
    """
    meets_score = float(score) >= float(confirmation_threshold)
    has_unphysical_anomaly = has_teleport or has_vertical_spike or has_altitude_glitch
    return meets_score and has_unphysical_anomaly
