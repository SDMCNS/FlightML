"""
FlightML Operational Performance, Flight Inefficiency & Navigation Infrastructure Module.

Provides exact formulation of:
- Eurocontrol PRU / ICAO GANP Horizontal Flight Inefficiency (KEA, KEP, KET)
- Radio Line-of-Sight (RLOS) VHF/UHF and DME/VOR radio horizon
- Continuous Descent Operations (CDO) / Continuous Climb Operations (CCO) metrics
- Descent flight path angles (FPA) and level-off detection
- Runway throughput and approach separation spacing kinematics
"""

from __future__ import annotations

import math

FT_TO_M: float = 0.3048
NM_TO_M: float = 1852.0


def kea_horizontal_inefficiency_pct(
    actual_flown_nm: float,
    great_circle_nm: float,
) -> float:
    """
    Calculate Eurocontrol PRU Key Performance Indicator KEA (Actual Trajectory Horizontal Inefficiency):
    KEA (%) = ((ActualDistance - GreatCircleDistance) / GreatCircleDistance) * 100.0.
    A KEA of 3.2% means the actual flight path was 3.2% longer than the theoretical direct geodesic.
    """
    gc = max(1.0, float(great_circle_nm))
    act = max(gc, float(actual_flown_nm))
    return ((act - gc) / gc) * 100.0


def kep_planned_inefficiency_pct(
    filed_planned_nm: float,
    great_circle_nm: float,
) -> float:
    """
    Calculate Eurocontrol PRU Key Performance Indicator KEP (Filed Flight Plan Horizontal Inefficiency):
    KEP (%) = ((PlannedDistance - GreatCircleDistance) / GreatCircleDistance) * 100.0.
    Reflects the constraint penalty of airway network structures and route charges avoidance.
    """
    gc = max(1.0, float(great_circle_nm))
    plan = max(gc, float(filed_planned_nm))
    return ((plan - gc) / gc) * 100.0


def ket_terminal_extension_nm(
    actual_flown_nm: float,
    filed_planned_nm: float,
) -> float:
    """
    Calculate tactical ATC radar vectoring extension distance:
    KET = ActualFlown - FiledPlanned (NM).
    """
    return float(actual_flown_nm) - float(filed_planned_nm)


def rlos_radio_horizon_nm(
    station_elevation_ft: float,
    aircraft_altitude_ft: float,
    effective_earth_radius_factor: float = 4.0 / 3.0,
) -> float:
    """
    Calculate geometric Radio Line-of-Sight (RLOS) horizon in nautical miles
    accounting for atmospheric refraction (standard 4/3 effective Earth radius):
    d_RLOS (NM) = 1.23 * (sqrt(h_station_ft) + sqrt(h_aircraft_ft)).
    At FL350 (35,000 ft) with a 150 ft antenna mast, d_RLOS ~ 245 nm.
    """
    h_st = max(0.0, float(station_elevation_ft))
    h_ac = max(0.0, float(aircraft_altitude_ft))
    coefficient = 1.064 * math.sqrt(effective_earth_radius_factor)  # ~1.2285
    return coefficient * (math.sqrt(h_st) + math.sqrt(h_ac))


def rlos_radio_horizon_km(
    station_elevation_m: float,
    aircraft_altitude_m: float,
) -> float:
    """
    Calculate Radio Line-of-Sight horizon in kilometers:
    d_RLOS (km) = 4.12 * (sqrt(h_station_m) + sqrt(h_aircraft_m)).
    """
    h_st = max(0.0, float(station_elevation_m))
    h_ac = max(0.0, float(aircraft_altitude_m))
    return 4.12 * (math.sqrt(h_st) + math.sqrt(h_ac))


def descent_flight_path_angle_deg(
    delta_altitude_ft: float,
    distance_nm: float,
) -> float:
    """
    Calculate vertical flight path angle (FPA) in degrees:
    theta = arctan(delta_altitude / distance).
    Standard 3.0° Instrument Landing System (ILS) glideslope corresponds to ~318 ft/nm.
    """
    d_nm = max(0.01, float(distance_nm))
    d_m = d_nm * NM_TO_M
    dh_m = float(delta_altitude_ft) * FT_TO_M
    return math.degrees(math.atan2(dh_m, d_m))


def is_cdo_level_off_segment(
    delta_fl: float,
    duration_s: float,
    fl_threshold: float = 2.5,
    duration_threshold_s: float = 150.0,
) -> bool:
    """
    Evaluate whether a descent segment constitutes a non-CDO level-off phase.
    According to Eurocontrol / ICAO CDO definitions:
    Vertical displacement < 250 ft (2.5 FL) sustained for > 2.5 minutes (150 seconds)
    indicates level flight requiring throttled engine thrust, terminating continuous idle descent.
    """
    return abs(float(delta_fl)) <= float(fl_threshold) and float(duration_s) >= float(duration_threshold_s)


def runway_separation_time_s(
    distance_spacing_nm: float,
    approach_groundspeed_kts: float,
) -> float:
    """
    Calculate temporal runway threshold separation time in seconds:
    t = distance / (speed / 3600).
    E.g., 3.0 nm separation at 140 kts approach speed = 77.1 seconds.
    """
    speed = max(30.0, float(approach_groundspeed_kts))
    dist = max(0.1, float(distance_spacing_nm))
    return (dist / (speed / 3600.0))


def runway_capacity_movements_per_hour(
    average_inter_arrival_time_s: float,
    buffer_s: float = 15.0,
) -> float:
    """
    Estimate single-runway peak arrival throughput capacity in movements per hour:
    Capacity = 3600 / (inter_arrival_time + buffer).
    """
    t_cycle = max(30.0, float(average_inter_arrival_time_s) + float(buffer_s))
    return 3600.0 / t_cycle
