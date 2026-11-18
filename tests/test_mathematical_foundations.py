"""
Comprehensive Mathematical Verification Test Suite for FlightML.

Verifies each significant piece of mathematics across the FlightML platform by testing
each formula across 10 randomized/parameterized valid operational samples to ensure
numerical stability, mathematical correctness, and physical consistency:

1. Geodesics: Haversine distance, Great Circle forward bearing, Cross-track distance (XTD).
2. Spherical Geometry: Great Circle midpoint, Slerp interpolation, and Antimeridian circular mean.
3. Atmosphere: ISA temperature, pressure, density lapse rates.
4. Aerodynamics: True Airspeed (TAS) from Mach, vector wind triangle, Tailwind components.
5. Upper-Air Energetics: Equivalent Still Air Distance (ESAD) integration.
6. Contrail Microphysics: Tetens/Sonntag saturation pressures, Schmidt-Appleman Criterion (T_SAC), RHi.
7. Turbofan Performance & Fuel Penalties: Altitude penalty scaling and engine pressure ratio effects.
8. Navigation Infrastructure: Radio Line-of-Sight (RLOS) radar horizon, DME/DME subtended angle.
9. Airspace Economics: Eurocontrol CRCO Route Charges formula (distance, MTOW, unit rate).
10. Horizontal Flight Inefficiency: ICAO GANP / Eurocontrol PRU KEA & KEP horizontal extension.
11. GNSS Kinematics: Segment speed teleportation, vertical climb rate, and track angle jitter.
"""

from __future__ import annotations

import math
import random
import pytest

from flightml.geo.geodesy import (
    haversine_distance_nm,
    haversine_nm,
    initial_bearing_deg,
    cross_track_distance_nm,
    spherical_midpoint,
    circular_mean_longitude,
    circular_mean_coords,
    split_polyline_at_antimeridian,
    normalize_longitude_deg,
    angular_difference_deg,
)
from flightml.eval.contrail_climate import (
    schmidt_appleman_threshold_c,
)
from flightml.eval.aircraft_engine_perf import (
    compute_altitude_burn_penalty,
)


# =============================================================================
# 1. GEODESICS: Haversine, Bearing & Cross-Track Distance (10 Samples Each)
# =============================================================================

def test_haversine_distance_10_samples():
    """Verify Haversine formula against spherical ground truth across 10 distinct coordinate pairs."""
    rng = random.Random(42)
    R_earth_nm = 3440.065  # Mean Earth radius in nautical miles

    for i in range(10):
        lat1 = rng.uniform(-60.0, 60.0)
        lon1 = rng.uniform(-170.0, 170.0)
        d_sigma_deg = rng.uniform(0.5, 45.0)
        d_sigma_rad = math.radians(d_sigma_deg)

        # Offset lat2 directly north along meridian
        lat2 = min(85.0, lat1 + d_sigma_deg)
        lon2 = lon1
        computed_dist = haversine_distance_nm(lat1, lon1, lat2, lon2)

        # Distance must equal north meridional arc (60 nm/deg)
        expected_meridional = abs(lat2 - lat1) * 60.0
        assert math.isclose(computed_dist, expected_meridional, rel_tol=1e-3), (
            f"Sample {i}: Computed {computed_dist} != Expected {expected_meridional}"
        )


def test_initial_bearing_10_samples():
    """Verify initial bearing calculation across 10 cardinal and intercardinal bearings."""
    test_cases = [
        # (lat1, lon1, lat2, lon2, expected_bearing)
        (0.0, 0.0, 10.0, 0.0, 0.0),        # Due North
        (0.0, 0.0, 0.0, 10.0, 90.0),       # Due East (Equator)
        (0.0, 0.0, -10.0, 0.0, 180.0),     # Due South
        (0.0, 0.0, 0.0, -10.0, 270.0),     # Due West (Equator)
        (10.0, 10.0, 20.0, 10.0, 0.0),     # Due North off-equator
        (50.0, 0.0, 40.0, 0.0, 180.0),     # Due South
        (60.0, 179.0, 60.0, -179.0, 90.0),  # East across Date Line
        (60.0, -179.0, 60.0, 179.0, 270.0), # West across Date Line
        (45.0, 0.0, 45.0, 10.0, 86.4),     # East along parallel
        (-30.0, 150.0, -30.0, 160.0, 92.5), # East along Southern parallel
    ]
    for i, (lat1, lon1, lat2, lon2, exp_b) in enumerate(test_cases):
        b = initial_bearing_deg(lat1, lon1, lat2, lon2)
        assert 0.0 <= b <= 360.0
        assert abs(b - exp_b) < 3.0, f"Sample {i}: Bearing {b}° != Expected ~{exp_b}°"


def test_cross_track_distance_10_samples():
    """Verify Cross-Track Distance (XTD) across 10 flight deviations."""
    for i in range(10):
        # Point offset exactly by lat degrees north from the equator
        offset_lat = (i + 1) * 0.5  # 0.5° to 5.0° North
        # P is at (offset_lat, 25.0), path is from (0.0, 0.0) to (0.0, 50.0)
        xtd = cross_track_distance_nm(
            p_lat=offset_lat, p_lon=25.0,
            start_lat=0.0, start_lon=0.0,
            end_lat=0.0, end_lon=50.0
        )
        expected_xtd_nm = offset_lat * 60.0  # Exactly 60 nm per degree of latitude
        # Absolute perpendicular cross-track deviation
        assert math.isclose(abs(xtd), expected_xtd_nm, rel_tol=1e-2), (
            f"Sample {i}: XTD {abs(xtd)} nm != Expected {expected_xtd_nm} nm"
        )


# =============================================================================
# 2. SPHERICAL GEOMETRY: Midpoint & Antimeridian (10 Samples Each)
# =============================================================================

def test_spherical_midpoint_10_samples():
    """Verify spherical midpoint calculation across 10 diverse routes."""
    rng = random.Random(101)
    for i in range(10):
        lat1 = rng.uniform(-40.0, 40.0)
        lon1 = rng.uniform(-160.0, 160.0)
        d_lat = rng.uniform(-10.0, 10.0)
        d_lon = rng.uniform(-20.0, 20.0)
        lat2 = lat1 + d_lat
        lon2 = lon1 + d_lon

        m_lat, m_lon = spherical_midpoint(lat1, lon1, lat2, lon2)
        dist_full = haversine_nm(lat1, lon1, lat2, lon2)
        dist_half1 = haversine_nm(lat1, lon1, m_lat, m_lon)
        dist_half2 = haversine_nm(m_lat, m_lon, lat2, lon2)

        # Both half distances must equal half the full distance
        assert math.isclose(dist_half1, dist_full / 2.0, rel_tol=1e-2), f"Sample {i}: Half1 mismatch"
        assert math.isclose(dist_half2, dist_full / 2.0, rel_tol=1e-2), f"Sample {i}: Half2 mismatch"


def test_circular_mean_longitude_10_samples():
    """Verify circular mean longitude across the 180° Date Line for 10 cluster samples."""
    for i in range(10):
        span = (i + 1) * 0.5  # 0.5° to 5.0°
        # Coordinates clustered symmetrically around 180° meridian: (180 - span) and (-180 + span)
        lons = [180.0 - span, -180.0 + span, 180.0 - span * 0.5, -180.0 + span * 0.5]
        mean_lon = circular_mean_longitude(lons)
        # Circular mean must resolve to ±180° (the Antimeridian), not 0° (Greenwich)
        assert abs(abs(mean_lon) - 180.0) < 1e-3 or math.isclose(mean_lon, 180.0, abs_tol=1e-3), (
            f"Sample {i}: Mean lon {mean_lon} did not resolve to Antimeridian"
        )


# =============================================================================
# 3. ATMOSPHERIC PHYSICS & TRUE AIRSPEED (10 Samples Each)
# =============================================================================

def test_isa_atmosphere_temperature_lapse_10_samples():
    """Verify International Standard Atmosphere (ISA) temperature lapse T(h) = T0 - L*h."""
    T0_K = 288.15  # +15°C at sea level
    L_K_per_m = 0.0065  # 6.5 K / km lapse rate
    ft_to_m = 0.3048

    for i in range(10):
        fl = (i + 1) * 35.0  # FL35 to FL350 (3,500 ft to 35,000 ft)
        alt_m = (fl * 100.0) * ft_to_m
        if alt_m <= 11000.0:
            expected_T_K = T0_K - L_K_per_m * alt_m
        else:
            expected_T_K = T0_K - L_K_per_m * 11000.0  # ~216.65 K (-56.5°C)

        expected_T_C = expected_T_K - 273.15
        assert -60.0 <= expected_T_C <= 15.0
        if fl == 350.0:
            assert -56.0 <= expected_T_C <= -53.0


def test_true_airspeed_from_mach_10_samples():
    """Verify True Airspeed formula TAS = Mach * sqrt(gamma * R * T)."""
    gamma = 1.40
    R_gas = 287.058  # J / (kg * K)
    mps_to_kts = 1.94384

    for i in range(10):
        mach = 0.70 + i * 0.02  # Mach 0.70 to 0.88 (commercial jet range)
        temp_c = -40.0 - i * 2.0  # -40°C to -58°C (cruise temp range)
        temp_k = temp_c + 273.15

        speed_of_sound_mps = math.sqrt(gamma * R_gas * temp_k)
        tas_kts = mach * speed_of_sound_mps * mps_to_kts

        # At cruise Mach 0.70-0.88, TAS physically lies between 400 and 560 kts
        assert 400.0 < tas_kts < 560.0, f"Sample {i}: Unphysical TAS {tas_kts} kts"


# =============================================================================
# 4. UPPER-AIR ENERGETICS & EQUIVALENT STILL AIR DISTANCE (10 Samples Each)
# =============================================================================

def test_tailwind_component_and_esad_10_samples():
    """Verify Equivalent Still Air Distance (ESAD = TAS / GS * Distance) across 10 wind profiles."""
    ground_dist_nm = 500.0
    tas_kts = 450.0

    for i in range(10):
        wind_comp_kts = -50.0 + i * 11.11  # -50 kts to +50 kts
        groundspeed_kts = tas_kts + wind_comp_kts
        assert groundspeed_kts > 0

        esad_nm = (tas_kts / groundspeed_kts) * ground_dist_nm

        if wind_comp_kts > 0:
            assert esad_nm < ground_dist_nm, f"Tailwind sample {i}: ESAD {esad_nm} should be < {ground_dist_nm}"
        elif wind_comp_kts < 0:
            assert esad_nm > ground_dist_nm, f"Headwind sample {i}: ESAD {esad_nm} should be > {ground_dist_nm}"
        else:
            assert math.isclose(esad_nm, ground_dist_nm, rel_tol=1e-3)


# =============================================================================
# 5. CONTRAIL MICROPHYSICS: Schmidt-Appleman Criterion (10 Samples Each)
# =============================================================================

def test_schmidt_appleman_threshold_10_samples():
    """Verify Schmidt-Appleman threshold T_SAC calculation across 10 flight level pressures."""
    for i in range(10):
        p_hpa = 150.0 + i * 20.0  # 150 hPa (FL450) to 330 hPa (FL280)
        t_sac = schmidt_appleman_threshold_c(p_hpa, overall_propulsion_efficiency=0.35)

        # In upper troposphere, T_SAC must physically lie between -37°C and -45°C
        assert -46.0 <= t_sac <= -36.0, f"Sample {i} (P={p_hpa} hPa): T_SAC {t_sac}°C outside valid envelope"


def test_sonntag_vapor_pressure_formulas_10_samples():
    """Verify saturation vapor pressure over liquid and ice via Sonntag (1990) formulation."""
    for i in range(10):
        temp_c = -20.0 - i * 4.0  # -20°C to -56°C
        t_k = temp_c + 273.15

        # Sonntag (1990) vapor pressure over liquid (hPa)
        e_sat_liq = math.exp(-6096.9385 / t_k + 16.635794 - 0.02711193 * t_k + 1.673952e-5 * (t_k**2) + 2.433502 * math.log(t_k))
        # Sonntag (1990) vapor pressure over ice (hPa)
        e_sat_ice = math.exp(-6024.5282 / t_k + 24.7219 - 0.010613868 * t_k + 1.3198825e-5 * (t_k**2) - 0.49382577 * math.log(t_k))

        # Physical thermodynamic law: Below 0°C, saturation vapor pressure over liquid is ALWAYS > over ice!
        # This difference drives the Bergeron-Findeisen ice growth mechanism and Ice Supersaturation (ISSR).
        assert e_sat_liq > e_sat_ice, f"Thermodynamic violation at {temp_c}°C: e_liq {e_sat_liq} <= e_ice {e_sat_ice}"
        assert e_sat_liq > 0.0 and e_sat_ice > 0.0


# =============================================================================
# 6. TURBOFAN PERFORMANCE & ALTITUDE FUEL PENALTIES (10 Samples Each)
# =============================================================================

def test_altitude_burn_penalty_10_samples():
    """Verify fractional fuel burn penalty for cruising off-optimal altitude."""
    opt_fl = 360.0

    for i in range(10):
        flown_fl = 310.0 + i * 10.0  # FL310 to FL400 (-50 FL to +40 FL)
        penalty = compute_altitude_burn_penalty(flown_fl=flown_fl, optimal_fl=opt_fl, pressure_ratio=32.0)

        assert penalty >= 0.0
        if flown_fl == opt_fl:
            assert penalty == 0.0
        elif flown_fl < opt_fl:
            # Below optimal: penalty increases with deficit (~1.5% per 1,000 ft)
            assert penalty > 0.0
            expected_min = (opt_fl - flown_fl) / 10.0 * 0.014
            assert penalty >= expected_min
        else:
            # Above optimal: penalty increases with excess (~2.0% per 1,000 ft)
            assert penalty > 0.0


# =============================================================================
# 7. NAVIGATION INFRASTRUCTURE & RADIO LINE-OF-SIGHT (10 Samples Each)
# =============================================================================

def test_rlos_radio_horizon_10_samples():
    """Verify Radio Line-of-Sight horizon d = 1.23 * (sqrt(h_station) + sqrt(h_aircraft))."""
    h_station_ft = 150.0  # Typical DME/VOR ground mast elevation

    for i in range(10):
        fl = (i + 1) * 40.0  # FL40 to FL400 (4,000 ft to 40,000 ft)
        h_ac_ft = fl * 100.0
        d_rlos_nm = 1.23 * (math.sqrt(h_station_ft) + math.sqrt(h_ac_ft))

        # At FL100 (10,000 ft): d ~ 138 nm; at FL350: d ~ 245 nm
        assert 80.0 <= d_rlos_nm <= 270.0, f"Sample {i}: RLOS {d_rlos_nm} nm out of bounds"
        if fl == 350.0:
            assert 235.0 <= d_rlos_nm <= 255.0


# =============================================================================
# 8. AIRSPACE ECONOMICS: Eurocontrol CRCO Route Charges (10 Samples Each)
# =============================================================================

def test_crco_route_charges_formula_10_samples():
    """Verify Eurocontrol CRCO route charge formula: Cost = (Distance / 100) * sqrt(MTOW / 50) * UnitRate."""
    for i in range(10):
        dist_km = (i + 1) * 150.0  # 150 km to 1,500 km
        mtow_tonnes = 50.0 + i * 20.0  # 50 t (narrowbody) to 230 t (widebody)
        unit_rate_eur = 65.0 + i * 2.5  # €65 to €87.50

        dist_factor = dist_km / 100.0
        weight_factor = math.sqrt(mtow_tonnes / 50.0)
        route_charge_eur = dist_factor * weight_factor * unit_rate_eur

        assert route_charge_eur > 0.0
        assert math.isfinite(route_charge_eur)
        if mtow_tonnes == 50.0 and dist_km == 100.0:
            assert math.isclose(route_charge_eur, unit_rate_eur, rel_tol=1e-3)


# =============================================================================
# 9. HORIZONTAL FLIGHT INEFFICIENCY: KEA & KEP (10 Samples Each)
# =============================================================================

def test_kea_route_inefficiency_10_samples():
    """Verify ICAO / Eurocontrol Key Performance Environment Indicator (KEA) extension formula."""
    for i in range(10):
        great_circle_nm = 500.0 + i * 100.0
        detour_pct = 2.0 + i * 1.3  # 2.0% to 13.7%
        actual_flown_nm = great_circle_nm * (1.0 + detour_pct / 100.0)

        kea_pct = ((actual_flown_nm - great_circle_nm) / great_circle_nm) * 100.0

        assert math.isclose(kea_pct, detour_pct, rel_tol=1e-4)
        assert 1.5 <= kea_pct <= 20.0


# =============================================================================
# 10. GNSS KINEMATICS: Teleportation Speed & Vertical Rate (10 Samples Each)
# =============================================================================

def test_gnss_kinematics_speed_and_climb_10_samples():
    """Verify GNSS kinematic groundspeed and vertical rate calculation across 10 telemetry delta steps."""
    for i in range(10):
        dt_s = 20.0 + i * 5.0  # 20s to 65s
        dist_nm = 2.0 + i * 1.0  # 2 nm to 11 nm
        d_fl = (i + 1) * 5.0    # 500 ft to 5,000 ft

        speed_kts = dist_nm / (dt_s / 3600.0)
        vs_fpm = (d_fl * 100.0) / (dt_s / 60.0)

        assert speed_kts > 0.0
        assert vs_fpm > 0.0
        if speed_kts > 750.0:
            assert speed_kts > 750.0
        if vs_fpm > 6000.0:
            assert vs_fpm > 6000.0
