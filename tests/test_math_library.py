"""
Unit Tests for the FlightML Core Mathematics Library (`flightml.math`).

Validates correctness, edge cases, numerical precision, and physical boundaries across:
1. Geodesy & Spherical Trigonometry
2. Atmospheric Physics (ISA)
3. Aerodynamics & Kinematics
4. Climate Microphysics & Emissions
5. Aviation Economics & Passenger Liability
6. Operational Efficiency & Navigation
7. GNSS Kinematics & EW Gating
"""

import math
import pytest
import flightml.math as fmath


def test_geodesy_haversine_and_bearings():
    # Equator 10 degrees east: 600 NM
    dist = fmath.haversine_distance_nm(0.0, 0.0, 0.0, 10.0)
    assert math.isclose(dist, 600.0, rel_tol=1e-2)

    # Initial bearing due East
    bearing = fmath.initial_bearing_deg(0.0, 0.0, 0.0, 10.0)
    assert math.isclose(bearing, 90.0, abs_tol=1e-3)

    # Cross track: point 1 deg north of equator path
    xtd = fmath.cross_track_distance_nm(1.0, 5.0, 0.0, 0.0, 0.0, 10.0)
    assert math.isclose(abs(xtd), 60.0, rel_tol=1e-2)

    # Midpoint
    m_lat, m_lon = fmath.spherical_midpoint(0.0, 0.0, 0.0, 10.0)
    assert math.isclose(m_lat, 0.0, abs_tol=1e-3)
    assert math.isclose(m_lon, 5.0, abs_tol=1e-3)

    # Slerp
    s_lat, s_lon = fmath.slerp_interpolation(0.0, 0.0, 0.0, 10.0, 0.25)
    assert math.isclose(s_lat, 0.0, abs_tol=1e-3)
    assert math.isclose(s_lon, 2.5, abs_tol=1e-2)


def test_geodesy_antimeridian_handling():
    # Clustered across 180°
    mean_lon = fmath.circular_mean_longitude([179.0, -179.0, 178.5, -178.5])
    assert math.isclose(abs(mean_lon), 180.0, abs_tol=1e-2)

    # Polyline splitting
    coords = [(50.0, 170.0), (52.0, 179.0), (53.0, -179.0), (55.0, -170.0)]
    segments = fmath.split_polyline_at_antimeridian(coords, threshold_deg=180.0)
    assert len(segments) == 2
    assert len(segments[0]) == 2
    assert len(segments[1]) == 2


def test_atmosphere_isa_physics():
    # Sea level
    assert math.isclose(fmath.isa_temperature_c(0.0), 15.0, abs_tol=1e-3)
    assert math.isclose(fmath.isa_pressure_hpa(0.0), 1013.25, abs_tol=1e-2)
    assert math.isclose(fmath.isa_density_kg_m3(0.0), 1.225, rel_tol=1e-2)

    # Tropopause FL360
    alt_m = fmath.flight_level_to_altitude_m(360.89)
    assert math.isclose(fmath.isa_temperature_c(alt_m), -56.5, abs_tol=0.5)

    # Speed of sound
    sos_0 = fmath.speed_of_sound_kts(0.0)
    assert 660.0 < sos_0 < 665.0  # ~661.5 kts at sea level


def test_aerodynamics_airspeed_and_wind():
    # Mach 0.80 at -50 °C (223.15 K)
    tas = fmath.mach_to_tas_kts(0.80, 223.15)
    assert 460.0 < tas < 480.0

    # Invert Mach
    m_inv = fmath.tas_to_mach(tas, 223.15)
    assert math.isclose(m_inv, 0.80, rel_tol=1e-3)

    # Wind components: heading East (90°), wind from East (90°) -> 40 kt Headwind
    hw = fmath.headwind_tailwind_component_kts(90.0, 90.0, 40.0)
    assert math.isclose(hw, -40.0, abs_tol=1e-3)

    # Wind from North (0°) -> 40 kt Crosswind
    xw = fmath.crosswind_component_kts(90.0, 0.0, 40.0)
    assert math.isclose(xw, 40.0, abs_tol=1e-3)

    # ESAD in tailwind (400 nm ground, TAS 450, GS 500) -> 360 nm ESAD
    esad = fmath.equivalent_still_air_distance_nm(400.0, 450.0, 500.0)
    assert math.isclose(esad, 360.0, rel_tol=1e-3)

    # Turn radius at 450 kts, 30° bank
    r = fmath.turn_radius_nm(450.0, 30.0)
    assert 4.0 < r < 8.0

    # Load factor at 60° bank -> 2.0g
    assert math.isclose(fmath.load_factor_g(60.0), 2.0, rel_tol=1e-3)


def test_climate_sonntag_and_contrail_sac():
    # Sonntag vapor pressure: e_liq > e_ice below 0°C
    t_k = 233.15  # -40 °C
    e_liq = fmath.sonntag_saturation_vapor_pressure_liquid_hpa(t_k)
    e_ice = fmath.sonntag_saturation_vapor_pressure_ice_hpa(t_k)
    assert e_liq > e_ice
    assert e_ice > 0.0

    # Relative humidity over ice
    rhi = fmath.relative_humidity_over_ice_pct(80.0, t_k)
    assert rhi > 80.0  # RHi is higher than RH_liquid at sub-zero temperatures

    # T_SAC threshold at 250 hPa
    t_sac = fmath.schmidt_appleman_threshold_c(250.0, 0.35)
    assert -43.0 < t_sac < -39.0

    # Persistent contrail formation
    assert fmath.is_persistent_contrail_forming(-45.0, 250.0, 105.0) is True
    assert fmath.is_persistent_contrail_forming(-35.0, 250.0, 105.0) is False  # Too warm
    assert fmath.is_persistent_contrail_forming(-45.0, 250.0, 90.0) is False   # Too dry (RHi < 100%)

    # Emissions
    assert math.isclose(fmath.fuel_to_co2_kg(1000.0), 3155.0, rel_tol=1e-2)
    assert math.isclose(fmath.fuel_to_h2o_kg(1000.0), 1230.0, rel_tol=1e-2)


def test_economics_crco_and_passenger():
    # CRCO: 50-tonne aircraft (weight factor 1.0), 100 km (distance factor 1.0) at €75
    charge = fmath.crco_route_charge_eur(100.0, 50.0, 75.0)
    assert math.isclose(charge, 75.0, rel_tol=1e-3)

    # A320 (77t), 1000 km, €75 unit rate
    charge_a320 = fmath.crco_route_charge_eur(1000.0, 77.0, 75.0)
    assert math.isclose(charge_a320, 10.0 * math.sqrt(77.0 / 50.0) * 75.0, rel_tol=1e-3)

    # EU261 statutory bands
    assert fmath.eu261_statutory_compensation_eur(1000.0, arrival_delay_min=190.0) == 250.0
    assert fmath.eu261_statutory_compensation_eur(2500.0, arrival_delay_min=200.0) == 400.0
    assert fmath.eu261_statutory_compensation_eur(5000.0, arrival_delay_min=240.0) == 600.0
    assert fmath.eu261_statutory_compensation_eur(5000.0, arrival_delay_min=60.0) == 0.0  # < 3 hours

    # Passenger connection sigmoid
    p_safe = fmath.passenger_missed_connection_probability(inbound_delay_min=10.0, scheduled_layover_min=90.0)
    p_critical = fmath.passenger_missed_connection_probability(inbound_delay_min=70.0, scheduled_layover_min=90.0)
    assert p_safe < 0.05
    assert p_critical > 0.90


def test_operations_kea_and_rlos():
    # KEA: 1050 nm flown vs 1000 nm direct -> 5.0%
    kea = fmath.kea_horizontal_inefficiency_pct(1050.0, 1000.0)
    assert math.isclose(kea, 5.0, rel_tol=1e-3)

    # RLOS at FL350 (35,000 ft) with 100 ft mast
    rlos = fmath.rlos_radio_horizon_nm(100.0, 35000.0)
    assert 230.0 < rlos < 255.0

    # CDO level-off detection
    assert fmath.is_cdo_level_off_segment(delta_fl=1.0, duration_s=180.0) is True
    assert fmath.is_cdo_level_off_segment(delta_fl=10.0, duration_s=180.0) is False  # Descending cleanly


def test_gnss_kinematics_and_screening():
    # Teleportation: 20 nm in 30 seconds -> 2400 kts
    speed = fmath.teleportation_groundspeed_kts(20.0, 30.0)
    assert math.isclose(speed, 2400.0, rel_tol=1e-3)

    # Vertical spike: 5000 ft in 20 seconds -> 15,000 fpm
    vs = fmath.vertical_rate_fpm(5000.0, 20.0)
    assert math.isclose(vs, 15000.0, rel_tol=1e-3)

    # Terminal vectoring gate: 30 nm from origin -> inside terminal zone
    assert fmath.is_terminal_vectoring_zone(30.0, 200.0) is True
    # 80 nm from origin and 120 nm from dest -> en-route
    assert fmath.is_terminal_vectoring_zone(80.0, 120.0) is False

    # Score calculation
    score = fmath.gnss_composite_spoofing_score(teleport_count=2, vertical_spike_count=1, altitude_glitch_count=0, track_jitter_count=0)
    assert score >= 75.0
    assert fmath.is_confirmed_spoofing(score, has_teleport=True, has_vertical_spike=True, has_altitude_glitch=False) is True
