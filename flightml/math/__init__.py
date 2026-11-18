"""
FlightML Core Mathematical Library.

A standalone, high-precision engineering mathematics and physics library serving:
1. Geodesy & Spherical Trigonometry (`flightml.math.geodesy`)
2. Atmospheric Physics & ISA (`flightml.math.atmosphere`)
3. Aerodynamics & Upper-Air Energetics (`flightml.math.aerodynamics`)
4. Climate Microphysics & Emissions (`flightml.math.climate`)
5. Aviation Economics & Passenger Liability (`flightml.math.economics`)
6. Operational Efficiency & Flight Inefficiency (`flightml.math.operations`)
7. GNSS Kinematics & Electronic Warfare Shield (`flightml.math.gnss_kinematics`)
"""

from __future__ import annotations

# Geodesy
from flightml.math.geodesy import (
    EARTH_RADIUS_KM,
    EARTH_RADIUS_NM,
    along_track_distance_nm,
    angular_difference_deg,
    circular_mean_coords,
    circular_mean_longitude,
    cross_track_distance_nm,
    final_bearing_deg,
    haversine_distance_km,
    haversine_distance_nm,
    initial_bearing_deg,
    normalize_longitude_deg,
    slerp_interpolation,
    spherical_midpoint,
    split_polyline_at_antimeridian,
)

# Atmosphere
from flightml.math.atmosphere import (
    FT_PER_M,
    G_GRAVITY,
    GAMMA_AIR,
    LAPSE_RATE_K_PER_M,
    M_PER_FT,
    MPS_TO_KTS,
    P0_HPA,
    R_GAS_AIR,
    RHO0_KG_M3,
    T_TROPOPAUSE_K,
    T0_K,
    TROPOPAUSE_ALT_M,
    altitude_m_to_flight_level,
    flight_level_to_altitude_ft,
    flight_level_to_altitude_m,
    isa_density_kg_m3,
    isa_pressure_hpa,
    isa_temperature_c,
    isa_temperature_k,
    pressure_to_altitude_m,
    speed_of_sound_kts,
    speed_of_sound_mps,
)

# Aerodynamics
from flightml.math.aerodynamics import (
    crosswind_component_kts,
    equivalent_still_air_distance_nm,
    headwind_tailwind_component_kts,
    load_factor_g,
    mach_to_tas_kts,
    rate_of_turn_deg_per_s,
    tas_to_mach,
    turn_radius_nm,
    wind_triangle_decomposition,
)

# Climate
from flightml.math.climate import (
    EI_CO2_KG_PER_KG_FUEL,
    EI_H2O_KG_PER_KG_FUEL,
    EI_NOX_G_PER_KG_FUEL_CRUISE,
    JET_FUEL_LOWER_HEATING_VALUE_MJ_KG,
    altitude_fuel_burn_penalty,
    contrail_energy_forcing_rf_j,
    fuel_to_co2_kg,
    fuel_to_h2o_kg,
    fuel_to_nox_g,
    is_ice_supersaturated,
    is_persistent_contrail_forming,
    relative_humidity_over_ice_pct,
    schmidt_appleman_threshold_c,
    sonntag_saturation_vapor_pressure_ice_hpa,
    sonntag_saturation_vapor_pressure_liquid_hpa,
)

# Economics
from flightml.math.economics import (
    COST_AIRBORNE_DELAY_EUR_PER_MIN,
    COST_GATE_DELAY_EUR_PER_MIN,
    COST_TAXI_DELAY_EUR_PER_MIN,
    DEFAULT_MTOW_TONNES,
    crco_distance_factor,
    crco_route_charge_eur,
    crco_weight_factor,
    delay_cost_by_phase_eur,
    eu261_statutory_compensation_eur,
    fuel_cost_eur,
    passenger_disruption_liability_eur,
    passenger_missed_connection_probability,
)

# Operations
from flightml.math.operations import (
    descent_flight_path_angle_deg,
    is_cdo_level_off_segment,
    kea_horizontal_inefficiency_pct,
    kep_planned_inefficiency_pct,
    ket_terminal_extension_nm,
    rlos_radio_horizon_km,
    rlos_radio_horizon_nm,
    runway_capacity_movements_per_hour,
    runway_separation_time_s,
)

# GNSS Kinematics
from flightml.math.gnss_kinematics import (
    gnss_composite_spoofing_score,
    is_confirmed_spoofing,
    is_terminal_vectoring_zone,
    teleportation_groundspeed_kts,
    track_jitter_rate_deg_per_s,
    vertical_rate_fpm,
)

__all__ = [
    # Geodesy
    "EARTH_RADIUS_NM",
    "EARTH_RADIUS_KM",
    "normalize_longitude_deg",
    "angular_difference_deg",
    "haversine_distance_nm",
    "haversine_distance_km",
    "initial_bearing_deg",
    "final_bearing_deg",
    "cross_track_distance_nm",
    "along_track_distance_nm",
    "spherical_midpoint",
    "slerp_interpolation",
    "circular_mean_longitude",
    "circular_mean_coords",
    "split_polyline_at_antimeridian",
    # Atmosphere
    "T0_K",
    "P0_HPA",
    "RHO0_KG_M3",
    "LAPSE_RATE_K_PER_M",
    "TROPOPAUSE_ALT_M",
    "T_TROPOPAUSE_K",
    "R_GAS_AIR",
    "GAMMA_AIR",
    "G_GRAVITY",
    "flight_level_to_altitude_m",
    "flight_level_to_altitude_ft",
    "altitude_m_to_flight_level",
    "isa_temperature_k",
    "isa_temperature_c",
    "isa_pressure_hpa",
    "isa_density_kg_m3",
    "speed_of_sound_mps",
    "speed_of_sound_kts",
    "pressure_to_altitude_m",
    # Aerodynamics
    "mach_to_tas_kts",
    "tas_to_mach",
    "headwind_tailwind_component_kts",
    "crosswind_component_kts",
    "wind_triangle_decomposition",
    "equivalent_still_air_distance_nm",
    "turn_radius_nm",
    "load_factor_g",
    "rate_of_turn_deg_per_s",
    # Climate
    "EI_CO2_KG_PER_KG_FUEL",
    "EI_H2O_KG_PER_KG_FUEL",
    "EI_NOX_G_PER_KG_FUEL_CRUISE",
    "JET_FUEL_LOWER_HEATING_VALUE_MJ_KG",
    "sonntag_saturation_vapor_pressure_liquid_hpa",
    "sonntag_saturation_vapor_pressure_ice_hpa",
    "relative_humidity_over_ice_pct",
    "is_ice_supersaturated",
    "schmidt_appleman_threshold_c",
    "is_persistent_contrail_forming",
    "altitude_fuel_burn_penalty",
    "fuel_to_co2_kg",
    "fuel_to_nox_g",
    "fuel_to_h2o_kg",
    "contrail_energy_forcing_rf_j",
    # Economics
    "DEFAULT_MTOW_TONNES",
    "COST_GATE_DELAY_EUR_PER_MIN",
    "COST_TAXI_DELAY_EUR_PER_MIN",
    "COST_AIRBORNE_DELAY_EUR_PER_MIN",
    "crco_weight_factor",
    "crco_distance_factor",
    "crco_route_charge_eur",
    "eu261_statutory_compensation_eur",
    "passenger_missed_connection_probability",
    "passenger_disruption_liability_eur",
    "delay_cost_by_phase_eur",
    "fuel_cost_eur",
    # Operations
    "kea_horizontal_inefficiency_pct",
    "kep_planned_inefficiency_pct",
    "ket_terminal_extension_nm",
    "rlos_radio_horizon_nm",
    "rlos_radio_horizon_km",
    "descent_flight_path_angle_deg",
    "is_cdo_level_off_segment",
    "runway_separation_time_s",
    "runway_capacity_movements_per_hour",
    # GNSS
    "teleportation_groundspeed_kts",
    "vertical_rate_fpm",
    "track_jitter_rate_deg_per_s",
    "is_terminal_vectoring_zone",
    "gnss_composite_spoofing_score",
    "is_confirmed_spoofing",
]
