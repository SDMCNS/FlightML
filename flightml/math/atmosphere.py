"""
FlightML Atmospheric Physics & International Standard Atmosphere (ISA) Module.

Provides exact thermodynamic and fluid dynamic properties according to ICAO Doc 7488:
- ISA temperature lapse rates across troposphere and lower stratosphere
- Barometric pressure equation and geopotential altitude relations
- Air density calculation via ideal gas law
- Local speed of sound in m/s and knots
- Flight Level (FL), meter, and foot unit conversions
"""

from __future__ import annotations

import math

# Standard Sea Level Atmosphere Constants (ICAO Doc 7488)
T0_K: float = 288.15           # Standard sea-level temperature (+15.0 °C)
P0_HPA: float = 1013.25        # Standard sea-level pressure (hPa / mbar)
RHO0_KG_M3: float = 1.2250     # Standard sea-level air density (kg/m³)
LAPSE_RATE_K_PER_M: float = 0.0065  # Tropospheric lapse rate (6.5 K / km)
TROPOPAUSE_ALT_M: float = 11000.0   # Tropopause boundary (11,000 m / 36,089 ft / ~FL361)
T_TROPOPAUSE_K: float = 216.65      # Isothermal tropopause temperature (-56.5 °C)

# Thermodynamic Gas Properties
R_GAS_AIR: float = 287.05287    # Specific gas constant for dry air (J / (kg * K))
GAMMA_AIR: float = 1.400        # Specific heat ratio (isentropic expansion factor)
G_GRAVITY: float = 9.80665      # Standard acceleration due to gravity (m/s²)

# Unit conversion constants
M_PER_FT: float = 0.3048
FT_PER_M: float = 1.0 / 0.3048
MPS_TO_KTS: float = 1.94384449
KTS_TO_MPS: float = 1.0 / 1.94384449


def flight_level_to_altitude_m(flight_level: float) -> float:
    """Convert Flight Level (e.g. FL350) to geometric altitude in meters."""
    return float(flight_level) * 100.0 * M_PER_FT


def flight_level_to_altitude_ft(flight_level: float) -> float:
    """Convert Flight Level (e.g. FL350) to altitude in feet (35,000 ft)."""
    return float(flight_level) * 100.0


def altitude_m_to_flight_level(altitude_m: float) -> float:
    """Convert altitude in meters to standard Flight Level (FL)."""
    return (float(altitude_m) * FT_PER_M) / 100.0


def isa_temperature_k(altitude_m: float) -> float:
    """
    Calculate International Standard Atmosphere (ISA) temperature in Kelvin.
    - Troposphere (0 to 11,000 m): T(h) = T0 - L * h
    - Lower Stratosphere (11,000 to 20,000 m): Isothermal at 216.65 K (-56.5 °C)
    """
    h = max(0.0, float(altitude_m))
    if h <= TROPOPAUSE_ALT_M:
        return T0_K - LAPSE_RATE_K_PER_M * h
    else:
        # Lower stratosphere isothermal layer up to 20,000m
        return T_TROPOPAUSE_K


def isa_temperature_c(altitude_m: float) -> float:
    """Calculate ISA temperature in degrees Celsius."""
    return isa_temperature_k(altitude_m) - 273.15


def isa_pressure_hpa(altitude_m: float) -> float:
    """
    Calculate International Standard Atmosphere (ISA) barometric pressure in hPa (mbar).
    - Troposphere: P = P0 * (1 - (L*h)/T0) ^ (g / (R*L))
    - Lower Stratosphere: P = P_tropo * exp(-g*(h - h_tropo) / (R * T_tropo))
    """
    h = max(0.0, float(altitude_m))
    exponent_tropo = G_GRAVITY / (R_GAS_AIR * LAPSE_RATE_K_PER_M)  # ~5.25588

    if h <= TROPOPAUSE_ALT_M:
        temp_ratio = 1.0 - (LAPSE_RATE_K_PER_M * h) / T0_K
        return P0_HPA * (temp_ratio ** exponent_tropo)
    else:
        p_tropo = P0_HPA * ((T_TROPOPAUSE_K / T0_K) ** exponent_tropo)  # ~226.32 hPa
        delta_h = h - TROPOPAUSE_ALT_M
        return p_tropo * math.exp(- (G_GRAVITY * delta_h) / (R_GAS_AIR * T_TROPOPAUSE_K))


def isa_density_kg_m3(altitude_m: float) -> float:
    """
    Calculate standard air density rho in kg/m³ using the Ideal Gas Equation: rho = P / (R * T).
    """
    p_pa = isa_pressure_hpa(altitude_m) * 100.0  # hPa to Pa
    t_k = isa_temperature_k(altitude_m)
    return p_pa / (R_GAS_AIR * t_k)


def speed_of_sound_mps(altitude_m: float) -> float:
    """
    Calculate local speed of sound in meters per second: a = sqrt(gamma * R * T).
    Depends strictly on absolute thermodynamic temperature.
    """
    t_k = isa_temperature_k(altitude_m)
    return math.sqrt(GAMMA_AIR * R_GAS_AIR * t_k)


def speed_of_sound_kts(altitude_m: float) -> float:
    """Calculate local speed of sound in nautical miles per hour (knots)."""
    return speed_of_sound_mps(altitude_m) * MPS_TO_KTS


def pressure_to_altitude_m(pressure_hpa: float) -> float:
    """
    Invert standard tropospheric barometric equation: returns altitude in meters for given pressure in hPa.
    """
    p = max(50.0, min(1050.0, float(pressure_hpa)))
    exponent = (R_GAS_AIR * LAPSE_RATE_K_PER_M) / G_GRAVITY  # ~0.19026
    alt_m = (T0_K / LAPSE_RATE_K_PER_M) * (1.0 - (p / P0_HPA) ** exponent)
    return max(0.0, alt_m)
