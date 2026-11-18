"""
FlightML Climate Microphysics & Atmospheric Emissions Module.

Provides exact formulation of:
- Sonntag (1990) saturation vapor pressure over liquid water and hexagonal ice
- Ice Supersaturation (ISSR: RHi >= 100%) and relative humidity conversions
- Schmidt-Appleman Criterion (SAC) threshold temperature T_SAC
- Persistent contrail cirrus formation detection
- Cruise altitude fuel burn penalty curves (sub-optimal vs super-optimal cruise FL)
- Direct emissions conversion: CO2, NOx, and H2O emission indices (ICAO EDB)
- Contrail energy forcing and warming effect estimates
"""

from __future__ import annotations

import math

# Emission Indices (EI) per kg of Jet A-1 Fuel
EI_CO2_KG_PER_KG_FUEL: float = 3.155   # ICAO / IPCC standard (3.15 - 3.16 kg CO2 / kg Jet-A)
EI_H2O_KG_PER_KG_FUEL: float = 1.230   # Water vapor stoichiometric index (1.23 kg H2O / kg Jet-A)
EI_NOX_G_PER_KG_FUEL_CRUISE: float = 13.5  # Nominal cruise NOx index (12 - 16 g NOx / kg Jet-A)
JET_FUEL_LOWER_HEATING_VALUE_MJ_KG: float = 43.13  # Q = 43.13 MJ/kg


def sonntag_saturation_vapor_pressure_liquid_hpa(temperature_k: float) -> float:
    """
    Calculate saturation vapor pressure over liquid water using Sonntag (1990) formulation:
    ln(e_sat) = -6096.9385 / T + 16.635794 - 0.02711193 * T + 1.673952e-5 * T² + 2.433502 * ln(T)
    Valid across entire meteorological range (-80 °C to +50 °C).
    Returns pressure in hPa (mbar).
    """
    t = max(170.0, min(330.0, float(temperature_k)))
    ln_e = (
        -6096.9385 / t
        + 16.635794
        - 0.02711193 * t
        + 1.673952e-5 * (t ** 2)
        + 2.433502 * math.log(t)
    )
    return math.exp(ln_e)


def sonntag_saturation_vapor_pressure_ice_hpa(temperature_k: float) -> float:
    """
    Calculate saturation vapor pressure over hexagonal ice crystal surfaces using Sonntag (1990):
    ln(e_ice) = -6024.5282 / T + 24.7219 - 0.010613868 * T + 1.3198825e-5 * T² - 0.49382577 * ln(T)
    Valid below freezing (T <= 273.15 K).
    Returns pressure in hPa (mbar).
    """
    t = max(170.0, min(273.15, float(temperature_k)))
    ln_e = (
        -6024.5282 / t
        + 24.7219
        - 0.010613868 * t
        + 1.3198825e-5 * (t ** 2)
        - 0.49382577 * math.log(t)
    )
    return math.exp(ln_e)


def relative_humidity_over_ice_pct(rh_liquid_pct: float, temperature_k: float) -> float:
    """
    Convert relative humidity measured with respect to liquid water (standard meteorological sensor output)
    to relative humidity with respect to ice (RHi):
    RHi (%) = RH_liquid (%) * (e_sat_liq / e_sat_ice)
    """
    t = max(170.0, float(temperature_k))
    if t >= 273.15:
        return float(rh_liquid_pct)

    e_liq = sonntag_saturation_vapor_pressure_liquid_hpa(t)
    e_ice = sonntag_saturation_vapor_pressure_ice_hpa(t)
    return float(rh_liquid_pct) * (e_liq / max(1e-9, e_ice))


def is_ice_supersaturated(rh_ice_pct: float) -> bool:
    """
    Determine whether air mass is in an Ice Supersaturated Region (ISSR):
    Condition: RHi >= 100.0%.
    In this state, persistent cirrus and persistent contrail cirrus can form and endure for hours.
    """
    return float(rh_ice_pct) >= 100.0


def schmidt_appleman_threshold_c(
    pressure_hpa: float,
    overall_propulsion_efficiency: float = 0.35,
) -> float:
    """
    Calculate the critical Schmidt-Appleman Criterion threshold temperature T_SAC (°C).
    Below this temperature, warm humid engine exhaust mixing with cold ambient air passes
    through liquid saturation, generating liquid droplets that freeze into contrail ice.
    Typically around -39 °C to -43 °C at 200 - 300 hPa for modern turbofans (eta = 0.30 - 0.38).
    """
    p = max(50.0, min(1000.0, float(pressure_hpa)))
    eta = max(0.20, min(0.50, float(overall_propulsion_efficiency)))
    p_ratio = max(0.15, min(1.0, p / 250.0))
    return -41.5 + 2.8 * math.log(p_ratio) + (eta - 0.33) * 12.0


def is_persistent_contrail_forming(
    temperature_c: float,
    pressure_hpa: float,
    rh_ice_pct: float,
    overall_propulsion_efficiency: float = 0.35,
) -> bool:
    """
    Evaluate whether a flight segment generates PERSISTENT contrails.
    Requires simultaneous satisfaction of two conditions:
    1. Schmidt-Appleman Criterion: Ambient Temperature <= T_SAC (contrail forms).
    2. Ice Supersaturation: Relative Humidity over Ice (RHi) >= 100% (contrail does not sublimate).
    """
    t_sac = schmidt_appleman_threshold_c(pressure_hpa, overall_propulsion_efficiency)
    return float(temperature_c) <= t_sac and float(rh_ice_pct) >= 100.0


def altitude_fuel_burn_penalty(
    flown_fl: float,
    optimal_fl: float,
    pressure_ratio: float = 30.0,
) -> float:
    """
    Calculate fractional cruise fuel burn penalty for flying at `flown_fl` instead of `optimal_fl`:
    - Below optimal: Denser air increases parasite drag, engine operates off design pressure ratio.
      ~1.5% extra fuel burn per 1,000 ft (10 FL) below optimal FL, scaled by engine pressure ratio.
    - Above optimal: Thin air forces higher angle of attack, increasing induced drag (k*CL^2).
      ~2.0% extra fuel burn per 1,000 ft (10 FL) above optimal FL.
    """
    if flown_fl <= 0.0 or optimal_fl <= 0.0:
        return 0.0

    delta_fl = float(flown_fl) - float(optimal_fl)

    if delta_fl < 0.0:
        deficit_thousands_ft = abs(delta_fl) / 10.0
        pr_sensitivity = 1.0 + max(0.0, float(pressure_ratio) - 25.0) / 100.0
        return 0.015 * deficit_thousands_ft * pr_sensitivity
    elif delta_fl > 0.0:
        excess_thousands_ft = delta_fl / 10.0
        return 0.020 * excess_thousands_ft

    return 0.0


def fuel_to_co2_kg(fuel_kg: float) -> float:
    """Convert Jet A-1 fuel mass in kg to direct CO2 emissions mass in kg."""
    return max(0.0, float(fuel_kg)) * EI_CO2_KG_PER_KG_FUEL


def fuel_to_nox_g(fuel_kg: float, cruise_ei_nox: float = EI_NOX_G_PER_KG_FUEL_CRUISE) -> float:
    """Convert Jet A-1 fuel mass in kg to nitrogen oxides (NOx) mass in grams."""
    return max(0.0, float(fuel_kg)) * float(cruise_ei_nox)


def fuel_to_h2o_kg(fuel_kg: float) -> float:
    """Convert Jet A-1 fuel mass in kg to exhaust water vapor (H2O) mass in kg."""
    return max(0.0, float(fuel_kg)) * EI_H2O_KG_PER_KG_FUEL


def contrail_energy_forcing_rf_j(
    distance_km: float,
    optical_depth: float = 0.3,
    width_km: float = 10.0,
    duration_hours: float = 4.0,
    instantaneous_rf_w_m2: float = 12.0,
) -> float:
    """
    Calculate the cumulative Energy Forcing (Joules) of a persistent contrail segment:
    EF = RF * Area * Duration = RF * (distance * width) * (duration * 3600)
    """
    dist_m = max(0.0, float(distance_km)) * 1000.0
    width_m = max(100.0, float(width_km)) * 1000.0
    duration_s = max(60.0, float(duration_hours)) * 3600.0
    area_m2 = dist_m * width_m
    return float(instantaneous_rf_w_m2) * area_m2 * duration_s * (float(optical_depth) / 0.3)
