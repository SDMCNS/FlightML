"""
FlightML Aviation Economics, Route Billing & Passenger Liability Module.

Provides exact formulation of:
- Eurocontrol Central Route Charges Office (CRCO) En-Route Charges formula
- Aircraft weight factors (MTOW) and charging service units
- EU261/2004 passenger compensation statutory bands and delay thresholds
- Passenger connection misconnect probability via logistic sigmoid response
- Phase-dependent airline operational delay cost curves (Gate, Taxi, Airborne)
- Jet fuel price and carbon allowance cost conversions
"""

from __future__ import annotations

import math

# Reference Aircraft MTOW in Tonnes (Eurocontrol PRU Standard Fleet)
DEFAULT_MTOW_TONNES: dict[str, float] = {
    "A320": 77.0,
    "A321": 89.0,
    "A20N": 79.0,
    "A21N": 97.0,
    "B738": 79.0,
    "B38M": 82.0,
    "B77W": 351.5,
    "B789": 254.0,
    "A359": 280.0,
    "A388": 575.0,
    "AT76": 23.0,
    "DH8D": 29.5,
    "E195": 52.3,
}

# University of Westminster / Eurocontrol Standard Cost of Delay (€/min)
COST_GATE_DELAY_EUR_PER_MIN: float = 45.0
COST_TAXI_DELAY_EUR_PER_MIN: float = 65.0
COST_AIRBORNE_DELAY_EUR_PER_MIN: float = 95.0


def crco_weight_factor(mtow_tonnes: float) -> float:
    """
    Calculate the Eurocontrol CRCO Aircraft Weight Factor (P_factor):
    P = sqrt(MTOW_metric_tonnes / 50.0).
    For a 50-tonne aircraft (reference baseline), P = 1.000.
    For an A320 (77 tonnes), P = 1.241.
    For an A380 (575 tonnes), P = 3.391.
    """
    m = max(1.0, float(mtow_tonnes))
    return math.sqrt(m / 50.0)


def crco_distance_factor(distance_km: float) -> float:
    """
    Calculate Eurocontrol CRCO Distance Factor (d_factor):
    d = distance_km / 100.0.
    Eurocontrol charges per 100 km of great-circle distance flown through national airspace.
    """
    return max(0.0, float(distance_km)) / 100.0


def crco_route_charge_eur(
    distance_km: float,
    mtow_tonnes: float,
    unit_rate_eur: float,
) -> float:
    """
    Calculate Eurocontrol CRCO route charge in Euros:
    Cost = (Distance_km / 100.0) * sqrt(MTOW / 50.0) * UnitRate_EUR.
    """
    dist_factor = crco_distance_factor(distance_km)
    weight_fac = crco_weight_factor(mtow_tonnes)
    rate = max(0.0, float(unit_rate_eur))
    return dist_factor * weight_fac * rate


def eu261_statutory_compensation_eur(
    great_circle_km: float,
    arrival_delay_min: float,
    is_cancelled: bool = False,
) -> float:
    """
    Calculate passenger statutory compensation entitlement under Regulation (EC) No 261/2004:
    Entitlement triggers on >= 3 hours (180 minutes) delay at final destination or short-notice cancellation:
    - Band A (<= 1,500 km): €250
    - Band B (1,500 km to 3,500 km, or all intra-EU flights > 1,500 km): €400
    - Band C (> 3,500 km non-EU): €600
    """
    if not is_cancelled and float(arrival_delay_min) < 180.0:
        return 0.0

    dist = max(0.0, float(great_circle_km))
    if dist <= 1500.0:
        return 250.0
    elif dist <= 3500.0:
        return 400.0
    else:
        return 600.0


def passenger_missed_connection_probability(
    inbound_delay_min: float,
    scheduled_layover_min: float,
    minimum_connect_time_min: float = 45.0,
    sigmoid_steepness: float = 0.18,
) -> float:
    """
    Calculate the mathematical probability of a passenger missing an onward flight connection
    using a calibrated logistic sigmoid transfer function:
    P(miss) = 1.0 / (1.0 + exp(-k * (EffectiveDelay - Buffer)))
    where Buffer = ScheduledLayover - MinimumConnectTime.
    """
    buffer_min = float(scheduled_layover_min) - float(minimum_connect_time_min)
    delay_excess_min = float(inbound_delay_min) - buffer_min

    # Prevent numerical overflow in exponential
    clamped_arg = max(-25.0, min(25.0, -sigmoid_steepness * delay_excess_min))
    return 1.0 / (1.0 + math.exp(clamped_arg))


def passenger_disruption_liability_eur(
    pax_count: int,
    great_circle_km: float,
    missed_probability: float,
    duty_of_care_hotel_meal_pax_eur: float = 160.0,
) -> float:
    """
    Calculate expected airline financial liability for missed connections across passenger manifest:
    ExpectedCost = Pax * P(miss) * (EU261_Compensation + DutyOfCareCost).
    """
    if pax_count <= 0 or missed_probability <= 0.0:
        return 0.0

    p_miss = max(0.0, min(1.0, float(missed_probability)))
    eu261 = eu261_statutory_compensation_eur(great_circle_km, arrival_delay_min=240.0)
    total_cost_per_missed_pax = eu261 + float(duty_of_care_hotel_meal_pax_eur)
    return float(pax_count) * p_miss * total_cost_per_missed_pax


def delay_cost_by_phase_eur(
    gate_delay_min: float = 0.0,
    taxi_delay_min: float = 0.0,
    airborne_delay_min: float = 0.0,
) -> float:
    """
    Calculate tactical delay costs across operational phases:
    Total = (GateMin * €45) + (TaxiMin * €65) + (AirborneMin * €95).
    """
    return (
        max(0.0, float(gate_delay_min)) * COST_GATE_DELAY_EUR_PER_MIN
        + max(0.0, float(taxi_delay_min)) * COST_TAXI_DELAY_EUR_PER_MIN
        + max(0.0, float(airborne_delay_min)) * COST_AIRBORNE_DELAY_EUR_PER_MIN
    )


def fuel_cost_eur(fuel_kg: float, price_eur_per_kg: float = 0.78) -> float:
    """Calculate total fuel expenditure given fuel mass in kg and spot price in €/kg."""
    return max(0.0, float(fuel_kg)) * max(0.0, float(price_eur_per_kg))
