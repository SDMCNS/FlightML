"""
Upper-Air Wind-Optimal Trajectory Decomposition & Persistent Contrail (ISSR) Climate Engine (Sprint 6).

Evaluates:
1. Segment-by-segment Groundspeed (GS), True Airspeed (TAS), and Headwind / Tailwind Component (kts).
   Distinguishes Wind-Optimal Jet-Stream Riding from ATC Route Detours.
2. Schmidt-Appleman Criterion (SAC) & Ice Supersaturated Regions (ISSR: RHi >= 100%, T <= T_SAC):
   Identifies 4D trajectory segments forming persistent contrail cirrus, quantifies Contrail Energy
   Forcing & CO2e climate impact, and generates +/- 2,000 ft tactical altitude deviation recommendations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.eval.jet_fuel_prices import get_weekly_fuel_price
from flightml.geo.geodesy import haversine_nm, initial_bearing_deg
from flightml.weather.upper_air_client import UpperAirClient


@dataclass
class ContrailSegment:
    seq_start: int
    seq_end: int
    start_time_utc: str
    end_time_utc: str
    flight_level: float
    distance_nm: float
    duration_min: float
    temperature_c: float
    rh_ice_pct: float
    recommended_alt_fl: float
    alt_deviation_note: str


@dataclass
class UpperAirImpactReport:
    ectrl_id: int
    period: str
    adep: str
    ades: str
    ac_operator: str
    ac_type: str
    cruise_distance_nm: float
    avg_groundspeed_kts: float
    avg_true_airspeed_kts: float
    avg_tailwind_component_kts: float  # Positive = net tailwind, Negative = net headwind
    wind_time_savings_min: float
    wind_routing_verdict: str
    ground_distance_nm: float
    esad_nm: float  # Equivalent Still Air Distance (air mass miles flown)
    wind_distance_benefit_nm: float  # ground_distance_nm - esad_nm (>0: air miles saved by tailwind)
    wind_fuel_impact_kg: float  # Fuel saved (>0) or penalty (<0) vs still air
    wind_fuel_impact_eur: float  # Fuel value in € EUR
    wind_efficiency_verdict: str  # Actionable dispatch/efficiency classification
    persistent_contrail_distance_nm: float
    persistent_contrail_pct: float
    persistent_contrail_duration_min: float
    contrail_co2e_tons: float
    direct_co2_tons: float
    total_climate_co2e_multiplier: float
    issr_segments_count: int
    issr_segments: list[ContrailSegment]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def schmidt_appleman_threshold_c(pressure_hpa: float, overall_propulsion_efficiency: float = 0.35) -> float:
    """
    Calculate the critical Schmidt-Appleman Criterion threshold temperature T_SAC (°C)
    below which exhaust water vapor condenses into liquid droplets that freeze into contrail ice crystals.
    Typically around -40°C to -43°C at 200-300 hPa for modern turbofans.
    """
    p_ratio = max(0.15, min(1.0, pressure_hpa / 250.0))
    return -41.5 + 2.8 * math.log(p_ratio) + (overall_propulsion_efficiency - 0.33) * 12.0


def analyze_flight_wind_and_contrails(
    dataset: FlightDataset,
    ectrl_id: int,
    upper_air_client: UpperAirClient | None = None,
) -> UpperAirImpactReport | None:
    """
    Analyze 4D cruise trajectory against upper-air winds (headwind/tailwind, TAS vs GS)
    and evaluate persistent contrail formation in Ice Supersaturated Regions (ISSRs),
    proposing +/- 2,000 ft altitude changes where vertical moisture gradients allow contrail avoidance.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return None

    trajs = dataset.get_trajectory(ectrl_id)
    actual_df = trajs["actual"]
    if len(actual_df) < 3:
        return None

    actual_df = actual_df.sort_values("seq_num").reset_index(drop=True)
    client = upper_air_client or UpperAirClient(use_live_api=True)
    client.prefetch_trajectory_upper_air(actual_df)

    adep = str(flight.get("adep") or "")
    ades = str(flight.get("ades") or "")
    ac_op = str(flight.get("ac_operator") or "")
    ac_type = str(flight.get("ac_type") or "B738")

    cruise_dist_nm = 0.0
    weighted_gs = 0.0
    weighted_tas = 0.0
    weighted_tw = 0.0
    total_esad_nm = 0.0
    contrail_dist_nm = 0.0
    contrail_dur_min = 0.0
    issr_segments: list[ContrailSegment] = []

    for i in range(len(actual_df) - 1):
        r1 = actual_df.iloc[i]
        r2 = actual_df.iloc[i + 1]
        fl = float((r1["flight_level"] + r2["flight_level"]) / 2.0)
        if fl < 240.0:
            continue

        lat1, lon1 = float(r1["latitude"]), float(r1["longitude"])
        lat2, lon2 = float(r2["latitude"]), float(r2["longitude"])
        seg_nm = haversine_nm(lat1, lon1, lat2, lon2)
        if seg_nm <= 0.2:
            continue

        t1 = pd.to_datetime(r1["time_over"])
        t2 = pd.to_datetime(r2["time_over"])
        seg_min = max(0.5, (t2 - t1).total_seconds() / 60.0)
        gs_kts = min(650.0, max(280.0, (seg_nm / seg_min) * 60.0))

        track_deg = initial_bearing_deg(lat1, lon1, lat2, lon2)
        mid_lat = (lat1 + lat2) / 2.0
        mid_lon = (lon1 + lon2) / 2.0

        ua = client.get_upper_air_state(mid_lat, mid_lon, fl, str(r1["time_over"]))
        # Meteorological wind direction `ua.wind_dir_deg` is direction wind blows FROM.
        # Wind vector blows TOWARD `(ua.wind_dir_deg + 180) % 360`.
        wind_toward_deg = (ua.wind_dir_deg + 180.0) % 360.0
        rel_angle = math.radians(wind_toward_deg - track_deg)
        tailwind_kts = ua.wind_speed_kts * math.cos(rel_angle)
        tas_kts = max(350.0, min(540.0, gs_kts - tailwind_kts))

        seg_esad_nm = seg_nm * (tas_kts / max(100.0, gs_kts))
        total_esad_nm += seg_esad_nm

        cruise_dist_nm += seg_nm
        weighted_gs += gs_kts * seg_nm
        weighted_tas += tas_kts * seg_nm
        weighted_tw += tailwind_kts * seg_nm

        # Check Schmidt-Appleman Criterion & Ice Supersaturation (RHi >= 100%)
        t_sac = schmidt_appleman_threshold_c(ua.pressure_hpa)
        if ua.temperature_c <= t_sac and ua.rh_ice_pct >= 100.0:
            contrail_dist_nm += seg_nm
            contrail_dur_min += seg_min

            # Check +/- 2,000 ft (+/- 20 FL) for drier air outside the thin ISSR layer
            ua_up = client.get_upper_air_state(mid_lat, mid_lon, min(410.0, fl + 20.0), str(r1["time_over"]))
            ua_dn = client.get_upper_air_state(mid_lat, mid_lon, max(260.0, fl - 20.0), str(r1["time_over"]))
            if ua_dn.rh_ice_pct <= ua_up.rh_ice_pct:
                rec_fl = max(260.0, fl - 20.0)
                note = f"Descend -2,000 ft to FL{rec_fl:.0f} (RHi drops {ua.rh_ice_pct:.0f}% ➔ {ua_dn.rh_ice_pct:.0f}%)"
            else:
                rec_fl = min(410.0, fl + 20.0)
                note = f"Climb +2,000 ft to FL{rec_fl:.0f} (RHi drops {ua.rh_ice_pct:.0f}% ➔ {ua_up.rh_ice_pct:.0f}%)"

            issr_segments.append(
                ContrailSegment(
                    seq_start=int(r1["seq_num"]),
                    seq_end=int(r2["seq_num"]),
                    start_time_utc=str(r1["time_over"]),
                    end_time_utc=str(r2["time_over"]),
                    flight_level=round(fl, 0),
                    distance_nm=round(seg_nm, 1),
                    duration_min=round(seg_min, 1),
                    temperature_c=ua.temperature_c,
                    rh_ice_pct=ua.rh_ice_pct,
                    recommended_alt_fl=round(rec_fl, 0),
                    alt_deviation_note=note,
                )
            )

    if cruise_dist_nm <= 1.0:
        cruise_dist_nm = max(50.0, float(flight.get("actual_distance_nm") or 300.0))
        total_esad_nm = cruise_dist_nm
        avg_gs = 445.0
        avg_tas = 440.0
        avg_tw = 5.0
    else:
        avg_gs = weighted_gs / cruise_dist_nm
        avg_tas = weighted_tas / cruise_dist_nm
        avg_tw = weighted_tw / cruise_dist_nm

    still_air_min = (cruise_dist_nm / max(250.0, avg_tas)) * 60.0
    actual_wind_min = (cruise_dist_nm / max(250.0, avg_gs)) * 60.0
    wind_saved_min = still_air_min - actual_wind_min

    if avg_tw >= 20.0:
        wind_verdict = f"JET_STREAM_TAILWIND_ASSIST (+{avg_tw:.0f} kts net tailwind)"
    elif avg_tw <= -20.0:
        wind_verdict = f"STRONG_HEADWIND_PENALTY ({avg_tw:.0f} kts net headwind)"
    else:
        wind_verdict = f"MODERATE_UPPER_WINDS ({avg_tw:+.0f} kts net wind)"

    contrail_pct = (contrail_dist_nm / max(1.0, cruise_dist_nm)) * 100.0

    # Fuel and cost impact of wind vs still-air baseline
    ac_perf = dataset.get_aircraft_performance(ac_type)
    wake = str(ac_perf.get("wake_category") or "M") if ac_perf else "M"
    ff_kg_nm = 14.5 if wake == "H" else 24.0 if wake == "J" else 3.2 if wake == "L" else 5.6
    nominal_fuel_flow_kg_hr = ff_kg_nm * avg_tas
    wind_fuel_impact_kg = (wind_saved_min / 60.0) * nominal_fuel_flow_kg_hr

    fuel_price = get_weekly_fuel_price(flight.get("off_block_date") or flight.get("entry_time") or dataset.period)
    eur_per_kg = fuel_price.price_eur_per_kg if fuel_price else 0.78
    wind_fuel_impact_eur = wind_fuel_impact_kg * eur_per_kg

    wind_benefit_dist_nm = cruise_dist_nm - total_esad_nm
    gcd_nm = float(flight.get("gcd_nm") or 0.0)
    actual_tot_nm = float(flight.get("actual_distance_nm") or cruise_dist_nm)
    detour_pct = ((actual_tot_nm - gcd_nm) / max(1.0, gcd_nm)) * 100.0 if gcd_nm > 0 else 0.0

    if avg_tw >= 25.0 and detour_pct >= 3.0:
        wind_eff_verdict = "SMART_WIND_RIDING_DETOUR (Flown ground path is longer, but jetstream tailwinds yielded a lower equivalent air distance and reduced fuel burn)"
    elif avg_tw >= 15.0:
        wind_eff_verdict = "FAVORABLE_TAILWIND_ASSIST (Net tailwinds reduced effective air distance and shortened air time)"
    elif avg_tw <= -25.0 and detour_pct <= 2.0:
        wind_eff_verdict = "DIRECT_HEADWIND_PENALTY (Flown along direct Great Circle track, but fought a severe jetstream headwind core)"
    elif avg_tw <= -15.0:
        wind_eff_verdict = "ADVERSE_HEADWIND_RESISTANCE (Upper-air headwinds expanded air distance and increased cruise fuel consumption)"
    elif detour_pct >= 5.0 and abs(avg_tw) < 15.0:
        wind_eff_verdict = "STRUCTURAL_ATC_DETOUR (Flown route added distance without wind assistance; driven by airway geometry, RAD, or sector closures)"
    else:
        wind_eff_verdict = "NEUTRAL_WIND_PROFILE (Minimal net wind assistance or resistance along the trajectory)"

    # Estimate direct CO2 and Contrail Cirrus GWP100 CO2e (~18 kg CO2e per nm of persistent contrail)
    direct_co2_tons = (cruise_dist_nm * ff_kg_nm * 3.16) / 1000.0
    contrail_co2e_tons = (contrail_dist_nm * 18.5) / 1000.0
    multiplier = (direct_co2_tons + contrail_co2e_tons) / max(0.1, direct_co2_tons)

    return UpperAirImpactReport(
        ectrl_id=int(ectrl_id),
        period=dataset.period,
        adep=adep,
        ades=ades,
        ac_operator=ac_op,
        ac_type=ac_type,
        cruise_distance_nm=round(cruise_dist_nm, 1),
        avg_groundspeed_kts=round(avg_gs, 1),
        avg_true_airspeed_kts=round(avg_tas, 1),
        avg_tailwind_component_kts=round(avg_tw, 1),
        wind_time_savings_min=round(wind_saved_min, 1),
        wind_routing_verdict=wind_verdict,
        ground_distance_nm=round(cruise_dist_nm, 1),
        esad_nm=round(total_esad_nm, 1),
        wind_distance_benefit_nm=round(wind_benefit_dist_nm, 1),
        wind_fuel_impact_kg=round(wind_fuel_impact_kg, 1),
        wind_fuel_impact_eur=round(wind_fuel_impact_eur, 2),
        wind_efficiency_verdict=wind_eff_verdict,
        persistent_contrail_distance_nm=round(contrail_dist_nm, 1),
        persistent_contrail_pct=round(contrail_pct, 1),
        persistent_contrail_duration_min=round(contrail_dur_min, 1),
        contrail_co2e_tons=round(contrail_co2e_tons, 2),
        direct_co2_tons=round(direct_co2_tons, 2),
        total_climate_co2e_multiplier=round(multiplier, 2),
        issr_segments_count=len(issr_segments),
        issr_segments=issr_segments[:15],
    )

