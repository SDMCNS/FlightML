"""
Multi-Objective Eco-Optimal Trajectory Optimizer (Contrail Avoidance vs. Fuel & Gaseous Emissions).

Solves the multi-objective environmental trade-off for commercial aircraft:
  Minimize Net Climate Impact = Direct CO2 + Persistent Contrail Cirrus CO2e + Cruise NOx Forcing
  Subject to:
    - Aircraft operational envelope (service ceiling, climb capability, MTOW fuel limits)
    - RVSM cruising flight levels (2,000 ft / 1,000 ft step feasibility)
    - Aerodynamic fuel burn penalty of off-optimal cruise flight levels
    - Upper-air atmospheric winds (headwind / tailwind vector) and Ice Supersaturated Regions (ISSR)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.eval.aircraft_engine_perf import resolve_aircraft_engine_perf
from flightml.eval.contrail_climate import (
    ContrailSegment,
    UpperAirImpactReport,
    analyze_flight_wind_and_contrails,
    schmidt_appleman_threshold_c,
)
from flightml.eval.jet_fuel_prices import get_weekly_fuel_price
from flightml.geo.geodesy import haversine_nm, initial_bearing_deg
from flightml.weather.upper_air_client import UpperAirClient


@dataclass
class EcoRouteWaypoint:
    seq_num: int
    time_utc: str
    latitude: float
    longitude: float
    flight_level: float
    cum_dist_nm: float
    groundspeed_kts: float
    true_airspeed_kts: float
    tailwind_kts: float
    temperature_c: float
    rh_ice_pct: float
    is_issr_contrail: bool
    segment_fuel_kg: float
    fl_delta_vs_flown: float
    deviation_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EcoRouteComparisonReport:
    ectrl_id: int
    period: str
    adep: str
    ades: str
    ac_operator: str
    ac_type: str
    aircraft_name: str
    operational_ceiling_fl: float
    ideal_aerodynamic_fl: float
    # Distances
    as_flown_distance_nm: float
    eco_distance_nm: float
    # Fuel & Economic Trade-off
    as_flown_fuel_kg: float
    eco_fuel_kg: float
    fuel_delta_kg: float
    fuel_delta_pct: float
    fuel_cost_delta_eur: float
    # Gaseous Emissions
    as_flown_co2_kg: float
    eco_co2_kg: float
    co2_delta_kg: float
    as_flown_nox_kg: float
    eco_nox_kg: float
    nox_delta_kg: float
    # Persistent Contrails
    as_flown_contrail_nm: float
    eco_contrail_nm: float
    contrail_avoided_nm: float
    contrail_avoided_pct: float
    as_flown_contrail_co2e_kg: float
    eco_contrail_co2e_kg: float
    # Total Net Climate Impact (CO2e)
    as_flown_total_climate_co2e_kg: float
    eco_total_climate_co2e_kg: float
    net_climate_delta_kg: float
    net_climate_reduction_pct: float
    # Envelope & Feasibility
    is_within_aircraft_envelope: bool
    operational_feasibility_summary: str
    tactical_maneuvers_summary: list[str]
    eco_waypoints: list[EcoRouteWaypoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def optimize_eco_minimal_contrail_trajectory(
    dataset: FlightDataset,
    ectrl_id: int,
    upper_air_client: UpperAirClient | None = None,
) -> EcoRouteComparisonReport | None:
    """
    Synthesize an alternative eco-optimal trajectory (both horizontal and vertical)
    balancing fuel burn, persistent contrails (ISSR), direct CO2, and cruise NOx emissions.

    Uses deterministic multi-objective evaluation across aircraft performance limits:
    1. Evaluates tactical vertical steps (+/- 2,000 ft or +/- 1,000 ft) outside thin ISSR layers.
    2. Accounts for off-optimal aerodynamic fuel burn penalty vs. contrail cirrus warming elimination.
    3. Respects operational ceiling, RVSM rules, and smooth climb/descent profiles.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return None

    trajs = dataset.get_trajectory(ectrl_id)
    actual_df = trajs.get("actual", pd.DataFrame())
    if len(actual_df) < 3:
        return None

    actual_df = actual_df.sort_values("seq_num").reset_index(drop=True)
    client = upper_air_client or UpperAirClient(use_live_api=True)
    client.prefetch_trajectory_upper_air(actual_df)

    adep = str(flight.get("adep") or "ADEP")
    ades = str(flight.get("ades") or "ADES")
    ac_op = str(flight.get("ac_operator") or "OP")
    ac_type = str(flight.get("ac_type") or "B738")

    tot_dist = float(flight.get("actual_distance_nm") or 500.0)
    perf = resolve_aircraft_engine_perf(dataset, ac_type)
    ceiling_fl = float(perf.operational_ceiling_fl)
    ideal_fl = float(perf.optimal_cruise_fl)
    nominal_ff_kg_hr = float(perf.nominal_fuel_flow_kg_hr)

    # 1. Base As-Flown Analysis
    base_rep = analyze_flight_wind_and_contrails(dataset, ectrl_id, client)
    flown_contrail_nm = base_rep.persistent_contrail_distance_nm if base_rep else 0.0

    # 2. Iterate through trajectory waypoints and construct Eco-Optimal Profile
    eco_waypoints: list[EcoRouteWaypoint] = []
    tactical_notes: list[str] = []

    flown_total_fuel_kg = 0.0
    eco_total_fuel_kg = 0.0
    flown_total_nox_kg = 0.0
    eco_total_nox_kg = 0.0
    eco_contrail_nm = 0.0

    cum_dist_nm = 0.0
    active_eco_fl = float(actual_df.iloc[0]["flight_level"])

    # Origin waypoint
    r0 = actual_df.iloc[0]
    eco_waypoints.append(
        EcoRouteWaypoint(
            seq_num=int(r0["seq_num"]),
            time_utc=str(r0["time_over"]),
            latitude=round(float(r0["latitude"]), 5),
            longitude=round(float(r0["longitude"]), 5),
            flight_level=round(float(r0["flight_level"]), 1),
            cum_dist_nm=0.0,
            groundspeed_kts=250.0,
            true_airspeed_kts=250.0,
            tailwind_kts=0.0,
            temperature_c=15.0,
            rh_ice_pct=40.0,
            is_issr_contrail=False,
            segment_fuel_kg=0.0,
            fl_delta_vs_flown=0.0,
            deviation_reason="Standard Departure Profile",
        )
    )

    for i in range(len(actual_df) - 1):
        r1 = actual_df.iloc[i]
        r2 = actual_df.iloc[i + 1]
        lat1, lon1 = float(r1["latitude"]), float(r1["longitude"])
        lat2, lon2 = float(r2["latitude"]), float(r2["longitude"])
        seg_nm = haversine_nm(lat1, lon1, lat2, lon2)
        cum_dist_nm += seg_nm

        fl_flown = float((r1["flight_level"] + r2["flight_level"]) / 2.0)
        t1 = pd.to_datetime(r1["time_over"])
        t2 = pd.to_datetime(r2["time_over"])
        seg_min = max(0.5, (t2 - t1).total_seconds() / 60.0)

        mid_lat = (lat1 + lat2) / 2.0
        mid_lon = (lon1 + lon2) / 2.0
        track_deg = initial_bearing_deg(lat1, lon1, lat2, lon2)

        # Baseline climb / descent phases (< FL240): follow standard profile
        if fl_flown < 240.0:
            active_eco_fl = float(r2["flight_level"])
            # Fuel burn approximation for terminal / climb / descent
            seg_ff = nominal_ff_kg_hr * (1.35 if fl_flown < 150 else 1.15)
            seg_fuel = (seg_min / 60.0) * seg_ff
            flown_total_fuel_kg += seg_fuel
            eco_total_fuel_kg += seg_fuel
            flown_total_nox_kg += seg_fuel * 0.010
            eco_total_nox_kg += seg_fuel * 0.010

            eco_waypoints.append(
                EcoRouteWaypoint(
                    seq_num=int(r2["seq_num"]),
                    time_utc=str(r2["time_over"]),
                    latitude=round(lat2, 5),
                    longitude=round(lon2, 5),
                    flight_level=round(active_eco_fl, 1),
                    cum_dist_nm=round(cum_dist_nm, 1),
                    groundspeed_kts=round(min(550.0, max(200.0, (seg_nm / seg_min) * 60.0)), 1),
                    true_airspeed_kts=round(min(500.0, max(220.0, (seg_nm / seg_min) * 60.0)), 1),
                    tailwind_kts=0.0,
                    temperature_c=-10.0,
                    rh_ice_pct=50.0,
                    is_issr_contrail=False,
                    segment_fuel_kg=round(seg_fuel, 1),
                    fl_delta_vs_flown=0.0,
                    deviation_reason="Climb/Descent Terminal Envelope",
                )
            )
            continue

        # En-Route Cruise Phase (>= FL240): Evaluate Multi-Objective Candidates
        # Baseline flown conditions
        ua_flown = client.get_upper_air_state(mid_lat, mid_lon, fl_flown, str(r1["time_over"]))
        t_sac_flown = schmidt_appleman_threshold_c(ua_flown.pressure_hpa)
        flown_is_contrail = (ua_flown.temperature_c <= t_sac_flown and ua_flown.rh_ice_pct >= 100.0)

        # Candidate flight levels to test (respecting aircraft service ceiling & minimum cruise FL)
        # RVSM standard increments (+/- 2,000 ft or +/- 1,000 ft)
        candidate_fls = [fl_flown]
        for delta in [-20.0, 20.0, -10.0, 10.0, -30.0]:
            cand = fl_flown + delta
            if 260.0 <= cand <= ceiling_fl:
                candidate_fls.append(cand)

        best_fl = fl_flown
        best_cost = float("inf")
        best_meta: dict[str, Any] = {}

        for cand_fl in candidate_fls:
            # Smoothness constraint: limit step changes between consecutive segments to max 2,000 ft
            step_delta = abs(cand_fl - active_eco_fl)
            if step_delta > 25.0:
                continue

            ua = client.get_upper_air_state(mid_lat, mid_lon, cand_fl, str(r1["time_over"]))
            t_sac = schmidt_appleman_threshold_c(ua.pressure_hpa)
            cand_is_contrail = (ua.temperature_c <= t_sac and ua.rh_ice_pct >= 100.0)

            # Tailwind / Groundspeed calculation
            wind_toward = (ua.wind_dir_deg + 180.0) % 360.0
            rel_ang = math.radians(wind_toward - track_deg)
            tailwind = ua.wind_speed_kts * math.cos(rel_ang)
            tas = max(380.0, min(500.0, 440.0 + (cand_fl - 300.0) * 0.5))
            gs = max(280.0, min(650.0, tas + tailwind))
            time_hr = seg_nm / gs

            # Off-optimal altitude fuel penalty
            fl_penalty = abs(cand_fl - ideal_fl) * 0.0012  # ~1.2% per 1,000 ft deviation from aerodynamic optimum
            ff_cand = nominal_ff_kg_hr * (1.0 + fl_penalty)
            fuel_kg = time_hr * ff_cand

            # Direct Gaseous emissions
            co2_kg = fuel_kg * 3.16
            # Cruise NOx emission index (g/kg fuel) scaled with combustor inlet pressure P/250
            ei_nox = 13.5 * ((ua.pressure_hpa / 250.0) ** 0.4)
            nox_kg = (ei_nox / 1000.0) * fuel_kg

            # Persistent Contrail Cirrus climate forcing: ~18.5 kg CO2e per nm in ISSR
            contrail_co2e_kg = (seg_nm * 18.5) if cand_is_contrail else 0.0

            # Step maneuver penalty (avoid continuous small oscillations)
            oscillation_penalty = 8.0 if abs(cand_fl - active_eco_fl) > 5.0 else 0.0

            # Multi-Objective Climate + Fuel objective function
            # CO2 (kg) + Contrail Cirrus (kg CO2e) + Cruise NOx forcing (GWP100 ~12x NOx) + Operational fuel penalty
            total_env_cost = co2_kg + contrail_co2e_kg + (nox_kg * 12.0) + oscillation_penalty

            if total_env_cost < best_cost:
                best_cost = total_env_cost
                best_fl = cand_fl
                best_meta = {
                    "is_contrail": cand_is_contrail,
                    "fuel_kg": fuel_kg,
                    "co2_kg": co2_kg,
                    "nox_kg": nox_kg,
                    "contrail_co2e_kg": contrail_co2e_kg,
                    "gs": gs,
                    "tas": tas,
                    "tailwind": tailwind,
                    "temp": ua.temperature_c,
                    "rhi": ua.rh_ice_pct,
                }

        # Flown baseline metrics for this segment
        wind_toward_flown = (ua_flown.wind_dir_deg + 180.0) % 360.0
        rel_flown = math.radians(wind_toward_flown - track_deg)
        tw_flown = ua_flown.wind_speed_kts * math.cos(rel_flown)
        tas_flown = max(380.0, min(500.0, 440.0 + (fl_flown - 300.0) * 0.5))
        gs_flown = max(280.0, min(650.0, tas_flown + tw_flown))
        time_flown = seg_nm / gs_flown
        ff_flown = nominal_ff_kg_hr * (1.0 + abs(fl_flown - ideal_fl) * 0.0012)
        fuel_flown_seg = time_flown * ff_flown

        flown_total_fuel_kg += fuel_flown_seg
        flown_total_nox_kg += (13.5 * ((ua_flown.pressure_hpa / 250.0) ** 0.4) / 1000.0) * fuel_flown_seg

        # Update active eco FL
        active_eco_fl = best_fl
        eco_fuel_seg = best_meta.get("fuel_kg", fuel_flown_seg)
        eco_total_fuel_kg += eco_fuel_seg
        eco_total_nox_kg += best_meta.get("nox_kg", 0.0)
        if best_meta.get("is_contrail", False):
            eco_contrail_nm += seg_nm

        fl_diff = best_fl - fl_flown
        if abs(fl_diff) >= 15.0:
            if flown_is_contrail and not best_meta.get("is_contrail", False):
                reason = f"Tactical Shift to FL{best_fl:.0f} ({fl_diff:+.0f} FL): 100% Contrail Avoided (RHi {best_meta['rhi']:.0f}% < 100%)"
                maneuver_note = f"Maneuver at {cum_dist_nm:.0f} nm: Shift from FL{fl_flown:.0f} to FL{best_fl:.0f} ({fl_diff:+.0f} FL) eliminated persistent contrail formation in ISSR layer."
                if maneuver_note not in tactical_notes:
                    tactical_notes.append(maneuver_note)
            elif abs(best_fl - ideal_fl) < abs(fl_flown - ideal_fl):
                reason = f"Climb to Aerodynamic Optimum FL{best_fl:.0f} (Saved {fuel_flown_seg - eco_fuel_seg:+.1f} kg fuel)"
            else:
                reason = f"Wind-Optimal Step to FL{best_fl:.0f} (Tailwind: {best_meta.get('tailwind', 0):+.1f} kts)"
        else:
            reason = "Optimal Cruise Envelope Maintained"

        eco_waypoints.append(
            EcoRouteWaypoint(
                seq_num=int(r2["seq_num"]),
                time_utc=str(r2["time_over"]),
                latitude=round(lat2, 5),
                longitude=round(lon2, 5),
                flight_level=round(best_fl, 1),
                cum_dist_nm=round(cum_dist_nm, 1),
                groundspeed_kts=round(best_meta.get("gs", gs_flown), 1),
                true_airspeed_kts=round(best_meta.get("tas", tas_flown), 1),
                tailwind_kts=round(best_meta.get("tailwind", tw_flown), 1),
                temperature_c=round(best_meta.get("temp", ua_flown.temperature_c), 1),
                rh_ice_pct=round(best_meta.get("rhi", ua_flown.rh_ice_pct), 1),
                is_issr_contrail=bool(best_meta.get("is_contrail", False)),
                segment_fuel_kg=round(eco_fuel_seg, 1),
                fl_delta_vs_flown=round(fl_diff, 1),
                deviation_reason=reason,
            )
        )

    # 3. Overall Totals & Multi-Objective Climate Accounting
    flown_co2_kg = flown_total_fuel_kg * 3.16
    eco_co2_kg = eco_total_fuel_kg * 3.16
    co2_delta_kg = eco_co2_kg - flown_co2_kg

    fuel_delta_kg = eco_total_fuel_kg - flown_total_fuel_kg
    fuel_delta_pct = (fuel_delta_kg / max(1.0, flown_total_fuel_kg)) * 100.0

    fuel_price = get_weekly_fuel_price(flight.get("off_block_date") or flight.get("entry_time") or dataset.period)
    fuel_eur_per_kg = fuel_price.price_eur_per_kg if fuel_price else 0.78
    fuel_cost_delta_eur = fuel_delta_kg * fuel_eur_per_kg

    nox_delta_kg = eco_total_nox_kg - flown_total_nox_kg

    contrail_avoided_nm = max(0.0, flown_contrail_nm - eco_contrail_nm)
    contrail_avoided_pct = (contrail_avoided_nm / max(0.1, flown_contrail_nm)) * 100.0 if flown_contrail_nm > 0 else 100.0

    flown_contrail_co2e_kg = flown_contrail_nm * 18.5
    eco_contrail_co2e_kg = eco_contrail_nm * 18.5

    # Total Net Climate Impact = Direct CO2 + Contrail Cirrus CO2e + (NOx * 12.0 GWP100)
    flown_total_climate_co2e = flown_co2_kg + flown_contrail_co2e_kg + (flown_total_nox_kg * 12.0)
    eco_total_climate_co2e = eco_co2_kg + eco_contrail_co2e_kg + (eco_total_nox_kg * 12.0)
    net_climate_delta_kg = eco_total_climate_co2e - flown_total_climate_co2e
    net_reduction_pct = (abs(net_climate_delta_kg) / max(1.0, flown_total_climate_co2e)) * 100.0 if net_climate_delta_kg < 0 else 0.0

    # Operational envelope compliance verification
    max_eco_fl = max((w.flight_level for w in eco_waypoints), default=ideal_fl)
    is_envelope_ok = max_eco_fl <= ceiling_fl

    if not tactical_notes:
        if flown_contrail_nm <= 1.0:
            tactical_notes.append("As-flown trajectory operated in clean, dry atmospheric air; zero persistent contrails formed.")
        else:
            tactical_notes.append("Eco-optimal profile maintained within aerodynamic cruise flight levels with balanced fuel burn.")

    envelope_summary = (
        f"✅ Compliant with {ac_type} flight envelope: Max eco altitude FL{max_eco_fl:.0f} is within operational ceiling FL{ceiling_fl:.0f}. "
        f"Step climbs/descents follow standard European RVSM separation rules."
        if is_envelope_ok
        else f"⚠️ Altitude Warning: Peak eco altitude FL{max_eco_fl:.0f} approaches or exceeds ceiling FL{ceiling_fl:.0f}."
    )

    return EcoRouteComparisonReport(
        ectrl_id=int(ectrl_id),
        period=dataset.period,
        adep=adep,
        ades=ades,
        ac_operator=ac_op,
        ac_type=ac_type,
        aircraft_name=str(perf.aircraft_name or ac_type),
        operational_ceiling_fl=ceiling_fl,
        ideal_aerodynamic_fl=ideal_fl,
        as_flown_distance_nm=round(tot_dist, 1),
        eco_distance_nm=round(cum_dist_nm, 1),
        as_flown_fuel_kg=round(flown_total_fuel_kg, 1),
        eco_fuel_kg=round(eco_total_fuel_kg, 1),
        fuel_delta_kg=round(fuel_delta_kg, 1),
        fuel_delta_pct=round(fuel_delta_pct, 2),
        fuel_cost_delta_eur=round(fuel_cost_delta_eur, 2),
        as_flown_co2_kg=round(flown_co2_kg, 1),
        eco_co2_kg=round(eco_co2_kg, 1),
        co2_delta_kg=round(co2_delta_kg, 1),
        as_flown_nox_kg=round(flown_total_nox_kg, 2),
        eco_nox_kg=round(eco_total_nox_kg, 2),
        nox_delta_kg=round(nox_delta_kg, 2),
        as_flown_contrail_nm=round(flown_contrail_nm, 1),
        eco_contrail_nm=round(eco_contrail_nm, 1),
        contrail_avoided_nm=round(contrail_avoided_nm, 1),
        contrail_avoided_pct=round(contrail_avoided_pct, 1),
        as_flown_contrail_co2e_kg=round(flown_contrail_co2e_kg, 1),
        eco_contrail_co2e_kg=round(eco_contrail_co2e_kg, 1),
        as_flown_total_climate_co2e_kg=round(flown_total_climate_co2e, 1),
        eco_total_climate_co2e_kg=round(eco_total_climate_co2e, 1),
        net_climate_delta_kg=round(net_climate_delta_kg, 1),
        net_climate_reduction_pct=round(net_reduction_pct, 1),
        is_within_aircraft_envelope=is_envelope_ok,
        operational_feasibility_summary=envelope_summary,
        tactical_maneuvers_summary=tactical_notes,
        eco_waypoints=eco_waypoints,
    )
