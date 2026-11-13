from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.eval.fuel_carbon import DEFAULT_FUEL_FLOW_KG_HR, CO2_FACTOR


@dataclass
class DescentInefficiencyEvent:
    event_type: str  # 'LEVEL_OFF' or 'ALTITUDE_REVERSAL_CLIMB'
    start_time: str
    end_time: str
    duration_min: float
    flight_level: float
    latitude: float
    longitude: float


@dataclass
class CDOReport:
    ectrl_id: int
    ac_type: str
    adep: str
    ades: str
    top_of_descent_time: str
    touchdown_time: str
    total_descent_duration_min: float
    cruise_flight_level: float
    inefficiency_events_count: int
    total_level_or_climb_min: float
    cdo_compliance_score: float  # 0 to 100%
    excess_fuel_burned_kg: float
    excess_co2_kg: float
    inefficiency_events: list[DescentInefficiencyEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ectrl_id": self.ectrl_id,
            "ac_type": self.ac_type,
            "route": f"{self.adep} ➔ {self.ades}",
            "top_of_descent_time": self.top_of_descent_time,
            "touchdown_time": self.touchdown_time,
            "total_descent_duration_min": round(self.total_descent_duration_min, 1),
            "cruise_flight_level": round(self.cruise_flight_level, 0),
            "inefficiency_events_count": self.inefficiency_events_count,
            "total_level_or_climb_min": round(self.total_level_or_climb_min, 1),
            "cdo_compliance_score_pct": round(self.cdo_compliance_score, 1),
            "excess_fuel_burned_kg": round(self.excess_fuel_burned_kg, 1),
            "excess_co2_kg": round(self.excess_co2_kg, 1),
            "events": [
                {
                    "type": e.event_type,
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                    "duration_min": round(e.duration_min, 1),
                    "flight_level": round(e.flight_level, 0),
                    "lat": round(e.latitude, 3),
                    "lon": round(e.longitude, 3),
                }
                for e in self.inefficiency_events
            ],
        }


def analyze_cdo_descent(dataset: FlightDataset, ectrl_id: int) -> CDOReport | None:
    """
    Analyze arrival descent profile to evaluate Continuous Descent Operations (CDO/CDA)
    compliance and detect fuel-wasting intermediate staircase level-offs and altitude reversals.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return None

    trajs = dataset.get_trajectory(ectrl_id)
    actual_df = trajs["actual"]
    if len(actual_df) < 5:
        return None

    actual_df = actual_df.sort_values("seq_num").reset_index(drop=True)
    max_fl = float(actual_df["flight_level"].max())
    ac_type = str(flight.get("ac_type") or "DEFAULT_JET").upper()
    fuel_flow = DEFAULT_FUEL_FLOW_KG_HR.get(ac_type, DEFAULT_FUEL_FLOW_KG_HR["DEFAULT_JET"])

    # 1. Identify Top of Descent (TOD)
    tod_idx = 0
    for i in range(len(actual_df) - 1, 0, -1):
        if actual_df.iloc[i]["flight_level"] >= max_fl - 15:
            tod_idx = i
            break

    descent_pts = actual_df.iloc[tod_idx:].copy().reset_index(drop=True)
    if len(descent_pts) < 3:
        return None

    tod_time = str(descent_pts.iloc[0]["time_over"])
    touchdown_time = str(descent_pts.iloc[-1]["time_over"])
    total_descent_min = max(
        1.0,
        (pd.to_datetime(touchdown_time) - pd.to_datetime(tod_time)).total_seconds() / 60.0,
    )

    events: list[DescentInefficiencyEvent] = []
    total_inefficient_min = 0.0

    # 2. Iterate through descent legs and identify level-offs and altitude climbs
    for i in range(len(descent_pts) - 1):
        fl1 = float(descent_pts.iloc[i]["flight_level"])
        fl2 = float(descent_pts.iloc[i + 1]["flight_level"])
        t1 = pd.to_datetime(descent_pts.iloc[i]["time_over"])
        t2 = pd.to_datetime(descent_pts.iloc[i + 1]["time_over"])
        duration = max(0.0, (t2 - t1).total_seconds() / 60.0)

        if fl1 <= 5 and fl2 <= 5:  # Runway ground roll
            continue

        # Case A: Level off (|delta FL| <= 4 for >= 1 min)
        if abs(fl2 - fl1) <= 4 and duration >= 1.0:
            events.append(
                DescentInefficiencyEvent(
                    event_type="LEVEL_OFF",
                    start_time=str(t1),
                    end_time=str(t2),
                    duration_min=duration,
                    flight_level=round((fl1 + fl2) / 2.0, 0),
                    latitude=float(descent_pts.iloc[i]["latitude"]),
                    longitude=float(descent_pts.iloc[i]["longitude"]),
                )
            )
            total_inefficient_min += duration

        # Case B: Altitude climb during descent (step-up or holding stack climb)
        elif fl2 > fl1 + 5:
            events.append(
                DescentInefficiencyEvent(
                    event_type="ALTITUDE_REVERSAL_CLIMB",
                    start_time=str(t1),
                    end_time=str(t2),
                    duration_min=duration,
                    flight_level=fl2,
                    latitude=float(descent_pts.iloc[i + 1]["latitude"]),
                    longitude=float(descent_pts.iloc[i + 1]["longitude"]),
                )
            )
            total_inefficient_min += duration

    # 3. CDO Compliance Score:
    cdo_score = max(0.0, 100.0 * (1.0 - (total_inefficient_min / total_descent_min)))

    # 4. Excess Fuel Burn:
    # Level flight & climbing consume nominal power (~85% flow) vs idle glide (~15% flow)
    excess_flow_rate = fuel_flow * 0.70  # kg / hr
    excess_fuel_kg = (total_inefficient_min / 60.0) * excess_flow_rate
    excess_co2_kg = excess_fuel_kg * CO2_FACTOR

    return CDOReport(
        ectrl_id=ectrl_id,
        ac_type=ac_type,
        adep=str(flight.get("adep")),
        ades=str(flight.get("ades")),
        top_of_descent_time=tod_time,
        touchdown_time=touchdown_time,
        total_descent_duration_min=total_descent_min,
        cruise_flight_level=max_fl,
        inefficiency_events_count=len(events),
        total_level_or_climb_min=total_inefficient_min,
        cdo_compliance_score=cdo_score,
        excess_fuel_burned_kg=excess_fuel_kg,
        excess_co2_kg=excess_co2_kg,
        inefficiency_events=events,
    )


@dataclass
class ClimbInefficiencyEvent:
    event_type: str  # 'CLIMB_LEVEL_OFF' or 'ALTITUDE_REVERSAL_DESCENT'
    start_time: str
    end_time: str
    duration_min: float
    flight_level: float
    latitude: float
    longitude: float


@dataclass
class CCOReport:
    ectrl_id: int
    ac_type: str
    adep: str
    ades: str
    takeoff_time: str
    top_of_climb_time: str
    total_climb_duration_min: float
    cruise_flight_level: float
    inefficiency_events_count: int
    total_level_off_min: float
    cco_compliance_score: float
    excess_fuel_burned_kg: float
    excess_co2_kg: float
    inefficiency_events: list[ClimbInefficiencyEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ectrl_id": self.ectrl_id,
            "ac_type": self.ac_type,
            "route": f"{self.adep} ➔ {self.ades}",
            "takeoff_time": self.takeoff_time,
            "top_of_climb_time": self.top_of_climb_time,
            "total_climb_duration_min": round(self.total_climb_duration_min, 1),
            "cruise_flight_level": round(self.cruise_flight_level, 0),
            "inefficiency_events_count": self.inefficiency_events_count,
            "total_level_off_min": round(self.total_level_off_min, 1),
            "cco_compliance_score_pct": round(self.cco_compliance_score, 1),
            "excess_fuel_burned_kg": round(self.excess_fuel_burned_kg, 1),
            "excess_co2_kg": round(self.excess_co2_kg, 1),
            "events": [
                {
                    "type": e.event_type,
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                    "duration_min": round(e.duration_min, 1),
                    "flight_level": round(e.flight_level, 0),
                    "lat": round(e.latitude, 3),
                    "lon": round(e.longitude, 3),
                }
                for e in self.inefficiency_events
            ],
        }


def analyze_cco_climb(dataset: FlightDataset, ectrl_id: int) -> CCOReport | None:
    """
    Analyze departure climb profile from takeoff to Top of Climb (TOC) to evaluate
    ICAO GANP KPI17 Continuous Climb Operations (CCO) compliance and detect
    intermediate SID altitude restrictions / level-offs below cruise altitude.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return None

    trajs = dataset.get_trajectory(ectrl_id)
    actual_df = trajs["actual"]
    if len(actual_df) < 5:
        return None

    actual_df = actual_df.sort_values("seq_num").reset_index(drop=True)
    max_fl = float(actual_df["flight_level"].max())
    ac_type = str(flight.get("ac_type") or "DEFAULT_JET").upper()
    fuel_flow = DEFAULT_FUEL_FLOW_KG_HR.get(ac_type, DEFAULT_FUEL_FLOW_KG_HR["DEFAULT_JET"])

    # 1. Identify Top of Climb (TOC): first point reaching within 15 FL of initial cruise ceiling
    toc_idx = len(actual_df) - 1
    for i in range(len(actual_df)):
        if float(actual_df.iloc[i]["flight_level"]) >= max_fl - 15:
            toc_idx = max(2, i)
            break

    climb_pts = actual_df.iloc[: toc_idx + 1].copy().reset_index(drop=True)
    if len(climb_pts) < 3:
        return None

    takeoff_time = str(climb_pts.iloc[0]["time_over"])
    toc_time = str(climb_pts.iloc[-1]["time_over"])
    total_climb_min = max(
        1.0,
        (pd.to_datetime(toc_time) - pd.to_datetime(takeoff_time)).total_seconds() / 60.0,
    )

    events: list[ClimbInefficiencyEvent] = []
    total_level_min = 0.0

    for i in range(len(climb_pts) - 1):
        fl1 = float(climb_pts.iloc[i]["flight_level"])
        fl2 = float(climb_pts.iloc[i + 1]["flight_level"])
        t1 = pd.to_datetime(climb_pts.iloc[i]["time_over"])
        t2 = pd.to_datetime(climb_pts.iloc[i + 1]["time_over"])
        duration = max(0.0, (t2 - t1).total_seconds() / 60.0)

        if fl1 <= 5 and fl2 <= 5:
            continue

        # Level-off below TOC (|delta FL| <= 4 for >= 1.0 min below max_fl - 15)
        if abs(fl2 - fl1) <= 4 and duration >= 1.0 and max(fl1, fl2) < max_fl - 15:
            events.append(
                ClimbInefficiencyEvent(
                    event_type="CLIMB_LEVEL_OFF",
                    start_time=str(t1),
                    end_time=str(t2),
                    duration_min=duration,
                    flight_level=round((fl1 + fl2) / 2.0, 0),
                    latitude=float(climb_pts.iloc[i]["latitude"]),
                    longitude=float(climb_pts.iloc[i]["longitude"]),
                )
            )
            total_level_min += duration
        elif fl2 < fl1 - 5 and fl1 < max_fl - 15:
            events.append(
                ClimbInefficiencyEvent(
                    event_type="ALTITUDE_REVERSAL_DESCENT",
                    start_time=str(t1),
                    end_time=str(t2),
                    duration_min=duration,
                    flight_level=fl2,
                    latitude=float(climb_pts.iloc[i + 1]["latitude"]),
                    longitude=float(climb_pts.iloc[i + 1]["longitude"]),
                )
            )
            total_level_min += duration

    cco_score = max(0.0, 100.0 * (1.0 - (total_level_min / total_climb_min)))
    # Leveling off at low/medium altitude burns ~22% more fuel per unit time than cruising at optimal FL
    excess_fuel_kg = (total_level_min / 60.0) * (fuel_flow * 0.25)
    excess_co2_kg = excess_fuel_kg * CO2_FACTOR

    return CCOReport(
        ectrl_id=ectrl_id,
        ac_type=ac_type,
        adep=str(flight.get("adep")),
        ades=str(flight.get("ades")),
        takeoff_time=takeoff_time,
        top_of_climb_time=toc_time,
        total_climb_duration_min=total_climb_min,
        cruise_flight_level=max_fl,
        inefficiency_events_count=len(events),
        total_level_off_min=total_level_min,
        cco_compliance_score=cco_score,
        excess_fuel_burned_kg=excess_fuel_kg,
        excess_co2_kg=excess_co2_kg,
        inefficiency_events=events,
    )


def analyze_vertical_profile_efficiency(dataset: FlightDataset, ectrl_id: int) -> dict[str, Any] | None:
    """
    Unified 3-Phase Vertical Flight Efficiency Scorecard combining:
    - Phase 1: Departure Climb (KPI17 CCO)
    - Phase 2: En-Route Cruise Level-Capping (KPI19)
    - Phase 3: Arrival Descent (KPI18 CDO)
    """
    from flightml.eval.pru_kpis import compute_flight_pru_kpis

    cco = analyze_cco_climb(dataset, ectrl_id)
    cdo = analyze_cdo_descent(dataset, ectrl_id)
    pru = compute_flight_pru_kpis(dataset, ectrl_id)

    if not cco and not cdo and not pru:
        return None

    total_vert_fuel_kg = (
        (cco.excess_fuel_burned_kg if cco else 0.0)
        + (pru.level_cap_excess_fuel_kg if pru else 0.0)
        + (cdo.excess_fuel_burned_kg if cdo else 0.0)
    )

    return {
        "ectrl_id": ectrl_id,
        "period": dataset.period,
        "cco_climb": cco.to_dict() if cco else None,
        "cruise_level_capping": {
            "requested_fl": pru.requested_fl if pru else None,
            "max_actual_fl": pru.max_actual_fl if pru else None,
            "level_cap_delta_fl": pru.level_cap_delta_fl if pru else 0.0,
            "is_level_capped": pru.is_level_capped if pru else False,
            "excess_fuel_kg": pru.level_cap_excess_fuel_kg if pru else 0.0,
        },
        "cdo_descent": cdo.to_dict() if cdo else None,
        "total_vertical_excess_fuel_kg": round(total_vert_fuel_kg, 1),
        "total_vertical_excess_co2_kg": round(total_vert_fuel_kg * CO2_FACTOR, 1),
    }
