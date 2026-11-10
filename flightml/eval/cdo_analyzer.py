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
