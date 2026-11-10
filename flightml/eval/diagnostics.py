from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.geo.trajectory import analyze_trajectory
from flightml.weather.metar_client import MetarClient
from flightml.weather.delay_weather import analyze_delayed_flight_weather


@dataclass
class DelayAttribution:
    ectrl_id: int
    adep: str
    ades: str
    ac_operator: str
    ac_reg: str
    total_arr_delay_min: float
    total_dep_delay_min: float
    reactionary_delay_min: float
    holding_delay_min: float
    weather_attribution: str
    primary_bottleneck_factor: str
    mitigation_recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ectrl_id": self.ectrl_id,
            "adep": self.adep,
            "ades": self.ades,
            "ac_operator": self.ac_operator,
            "ac_reg": self.ac_reg,
            "total_arr_delay_min": round(self.total_arr_delay_min, 1),
            "total_dep_delay_min": round(self.total_dep_delay_min, 1),
            "reactionary_delay_min": round(self.reactionary_delay_min, 1),
            "holding_delay_min": round(self.holding_delay_min, 1),
            "weather_attribution": self.weather_attribution,
            "primary_bottleneck_factor": self.primary_bottleneck_factor,
            "mitigation_recommendation": self.mitigation_recommendation,
        }


def diagnose_flight_bottlenecks(
    dataset: FlightDataset,
    metar_client: MetarClient,
    ectrl_id: int,
) -> DelayAttribution:
    """
    Synthesize 4D trajectory holding loops, aircraft tail rotation history,
    and aerodrome weather observations to diagnose the root cause of delays.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        raise ValueError(f"Flight {ectrl_id} not found")

    dep_delay = float(flight.get("dep_delay_min") or 0.0)
    arr_delay = float(flight.get("arr_delay_min") or 0.0)
    ac_reg = str(flight.get("ac_reg") or "")
    adep = str(flight.get("adep"))
    ades = str(flight.get("ades"))
    op = str(flight.get("ac_operator"))

    # 1. Trajectory & Holding detection
    trajs = dataset.get_trajectory(ectrl_id)
    metrics = analyze_trajectory(trajs["actual"], trajs["planned"])
    holding_min = metrics.total_holding_duration_min

    # 2. Reactionary delay from tail rotation
    reactionary_min = 0.0
    if ac_reg:
        off_block = flight.get("actual_off_block_time")
        if off_block:
            prev_leg_df = dataset.query(
                """
                SELECT arr_delay_min, ades, actual_arrival_time
                FROM flights
                WHERE ac_reg = ?
                  AND actual_arrival_time < ?
                ORDER BY actual_arrival_time DESC
                LIMIT 1;
                """,
                [ac_reg, off_block],
            )
            if not prev_leg_df.empty:
                prev_arr_delay = float(prev_leg_df.iloc[0]["arr_delay_min"] or 0.0)
                if prev_arr_delay > 0 and dep_delay > 0:
                    reactionary_min = min(prev_arr_delay, dep_delay)

    # 3. Weather check at origin and destination
    weather_res = analyze_delayed_flight_weather(dataset, metar_client, ectrl_id)
    weather_desc = "Normal VMC"
    if weather_res.get("adverse_weather_detected"):
        reasons = weather_res.get("weather_attribution_reasons", [])
        weather_desc = "; ".join(reasons) if reasons else "Adverse weather reported"

    # 4. Determine Primary Bottleneck Factor & Actionable Mitigation
    factors = []
    if reactionary_min >= 15:
        factors.append(("Reactionary Delay (Inbound Aircraft)", reactionary_min))
    if holding_min >= 10:
        factors.append(("Airspace Congestion / Terminal Holding", holding_min))
    if weather_res.get("adverse_weather_detected"):
        factors.append(("Aerodrome Adverse Weather (Crosswind/Ceiling)", 20.0))

    if not factors:
        if dep_delay > 15:
            primary_factor = "Ground Handling / Airport Departure Congestion"
            mitigation = f"Audit ground turnaround operations and gate pushback slot at {adep}."
        elif arr_delay > 15:
            primary_factor = "En-Route Speed Reductions / Airway Flow Restrictions"
            mitigation = f"Optimize flight plan cruise profile and coordinate slot compliance with en-route ATC."
        else:
            primary_factor = "Normal Operating Tolerances (Minor Delay)"
            mitigation = "No intervention necessary; operation within schedule buffers."
    else:
        # Sort factors by magnitude
        factors.sort(key=lambda x: -x[1])
        top_name, top_mag = factors[0]
        primary_factor = top_name

        if "Reactionary" in top_name:
            mitigation = f"Increase scheduled turnaround buffer for tail {ac_reg} at {adep} by at least {int(reactionary_min)} mins to prevent schedule cascading."
        elif "Holding" in top_name:
            mitigation = f"Evaluate tactical pre-departure ground delay program (GDP) to absorb {int(holding_min)} mins on ground rather than burning fuel in holding stacks near {ades}."
        elif "Weather" in top_name:
            mitigation = f"Coordinate secondary runway configuration with {ades} ATC or plan fuel contingency for expected crosswinds/instrument approaches."
        else:
            mitigation = f"Review operational schedule margin for route {adep}->{ades}."

    return DelayAttribution(
        ectrl_id=ectrl_id,
        adep=adep,
        ades=ades,
        ac_operator=op,
        ac_reg=ac_reg,
        total_arr_delay_min=arr_delay,
        total_dep_delay_min=dep_delay,
        reactionary_delay_min=reactionary_min,
        holding_delay_min=holding_min,
        weather_attribution=weather_desc,
        primary_bottleneck_factor=primary_factor,
        mitigation_recommendation=mitigation,
    )

