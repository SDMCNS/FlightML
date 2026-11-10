from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset


@dataclass
class RotationLegPair:
    ac_reg: str
    ac_operator: str
    leg1_ectrl_id: int
    leg1_adep: str
    leg1_ades: str
    leg1_arr_delay_min: float
    leg2_ectrl_id: int
    leg2_adep: str
    leg2_ades: str
    leg2_dep_delay_min: float
    scheduled_turnaround_min: float
    actual_turnaround_min: float
    delay_propagated_min: float
    delay_absorbed_min: float
    is_reactionary: bool


class ReactionaryDelayAnalyzer:
    """
    Analyzes delay cascades and reactionary delay propagation along aircraft rotations (tail turns).
    """

    def __init__(self, dataset: FlightDataset) -> None:
        self.dataset = dataset

    def extract_tail_rotations(
        self,
        operator: str | None = None,
        date_start: str = "2016-11-01",
        date_end: str = "2016-11-05",
        min_legs: int = 5,
    ) -> list[RotationLegPair]:
        """
        Extract consecutive flight leg pairs for aircraft tails to measure delay propagation.
        """
        op_filter = f"AND ac_operator = '{operator.upper()}'" if operator else ""

        query = f"""
            SELECT
                ac_reg,
                ac_operator,
                ectrl_id,
                adep,
                ades,
                filed_off_block_time,
                filed_arrival_time,
                actual_off_block_time,
                actual_arrival_time,
                dep_delay_min,
                arr_delay_min
            FROM flights
            WHERE ac_reg IS NOT NULL AND ac_reg != ''
              AND actual_off_block_time >= '{date_start}'
              AND actual_off_block_time < '{date_end}'
              {op_filter}
            ORDER BY ac_reg, actual_off_block_time ASC;
        """
        df = self.dataset.query(query)
        if df.empty:
            return []

        leg_pairs: list[RotationLegPair] = []
        grouped = df.groupby("ac_reg")

        for ac_reg, group in grouped:
            if len(group) < 2:
                continue

            records = group.to_dict(orient="records")
            for i in range(len(records) - 1):
                leg1 = records[i]
                leg2 = records[i + 1]

                # Ensure consecutive connection (leg 1 destination matches leg 2 departure)
                if leg1["ades"] != leg2["adep"]:
                    continue

                # Compute turnaround times
                t1_arr = pd.to_datetime(leg1["actual_arrival_time"])
                t2_dep = pd.to_datetime(leg2["actual_off_block_time"])
                turnaround_act = (t2_dep - t1_arr).total_seconds() / 60.0

                t1_farr = pd.to_datetime(leg1["filed_arrival_time"])
                t2_fdep = pd.to_datetime(leg2["filed_off_block_time"])
                turnaround_sched = (t2_fdep - t1_farr).total_seconds() / 60.0

                # Filter out overnight stays or long maintenance (> 4 hours)
                if turnaround_act > 240 or turnaround_act < 0:
                    continue

                arr_delay1 = float(leg1["arr_delay_min"] or 0.0)
                dep_delay2 = float(leg2["dep_delay_min"] or 0.0)

                # Propagation calculation
                if arr_delay1 > 0 and dep_delay2 > 0:
                    propagated = min(arr_delay1, dep_delay2)
                    absorbed = max(0.0, arr_delay1 - dep_delay2)
                    is_reactionary = True
                else:
                    propagated = 0.0
                    absorbed = max(0.0, arr_delay1)
                    is_reactionary = False

                leg_pairs.append(
                    RotationLegPair(
                        ac_reg=str(ac_reg),
                        ac_operator=str(leg1["ac_operator"]),
                        leg1_ectrl_id=int(leg1["ectrl_id"]),
                        leg1_adep=str(leg1["adep"]),
                        leg1_ades=str(leg1["ades"]),
                        leg1_arr_delay_min=round(arr_delay1, 1),
                        leg2_ectrl_id=int(leg2["ectrl_id"]),
                        leg2_adep=str(leg2["adep"]),
                        leg2_ades=str(leg2["ades"]),
                        leg2_dep_delay_min=round(dep_delay2, 1),
                        scheduled_turnaround_min=round(turnaround_sched, 1),
                        actual_turnaround_min=round(turnaround_act, 1),
                        delay_propagated_min=round(propagated, 1),
                        delay_absorbed_min=round(absorbed, 1),
                        is_reactionary=is_reactionary,
                    )
                )

        return leg_pairs

    def compute_network_reactionary_multiplier(
        self,
        operator: str | None = None,
        date_start: str = "2016-11-01",
        date_end: str = "2016-11-05",
    ) -> dict[str, Any]:
        """
        Compute airline-level reactionary delay multiplier and buffer absorption efficiency.
        """
        pairs = self.extract_tail_rotations(
            operator=operator, date_start=date_start, date_end=date_end
        )
        if not pairs:
            return {"error": "No rotation pairs found"}

        total_pairs = len(pairs)
        reactionary_pairs = [p for p in pairs if p.is_reactionary]
        reactionary_count = len(reactionary_pairs)

        total_incoming_delays = sum(p.leg1_arr_delay_min for p in pairs if p.leg1_arr_delay_min > 0)
        total_propagated_delays = sum(p.delay_propagated_min for p in pairs)
        total_absorbed_delays = sum(p.delay_absorbed_min for p in pairs)

        propagation_ratio = (
            total_propagated_delays / total_incoming_delays if total_incoming_delays > 0 else 0.0
        )
        absorption_rate = (
            total_absorbed_delays / total_incoming_delays if total_incoming_delays > 0 else 0.0
        )

        return {
            "total_turnarounds_analyzed": total_pairs,
            "reactionary_delay_incidents": reactionary_count,
            "reactionary_incident_rate": round(reactionary_count / total_pairs * 100.0, 1),
            "total_incoming_delay_min": round(total_incoming_delays, 1),
            "total_propagated_delay_min": round(total_propagated_delays, 1),
            "total_absorbed_delay_min": round(total_absorbed_delays, 1),
            "delay_propagation_ratio": round(propagation_ratio, 3),
            "ground_buffer_absorption_rate": round(absorption_rate, 3),
        }

