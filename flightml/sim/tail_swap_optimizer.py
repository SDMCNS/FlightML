"""
Prescriptive Hub Tail-Swap Recovery Optimizer (Sprint 6).

Identifies delayed inbound aircraft rotations (`ac_reg`) arriving at a hub airport that will
infect their next scheduled departure leg with reactionary delay, and searches for compatible
same-operator (`ac_operator`), same-type (`ac_type`) donor tails on the ground at the same hub
with sufficient turnaround slack to execute a tail swap (`Tail A <-> Tail B`).

Quantifies:
- Reactionary Delay Minutes Saved
- Downstream Infected Rotations Prevented
- Estimated EU261 / Operational Delay Cost Saved (€80 / delay min in Euros)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from flightml.data.dataset import FlightDataset


@dataclass
class TailSwapRecommendation:
    hub_airport: str
    operator: str
    ac_type: str
    delayed_tail_reg: str
    donor_tail_reg: str
    delayed_outbound_flight_id: int
    delayed_outbound_ades: str
    delayed_outbound_std: str
    donor_outbound_flight_id: int
    donor_outbound_ades: str
    donor_outbound_std: str
    baseline_reactionary_delay_min: float
    post_swap_reactionary_delay_min: float
    net_delay_saved_min: float
    estimated_cost_saved_eur: float
    swap_date: str = ""
    estimated_cost_saved_usd: float = 0.0


@dataclass
class TailSwapOptimizationResult:
    period: str
    operator: str
    hub_airport: str
    date_str: str
    min_turnaround_min: float
    candidate_rotations_analyzed: int
    recommended_swaps_count: int
    total_delay_minutes_saved: float
    infected_departures_prevented: int
    estimated_cost_saved_eur: float
    swaps: list[TailSwapRecommendation]
    estimated_cost_saved_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("estimated_cost_saved_usd", None)
        for s in d.get("swaps", []):
            s.pop("estimated_cost_saved_usd", None)
        return d


def optimize_hub_tail_swaps(
    dataset: FlightDataset,
    operator: str | None = None,
    hub_airport: str | None = None,
    date_str: str | None = None,
    min_turnaround_min: float = 35.0,
    min_delay_trigger_min: float = 20.0,
    max_swaps: int = 30,
) -> TailSwapOptimizationResult:
    """
    Solve for optimal same-operator, same-aircraft-type tail swaps (`ac_reg`) at hub airports
    to absorb inbound delays and minimize total downstream reactionary departure delay minutes.
    When `operator`, `hub_airport`, or `date_str` are omitted (or `'ALL'`), scans across the
    network to surface the highest-impact tail-swap pairs in general.
    """
    op = operator.strip().upper() if operator and operator.strip() and operator.strip().upper() != "ALL" else "ALL"
    hub_filter = hub_airport.strip().upper() if hub_airport and hub_airport.strip() and hub_airport.strip().upper() != "ALL" else "ALL"
    dt_filter = date_str.strip() if date_str and date_str.strip() and date_str.strip().upper() != "ALL" else "ALL"

    where_parts = ["ac_reg IS NOT NULL", "filed_off_block_time IS NOT NULL", "actual_arrival_time IS NOT NULL"]
    params: list[Any] = []
    if op != "ALL":
        where_parts.append("ac_operator = ?")
        params.append(op)
    if dt_filter != "ALL":
        where_parts.append("CAST(filed_off_block_time AS DATE) = CAST(? AS DATE)")
        params.append(dt_filter)

    where_sql = " AND ".join(where_parts)
    hub_clause = "AND ib.ades = ?" if hub_filter != "ALL" else ""
    if hub_filter != "ALL":
        params.append(hub_filter)

    sql = f"""
        WITH op_flights AS (
            SELECT
                ectrl_id,
                adep,
                ades,
                ac_operator,
                ac_type,
                ac_reg,
                filed_off_block_time,
                actual_off_block_time,
                filed_arrival_time,
                actual_arrival_time,
                arr_delay_min,
                dep_delay_min,
                lead(ectrl_id) OVER (PARTITION BY ac_operator, ac_reg ORDER BY actual_off_block_time) AS next_flight_id,
                lead(adep) OVER (PARTITION BY ac_operator, ac_reg ORDER BY actual_off_block_time) AS next_adep,
                lead(ades) OVER (PARTITION BY ac_operator, ac_reg ORDER BY actual_off_block_time) AS next_ades,
                lead(filed_off_block_time) OVER (PARTITION BY ac_operator, ac_reg ORDER BY actual_off_block_time) AS next_std,
                lead(actual_off_block_time) OVER (PARTITION BY ac_operator, ac_reg ORDER BY actual_off_block_time) AS next_atd
            FROM flights
            WHERE {where_sql}
        )
        SELECT
            CAST(ib.next_std AS DATE) AS turn_date,
            ib.ac_operator AS operator,
            ib.ades AS hub,
            ib.ac_type,
            ib.ac_reg,
            ib.ectrl_id AS inbound_id,
            ib.actual_arrival_time AS inbound_ata,
            ib.arr_delay_min AS inbound_arr_delay,
            ib.next_flight_id AS outbound_id,
            ib.next_ades AS outbound_ades,
            ib.next_std AS outbound_std,
            datediff('minute', ib.actual_arrival_time, ib.next_std) AS available_turn_min
        FROM op_flights ib
        WHERE ib.next_flight_id IS NOT NULL
          AND ib.ades = ib.next_adep
          {hub_clause}
          AND datediff('minute', ib.actual_arrival_time, ib.next_std) BETWEEN -180 AND 240
        ORDER BY turn_date, ib.ac_operator, ib.ades, ib.ac_type, ib.next_std
        LIMIT 8000;
    """
    turns_df = dataset.cursor.execute(sql, params).df()

    # Fallback if a specific hub/date filter yielded too few turns
    if len(turns_df) < 4 and (hub_filter != "ALL" or dt_filter != "ALL"):
        fallback_where = "ac_reg IS NOT NULL AND filed_off_block_time IS NOT NULL AND actual_arrival_time IS NOT NULL"
        fb_params: list[Any] = []
        if op != "ALL":
            fallback_where += " AND ac_operator = ?"
            fb_params.append(op)
        turns_df = dataset.cursor.execute(
            sql.replace(where_sql, fallback_where).replace("AND ib.ades = ?", ""),
            fb_params,
        ).df()

    swaps: list[TailSwapRecommendation] = []
    total_saved_min = 0.0
    prevented_count = 0

    if not turns_df.empty:
        for (turn_dt, grp_op, hub, ac_t), grp in turns_df.groupby(["turn_date", "operator", "hub", "ac_type"]):
            if len(grp) < 2:
                continue
            used_tails: set[str] = set()
            rows = grp.to_dict(orient="records")
            for i, delayed in enumerate(rows):
                reg_a = str(delayed["ac_reg"])
                if reg_a in used_tails:
                    continue

                ata_a = pd.to_datetime(delayed["inbound_ata"])
                std_a = pd.to_datetime(delayed["outbound_std"])
                ready_a = ata_a + pd.Timedelta(minutes=min_turnaround_min)
                base_delay_a = max(0.0, (ready_a - std_a).total_seconds() / 60.0)

                if base_delay_a < min_delay_trigger_min:
                    continue

                best_donor = None
                best_net_saving = 0.0
                best_post_delay = base_delay_a

                for j, cand in enumerate(rows):
                    if i == j:
                        continue
                    reg_b = str(cand["ac_reg"])
                    if reg_b == reg_a or reg_b in used_tails:
                        continue

                    ata_b = pd.to_datetime(cand["inbound_ata"])
                    std_b = pd.to_datetime(cand["outbound_std"])
                    # Candidate donor must be at the hub within a reasonable 4-hour window
                    if abs((std_b - std_a).total_seconds()) > 4 * 3600:
                        continue
                    ready_b = ata_b + pd.Timedelta(minutes=min_turnaround_min)

                    base_delay_b = max(0.0, (ready_b - std_b).total_seconds() / 60.0)
                    total_base = base_delay_a + base_delay_b

                    swap_delay_a_leg = max(0.0, (ready_b - std_a).total_seconds() / 60.0)
                    swap_delay_b_leg = max(0.0, (ready_a - std_b).total_seconds() / 60.0)
                    total_swap = swap_delay_a_leg + swap_delay_b_leg

                    net_saving = total_base - total_swap
                    if net_saving > best_net_saving and net_saving >= 10.0:
                        best_net_saving = net_saving
                        best_donor = cand
                        best_post_delay = total_swap

                if best_donor is not None:
                    reg_b = str(best_donor["ac_reg"])
                    used_tails.add(reg_a)
                    used_tails.add(reg_b)
                    cost_saved = best_net_saving * 80.0  # Eurocontrol standard cost of reactionary delay (€80/min)
                    total_saved_min += best_net_saving
                    if best_post_delay < 15.0:
                        prevented_count += 1

                    swaps.append(
                        TailSwapRecommendation(
                            hub_airport=str(hub),
                            operator=str(grp_op),
                            ac_type=str(ac_t),
                            delayed_tail_reg=reg_a,
                            donor_tail_reg=reg_b,
                            delayed_outbound_flight_id=int(delayed["outbound_id"]),
                            delayed_outbound_ades=str(delayed["outbound_ades"]),
                            delayed_outbound_std=str(delayed["outbound_std"]),
                            donor_outbound_flight_id=int(best_donor["outbound_id"]),
                            donor_outbound_ades=str(best_donor["outbound_ades"]),
                            donor_outbound_std=str(best_donor["outbound_std"]),
                            baseline_reactionary_delay_min=round(base_delay_a, 1),
                            post_swap_reactionary_delay_min=round(best_post_delay, 1),
                            net_delay_saved_min=round(best_net_saving, 1),
                            estimated_cost_saved_eur=round(cost_saved, 2),
                            swap_date=str(turn_dt),
                            estimated_cost_saved_usd=round(cost_saved, 2),
                        )
                    )

    swaps.sort(key=lambda s: s.net_delay_saved_min, reverse=True)
    top_swaps = swaps[: int(max_swaps)]
    shown_saved_min = sum(s.net_delay_saved_min for s in top_swaps)

    return TailSwapOptimizationResult(
        period=dataset.period,
        operator=op,
        hub_airport=hub_filter,
        date_str=dt_filter,
        min_turnaround_min=min_turnaround_min,
        candidate_rotations_analyzed=len(turns_df),
        recommended_swaps_count=len(top_swaps),
        total_delay_minutes_saved=round(shown_saved_min, 1),
        infected_departures_prevented=prevented_count,
        estimated_cost_saved_eur=round(shown_saved_min * 80.0, 2),
        swaps=top_swaps,
        estimated_cost_saved_usd=round(shown_saved_min * 80.0, 2),
    )

