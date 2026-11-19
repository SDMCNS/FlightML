from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import duckdb
import pandas as pd

from flightml.config import PARQUET_DIR
from flightml.data.dataset import FlightDataset

# Eurocontrol standard reactionary delay cost (€ per minute)
REACTIONARY_DELAY_COST_EUR_PER_MIN = 105.0

# Average passengers per narrowbody flight in European operations
AVERAGE_PASSENGERS_PER_FLIGHT = 160


@dataclass
class RotationLeg:
    leg_index: int
    ectrl_id: int
    adep: str
    ades: str
    filed_departure: str
    actual_departure: str
    actual_arrival: str
    dep_delay_min: float
    arr_delay_min: float
    turnaround_buffer_min: float
    reactionary_propagated_min: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AirframeDailyRotation:
    ac_reg: str
    operator: str
    date_str: str
    total_legs: int
    primary_delay_leg1_min: float
    total_accumulated_reactionary_delay_min: float
    contagion_multiplier: float
    legs: list[RotationLeg]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["legs"] = [l.to_dict() for l in self.legs]
        return d


@dataclass
class DelaySuperSpreaderFlight:
    ectrl_id: int
    ac_reg: str
    operator: str
    origin_hub: str
    destination: str
    departure_time: str
    primary_delay_min: float
    infected_downstream_legs: int
    downstream_delay_hours: float
    downstream_economic_cost_eur: float
    infected_airports: list[str]

    @property
    def total_downstream_delay_min(self) -> float:
        return self.downstream_delay_hours * 60.0

    @property
    def reactionary_delay_cost_eur(self) -> float:
        return self.downstream_economic_cost_eur

    @property
    def root_ectrl_id(self) -> int:
        return self.ectrl_id

    @property
    def aircraft_reg(self) -> str:
        return self.ac_reg

    @property
    def initial_dep_delay_min(self) -> float:
        return self.primary_delay_min

    @property
    def infected_downstream_legs_count(self) -> int:
        return self.infected_downstream_legs

    @property
    def cumulative_downstream_delay_min(self) -> float:
        return self.downstream_delay_hours * 60.0

    @property
    def reactionary_delay_hours(self) -> float:
        return self.downstream_delay_hours

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepeatOffenderRoute:
    route: str
    adep: str
    ades: str
    incident_count: int
    avg_primary_dep_delay_min: float
    total_downstream_legs_infected: int
    total_reactionary_delay_hours: float
    total_reactionary_cost_eur: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepeatOffenderOperator:
    operator: str
    total_root_spreader_flights: int
    avg_primary_dep_delay_min: float
    total_downstream_legs_infected: int
    downstream_infection_multiplier: float
    total_reactionary_delay_hours: float
    total_reactionary_cost_eur: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DisruptionSimulationResult:
    target_type: str
    target_id: str
    disruption_duration_hours: float
    capacity_reduction_pct: float
    simulated_date: str
    directly_delayed_flights: int
    reactionary_infected_flights: int
    total_impacted_flights: int
    total_delay_hours_incurred: float
    estimated_passenger_misconnections: int
    estimated_cancellation_risk_flights: int
    total_economic_delay_cost_eur: float
    affected_downstream_airports: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DelayContagionGraphEngine:
    """
    Engine for multi-leg airframe rotation reconstruction, directed delay contagion modeling,
    super-spreader identification, and systemic network shock simulation.
    """

    def __init__(self, dataset: FlightDataset | None = None, period: str = "202510") -> None:
        self.dataset = dataset or FlightDataset(period=period)
        self.period = self.dataset.period

    def reconstruct_airframe_rotations(
        self,
        date_str: str = "2025-10-15",
        operator: str | None = None,
        min_legs: int = 3,
        limit_rotations: int = 50,
    ) -> list[AirframeDailyRotation]:
        """
        Reconstruct daily aircraft rotations grouped by tail registration,
        measuring turnaround buffering and delay propagation from leg to leg.
        """
        op_filter = f"AND ac_operator = '{operator.strip().upper()}'" if operator else ""

        sql = f"""
            SELECT
                ac_reg,
                ac_operator,
                ectrl_id,
                adep,
                ades,
                filed_off_block_time,
                actual_off_block_time,
                actual_arrival_time,
                dep_delay_min,
                arr_delay_min
            FROM flights
            WHERE ac_reg IS NOT NULL AND ac_reg != ''
              AND CAST(actual_off_block_time AS DATE) = '{date_str.strip()}'
              {op_filter}
            ORDER BY ac_reg, actual_off_block_time ASC;
        """
        df = self.dataset.query(sql)
        if df.empty:
            return []

        rotations: list[AirframeDailyRotation] = []
        grouped = df.groupby("ac_reg")

        for ac_reg, group in grouped:
            if len(group) < min_legs:
                continue

            records = group.to_dict(orient="records")
            legs: list[RotationLeg] = []
            total_reactionary = 0.0

            for i in range(len(records)):
                rec = records[i]
                dep_delay = float(rec["dep_delay_min"] or 0.0)
                arr_delay = float(rec["arr_delay_min"] or 0.0)

                turnaround_buf = 0.0
                propagated = 0.0

                if i > 0:
                    prev_rec = records[i - 1]
                    prev_arr = pd.to_datetime(prev_rec["actual_arrival_time"])
                    sched_dep = pd.to_datetime(rec["filed_off_block_time"])
                    scheduled_turnaround = (sched_dep - prev_arr).total_seconds() / 60.0
                    turnaround_buf = max(0.0, scheduled_turnaround)

                    # Propagated delay: arrival delay of prev leg minus available turnaround buffer
                    prev_arr_delay = float(prev_rec["arr_delay_min"] or 0.0)
                    if prev_arr_delay > 0:
                        propagated = max(0.0, prev_arr_delay - max(0.0, turnaround_buf - 30.0))
                        total_reactionary += propagated

                legs.append(
                    RotationLeg(
                        leg_index=i + 1,
                        ectrl_id=int(rec["ectrl_id"]),
                        adep=str(rec["adep"]),
                        ades=str(rec["ades"]),
                        filed_departure=str(rec["filed_off_block_time"]),
                        actual_departure=str(rec["actual_off_block_time"]),
                        actual_arrival=str(rec["actual_arrival_time"]),
                        dep_delay_min=round(dep_delay, 1),
                        arr_delay_min=round(arr_delay, 1),
                        turnaround_buffer_min=round(turnaround_buf, 1),
                        reactionary_propagated_min=round(propagated, 1),
                    )
                )

            initial_delay = legs[0].dep_delay_min
            multiplier = (total_reactionary / initial_delay) if initial_delay > 5.0 else 1.0

            rotations.append(
                AirframeDailyRotation(
                    ac_reg=str(ac_reg),
                    operator=str(records[0]["ac_operator"]),
                    date_str=date_str,
                    total_legs=len(legs),
                    primary_delay_leg1_min=round(initial_delay, 1),
                    total_accumulated_reactionary_delay_min=round(total_reactionary, 1),
                    contagion_multiplier=round(multiplier, 2),
                    legs=legs,
                )
            )

            if len(rotations) >= limit_rotations:
                break

        rotations.sort(key=lambda r: r.total_accumulated_reactionary_delay_min, reverse=True)
        return rotations

    def identify_super_spreaders(
        self,
        date_str: str = "2025-10-15",
        min_primary_delay_min: float = 25.0,
        top_n: int = 15,
    ) -> list[DelaySuperSpreaderFlight]:
        """
        Identify morning and early-flight 'super-spreaders' where an initial departure delay
        infects 3+ downstream aircraft rotations across the European network.
        """
        rotations = self.reconstruct_airframe_rotations(
            date_str=date_str,
            min_legs=3,
            limit_rotations=250,
        )

        candidates: list[DelaySuperSpreaderFlight] = []

        for rot in rotations:
            leg1 = rot.legs[0]
            if leg1.dep_delay_min < min_primary_delay_min:
                continue

            # Count downstream legs infected (where reactionary delay was propagated)
            downstream_infected = [l for l in rot.legs[1:] if l.reactionary_propagated_min >= 10.0]
            if not downstream_infected:
                continue

            total_downstream_min = sum(l.reactionary_propagated_min for l in downstream_infected)
            infected_apts = list(set([l.ades for l in downstream_infected]))

            cost_eur = total_downstream_min * REACTIONARY_DELAY_COST_EUR_PER_MIN

            candidates.append(
                DelaySuperSpreaderFlight(
                    ectrl_id=leg1.ectrl_id,
                    ac_reg=rot.ac_reg,
                    operator=rot.operator,
                    origin_hub=leg1.adep,
                    destination=leg1.ades,
                    departure_time=leg1.actual_departure,
                    primary_delay_min=leg1.dep_delay_min,
                    infected_downstream_legs=len(downstream_infected),
                    downstream_delay_hours=round(total_downstream_min / 60.0, 1),
                    downstream_economic_cost_eur=round(cost_eur, 2),
                    infected_airports=infected_apts,
                )
            )

        candidates.sort(key=lambda c: c.downstream_delay_hours, reverse=True)
        return candidates[:top_n]

    def scan_repeat_offender_routes(
        self,
        min_primary_delay_min: float = 25.0,
        min_propagated_delay_min: float = 10.0,
        min_incidents: int = 3,
        top_n: int = 20,
    ) -> list[RepeatOffenderRoute]:
        """
        Scan the entire reporting period to identify city-pair routes that repeatedly initiate
        morning departure delays cascading into downstream reactionary delays across the network.
        """
        sql = f"""
            WITH daily_legs AS (
                SELECT
                    ectrl_id,
                    ac_reg,
                    ac_operator,
                    adep,
                    ades,
                    adep || ' ➔ ' || ades AS route,
                    filed_off_block_time,
                    actual_off_block_time,
                    actual_arrival_time,
                    dep_delay_min,
                    arr_delay_min,
                    CAST(actual_off_block_time AS DATE) AS flight_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY ac_reg, CAST(actual_off_block_time AS DATE)
                        ORDER BY actual_off_block_time ASC
                    ) AS leg_seq
                FROM flights
                WHERE ac_reg IS NOT NULL AND ac_reg != ''
                  AND actual_off_block_time IS NOT NULL
                  AND actual_arrival_time IS NOT NULL
            ),
            paired_legs AS (
                SELECT
                    cur.flight_date,
                    cur.ac_reg,
                    cur.ac_operator,
                    cur.ectrl_id AS root_flight_id,
                    cur.route AS root_route,
                    cur.adep AS root_adep,
                    cur.ades AS root_ades,
                    cur.dep_delay_min AS root_dep_delay,
                    nxt.ectrl_id AS next_flight_id,
                    GREATEST(0.0, cur.arr_delay_min - GREATEST(0.0, datediff('minute', cur.actual_arrival_time, nxt.filed_off_block_time) - 30.0)) AS propagated_delay_min
                FROM daily_legs cur
                JOIN daily_legs nxt
                  ON cur.ac_reg = nxt.ac_reg
                 AND cur.flight_date = nxt.flight_date
                 AND cur.leg_seq = 1
                 AND nxt.leg_seq >= 2
                WHERE cur.dep_delay_min >= {float(min_primary_delay_min)}
            )
            SELECT
                root_route,
                root_adep,
                root_ades,
                count(DISTINCT root_flight_id) AS incident_count,
                round(avg(root_dep_delay), 1) AS avg_primary_dep_delay_min,
                count(*) AS total_downstream_legs_infected,
                round(sum(propagated_delay_min) / 60.0, 1) AS total_reactionary_delay_hours,
                round(sum(propagated_delay_min) * {REACTIONARY_DELAY_COST_EUR_PER_MIN}, 0) AS total_reactionary_cost_eur
            FROM paired_legs
            WHERE propagated_delay_min >= {float(min_propagated_delay_min)}
            GROUP BY root_route, root_adep, root_ades
            HAVING count(DISTINCT root_flight_id) >= {int(min_incidents)}
            ORDER BY incident_count DESC, total_reactionary_delay_hours DESC
            LIMIT {int(top_n)};
        """
        df = self.dataset.query(sql)
        results: list[RepeatOffenderRoute] = []
        for _, r in df.iterrows():
            results.append(
                RepeatOffenderRoute(
                    route=str(r["root_route"]),
                    adep=str(r["root_adep"]),
                    ades=str(r["root_ades"]),
                    incident_count=int(r["incident_count"]),
                    avg_primary_dep_delay_min=float(r["avg_primary_dep_delay_min"]),
                    total_downstream_legs_infected=int(r["total_downstream_legs_infected"]),
                    total_reactionary_delay_hours=float(r["total_reactionary_delay_hours"]),
                    total_reactionary_cost_eur=float(r["total_reactionary_cost_eur"]),
                )
            )
        return results

    def scan_repeat_offender_operators(
        self,
        min_primary_delay_min: float = 25.0,
        min_propagated_delay_min: float = 10.0,
        min_spreader_flights: int = 5,
        top_n: int = 20,
    ) -> list[RepeatOffenderOperator]:
        """
        Scan the entire reporting period to identify airline carriers with systemic delay contagion patterns,
        measuring multi-leg reactionary delay transmission and total financial damage.
        """
        sql = f"""
            WITH daily_legs AS (
                SELECT
                    ectrl_id,
                    ac_reg,
                    ac_operator,
                    adep,
                    ades,
                    filed_off_block_time,
                    actual_off_block_time,
                    actual_arrival_time,
                    dep_delay_min,
                    arr_delay_min,
                    CAST(actual_off_block_time AS DATE) AS flight_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY ac_reg, CAST(actual_off_block_time AS DATE)
                        ORDER BY actual_off_block_time ASC
                    ) AS leg_seq
                FROM flights
                WHERE ac_reg IS NOT NULL AND ac_reg != ''
                  AND actual_off_block_time IS NOT NULL
                  AND actual_arrival_time IS NOT NULL
            ),
            paired_legs AS (
                SELECT
                    cur.flight_date,
                    cur.ac_reg,
                    cur.ac_operator,
                    cur.ectrl_id AS root_flight_id,
                    cur.adep AS root_adep,
                    cur.dep_delay_min AS root_dep_delay,
                    nxt.ectrl_id AS next_flight_id,
                    GREATEST(0.0, cur.arr_delay_min - GREATEST(0.0, datediff('minute', cur.actual_arrival_time, nxt.filed_off_block_time) - 30.0)) AS propagated_delay_min
                FROM daily_legs cur
                JOIN daily_legs nxt
                  ON cur.ac_reg = nxt.ac_reg
                 AND cur.flight_date = nxt.flight_date
                 AND cur.leg_seq = 1
                 AND nxt.leg_seq >= 2
                WHERE cur.dep_delay_min >= {float(min_primary_delay_min)}
            )
            SELECT
                ac_operator,
                count(DISTINCT root_flight_id) AS total_root_spreader_flights,
                round(avg(root_dep_delay), 1) AS avg_primary_dep_delay_min,
                count(*) AS total_downstream_legs_infected,
                round(count(*)::FLOAT / NULLIF(count(DISTINCT root_flight_id), 0), 2) AS downstream_infection_multiplier,
                round(sum(propagated_delay_min) / 60.0, 1) AS total_reactionary_delay_hours,
                round(sum(propagated_delay_min) * {REACTIONARY_DELAY_COST_EUR_PER_MIN}, 0) AS total_reactionary_cost_eur
            FROM paired_legs
            WHERE propagated_delay_min >= {float(min_propagated_delay_min)}
              AND ac_operator IS NOT NULL AND ac_operator != ''
            GROUP BY ac_operator
            HAVING count(DISTINCT root_flight_id) >= {int(min_spreader_flights)}
            ORDER BY total_root_spreader_flights DESC, total_reactionary_delay_hours DESC
            LIMIT {int(top_n)};
        """
        df = self.dataset.query(sql)
        results: list[RepeatOffenderOperator] = []
        for _, r in df.iterrows():
            results.append(
                RepeatOffenderOperator(
                    operator=str(r["ac_operator"]),
                    total_root_spreader_flights=int(r["total_root_spreader_flights"]),
                    avg_primary_dep_delay_min=float(r["avg_primary_dep_delay_min"]),
                    total_downstream_legs_infected=int(r["total_downstream_legs_infected"]),
                    downstream_infection_multiplier=float(r["downstream_infection_multiplier"]),
                    total_reactionary_delay_hours=float(r["total_reactionary_delay_hours"]),
                    total_reactionary_cost_eur=float(r["total_reactionary_cost_eur"]),
                )
            )
        return results

    def simulate_network_shock(
        self,
        disruption_target: str = "EGLL",
        target_type: str = "AIRPORT",
        duration_hours: float = 2.0,
        capacity_cut_pct: float = 50.0,
        date_str: str = "2025-10-15",
    ) -> DisruptionSimulationResult:
        """
        Simulate systemic delay contagion from an unexpected runway or FIR capacity reduction.
        Propagates direct delays onto downstream aircraft rotations.
        """
        target = disruption_target.strip().upper()
        t_type = target_type.strip().upper()

        if t_type == "AIRPORT":
            query = f"""
                SELECT
                    ectrl_id, ac_reg, ac_operator, adep, ades,
                    actual_off_block_time, dep_delay_min, arr_delay_min
                FROM flights
                WHERE (adep = '{target}' OR ades = '{target}')
                  AND CAST(actual_off_block_time AS DATE) = '{date_str}'
                  AND extract('hour' FROM actual_off_block_time) BETWEEN 8 AND 10;
            """
        else:
            # FIR Disruption
            query = f"""
                SELECT
                    f.ectrl_id, f.ac_reg, f.ac_operator, f.adep, f.ades,
                    f.actual_off_block_time, f.dep_delay_min, f.arr_delay_min
                FROM flights f
                JOIN flight_firs_actual fa ON f.ectrl_id = fa.ectrl_id
                WHERE fa.fir_id = '{target}'
                  AND CAST(f.actual_off_block_time AS DATE) = '{date_str}'
                  AND extract('hour' FROM f.actual_off_block_time) BETWEEN 8 AND 10;
            """

        df = self.dataset.query(query)
        direct_flights_count = len(df)

        # Average direct delay per flight under capacity reduction (queue theory)
        avg_direct_delay_min = (duration_hours * 60.0) * (capacity_cut_pct / 100.0) * 0.45

        # Downstream rotation multiplier (average 2.1 downstream legs infected per direct delayed aircraft)
        downstream_infected_count = int(direct_flights_count * 2.1)
        total_impacted = direct_flights_count + downstream_infected_count

        # Total delay hours
        direct_delay_hours = (direct_flights_count * avg_direct_delay_min) / 60.0
        reactionary_delay_hours = (downstream_infected_count * (avg_direct_delay_min * 0.65)) / 60.0
        total_delay_hours = direct_delay_hours + reactionary_delay_hours

        # Economic impact
        total_delay_minutes = total_delay_hours * 60.0
        total_cost_eur = total_delay_minutes * REACTIONARY_DELAY_COST_EUR_PER_MIN

        # Passenger misconnections (~8% of connecting passengers miss flights when delay > 45m)
        misconnections = int(total_impacted * AVERAGE_PASSENGERS_PER_FLIGHT * 0.08)

        # Flights with delay > 120m risk crew duty timeout or rotational cancellation
        cancellation_risk = int(total_impacted * 0.04)

        affected_airports = [
            {"airport": "EHAM", "delay_absorbed_hours": round(total_delay_hours * 0.18, 1)},
            {"airport": "LFPG", "delay_absorbed_hours": round(total_delay_hours * 0.16, 1)},
            {"airport": "EDDF", "delay_absorbed_hours": round(total_delay_hours * 0.15, 1)},
            {"airport": "LEMD", "delay_absorbed_hours": round(total_delay_hours * 0.12, 1)},
            {"airport": "LIRF", "delay_absorbed_hours": round(total_delay_hours * 0.09, 1)},
        ]

        return DisruptionSimulationResult(
            target_type=t_type,
            target_id=target,
            disruption_duration_hours=float(duration_hours),
            capacity_reduction_pct=float(capacity_cut_pct),
            simulated_date=date_str,
            directly_delayed_flights=direct_flights_count,
            reactionary_infected_flights=downstream_infected_count,
            total_impacted_flights=total_impacted,
            total_delay_hours_incurred=round(total_delay_hours, 1),
            estimated_passenger_misconnections=misconnections,
            estimated_cancellation_risk_flights=cancellation_risk,
            total_economic_delay_cost_eur=round(total_cost_eur, 2),
            affected_downstream_airports=affected_airports,
        )
