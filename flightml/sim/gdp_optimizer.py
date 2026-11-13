from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.eval.fuel_carbon import DEFAULT_FUEL_FLOW_KG_HR, CO2_FACTOR
from flightml.eval.jet_fuel_prices import get_weekly_fuel_price


@dataclass
class SlotAssignment:
    ectrl_id: int
    adep: str
    ades: str
    ac_operator: str
    ac_type: str
    original_arrival_time: str
    assigned_slot_time: str
    ground_delay_assigned_min: float
    airborne_holding_saved_min: float
    fuel_saved_kg: float
    co2_saved_kg: float
    fuel_cost_saved_eur: float
    fuel_cost_saved_usd: float = 0.0


@dataclass
class GDPResult:
    airport: str
    program_start: str
    program_end: str
    degraded_capacity_rate_per_hr: int
    total_flights_regulated: int
    total_ground_delay_assigned_min: float
    total_airborne_delay_prevented_min: float
    total_fuel_saved_kg: float
    total_co2_saved_kg: float
    total_fuel_cost_saved_eur: float
    total_fuel_cost_saved_usd: float = 0.0
    slot_assignments: list[SlotAssignment] = None  # type: ignore

    def to_dict(self) -> dict[str, Any]:
        return {
            "airport": self.airport,
            "program_start": self.program_start,
            "program_end": self.program_end,
            "degraded_capacity_rate_per_hr": self.degraded_capacity_rate_per_hr,
            "total_flights_regulated": self.total_flights_regulated,
            "total_ground_delay_assigned_min": round(self.total_ground_delay_assigned_min, 1),
            "total_airborne_delay_prevented_min": round(self.total_airborne_delay_prevented_min, 1),
            "total_fuel_saved_kg": round(self.total_fuel_saved_kg, 1),
            "total_co2_saved_kg": round(self.total_co2_saved_kg, 1),
            "total_fuel_cost_saved_eur": round(self.total_fuel_cost_saved_eur, 2),
            "sample_assignments": [
                {
                    "ectrl_id": s.ectrl_id,
                    "operator": s.ac_operator,
                    "origin": s.adep,
                    "ac_type": s.ac_type,
                    "original_eta": s.original_arrival_time,
                    "slot_time": s.assigned_slot_time,
                    "ground_delay_min": round(s.ground_delay_assigned_min, 1),
                    "fuel_saved_kg": round(s.fuel_saved_kg, 1),
                }
                for s in (self.slot_assignments or [])[:10]
            ],
        }


def optimize_ground_delay_program(
    dataset: FlightDataset,
    target_airport: str | None = None,
    program_date: str | None = None,
    start_hour: int = 6,
    end_hour: int = 14,
    degraded_capacity_rate_per_hr: int | None = None,
    fuel_price_per_kg: float | None = None,
    *,
    airport_icao: str | None = None,
    declared_aar_per_hour: int | None = None,
) -> GDPResult:
    """
    Ration-by-Schedule (RBS) Ground Delay Program (GDP) Optimizer.
    
    When an airport's acceptance rate (AAR) collapses due to fog/winds, this algorithm
    assigns Controlled Times of Arrival (CTA) and Calculated Take-Off Times (CTOT)
    to hold flights at their origin gates rather than burning fuel in holding stacks.
    Automatically resolves the EIA weekly jet fuel spot price (in EUR) for the program date.
    """
    resolved_airport = (airport_icao or target_airport or "EGLL").strip().upper()
    resolved_rate = int(declared_aar_per_hour if declared_aar_per_hour is not None else (degraded_capacity_rate_per_hr if degraded_capacity_rate_per_hr is not None else 25))

    # Auto-resolve busiest arrival date for the target airport if program_date is omitted or has 0 arrivals
    if not program_date:
        try:
            dt_df = dataset.query(
                """
                SELECT substr(CAST(filed_arrival_time AS VARCHAR), 1, 10) AS dt, COUNT(*) AS cnt
                FROM flights
                WHERE ades = ? AND filed_arrival_time IS NOT NULL
                GROUP BY dt
                ORDER BY cnt DESC
                LIMIT 1
                """,
                [resolved_airport],
            )
            if not dt_df.empty and dt_df.iloc[0]["dt"]:
                program_date = str(dt_df.iloc[0]["dt"])
        except Exception:
            pass
    if not program_date:
        program_date = "2025-10-16" if str(getattr(dataset, "period", "")).startswith("2025") else "2016-11-11"

    if fuel_price_per_kg is None:
        fuel_price_per_kg = get_weekly_fuel_price(program_date, period_hint=dataset.period).price_eur_per_kg
    target_airport = resolved_airport
    degraded_capacity_rate_per_hr = resolved_rate
    t_start = f"{program_date} {start_hour:02d}:00:00"
    t_end = f"{program_date} {end_hour:02d}:00:00"

    query = f"""
        SELECT
            ectrl_id,
            adep,
            ades,
            ac_operator,
            ac_type,
            filed_arrival_time,
            actual_arrival_time
        FROM flights
        WHERE ades = ?
          AND filed_arrival_time >= ?
          AND filed_arrival_time < ?
        ORDER BY filed_arrival_time ASC;
    """
    flights_df = dataset.query(query, [target_airport, t_start, t_end])
    if flights_df.empty:
        return GDPResult(
            airport=target_airport,
            program_start=t_start,
            program_end=t_end,
            degraded_capacity_rate_per_hr=degraded_capacity_rate_per_hr,
            total_flights_regulated=0,
            total_ground_delay_assigned_min=0.0,
            total_airborne_delay_prevented_min=0.0,
            total_fuel_saved_kg=0.0,
            total_co2_saved_kg=0.0,
            total_fuel_cost_saved_eur=0.0,
            total_fuel_cost_saved_usd=0.0,
            slot_assignments=[],
        )

    # Slot interval in minutes based on hourly capacity: e.g. 25/hr -> 60 / 25 = 2.4 min
    slot_interval_min = 60.0 / float(degraded_capacity_rate_per_hr)
    program_start_dt = pd.to_datetime(t_start)

    assignments: list[SlotAssignment] = []
    current_slot_dt = program_start_dt

    total_ground_delay = 0.0
    total_fuel_saved = 0.0

    for _, row in flights_df.iterrows():
        eta = pd.to_datetime(row["filed_arrival_time"])
        ac_type = str(row["ac_type"] or "DEFAULT_JET").upper()
        fuel_flow = DEFAULT_FUEL_FLOW_KG_HR.get(ac_type, DEFAULT_FUEL_FLOW_KG_HR["DEFAULT_JET"])

        # RBS slot assignment: Earliest slot at or after current available slot
        if current_slot_dt < eta:
            assigned_slot = eta
        else:
            assigned_slot = current_slot_dt

        delay_min = max(0.0, (assigned_slot - eta).total_seconds() / 60.0)

        # In an unmanaged scenario, this delay would be absorbed in airborne holding stacks
        # Fuel burned in holding stack ~ 85% of cruise fuel flow rate
        # Fuel burned on ground at gate with engines off = 0 kg!
        holding_fuel_prevented_kg = (delay_min / 60.0) * (fuel_flow * 0.85)
        co2_saved_kg = holding_fuel_prevented_kg * CO2_FACTOR
        cost_saved_eur = holding_fuel_prevented_kg * fuel_price_per_kg

        assignments.append(
            SlotAssignment(
                ectrl_id=int(row["ectrl_id"]),
                adep=str(row["adep"]),
                ades=str(row["ades"]),
                ac_operator=str(row["ac_operator"]),
                ac_type=ac_type,
                original_arrival_time=str(eta),
                assigned_slot_time=str(assigned_slot),
                ground_delay_assigned_min=delay_min,
                airborne_holding_saved_min=delay_min,
                fuel_saved_kg=holding_fuel_prevented_kg,
                co2_saved_kg=co2_saved_kg,
                fuel_cost_saved_eur=cost_saved_eur,
                fuel_cost_saved_usd=cost_saved_eur,
            )
        )

        total_ground_delay += delay_min
        total_fuel_saved += holding_fuel_prevented_kg

        # Advance slot
        current_slot_dt = assigned_slot + timedelta(minutes=slot_interval_min)

    total_co2_saved = total_fuel_saved * CO2_FACTOR
    total_cost_saved = total_fuel_saved * fuel_price_per_kg

    return GDPResult(
        airport=target_airport,
        program_start=t_start,
        program_end=t_end,
        degraded_capacity_rate_per_hr=degraded_capacity_rate_per_hr,
        total_flights_regulated=len(assignments),
        total_ground_delay_assigned_min=total_ground_delay,
        total_airborne_delay_prevented_min=total_ground_delay,
        total_fuel_saved_kg=total_fuel_saved,
        total_co2_saved_kg=total_co2_saved,
        total_fuel_cost_saved_eur=total_cost_saved,
        total_fuel_cost_saved_usd=total_cost_saved,
        slot_assignments=assignments,
    )

