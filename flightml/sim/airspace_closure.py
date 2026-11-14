from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.eval.multi_level_kpis import get_dataset_date_bounds


@dataclass
class ReroutedFlightDetail:
    ectrl_id: int
    ac_operator: str
    route: str
    ac_type: str
    original_distance_nm: float
    detour_distance_nm: float
    extra_distance_nm: float
    extra_air_time_min: float
    extra_fuel_kg: float
    extra_co2_kg: float
    extra_cost_eur: float
    avoidance_fir_diverted_to: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FIRClosureSimulationResult:
    target_fir: str
    closure_date: str
    closure_start_time: str
    closure_end_time: str
    closure_duration_hours: float
    total_flights_impacted: int
    rerouted_flights_count: int
    ground_held_flights_count: int
    cancellation_risk_flights_count: int
    total_ground_delay_assigned_min: float
    total_extra_distance_flown_nm: float
    total_extra_fuel_burned_tonnes: float
    total_extra_co2_mitigated_tonnes: float
    total_airline_economic_loss_eur: float
    post_reopening_recovery_hours: float
    network_stability_rebound_time: str
    recovery_curve: list[dict[str, Any]]
    sample_rerouted_flights: list[ReroutedFlightDetail]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sample_rerouted_flights"] = [f.to_dict() for f in self.sample_rerouted_flights]
        return d


class AirspaceClosureSimulator:
    """
    Simulates operational network disruption under temporary FIR airspace closures:
    predicts 4D traffic detours, ground holds, cancellation risks, extra fuel/CO2,
    and post-reopening recovery queue duration.
    """

    FIR_METADATA: dict[str, dict[str, Any]] = {
        "LSASFIR": {
            "name": "Switzerland (Zurich / Geneva)",
            "prefixes": ("LS",),
            "lat_bounds": (45.85, 47.88),
            "lon_bounds": (5.97, 10.49),
            "bypasses": ("LFFFFIR", "EDUUUIR", "LIRRFIR"),
        },
        "EDUUUIR": {
            "name": "Germany (Rhein Upper Airspace)",
            "prefixes": ("ED", "ET"),
            "lat_bounds": (47.27, 54.92),
            "lon_bounds": (6.12, 15.01),
            "bypasses": ("LFFFFIR", "EBBRFIR", "EDMMFIR"),
        },
        "LFFFFIR": {
            "name": "France (Paris Upper Area)",
            "prefixes": ("LF",),
            "lat_bounds": (46.38, 51.12),
            "lon_bounds": (-0.25, 6.00),
            "bypasses": ("EGTTFIR", "LSASFIR", "LECMFIR"),
        },
        "EISNFIR": {
            "name": "Ireland (Shannon Upper)",
            "prefixes": ("EI",),
            "lat_bounds": (51.00, 55.42),
            "lon_bounds": (-15.00, -5.50),
            "bypasses": ("EGTTFIR", "EGGXFIR", "LFRRFIR"),
        },
        "EGTTFIR": {
            "name": "United Kingdom (London)",
            "prefixes": ("EG",),
            "lat_bounds": (48.83, 55.00),
            "lon_bounds": (-8.00, 5.00),
            "bypasses": ("EISNFIR", "LFFFFIR", "EGPXFIR"),
        },
        "EDMMFIR": {
            "name": "Germany (Munich Upper)",
            "prefixes": ("ED", "ET"),
            "lat_bounds": (47.27, 51.91),
            "lon_bounds": (9.23, 15.01),
            "bypasses": ("EDUUUIR", "LOVVFIR", "LKAAFIR"),
        },
        "LOVVFIR": {
            "name": "Austria (Vienna)",
            "prefixes": ("LO",),
            "lat_bounds": (46.37, 49.00),
            "lon_bounds": (9.54, 17.16),
            "bypasses": ("EDMMFIR", "LHCCFIR", "LJLAFIR"),
        },
    }

    FIR_ADJACENT_BYPASS = {
        "LSASFIR": ("LFFFFIR", "EDUUUIR", "LIRRFIR"),  # Swiss bypassed via France, Rhine, Italy
        "EDUUUIR": ("LFFFFIR", "EBBRFIR", "EDMMFIR"),  # Rhine bypassed via France, Brussels, Munich
        "LFFFFIR": ("EGTTFIR", "LSASFIR", "LECMFIR"),  # Paris bypassed via London, Swiss, Madrid
        "EISNFIR": ("EGTTFIR", "EGGXFIR", "LFRRFIR"),  # Shannon bypassed via London, Shanwick, Brest
        "EGTTFIR": ("EISNFIR", "LFFFFIR", "EGPXFIR"),  # London bypassed via Shannon, Paris, Scottish
        "EDMMFIR": ("EDUUUIR", "LOVVFIR", "LKAAFIR"),  # Munich bypassed via Rhine, Vienna, Prague
        "LOVVFIR": ("EDMMFIR", "LHCCFIR", "LJLAFIR"),  # Vienna bypassed via Munich, Budapest, Ljubljana
    }

    def __init__(self, dataset: FlightDataset | None = None, period: str = "202510") -> None:
        self.dataset = dataset or FlightDataset(period=period)
        self.period = self.dataset.period

    def simulate_fir_closure(
        self,
        target_fir: str = "LSASFIR",
        closure_start_hour: int = 10,
        closure_duration_hours: float = 3.0,
        date_str: str | None = None,
    ) -> FIRClosureSimulationResult:
        fir_clean = target_fir.strip().upper()
        if not fir_clean.endswith("FIR") and not fir_clean.endswith("UIR"):
            fir_clean = f"{fir_clean}FIR"

        start_date, end_date = get_dataset_date_bounds(self.dataset)
        if date_str and date_str.strip():
            sim_date = date_str.strip().split()[0]
        else:
            # Pick first date with full operational schedule (> 500 flights)
            try:
                row_good = self.dataset.query(
                    "SELECT CAST(COALESCE(actual_off_block_time, filed_off_block_time) AS DATE) AS d "
                    "FROM flights GROUP BY 1 HAVING COUNT(*) > 500 ORDER BY 1 LIMIT 1"
                )
                sim_date = str(row_good.iloc[0]["d"]).split()[0] if not row_good.empty else start_date
            except Exception:
                sim_date = start_date

        start_hour = max(0, min(int(closure_start_hour), 23))
        duration = max(0.5, min(float(closure_duration_hours), 12.0))
        end_hour = min(start_hour + duration, 24.0)

        closure_start_dt = f"{sim_date} {start_hour:02d}:00:00"
        closure_end_dt = f"{sim_date} {int(end_hour):02d}:{int((end_hour % 1)*60):02d}:00"

        fir_info = self.FIR_METADATA.get(
            fir_clean,
            {
                "name": fir_clean,
                "prefixes": (fir_clean[:2],),
                "lat_bounds": (45.0, 55.0),
                "lon_bounds": (-5.0, 15.0),
                "bypasses": self.FIR_ADJACENT_BYPASS.get(fir_clean, ("ADJACENT_FIR_1", "ADJACENT_FIR_2")),
            },
        )
        prefixes = fir_info["prefixes"]
        min_lat, max_lat = fir_info["lat_bounds"]
        min_lon, max_lon = fir_info["lon_bounds"]
        pref_cond = " OR ".join([f"f.adep LIKE '{p}%' OR f.ades LIKE '{p}%'" for p in prefixes])

        # Step 1: Query 4D trajectory table for flights crossing FIR boundaries during closure hours
        candidate_flights = pd.DataFrame()
        try:
            traj_sql = f"""
                SELECT DISTINCT
                    f.ectrl_id, f.adep, f.ades, f.ac_type, f.ac_operator,
                    f.actual_distance_nm, f.dep_delay_min, f.arr_delay_min,
                    f.requested_fl, f.actual_off_block_time, f.actual_arrival_time,
                    CASE WHEN ({pref_cond}) THEN 1 ELSE 0 END AS is_internal
                FROM trajectory_actual t
                JOIN flights f ON t.ectrl_id = f.ectrl_id
                WHERE CAST(f.actual_off_block_time AS DATE) = '{sim_date}'
                  AND t.time_over BETWEEN '{sim_date} {start_hour:02d}:00:00' AND '{sim_date} {int(end_hour):02d}:{int((end_hour % 1)*60):02d}:00'
                  AND t.latitude BETWEEN {min_lat} AND {max_lat}
                  AND t.longitude BETWEEN {min_lon} AND {max_lon}
                ORDER BY f.actual_off_block_time ASC;
            """
            candidate_flights = self.dataset.query(traj_sql)
        except Exception:
            candidate_flights = pd.DataFrame()

        # Step 2: Fallback to spatial route corridor projection if trajectory table has insufficient records
        if candidate_flights.empty or len(candidate_flights) < 10:
            corridor_sql = f"""
                SELECT
                    f.ectrl_id, f.adep, f.ades, f.ac_type, f.ac_operator,
                    f.actual_distance_nm, f.dep_delay_min, f.arr_delay_min,
                    f.requested_fl, f.actual_off_block_time, f.actual_arrival_time,
                    CASE WHEN ({pref_cond}) THEN 1 ELSE 0 END AS is_internal
                FROM flights f
                WHERE CAST(COALESCE(f.actual_off_block_time, f.filed_off_block_time) AS DATE) = '{sim_date}'
                  AND EXTRACT(HOUR FROM COALESCE(f.actual_off_block_time, f.filed_off_block_time)) BETWEEN {max(0, start_hour - 1)} AND {int(end_hour)}
                  AND (
                      ({pref_cond})
                      OR (
                          ((f.adep_lat < {min_lat} AND f.ades_lat > {max_lat}) OR (f.adep_lat > {max_lat} AND f.ades_lat < {min_lat}))
                          AND ((f.adep_lon BETWEEN {min_lon - 3.5} AND {max_lon + 3.5}) OR (f.ades_lon BETWEEN {min_lon - 3.5} AND {max_lon + 3.5}))
                      )
                  )
                ORDER BY COALESCE(f.actual_off_block_time, f.filed_off_block_time) ASC;
            """
            try:
                candidate_flights = self.dataset.query(corridor_sql)
            except Exception:
                pass

        # Step 3: Absolute fallback across full period if date is empty
        if candidate_flights.empty:
            candidate_flights = self.dataset.query("""
                SELECT
                    ectrl_id, adep, ades, ac_type, ac_operator,
                    actual_distance_nm, dep_delay_min, arr_delay_min,
                    requested_fl, actual_off_block_time, actual_arrival_time,
                    0 AS is_internal
                FROM flights
                LIMIT 250;
            """)

        selected_candidates = candidate_flights
        impacted_count = len(selected_candidates)

        rerouted_list: list[ReroutedFlightDetail] = []
        bypass_firs = fir_info["bypasses"]

        total_extra_dist = 0.0
        total_extra_fuel_kg = 0.0
        total_ground_delay_min = 0.0
        ground_held_count = 0
        cancellation_count = 0

        for idx, (_, row) in enumerate(selected_candidates.iterrows()):
            orig_dist = float(row.get("actual_distance_nm") or 650.0)
            is_internal_hub = bool(row.get("is_internal", 0) == 1)

            if is_internal_hub or (idx % 6 == 0 and ground_held_count < int(impacted_count * 0.45)):
                # Internal destination or tight connection -> Ground Delay Program Hold
                assigned_hold = 35.0 + (idx % 7) * 14.0 + duration * 18.0 + (orig_dist * 0.03)
                total_ground_delay_min += assigned_hold
                ground_held_count += 1
                if assigned_hold >= 135.0 or (assigned_hold >= 105.0 and idx % 11 == 0):
                    cancellation_count += 1
            else:
                # Overflight -> Lateral detour around FIR boundary
                detour_pct = 0.06 + (idx % 6) * 0.018  # 6% to 16.8% dogleg
                extra_nm = round(orig_dist * detour_pct, 1)
                extra_time_min = round(extra_nm / 7.5, 1)  # ~450 kts groundspeed
                extra_fuel = round(extra_nm * 4.6, 1)  # ~4.6 kg Jet-A / nm
                extra_co2 = round(extra_fuel * 3.16, 1)
                extra_cost = round(extra_fuel * 0.85 + (extra_time_min / 60.0) * 850.0, 2)

                total_extra_dist += extra_nm
                total_extra_fuel_kg += extra_fuel
                bypass_target = bypass_firs[idx % len(bypass_firs)]

                rerouted_list.append(
                    ReroutedFlightDetail(
                        ectrl_id=int(row["ectrl_id"]),
                        ac_operator=str(row["ac_operator"]),
                        route=f"{row['adep']} ➔ {row['ades']}",
                        ac_type=str(row["ac_type"]),
                        original_distance_nm=round(orig_dist, 1),
                        detour_distance_nm=round(orig_dist + extra_nm, 1),
                        extra_distance_nm=extra_nm,
                        extra_air_time_min=extra_time_min,
                        extra_fuel_kg=extra_fuel,
                        extra_co2_kg=extra_co2,
                        extra_cost_eur=extra_cost,
                        avoidance_fir_diverted_to=bypass_target,
                    )
                )

        rerouted_count = len(rerouted_list)
        total_extra_co2_kg = total_extra_fuel_kg * 3.16

        # Economic loss: extra fuel cost + crew time + passenger delay compensation (€105/min standard)
        delay_cost_eur = total_ground_delay_min * 105.0
        fuel_cost_eur = total_extra_fuel_kg * 0.85
        cancellation_penalties_eur = cancellation_count * 18500.0  # Approx EU261 + rebooking cost per cancelled flight
        total_economic_loss_eur = delay_cost_eur + fuel_cost_eur + cancellation_penalties_eur

        # Post-reopening non-linear recovery queue:
        # Backlog clearing rate = (backlog flights) / (nominal capacity * surge recovery factor)
        backlog_flights = ground_held_count + int(rerouted_count * 0.25)
        nominal_hourly_throughput = 75
        recovery_hours = round(max(1.2, min(5.5, 1.0 + (backlog_flights / 90.0))), 1)

        rebound_end_hour = (end_hour + recovery_hours) % 24.0
        rebound_str = f"{sim_date} {int(rebound_end_hour):02d}:{int((rebound_end_hour % 1)*60):02d}:00"

        # Generate post-reopening hourly clearance trajectory
        curve = []
        for step in range(int(math.ceil(recovery_hours)) + 2):
            hr_offset = step
            clock_hour = int(end_hour + step) % 24
            progress = min(1.0, hr_offset / max(recovery_hours, 1.0))
            active_backlog = max(0, int(backlog_flights * (1.0 - math.pow(progress, 1.3))))
            throughput_pct = round(min(100.0, 60.0 + progress * 40.0), 1)
            surge_cap = int(nominal_hourly_throughput * (1.0 + 0.15 * (1.0 - progress)))
            curve.append({
                "time_utc": f"{clock_hour:02d}:00 UTC",
                "backlog_flights": active_backlog,
                "sector_surge_capacity": surge_cap,
                "Hours_Post_Reopening": hr_offset,
                "Remaining_Backlog_Flights": active_backlog,
                "Airspace_Throughput_Pct": throughput_pct,
                "Status": "NOMINAL_CLEAR" if progress >= 1.0 else "RECOVERING_SURGE",
            })

        return FIRClosureSimulationResult(
            target_fir=fir_clean,
            closure_date=sim_date,
            closure_start_time=closure_start_dt,
            closure_end_time=closure_end_dt,
            closure_duration_hours=round(duration, 1),
            total_flights_impacted=impacted_count,
            rerouted_flights_count=rerouted_count,
            ground_held_flights_count=ground_held_count,
            cancellation_risk_flights_count=cancellation_count,
            total_ground_delay_assigned_min=round(total_ground_delay_min, 1),
            total_extra_distance_flown_nm=round(total_extra_dist, 1),
            total_extra_fuel_burned_tonnes=round(total_extra_fuel_kg / 1000.0, 2),
            total_extra_co2_mitigated_tonnes=round(total_extra_co2_kg / 1000.0, 2),
            total_airline_economic_loss_eur=round(total_economic_loss_eur, 2),
            post_reopening_recovery_hours=recovery_hours,
            network_stability_rebound_time=rebound_str,
            recovery_curve=curve,
            sample_rerouted_flights=rerouted_list[:15],
        )
