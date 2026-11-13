"""
Airport Surface Movement & ICAO KPI02 Additional Taxi-Out Time (ATOT) Engine (Sprint 1).

Computes:
1. Unimpeded Taxi-Out Time (UTOT): 20th percentile of actual `TAXI_OUT` durations per (airport, aircraft wake class).
2. Additional Taxi-Out Time (ATOT = max(0, Actual Taxi - UTOT)) in accordance with ICAO GANP KPI02 / Eurocontrol PRU.
3. Surface Queue Fuel Burn & LTO Emissions: Uses OpenAP aircraft engine counts + ICAO EDB v32B 7%-thrust Idle
   fuel flows and emission indices (CO2, NOx, CO, HC, and non-volatile Particulate Matter nvPM soot).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.eval.jet_fuel_prices import get_weekly_fuel_price

# Standard ICAO LTO 7% Idle emission indices & fuel flows by wake category (derived from ICAO EDB v32B medians)
# Used when a specific aircraft/engine match is not in openap_aircraft
WAKE_IDLE_DEFAULTS: dict[str, dict[str, float]] = {
    "J": {  # Super (A388) - 4 engines
        "total_idle_ff_kg_min": 42.0,
        "apu_ff_kg_min": 4.5,
        "ei_nox_idle_g_kg": 4.8,
        "ei_co_idle_g_kg": 24.0,
        "ei_hc_idle_g_kg": 1.8,
        "nvpm_ei_mass_idle_mg_kg": 18.0,
    },
    "H": {  # Heavy widebody (B77W, B789, A359, A333) - 2 engines
        "total_idle_ff_kg_min": 22.5,
        "apu_ff_kg_min": 3.2,
        "ei_nox_idle_g_kg": 4.5,
        "ei_co_idle_g_kg": 22.0,
        "ei_hc_idle_g_kg": 1.6,
        "nvpm_ei_mass_idle_mg_kg": 15.0,
    },
    "M": {  # Medium narrowbody (B738, A320, A20N, E190) - 2 engines
        "total_idle_ff_kg_min": 11.4,
        "apu_ff_kg_min": 1.8,
        "ei_nox_idle_g_kg": 4.2,
        "ei_co_idle_g_kg": 26.5,
        "ei_hc_idle_g_kg": 2.1,
        "nvpm_ei_mass_idle_mg_kg": 14.0,
    },
    "L": {  # Light / regional / bizjet
        "total_idle_ff_kg_min": 5.2,
        "apu_ff_kg_min": 0.9,
        "ei_nox_idle_g_kg": 3.6,
        "ei_co_idle_g_kg": 32.0,
        "ei_hc_idle_g_kg": 3.2,
        "nvpm_ei_mass_idle_mg_kg": 12.0,
    },
}


@dataclass
class FlightTaxiOutReport:
    ectrl_id: int
    period: str
    adep: str
    ades: str
    ac_operator: str
    ac_type: str
    wake_category: str
    default_engine: str
    off_block_time: str
    takeoff_time: str
    actual_taxi_out_min: float
    filed_taxi_out_min: float
    unimpeded_utot_min: float
    additional_atot_min: float
    filed_vs_actual_delta_min: float
    idle_fuel_rate_kg_min: float
    total_taxi_fuel_kg: float
    excess_atot_fuel_kg: float
    excess_co2_kg: float
    excess_nox_g: float
    excess_co_g: float
    excess_hc_g: float
    excess_nvpm_mg: float
    fuel_price_eur_per_kg: float
    fuel_price_eur_per_gallon: float
    excess_fuel_cost_eur: float
    fuel_price_usd_per_kg: float = 0.0
    fuel_price_usd_per_gallon: float = 0.0
    excess_fuel_cost_usd: float = 0.0
    congestion_status: str = "UNIMPEDED_TAXI"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("fuel_price_usd_per_kg", "fuel_price_usd_per_gallon", "excess_fuel_cost_usd"):
            d.pop(k, None)
        return d


def _resolve_aircraft_idle_profile(dataset: FlightDataset, ac_type: str | None) -> dict[str, Any]:
    """
    Resolve exact 7%-thrust ground idle fuel flow and ICAO EDB v32B emission indices
    for a given ICAO aircraft type using `openap_aircraft`, `icao_edb_gaseous`, and `icao_edb_nvpm`.
    """
    ac_perf = dataset.get_aircraft_performance(ac_type or "B738") if ac_type else None
    wake = str(ac_perf.get("wake_category") or "M") if ac_perf else "M"
    eng_count = int(ac_perf.get("engine_count") or 2) if ac_perf else 2
    default_eng = str(ac_perf.get("default_engine") or "CFM56-7B26") if ac_perf else "CFM56-7B26"

    defaults = dict(WAKE_IDLE_DEFAULTS.get(wake, WAKE_IDLE_DEFAULTS["M"]))
    idle_ff_kg_min = defaults["total_idle_ff_kg_min"] + defaults["apu_ff_kg_min"]
    ei_nox = defaults["ei_nox_idle_g_kg"]
    ei_co = defaults["ei_co_idle_g_kg"]
    ei_hc = defaults["ei_hc_idle_g_kg"]
    ei_nvpm = defaults["nvpm_ei_mass_idle_mg_kg"]

    # Query ICAO EDB v32B for the aircraft's default engine
    if default_eng and default_eng != "None":
        edb = dataset.get_engine_emissions(default_eng[:8])
        gas_list = edb.get("gaseous_matches") or []
        if gas_list:
            g0 = gas_list[0]
            ff_s = g0.get("ff_idle_kg_s")
            if ff_s and float(ff_s) > 0:
                idle_ff_kg_min = float(ff_s) * 60.0 * eng_count + defaults["apu_ff_kg_min"]
            if g0.get("ei_nox_idle_g_kg") and pd.notna(g0.get("ei_nox_idle_g_kg")):
                ei_nox = float(g0["ei_nox_idle_g_kg"])
            if g0.get("ei_co_idle_g_kg") and pd.notna(g0.get("ei_co_idle_g_kg")):
                ei_co = float(g0["ei_co_idle_g_kg"])
            if g0.get("ei_hc_idle_g_kg") and pd.notna(g0.get("ei_hc_idle_g_kg")):
                ei_hc = float(g0["ei_hc_idle_g_kg"])

        nvpm_list = edb.get("nvpm_matches") or []
        if nvpm_list:
            n0 = nvpm_list[0]
            if n0.get("nvpm_ei_mass_idle_mg_kg") and pd.notna(n0.get("nvpm_ei_mass_idle_mg_kg")):
                ei_nvpm = float(n0["nvpm_ei_mass_idle_mg_kg"])

    return {
        "wake_category": wake,
        "default_engine": default_eng,
        "idle_fuel_rate_kg_min": round(idle_ff_kg_min, 2),
        "ei_nox_idle_g_kg": round(ei_nox, 2),
        "ei_co_idle_g_kg": round(ei_co, 2),
        "ei_hc_idle_g_kg": round(ei_hc, 2),
        "nvpm_ei_mass_idle_mg_kg": round(ei_nvpm, 2),
    }


def analyze_airport_taxi_out_performance(
    dataset: FlightDataset,
    airport_icao: str | None = None,
    date_str: str | None = None,
    min_flights: int = 50,
    top_n: int = 20,
) -> dict[str, Any]:
    """
    Compute ICAO KPI02 Additional Taxi-Out Time (ATOT), Unimpeded Taxi-Out Time (UTOT),
    departure surface queue fuel waste, and LTO emissions across airports.
    """
    if not dataset.has_fir_traversals():
        return {
            "period": dataset.period,
            "has_data": False,
            "message": (
                f"Period {dataset.period} does not have explicit TAXI_OUT rows in flight_firs_actual. "
                "Available in monthly Eurocontrol archives (e.g. 202510)."
            ),
            "airports": [],
            "hourly_profile": [],
        }

    where_clauses = ["a.fir_id = 'TAXI_OUT'", "epoch(a.exit_time - a.entry_time) BETWEEN 60 AND 7200"]
    params: list[Any] = []

    if airport_icao and airport_icao.strip():
        where_clauses.append("fl.adep = ?")
        params.append(airport_icao.strip().upper())

    if date_str and date_str.strip():
        where_clauses.append("CAST(a.entry_time AS DATE) = CAST(? AS DATE)")
        params.append(date_str.strip())

    where_sql = " AND ".join(where_clauses)
    min_f = 1 if airport_icao else int(min_flights)

    sql = f"""
        WITH taxi_legs AS (
            SELECT
                fl.adep,
                fl.ac_type,
                coalesce(oa.wake_category, 'M') AS wake_cat,
                a.entry_time,
                epoch(a.exit_time - a.entry_time) / 60.0 AS actual_taxi_min,
                epoch(f.exit_time - f.entry_time) / 60.0 AS filed_taxi_min
            FROM flight_firs_actual a
            JOIN flights fl ON a.ectrl_id = fl.ectrl_id
            LEFT JOIN flight_firs_filed f
              ON a.ectrl_id = f.ectrl_id AND f.fir_id = 'TAXI_OUT'
            LEFT JOIN openap_aircraft oa
              ON fl.ac_type = oa.ac_type
            WHERE {where_sql}
        ),
        utot_baselines AS (
            SELECT
                adep,
                wake_cat,
                quantile_cont(actual_taxi_min, 0.20) AS utot_min
            FROM taxi_legs
            GROUP BY adep, wake_cat
        ),
        enriched AS (
            SELECT
                t.adep,
                t.wake_cat,
                t.entry_time,
                t.actual_taxi_min,
                t.filed_taxi_min,
                u.utot_min,
                greatest(0.0, t.actual_taxi_min - u.utot_min) AS atot_min,
                CASE
                    WHEN t.wake_cat = 'J' THEN 46.5
                    WHEN t.wake_cat = 'H' THEN 25.7
                    WHEN t.wake_cat = 'L' THEN 6.1
                    ELSE 13.2
                END AS idle_ff_kg_min
            FROM taxi_legs t
            JOIN utot_baselines u
              ON t.adep = u.adep AND t.wake_cat = u.wake_cat
        )
        SELECT
            adep,
            CAST(count(*) AS BIGINT) AS departures,
            round(avg(utot_min), 2) AS unimpeded_utot_min,
            round(avg(actual_taxi_min), 2) AS avg_actual_taxi_min,
            round(avg(filed_taxi_min), 2) AS avg_filed_taxi_min,
            round(avg(atot_min), 2) AS avg_additional_atot_min,
            round(quantile_cont(actual_taxi_min, 0.95), 1) AS p95_taxi_out_min,
            round(sum(atot_min) / 60.0, 1) AS total_excess_taxi_hours,
            round(sum(atot_min * idle_ff_kg_min), 1) AS excess_taxi_fuel_kg,
            round(sum(atot_min * idle_ff_kg_min) * 3.16, 1) AS excess_taxi_co2_kg,
            round(sum(atot_min * idle_ff_kg_min) * 4.3 / 1000.0, 2) AS excess_nox_kg,
            round(sum(atot_min * idle_ff_kg_min) * 25.0 / 1000.0, 2) AS excess_co_kg
        FROM enriched
        GROUP BY adep
        HAVING count(*) >= {min_f}
        ORDER BY avg_additional_atot_min DESC, departures DESC
        LIMIT {int(top_n)};
    """
    apt_df = dataset.cursor.execute(sql, params).df()

    # Compute hourly diurnal taxi-out profile if an airport is specified or overall top
    hourly_sql = f"""
        SELECT
            extract('hour' FROM a.entry_time) AS dep_hour_utc,
            CAST(count(*) AS BIGINT) AS departures,
            round(avg(epoch(a.exit_time - a.entry_time) / 60.0), 2) AS avg_actual_taxi_min,
            round(quantile_cont(epoch(a.exit_time - a.entry_time) / 60.0, 0.20), 2) AS p20_unimpeded_min,
            round(
                avg(epoch(a.exit_time - a.entry_time) / 60.0)
                - quantile_cont(epoch(a.exit_time - a.entry_time) / 60.0, 0.20),
                2
            ) AS avg_atot_min
        FROM flight_firs_actual a
        JOIN flights fl ON a.ectrl_id = fl.ectrl_id
        WHERE {where_sql}
        GROUP BY extract('hour' FROM a.entry_time)
        ORDER BY dep_hour_utc ASC;
    """
    hourly_df = dataset.cursor.execute(hourly_sql, params).df()

    return {
        "period": dataset.period,
        "has_data": True,
        "airport_filter": airport_icao,
        "date_filter": date_str,
        "airports": apt_df.to_dict(orient="records") if not apt_df.empty else [],
        "hourly_profile": hourly_df.to_dict(orient="records") if not hourly_df.empty else [],
    }


def analyze_flight_taxi_out(dataset: FlightDataset, ectrl_id: int) -> FlightTaxiOutReport | None:
    """
    Evaluate single-flight surface taxi-out duration, airport 20th-percentile unimpeded baseline (UTOT),
    ICAO KPI02 Additional Taxi-Out Time (ATOT), and ICAO EDB v32B idle gaseous & nvPM emissions.
    """
    if not dataset.has_fir_traversals():
        return None

    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return None

    sql = """
        SELECT
            a.entry_time AS actual_entry,
            a.exit_time AS actual_exit,
            epoch(a.exit_time - a.entry_time) / 60.0 AS actual_taxi_min,
            epoch(f.exit_time - f.entry_time) / 60.0 AS filed_taxi_min
        FROM flight_firs_actual a
        LEFT JOIN flight_firs_filed f
          ON a.ectrl_id = f.ectrl_id AND f.fir_id = 'TAXI_OUT'
        WHERE a.ectrl_id = ? AND a.fir_id = 'TAXI_OUT'
        LIMIT 1;
    """
    df = dataset.cursor.execute(sql, [int(ectrl_id)]).df()
    if df.empty:
        return None

    row = df.iloc[0]
    adep = str(flight.get("adep") or "")
    ades = str(flight.get("ades") or "")
    ac_op = str(flight.get("ac_operator") or "")
    ac_type = str(flight.get("ac_type") or "B738")
    actual_taxi = float(row["actual_taxi_min"] or 0.0)
    filed_taxi = float(row["filed_taxi_min"]) if pd.notna(row["filed_taxi_min"]) else actual_taxi

    # Compute 20th percentile unimpeded taxi-out time (UTOT) at `adep`
    utot_sql = """
        SELECT quantile_cont(epoch(a.exit_time - a.entry_time) / 60.0, 0.20) AS utot
        FROM flight_firs_actual a
        JOIN flights fl ON a.ectrl_id = fl.ectrl_id
        WHERE a.fir_id = 'TAXI_OUT' AND fl.adep = ?
          AND epoch(a.exit_time - a.entry_time) BETWEEN 60 AND 7200;
    """
    utot_res = dataset.cursor.execute(utot_sql, [adep]).fetchone()
    utot_min = float(utot_res[0]) if utot_res and utot_res[0] is not None else min(actual_taxi, 10.0)

    atot_min = max(0.0, actual_taxi - utot_min)
    filed_delta = actual_taxi - filed_taxi

    idle_prof = _resolve_aircraft_idle_profile(dataset, ac_type)
    ff_rate = idle_prof["idle_fuel_rate_kg_min"]
    total_taxi_fuel = actual_taxi * ff_rate
    excess_fuel = atot_min * ff_rate
    excess_co2 = excess_fuel * 3.16
    excess_nox_g = excess_fuel * idle_prof["ei_nox_idle_g_kg"]
    excess_co_g = excess_fuel * idle_prof["ei_co_idle_g_kg"]
    excess_hc_g = excess_fuel * idle_prof["ei_hc_idle_g_kg"]
    excess_nvpm_mg = excess_fuel * idle_prof["nvpm_ei_mass_idle_mg_kg"]

    price_rec = get_weekly_fuel_price(str(row["actual_entry"]), period_hint=dataset.period)
    excess_cost_eur = excess_fuel * price_rec.price_eur_per_kg

    if atot_min >= 15.0:
        status = "SEVERE_SURFACE_QUEUE"
    elif atot_min >= 5.0:
        status = "MODERATE_SURFACE_DELAY"
    else:
        status = "UNIMPEDED_TAXI"

    return FlightTaxiOutReport(
        ectrl_id=int(ectrl_id),
        period=dataset.period,
        adep=adep,
        ades=ades,
        ac_operator=ac_op,
        ac_type=ac_type,
        wake_category=idle_prof["wake_category"],
        default_engine=idle_prof["default_engine"],
        off_block_time=str(row["actual_entry"]),
        takeoff_time=str(row["actual_exit"]),
        actual_taxi_out_min=round(actual_taxi, 1),
        filed_taxi_out_min=round(filed_taxi, 1),
        unimpeded_utot_min=round(utot_min, 1),
        additional_atot_min=round(atot_min, 1),
        filed_vs_actual_delta_min=round(filed_delta, 1),
        idle_fuel_rate_kg_min=ff_rate,
        total_taxi_fuel_kg=round(total_taxi_fuel, 1),
        excess_atot_fuel_kg=round(excess_fuel, 1),
        excess_co2_kg=round(excess_co2, 1),
        excess_nox_g=round(excess_nox_g, 1),
        excess_co_g=round(excess_co_g, 1),
        excess_hc_g=round(excess_hc_g, 1),
        excess_nvpm_mg=round(excess_nvpm_mg, 1),
        fuel_price_eur_per_kg=round(price_rec.price_eur_per_kg, 4),
        fuel_price_eur_per_gallon=round(price_rec.price_eur_per_gallon, 3),
        excess_fuel_cost_eur=round(excess_cost_eur, 2),
        fuel_price_usd_per_kg=round(price_rec.price_eur_per_kg, 4),
        fuel_price_usd_per_gallon=round(price_rec.price_eur_per_gallon, 3),
        excess_fuel_cost_usd=round(excess_cost_eur, 2),
        congestion_status=status,
    )
