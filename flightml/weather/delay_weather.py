from __future__ import annotations

from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.weather.metar_client import MetarClient
from flightml.weather.wind_calc import find_optimal_runways


def analyze_delayed_flight_weather(
    dataset: FlightDataset,
    client: MetarClient,
    ectrl_id: int,
) -> dict[str, Any]:
    """
    Examine whether adverse weather at origin or destination explains a delayed flight.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return {"error": f"Flight {ectrl_id} not found"}

    adep = flight["adep"]
    ades = flight["ades"]
    off_block = flight["actual_off_block_time"] or flight["filed_off_block_time"]
    arrival = flight["actual_arrival_time"] or flight["filed_arrival_time"]

    dep_metar = client.get_closest_metar(adep, off_block) if adep and off_block else None
    arr_metar = client.get_closest_metar(ades, arrival) if ades and arrival else None

    # Analyze destination runway wind impact
    arr_runways_df = dataset.query(
        "SELECT * FROM raw_runways WHERE airport_ident = ? ORDER BY length_ft DESC", [ades]
    )
    arr_runways = arr_runways_df.to_dict(orient="records")

    runway_impact = []
    adverse_weather_detected = False
    reasons = []

    if arr_metar:
        if arr_metar.flight_category in ("IFR", "LIFR"):
            adverse_weather_detected = True
            reasons.append(f"Low visibility/ceiling at {ades} ({arr_metar.flight_category})")

        if arr_metar.wind_dir_deg is not None and arr_metar.wind_speed_kts is not None:
            candidates = find_optimal_runways(
                arr_runways,
                arr_metar.wind_dir_deg,
                arr_metar.wind_speed_kts,
                arr_metar.wind_gust_kts,
            )
            runway_impact = [c.to_dict() for c in candidates]
            # Check if all runways had high crosswinds
            if candidates and min(c.crosswind_kts for c in candidates) > 20:
                adverse_weather_detected = True
                min_cw = min(c.crosswind_kts for c in candidates)
                reasons.append(f"High crosswind ({min_cw:.1f} kts) at {ades}")

    if dep_metar and dep_metar.flight_category in ("IFR", "LIFR"):
        adverse_weather_detected = True
        reasons.append(f"Low visibility/ceiling at departure {adep} ({dep_metar.flight_category})")

    return {
        "ectrl_id": ectrl_id,
        "adep": adep,
        "ades": ades,
        "dep_delay_min": flight.get("dep_delay_min"),
        "arr_delay_min": flight.get("arr_delay_min"),
        "dep_metar": dep_metar.to_dict() if dep_metar else None,
        "arr_metar": arr_metar.to_dict() if arr_metar else None,
        "adverse_weather_detected": adverse_weather_detected,
        "weather_attribution_reasons": reasons,
        "destination_runway_impact": runway_impact,
    }

