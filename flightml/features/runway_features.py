from __future__ import annotations

from typing import Any
import pandas as pd

from flightml.weather.wind_calc import find_optimal_runways
from flightml.weather.metar_parser import MetarObservation


def extract_airport_weather_features(
    runways_df: pd.DataFrame, metar: MetarObservation | None
) -> dict[str, Any]:
    """
    Combine airport physical runway layouts with current/forecast METAR weather
    to compute operational capacity and wind impedance features.
    """
    if metar is None:
        return {
            "wind_speed_kts": 0.0,
            "wind_gust_kts": 0.0,
            "min_crosswind_kts": 0.0,
            "max_headwind_kts": 0.0,
            "runways_crosswind_exceeded": 0,
            "is_ifr": 0,
            "visibility_sm": 10.0,
            "ceiling_ft": 10000.0,
        }

    runway_dicts = runways_df.to_dict(orient="records")
    w_dir = metar.wind_dir_deg or 0.0
    w_spd = metar.wind_speed_kts or 0.0
    w_gust = metar.wind_gust_kts

    candidates = find_optimal_runways(runway_dicts, w_dir, w_spd, w_gust)

    if candidates:
        min_crosswind = min(c.crosswind_kts for c in candidates)
        max_headwind = max(c.headwind_kts for c in candidates)
        exceeded_count = sum(1 for c in candidates if c.crosswind_limit_exceeded)
    else:
        min_crosswind = 0.0
        max_headwind = 0.0
        exceeded_count = 0

    is_ifr = 1 if metar.flight_category in ("IFR", "LIFR") else 0
    vis = metar.visibility_sm if metar.visibility_sm is not None else 10.0
    ceil = metar.ceiling_ft if metar.ceiling_ft is not None else 10000.0

    return {
        "wind_speed_kts": w_spd,
        "wind_gust_kts": w_gust or 0.0,
        "min_crosswind_kts": round(min_crosswind, 1),
        "max_headwind_kts": round(max_headwind, 1),
        "runways_crosswind_exceeded": exceeded_count,
        "is_ifr": is_ifr,
        "visibility_sm": round(vis, 1),
        "ceiling_ft": round(ceil, 0),
    }

