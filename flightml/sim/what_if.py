from __future__ import annotations

from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.models.reactionary_delay import ReactionaryDelayAnalyzer
from flightml.weather.wind_calc import find_optimal_runways


def simulate_turnaround_mitigation(
    dataset: FlightDataset,
    operator: str = "RYR",
    extra_buffer_min: float = 10.0,
    start_date: str = "2016-11-01",
    end_date: str = "2016-11-04",
) -> dict[str, Any]:
    """
    Counterfactual simulation: What if the airline had scheduled an extra X minutes of ground buffer?
    Calculates the reduction in propagated reactionary delays across the fleet.
    """
    analyzer = ReactionaryDelayAnalyzer(dataset)
    pairs = analyzer.extract_tail_rotations(
        operator=operator, date_start=start_date, date_end=end_date
    )

    if not pairs:
        return {"error": "No rotation pairs found"}

    baseline_propagated = sum(p.delay_propagated_min for p in pairs)
    baseline_incidents = sum(1 for p in pairs if p.is_reactionary)

    # Counterfactual with extra buffer
    counterfactual_propagated = 0.0
    counterfactual_incidents = 0

    for p in pairs:
        if p.leg1_arr_delay_min > 0:
            effective_delay_after_buffer = max(0.0, p.leg1_arr_delay_min - extra_buffer_min)
            new_propagated = min(effective_delay_after_buffer, p.leg2_dep_delay_min)
            if new_propagated > 0:
                counterfactual_propagated += new_propagated
                counterfactual_incidents += 1

    saved_minutes = max(0.0, baseline_propagated - counterfactual_propagated)
    prevented_incidents = baseline_incidents - counterfactual_incidents
    pct_reduction = (saved_minutes / baseline_propagated * 100.0) if baseline_propagated > 0 else 0.0

    return {
        "operator": operator.upper(),
        "extra_buffer_scheduled_min": extra_buffer_min,
        "baseline_reactionary_incidents": baseline_incidents,
        "counterfactual_reactionary_incidents": counterfactual_incidents,
        "incidents_prevented": prevented_incidents,
        "baseline_propagated_delay_min": round(baseline_propagated, 1),
        "counterfactual_propagated_delay_min": round(counterfactual_propagated, 1),
        "delay_minutes_saved": round(saved_minutes, 1),
        "delay_reduction_percentage": round(pct_reduction, 1),
    }


def simulate_runway_weather_scenario(
    dataset: FlightDataset,
    airport: str,
    wind_dir: float,
    wind_speed: float,
    gust: float | None = None,
) -> dict[str, Any]:
    """
    Simulate runway acceptance rate and available runway configurations under severe wind conditions.
    """
    airport = airport.upper()
    runways_df = dataset.query("SELECT * FROM raw_runways WHERE airport_ident = ?", [airport])
    if runways_df.empty:
        return {"error": f"Airport {airport} has no runway data"}

    candidates = find_optimal_runways(
        runways_df.to_dict(orient="records"), wind_dir, wind_speed, gust
    )

    total_runway_ends = len(candidates)
    usable_runways = [c for c in candidates if not (c.crosswind_limit_exceeded or c.tailwind_limit_exceeded)]
    usable_count = len(usable_runways)
    capacity_ratio = (usable_count / total_runway_ends) if total_runway_ends > 0 else 0.0

    return {
        "airport": airport,
        "simulated_wind": f"{wind_dir:.0f}° @ {wind_speed:.0f} kts" + (f" G{gust:.0f}" if gust else ""),
        "total_runway_ends": total_runway_ends,
        "usable_runway_ends": usable_count,
        "exceeded_runway_ends": total_runway_ends - usable_count,
        "estimated_capacity_retention_pct": round(capacity_ratio * 100.0, 1),
        "optimal_runway": usable_runways[0].to_dict() if usable_runways else None,
        "runways_status": [c.to_dict() for c in candidates],
    }

