from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.models.reactionary_delay import ReactionaryDelayAnalyzer


@dataclass
class StressTestResult:
    operator: str
    total_rotations: int
    mean_scheduled_turnaround_min: float
    fragility_curve: list[dict[str, Any]]
    resilience_threshold_shock_min: float
    resilience_grade: str  # A (Robust), B (Moderate), C (Fragile), D (Highly Fragile)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "total_rotations_analyzed": self.total_rotations,
            "mean_scheduled_turnaround_min": round(self.mean_scheduled_turnaround_min, 1),
            "resilience_threshold_shock_min": self.resilience_threshold_shock_min,
            "resilience_grade": self.resilience_grade,
            "fragility_curve": self.fragility_curve,
        }


def stress_test_airline(
    dataset: FlightDataset,
    operator: str = "RYR",
    shocks: list[float] | None = None,
    start_date: str = "2016-11-01",
    end_date: str = "2016-11-04",
) -> StressTestResult:
    """
    Stress-test an airline's scheduling model by injecting synthetic operational delays
    and simulating downstream propagation through the aircraft rotation network.
    """
    operator = operator.upper()
    if shocks is None:
        shocks = [15.0, 30.0, 45.0, 60.0, 90.0, 120.0]

    analyzer = ReactionaryDelayAnalyzer(dataset)
    pairs = analyzer.extract_tail_rotations(
        operator=operator, date_start=start_date, date_end=end_date
    )

    if not pairs:
        return StressTestResult(
            operator=operator,
            total_rotations=0,
            mean_scheduled_turnaround_min=0.0,
            fragility_curve=[],
            resilience_threshold_shock_min=0.0,
            resilience_grade="N/A",
        )

    mean_turnaround = sum(p.scheduled_turnaround_min for p in pairs) / len(pairs)

    curve = []
    resilience_threshold = 120.0

    for shock in shocks:
        propagated_sum = 0.0
        infected_legs = 0

        for p in pairs:
            # Inject shock onto leg 1 arrival
            effective_inbound_delay = shock
            # Available ground buffer = scheduled turnaround - minimum required turnaround (assumed 25m)
            min_turnaround = 25.0
            available_buffer = max(0.0, p.scheduled_turnaround_min - min_turnaround)

            unabsorbed_delay = max(0.0, effective_inbound_delay - available_buffer)
            if unabsorbed_delay > 0:
                propagated_sum += unabsorbed_delay
                infected_legs += 1

        infection_rate = (infected_legs / len(pairs)) * 100.0
        multiplier = propagated_sum / (shock * len(pairs)) if shock > 0 else 0.0

        curve.append({
            "injected_shock_min": shock,
            "infection_rate_pct": round(infection_rate, 1),
            "propagated_delay_total_min": round(propagated_sum, 1),
            "cascade_multiplier": round(multiplier, 3),
        })

        if infection_rate >= 50.0 and resilience_threshold == 120.0:
            resilience_threshold = shock

    # Assign Resilience Grade based on buffer and threshold
    if mean_turnaround >= 50 and resilience_threshold >= 60:
        grade = "A (High Resilience)"
    elif mean_turnaround >= 40:
        grade = "B (Moderate Resilience)"
    elif mean_turnaround >= 30:
        grade = "C (Lean / Fragile)"
    else:
        grade = "D (Ultra-Lean / Highly Fragile)"

    return StressTestResult(
        operator=operator,
        total_rotations=len(pairs),
        mean_scheduled_turnaround_min=mean_turnaround,
        fragility_curve=curve,
        resilience_threshold_shock_min=resilience_threshold,
        resilience_grade=grade,
    )

