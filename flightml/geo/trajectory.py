from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
import pandas as pd

from flightml.geo.geodesy import haversine_distance_nm, initial_bearing_deg


@dataclass
class HoldingEvent:
    start_time: str
    end_time: str
    duration_min: float
    center_lat: float
    center_lon: float
    flight_level: float
    total_turn_deg: float


@dataclass
class TrajectoryMetrics:
    ectrl_id: int
    actual_distance_nm: float
    planned_distance_nm: float
    great_circle_distance_nm: float
    horizontal_extension_ratio: float  # Actual / Great Circle (1.0 = ideal)
    max_vertical_deviation_fl: float
    mean_vertical_deviation_fl: float
    holding_events: list[HoldingEvent]
    total_holding_duration_min: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "ectrl_id": self.ectrl_id,
            "actual_distance_nm": round(self.actual_distance_nm, 1),
            "planned_distance_nm": round(self.planned_distance_nm, 1),
            "great_circle_distance_nm": round(self.great_circle_distance_nm, 1),
            "horizontal_extension_ratio": round(self.horizontal_extension_ratio, 3),
            "max_vertical_deviation_fl": round(self.max_vertical_deviation_fl, 1),
            "mean_vertical_deviation_fl": round(self.mean_vertical_deviation_fl, 1),
            "holding_count": len(self.holding_events),
            "total_holding_duration_min": round(self.total_holding_duration_min, 1),
            "holding_events": [
                {
                    "start_time": h.start_time,
                    "end_time": h.end_time,
                    "duration_min": round(h.duration_min, 1),
                    "center_lat": round(h.center_lat, 4),
                    "center_lon": round(h.center_lon, 4),
                    "flight_level": h.flight_level,
                }
                for h in self.holding_events
            ],
        }


def compute_path_distance_nm(df: pd.DataFrame) -> float:
    """Calculate cumulative distance along a sequence of waypoints."""
    if len(df) < 2:
        return 0.0
    total = 0.0
    lats = df["latitude"].values
    lons = df["longitude"].values
    for i in range(len(df) - 1):
        total += haversine_distance_nm(lats[i], lons[i], lats[i + 1], lons[i + 1])
    return total


def detect_holding_patterns(
    actual_df: pd.DataFrame, max_radius_nm: float = 15.0, min_turn_deg: float = 300.0
) -> list[HoldingEvent]:
    """
    Detect holding patterns (racetrack loops or orbits) from trajectory waypoints.
    Looks for localized segments with cumulative turn >= 300 deg inside max_radius_nm.
    """
    if len(actual_df) < 4:
        return []

    events: list[HoldingEvent] = []
    n = len(actual_df)

    i = 0
    while i < n - 3:
        # Check window starting at i
        window_pts = [i]
        curr_turn = 0.0
        start_lat = actual_df.iloc[i]["latitude"]
        start_lon = actual_df.iloc[i]["longitude"]

        prev_bearing = None
        j = i + 1
        is_holding = False

        while j < n:
            lat1 = actual_df.iloc[j - 1]["latitude"]
            lon1 = actual_df.iloc[j - 1]["longitude"]
            lat2 = actual_df.iloc[j]["latitude"]
            lon2 = actual_df.iloc[j]["longitude"]

            dist_from_start = haversine_distance_nm(start_lat, start_lon, lat2, lon2)
            if dist_from_start > max_radius_nm:
                break

            seg_dist = haversine_distance_nm(lat1, lon1, lat2, lon2)
            if seg_dist > 0.01:
                bearing = initial_bearing_deg(lat1, lon1, lat2, lon2)
                if prev_bearing is not None:
                    diff = (bearing - prev_bearing + 180) % 360 - 180
                    curr_turn += abs(diff)
                prev_bearing = bearing

            window_pts.append(j)

            if curr_turn >= min_turn_deg:
                is_holding = True
                break
            j += 1

        if is_holding:
            t_start = str(actual_df.iloc[window_pts[0]]["time_over"])
            t_end = str(actual_df.iloc[window_pts[-1]]["time_over"])
            dt1 = pd.to_datetime(t_start)
            dt2 = pd.to_datetime(t_end)
            duration_min = max(0.0, (dt2 - dt1).total_seconds() / 60.0)

            sub = actual_df.iloc[window_pts]
            events.append(
                HoldingEvent(
                    start_time=t_start,
                    end_time=t_end,
                    duration_min=duration_min,
                    center_lat=float(sub["latitude"].mean()),
                    center_lon=float(sub["longitude"].mean()),
                    flight_level=float(sub["flight_level"].mean()),
                    total_turn_deg=curr_turn,
                )
            )
            i = window_pts[-1] + 1
        else:
            i += 1

    return events


def analyze_trajectory(actual_df: pd.DataFrame, planned_df: pd.DataFrame) -> TrajectoryMetrics:
    """
    Perform deep comparative analytics on paired actual vs. planned 4D trajectories.
    """
    ectrl_id = int(actual_df.iloc[0]["ectrl_id"]) if not actual_df.empty else 0

    actual_dist = compute_path_distance_nm(actual_df)
    planned_dist = compute_path_distance_nm(planned_df)

    if not actual_df.empty:
        gc_dist = haversine_distance_nm(
            actual_df.iloc[0]["latitude"],
            actual_df.iloc[0]["longitude"],
            actual_df.iloc[-1]["latitude"],
            actual_df.iloc[-1]["longitude"],
        )
    else:
        gc_dist = actual_dist

    extension_ratio = actual_dist / gc_dist if gc_dist > 0 else 1.0

    # Vertical comparison
    max_vert = 0.0
    mean_vert = 0.0
    if not actual_df.empty and not planned_df.empty:
        # Match nearest in time or sequence
        act_fls = actual_df["flight_level"].values
        plan_fls = planned_df["flight_level"].values
        min_len = min(len(act_fls), len(plan_fls))
        if min_len > 0:
            diffs = [abs(act_fls[k] - plan_fls[k]) for k in range(min_len)]
            max_vert = float(max(diffs))
            mean_vert = float(sum(diffs) / len(diffs))

    holding_events = detect_holding_patterns(actual_df)
    total_holding_duration = sum(h.duration_min for h in holding_events)

    return TrajectoryMetrics(
        ectrl_id=ectrl_id,
        actual_distance_nm=actual_dist,
        planned_distance_nm=planned_dist,
        great_circle_distance_nm=gc_dist,
        horizontal_extension_ratio=extension_ratio,
        max_vertical_deviation_fl=max_vert,
        mean_vertical_deviation_fl=mean_vert,
        holding_events=holding_events,
        total_holding_duration_min=total_holding_duration,
    )

