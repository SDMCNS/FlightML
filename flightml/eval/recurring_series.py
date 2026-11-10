from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from flightml.data.dataset import FlightDataset


def categorize_time_of_day(hour: int) -> str:
    if 6 <= hour < 12:
        return "Morning (06:00-12:00)"
    elif 12 <= hour < 18:
        return "Afternoon (12:00-18:00)"
    elif 18 <= hour < 24:
        return "Evening (18:00-24:00)"
    else:
        return "Night (00:00-06:00)"


@dataclass
class RecurringFlightOccurrence:
    ectrl_id: int
    date_str: str
    off_block_time: str
    dep_hour: int
    time_of_day_bank: str
    ac_operator: str
    ac_type: str
    ac_reg: str
    actual_distance_nm: float
    actual_air_time_min: float
    dep_delay_min: float
    arr_delay_min: float


@dataclass
class DiurnalBankSummary:
    bank_name: str
    flight_count: int
    avg_arr_delay_min: float
    avg_dep_delay_min: float
    avg_air_time_min: float


@dataclass
class RecurringSeriesReport:
    route: str
    adep: str
    ades: str
    filter_operator: str | None
    filter_ac_type: str | None
    total_flights: int
    date_span: str
    mean_arr_delay_min: float
    std_arr_delay_min: float
    mean_dep_delay_min: float
    mean_air_time_min: float
    min_air_time_min: float
    max_air_time_min: float
    air_time_span_min: float
    mean_distance_nm: float
    distance_span_nm: float
    best_performing_day: dict[str, Any]
    worst_performing_day: dict[str, Any]
    diurnal_banks: list[DiurnalBankSummary]
    hourly_delay_curve: list[dict[str, Any]]
    occurrences: list[RecurringFlightOccurrence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "adep": self.adep,
            "ades": self.ades,
            "filter_operator": self.filter_operator or "ALL",
            "filter_ac_type": self.filter_ac_type or "ALL",
            "total_flights": self.total_flights,
            "date_span": self.date_span,
            "delays": {
                "mean_arr_delay_min": round(self.mean_arr_delay_min, 2),
                "std_arr_delay_min": round(self.std_arr_delay_min, 2),
                "mean_dep_delay_min": round(self.mean_dep_delay_min, 2),
            },
            "flight_duration": {
                "mean_air_time_min": round(self.mean_air_time_min, 1),
                "min_air_time_min": round(self.min_air_time_min, 1),
                "max_air_time_min": round(self.max_air_time_min, 1),
                "variance_span_min": round(self.air_time_span_min, 1),
            },
            "flown_distance": {
                "mean_distance_nm": round(self.mean_distance_nm, 1),
                "distance_span_nm": round(self.distance_span_nm, 1),
            },
            "best_performing_day": self.best_performing_day,
            "worst_performing_day": self.worst_performing_day,
            "diurnal_time_of_day_breakdown": [
                {
                    "bank": b.bank_name,
                    "flights": b.flight_count,
                    "avg_arr_delay_min": round(b.avg_arr_delay_min, 2),
                    "avg_dep_delay_min": round(b.avg_dep_delay_min, 2),
                    "avg_air_time_min": round(b.avg_air_time_min, 1),
                }
                for b in self.diurnal_banks
            ],
            "hourly_delay_curve": self.hourly_delay_curve,
            "occurrences_count": len(self.occurrences),
        }


def analyze_recurring_flight_series(
    ds: FlightDataset,
    adep: str,
    ades: str,
    operator: str | None = None,
    ac_type: str | None = None,
    time_of_day: str | None = None,
    limit: int = 500,
) -> RecurringSeriesReport | None:
    """
    Extract and benchmark recurring flights on a corridor across days and time-of-day banks.
    Computes day-over-day operational variability, distance deviation, and diurnal congestion curves.
    """
    df = ds.search_flights(
        adep=adep,
        ades=ades,
        ac_operator=operator,
        ac_type=ac_type,
        time_of_day=time_of_day,
        limit=limit,
    )

    if df.empty or len(df) < 2:
        return None

    df["dt"] = pd.to_datetime(df["actual_off_block_time"])
    df["date_str"] = df["dt"].dt.strftime("%Y-%m-%d")
    df["dep_hour"] = df["dt"].dt.hour
    df["tod_bank"] = df["dep_hour"].apply(categorize_time_of_day)

    # Clean numeric columns
    arr_delays = pd.to_numeric(df["arr_delay_min"], errors="coerce").fillna(0.0)
    dep_delays = pd.to_numeric(df["dep_delay_min"], errors="coerce").fillna(0.0)
    air_times = pd.to_numeric(df["actual_air_time_min"], errors="coerce").fillna(0.0)
    distances = pd.to_numeric(df["actual_distance_nm"], errors="coerce").fillna(0.0)

    # Occurrences list
    occurrences: list[RecurringFlightOccurrence] = []
    for _, row in df.iterrows():
        occurrences.append(
            RecurringFlightOccurrence(
                ectrl_id=int(row["ectrl_id"]),
                date_str=str(row["date_str"]),
                off_block_time=str(row["actual_off_block_time"])[:16],
                dep_hour=int(row["dep_hour"]),
                time_of_day_bank=str(row["tod_bank"]),
                ac_operator=str(row.get("ac_operator") or ""),
                ac_type=str(row.get("ac_type") or ""),
                ac_reg=str(row.get("ac_reg") or ""),
                actual_distance_nm=float(row.get("actual_distance_nm") or 0.0),
                actual_air_time_min=float(row.get("actual_air_time_min") or 0.0),
                dep_delay_min=float(row.get("dep_delay_min") or 0.0),
                arr_delay_min=float(row.get("arr_delay_min") or 0.0),
            )
        )

    # Diurnal Banks Aggregation
    bank_order = [
        "Morning (06:00-12:00)",
        "Afternoon (12:00-18:00)",
        "Evening (18:00-24:00)",
        "Night (00:00-06:00)",
    ]
    diurnal_banks: list[DiurnalBankSummary] = []
    for b_name in bank_order:
        sub = df[df["tod_bank"] == b_name]
        if not sub.empty:
            diurnal_banks.append(
                DiurnalBankSummary(
                    bank_name=b_name,
                    flight_count=len(sub),
                    avg_arr_delay_min=float(sub["arr_delay_min"].mean()),
                    avg_dep_delay_min=float(sub["dep_delay_min"].mean()),
                    avg_air_time_min=float(sub["actual_air_time_min"].mean()),
                )
            )

    # Hourly delay curve
    hourly_df = (
        df.groupby("dep_hour")
        .agg(
            flights=("ectrl_id", "count"),
            avg_arr_delay=("arr_delay_min", "mean"),
            avg_dep_delay=("dep_delay_min", "mean"),
        )
        .reset_index()
    )
    hourly_curve = [
        {
            "hour": int(r["dep_hour"]),
            "flights": int(r["flights"]),
            "avg_arr_delay_min": round(float(r["avg_arr_delay"]), 2),
            "avg_dep_delay_min": round(float(r["avg_dep_delay"]), 2),
        }
        for _, r in hourly_df.iterrows()
    ]

    # Daily aggregation to find best and worst days
    daily_grp = (
        df.groupby("date_str")
        .agg(
            flights=("ectrl_id", "count"),
            avg_delay=("arr_delay_min", "mean"),
            avg_distance=("actual_distance_nm", "mean"),
        )
        .reset_index()
    )

    best_row = daily_grp.sort_values("avg_delay", ascending=True).iloc[0]
    worst_row = daily_grp.sort_values("avg_delay", ascending=False).iloc[0]

    min_date = df["date_str"].min()
    max_date = df["date_str"].max()

    return RecurringSeriesReport(
        route=f"{adep.upper()} ➔ {ades.upper()}",
        adep=adep.upper(),
        ades=ades.upper(),
        filter_operator=operator.upper() if operator else None,
        filter_ac_type=ac_type.upper() if ac_type else None,
        total_flights=len(df),
        date_span=f"{min_date} to {max_date}",
        mean_arr_delay_min=float(arr_delays.mean()),
        std_arr_delay_min=float(arr_delays.std() if len(arr_delays) > 1 else 0.0),
        mean_dep_delay_min=float(dep_delays.mean()),
        mean_air_time_min=float(air_times.mean()),
        min_air_time_min=float(air_times.min()),
        max_air_time_min=float(air_times.max()),
        air_time_span_min=float(air_times.max() - air_times.min()),
        mean_distance_nm=float(distances.mean()),
        distance_span_nm=float(distances.max() - distances.min()),
        best_performing_day={
            "date": str(best_row["date_str"]),
            "flights": int(best_row["flights"]),
            "avg_arr_delay_min": round(float(best_row["avg_delay"]), 2),
        },
        worst_performing_day={
            "date": str(worst_row["date_str"]),
            "flights": int(worst_row["flights"]),
            "avg_arr_delay_min": round(float(worst_row["avg_delay"]), 2),
        },
        diurnal_banks=diurnal_banks,
        hourly_delay_curve=hourly_curve,
        occurrences=occurrences,
    )

