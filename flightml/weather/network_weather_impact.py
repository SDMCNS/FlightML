from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.weather.metar_client import MetarClient

DEFAULT_EUROPEAN_HUBS = ["EGLL", "LFPG", "EHAM", "EDDF", "LEMD", "EINN", "EDDM"]


@dataclass
class DailyWeatherTrafficRecord:
    date_str: str
    total_flights: int
    avg_arr_delay_min: float
    avg_dep_delay_min: float
    pct_flights_delayed_30m: float
    metar_obs_count: int
    ifr_hours_pct: float
    high_wind_hours_pct: float
    composite_weather_severity_index: float  # 0 to 10 scale


@dataclass
class WeatherTrafficImpactReport:
    start_date: str
    end_date: str
    analyzed_hubs: list[str]
    total_flights_analyzed: int
    pearson_corr_delay_vs_ifr: float
    pearson_corr_severe_delays_vs_weather: float
    delay_sensitivity_minutes_per_severity_point: float
    clear_sky_baseline_delay_min: float
    worst_weather_day: dict[str, Any]
    daily_time_series: list[DailyWeatherTrafficRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_window": f"{self.start_date} to {self.end_date}",
            "analyzed_hubs": self.analyzed_hubs,
            "total_flights_analyzed": self.total_flights_analyzed,
            "correlations": {
                "arr_delay_vs_ifr_conditions": round(self.pearson_corr_delay_vs_ifr, 3),
                "severe_delays_30m_vs_weather_severity": round(
                    self.pearson_corr_severe_delays_vs_weather, 3
                ),
            },
            "delay_sensitivity": {
                "minutes_delay_per_severity_point": round(
                    self.delay_sensitivity_minutes_per_severity_point, 2
                ),
                "clear_sky_baseline_delay_min": round(self.clear_sky_baseline_delay_min, 2),
            },
            "worst_weather_day": self.worst_weather_day,
            "daily_trends": [
                {
                    "date": r.date_str,
                    "flights": r.total_flights,
                    "avg_arr_delay_min": r.avg_arr_delay_min,
                    "pct_delayed_30m": r.pct_flights_delayed_30m,
                    "ifr_pct": r.ifr_hours_pct,
                    "high_wind_pct": r.high_wind_hours_pct,
                    "severity_index": r.composite_weather_severity_index,
                }
                for r in self.daily_time_series
            ],
        }


def analyze_network_weather_impact(
    ds: FlightDataset,
    client: MetarClient,
    hubs: list[str] | None = None,
    start_date: str = "2016-11-01",
    end_date: str = "2016-11-10",
) -> WeatherTrafficImpactReport:
    """
    Correlate historical European aerodrome weather disruptions with global air traffic delays.
    Computes Pearson correlation coefficients and delay sensitivity slopes.
    """
    if hubs is None:
        hubs = DEFAULT_EUROPEAN_HUBS

    # 1. Query daily traffic metrics from DuckDB
    traffic_df = ds.query(f"""
        SELECT 
            CAST(actual_off_block_time AS DATE) as date_val,
            count(*) as flights,
            round(avg(dep_delay_min), 2) as avg_dep_delay,
            round(avg(arr_delay_min), 2) as avg_arr_delay,
            round(sum(case when arr_delay_min > 30 then 1 else 0 end) * 100.0 / count(*), 2) as pct_delayed_30
        FROM flights
        WHERE actual_off_block_time >= '{start_date}' AND actual_off_block_time <= '{end_date} 23:59:59'
        GROUP BY CAST(actual_off_block_time AS DATE)
        ORDER BY date_val ASC
    """)

    traffic_df["date_str"] = traffic_df["date_val"].astype(str)

    # 2. Extract and aggregate historical METAR observations across specified hubs
    date_range = pd.date_range(start_date, end_date)
    daily_records: list[DailyWeatherTrafficRecord] = []

    for d in date_range:
        d_str = d.strftime("%Y-%m-%d")
        daily_ifr_obs = 0
        daily_wind_obs = 0
        total_obs = 0

        for h in hubs:
            try:
                df_h = client.fetch_station_metars(h, f"{d_str} 00:00:00", f"{d_str} 23:59:59")
                if not df_h.empty:
                    total_obs += len(df_h)
                    if "flight_category" in df_h.columns:
                        daily_ifr_obs += (df_h["flight_category"].isin(["IFR", "LIFR"])).sum()
                    if "gust" in df_h.columns:
                        daily_wind_obs += (df_h["gust"].fillna(0) >= 25).sum()
                    elif "sknt" in df_h.columns:
                        daily_wind_obs += (df_h["sknt"].fillna(0) >= 20).sum()
            except Exception:
                continue

        ifr_pct = (daily_ifr_obs / total_obs * 100.0) if total_obs > 0 else 0.0
        wind_pct = (daily_wind_obs / total_obs * 100.0) if total_obs > 0 else 0.0
        # Severity Index (0 - 10): 60% low ceiling/visibility, 40% high wind/gusts
        severity = round(min(10.0, (ifr_pct * 0.06) + (wind_pct * 0.04)), 2)

        # Match with traffic
        match_traffic = traffic_df[traffic_df["date_str"] == d_str]
        if not match_traffic.empty:
            t_row = match_traffic.iloc[0]
            daily_records.append(
                DailyWeatherTrafficRecord(
                    date_str=d_str,
                    total_flights=int(t_row["flights"]),
                    avg_arr_delay_min=float(t_row["avg_arr_delay"]),
                    avg_dep_delay_min=float(t_row["avg_dep_delay"]),
                    pct_flights_delayed_30m=float(t_row["pct_delayed_30"]),
                    metar_obs_count=total_obs,
                    ifr_hours_pct=round(ifr_pct, 1),
                    high_wind_hours_pct=round(wind_pct, 1),
                    composite_weather_severity_index=severity,
                )
            )

    if not daily_records:
        raise ValueError(f"No traffic or weather records found for date window {start_date} to {end_date}")

    rec_df = pd.DataFrame([
        {
            "arr_delay": r.avg_arr_delay_min,
            "severe_30": r.pct_flights_delayed_30m,
            "ifr_pct": r.ifr_hours_pct,
            "severity": r.composite_weather_severity_index,
        }
        for r in daily_records
    ])

    corr_ifr = float(rec_df["arr_delay"].corr(rec_df["ifr_pct"])) if len(rec_df) > 2 else 0.0
    corr_severe = float(rec_df["severe_30"].corr(rec_df["severity"])) if len(rec_df) > 2 else 0.0

    if np.isnan(corr_ifr):
        corr_ifr = 0.0
    if np.isnan(corr_severe):
        corr_severe = 0.0

    # Fit simple linear sensitivity slope
    if len(rec_df) >= 3 and rec_df["severity"].std() > 0:
        poly = np.polyfit(rec_df["severity"], rec_df["arr_delay"], deg=1)
        slope = float(poly[0])
        baseline = float(poly[1])
    else:
        slope = 0.0
        baseline = float(rec_df["arr_delay"].mean())

    worst_rec = max(daily_records, key=lambda r: r.composite_weather_severity_index)

    return WeatherTrafficImpactReport(
        start_date=start_date,
        end_date=end_date,
        analyzed_hubs=hubs,
        total_flights_analyzed=sum(r.total_flights for r in daily_records),
        pearson_corr_delay_vs_ifr=corr_ifr,
        pearson_corr_severe_delays_vs_weather=corr_severe,
        delay_sensitivity_minutes_per_severity_point=slope,
        clear_sky_baseline_delay_min=baseline,
        worst_weather_day={
            "date": worst_rec.date_str,
            "severity_index": worst_rec.composite_weather_severity_index,
            "ifr_pct": worst_rec.ifr_hours_pct,
            "avg_delay_min": worst_rec.avg_arr_delay_min,
            "pct_delayed_30m": worst_rec.pct_flights_delayed_30m,
        },
        daily_time_series=daily_records,
    )

