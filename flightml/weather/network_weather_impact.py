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
    start_date: str | None = None,
    end_date: str | None = None,
) -> WeatherTrafficImpactReport:
    """
    Correlate historical European aerodrome weather disruptions with global air traffic delays.
    Computes Pearson correlation coefficients and delay sensitivity slopes over the entire dataset time frame.
    """
    if hubs is None:
        hubs = DEFAULT_EUROPEAN_HUBS

    # 1. Resolve date boundaries from dataset if not provided or if provided window does not match dataset period
    bounds = ds.query("""
        SELECT 
            min(cast(coalesce(actual_off_block_time, filed_off_block_time) as date)) as min_date,
            max(cast(coalesce(actual_off_block_time, filed_off_block_time) as date)) as max_date
        FROM flights
    """)
    ds_min_date = str(bounds.iloc[0]["min_date"]) if not bounds.empty and bounds.iloc[0]["min_date"] is not None else ("2016-11-01" if ds.period == "201611" else "2025-10-01")
    ds_max_date = str(bounds.iloc[0]["max_date"]) if not bounds.empty and bounds.iloc[0]["max_date"] is not None else ("2016-11-30" if ds.period == "201611" else "2025-10-31")

    # If start_date / end_date are None, or if user requested window has no overlap with the dataset period
    if not start_date or not end_date:
        start_date = ds_min_date
        end_date = ds_max_date
    else:
        # Check if the requested start_date falls completely outside the period
        if start_date > ds_max_date or end_date < ds_min_date:
            start_date = ds_min_date
            end_date = ds_max_date
        else:
            start_date = max(start_date, ds_min_date)
            end_date = min(end_date, ds_max_date)

    # 2. Query daily traffic metrics from DuckDB across the resolved window
    traffic_df = ds.query(f"""
        SELECT 
            CAST(coalesce(actual_off_block_time, filed_off_block_time) AS DATE) as date_val,
            count(*) as flights,
            round(avg(dep_delay_min), 2) as avg_dep_delay,
            round(avg(arr_delay_min), 2) as avg_arr_delay,
            round(sum(case when arr_delay_min > 30 then 1 else 0 end) * 100.0 / count(*), 2) as pct_delayed_30
        FROM flights
        WHERE coalesce(actual_off_block_time, filed_off_block_time) >= '{start_date}' 
          AND coalesce(actual_off_block_time, filed_off_block_time) <= '{end_date} 23:59:59'
        GROUP BY CAST(coalesce(actual_off_block_time, filed_off_block_time) AS DATE)
        ORDER BY date_val ASC
    """)

    traffic_df["date_str"] = traffic_df["date_val"].astype(str) if not traffic_df.empty else []

    # 3. Extract and aggregate historical METAR observations across specified hubs in bulk
    all_hub_dfs: list[pd.DataFrame] = []
    for h in hubs:
        try:
            df_h = client.fetch_station_metars(h, f"{start_date} 00:00:00", f"{end_date} 23:59:59")
            if not df_h.empty:
                df_h = df_h.copy()
                ts_col = "timestamp" if "timestamp" in df_h.columns else "valid" if "valid" in df_h.columns else None
                if ts_col:
                    df_h["date_str"] = pd.to_datetime(df_h[ts_col]).dt.strftime("%Y-%m-%d")
                    all_hub_dfs.append(df_h)
        except Exception:
            continue

    weather_by_date: dict[str, dict[str, Any]] = {}
    if all_hub_dfs:
        combined_metars = pd.concat(all_hub_dfs, ignore_index=True)
        for d_str, g in combined_metars.groupby("date_str"):
            total_obs = len(g)
            daily_ifr_obs = (g["flight_category"].isin(["IFR", "LIFR"])).sum() if "flight_category" in g.columns else 0
            gust_col = "wind_gust_kts" if "wind_gust_kts" in g.columns else "gust" if "gust" in g.columns else None
            spd_col = "wind_speed_kts" if "wind_speed_kts" in g.columns else "sknt" if "sknt" in g.columns else None
            daily_wind_obs = 0
            if gust_col and gust_col in g.columns:
                daily_wind_obs = (g[gust_col].fillna(0) >= 25).sum()
            elif spd_col and spd_col in g.columns:
                daily_wind_obs = (g[spd_col].fillna(0) >= 20).sum()

            ifr_pct = (daily_ifr_obs / total_obs * 100.0) if total_obs > 0 else 0.0
            wind_pct = (daily_wind_obs / total_obs * 100.0) if total_obs > 0 else 0.0
            # Severity Index (0 - 10): 60% low ceiling/visibility, 40% high wind/gusts
            severity = round(min(10.0, (ifr_pct * 0.06) + (wind_pct * 0.04)), 2)
            weather_by_date[str(d_str)] = {
                "total_obs": int(total_obs),
                "ifr_pct": round(ifr_pct, 1),
                "wind_pct": round(wind_pct, 1),
                "severity": severity,
            }

    daily_records: list[DailyWeatherTrafficRecord] = []
    traffic_by_date = {row["date_str"]: row for _, row in traffic_df.iterrows()} if not traffic_df.empty else {}

    for d_str, t_row in traffic_by_date.items():
        w_info = weather_by_date.get(d_str, {"total_obs": 0, "ifr_pct": 0.0, "wind_pct": 0.0, "severity": 0.0})
        daily_records.append(
            DailyWeatherTrafficRecord(
                date_str=d_str,
                total_flights=int(t_row["flights"]),
                avg_arr_delay_min=float(t_row["avg_arr_delay"]),
                avg_dep_delay_min=float(t_row["avg_dep_delay"]),
                pct_flights_delayed_30m=float(t_row["pct_delayed_30"]),
                metar_obs_count=w_info["total_obs"],
                ifr_hours_pct=w_info["ifr_pct"],
                high_wind_hours_pct=w_info["wind_pct"],
                composite_weather_severity_index=w_info["severity"],
            )
        )

    if not daily_records:
        return WeatherTrafficImpactReport(
            start_date=start_date,
            end_date=end_date,
            analyzed_hubs=hubs,
            total_flights_analyzed=0,
            pearson_corr_delay_vs_ifr=0.0,
            pearson_corr_severe_delays_vs_weather=0.0,
            delay_sensitivity_minutes_per_severity_point=0.0,
            clear_sky_baseline_delay_min=0.0,
            worst_weather_day={"date": start_date, "severity_index": 0.0, "ifr_pct": 0.0, "avg_delay_min": 0.0, "pct_delayed_30m": 0.0},
            daily_time_series=[],
        )

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

