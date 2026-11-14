"""
Upper-Air Atmospheric Physics & ERA5 Pressure-Level Client (Sprint 6).

Provides 4D upper-air state (Temperature °C, Wind Speed kts, Wind Direction deg,
Relative Humidity over Water RHw %, and Relative Humidity over Ice RHi %) at cruise
Flight Levels (FL280 - FL410, corresponding to 300 hPa - 180 hPa).

Supports:
1. Optional live historical ERA5 reanalysis retrieval via Open-Meteo Archive API with local caching.
2. Fast, deterministic synoptic mid-latitude Rossby wave & Polar/Subtropical Jet Stream physics model
   calibrated to latitude, longitude, Flight Level, and day-of-year.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

CACHE_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "weather_cache" / "upper_air_era5.sqlite"


@dataclass
class UpperAirPointState:
    latitude: float
    longitude: float
    flight_level: float
    pressure_hpa: float
    temperature_c: float
    wind_speed_kts: float
    wind_dir_deg: float
    rh_water_pct: float
    rh_ice_pct: float
    is_issr: bool  # Ice Supersaturated Region (RHi >= 100% and T <= -40°C)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fl_to_pressure_hpa(flight_level: float) -> float:
    """Convert ICAO standard Flight Level (100s of ft) to atmospheric pressure in hPa."""
    alt_m = max(0.0, float(flight_level) * 100.0 * 0.3048)
    if alt_m <= 11000.0:
        return round(1013.25 * ((1.0 - 0.0065 * alt_m / 288.15) ** 5.25588), 1)
    # Stratosphere isothermal layer above 11 km (226.32 hPa at 11 km)
    return round(226.32 * math.exp(-0.000157688 * (alt_m - 11000.0)), 1)


def rh_water_to_rh_ice(rh_water_pct: float, temp_c: float) -> float:
    """
    Convert Relative Humidity over liquid water (RHw) to Relative Humidity over ice (RHi)
    using the Murphy & Koop (2005) saturation vapor pressure ratio (e_sat_water / e_sat_ice).
    At T = -45°C, e_sat_water / e_sat_ice ≈ 1.54, so RHw = 68% corresponds to RHi ≈ 105% (Ice Supersaturated!).
    """
    t_k = max(190.0, min(273.15, temp_c + 273.15))
    # Goff-Gratch / Murphy-Koop ratio approximation for -70°C <= T <= 0°C
    ratio = math.exp(0.00955 * (273.15 - t_k))
    return round(rh_water_pct * ratio, 1)


class UpperAirClient:
    """
    Retrieves upper-air wind, temperature, and ice-supersaturation state along 4D flight trajectories.
    Primary live sources:
    - 2021+ (`202510`): Open-Meteo Historical Forecast Pressure-Level API (`https://historical-forecast-api.open-meteo.com/v1/forecast`)
      providing direct 300/250/200/150 hPa temperature, RHw, wind speed, and wind direction.
    - Pre-2021 (`201611`): Open-Meteo ERA5 Reanalysis Archive API (`https://archive-api.open-meteo.com/v1/archive`)
      providing surface temperature, pressure, 100m winds, and high-altitude cloud cover (`cloud_cover_high` > 6 km)
      coupled with hydrostatic lapse-rate & thermal-wind vertical profile extrapolation.
    Cached locally in SQLite (`data/weather_cache/upper_air_era5.sqlite`).
    """

    OPEN_METEO_PRESSURE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, use_live_api: bool = True, cache_db: Path | None = None) -> None:
        self.use_live_api = use_live_api
        self.cache_db = cache_db or CACHE_DB_PATH
        self._mem_cache: dict[str, dict[str, Any]] = {}
        self._init_cache()

    def _init_cache(self) -> None:
        try:
            self.cache_db.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS era5_upper_air_cache (
                        cache_key TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL
                    )
                    """
                )
        except Exception:
            pass

    @staticmethod
    def _nearest_era5_pressure_level(p_hpa: float) -> int:
        levels = [300, 250, 200, 150]
        return min(levels, key=lambda lvl: abs(lvl - p_hpa))

    def prefetch_trajectory_upper_air(self, trajectory_df: pd.DataFrame) -> None:
        """
        Prefetch upper-air meteorological states for 4D waypoints along a flight trajectory
        using local SQLite cache first, then issuing a single batched HTTP request for any
        uncached coordinate points to the Open-Meteo European Reanalysis & Forecast API.
        """
        if not self.use_live_api or trajectory_df.empty:
            return

        # 1. Collect cruise midpoints
        cruise_midpoints: list[tuple[float, float, float, str]] = []
        for i in range(len(trajectory_df) - 1):
            r1 = trajectory_df.iloc[i]
            r2 = trajectory_df.iloc[i + 1]
            fl = float((r1.get("flight_level", 0.0) + r2.get("flight_level", 0.0)) / 2.0)
            if fl < 200.0:
                continue
            lat = (float(r1.get("latitude", 0.0)) + float(r2.get("latitude", 0.0))) / 2.0
            lon = (float(r1.get("longitude", 0.0)) + float(r2.get("longitude", 0.0))) / 2.0
            t_val = r1.get("time_over")
            if not t_val or pd.isna(t_val):
                continue
            cruise_midpoints.append((lat, lon, fl, str(t_val)))

        if not cruise_midpoints:
            return

        # 2. Sample 6 to 8 representative synoptic checkpoints along the route
        if len(cruise_midpoints) > 8:
            step = len(cruise_midpoints) / 7.0
            sampled = [cruise_midpoints[int(idx * step)] for idx in range(7)]
            if cruise_midpoints[-1] not in sampled:
                sampled.append(cruise_midpoints[-1])
        else:
            sampled = cruise_midpoints

        needed_by_group: dict[tuple[str, int], list[tuple[float, float, str]]] = {}
        for lat, lon, fl, t_str in sampled:
            try:
                dt = pd.to_datetime(t_str)
                d_str = dt.strftime("%Y-%m-%d")
            except Exception:
                continue

            lvl = self._nearest_era5_pressure_level(fl_to_pressure_hpa(fl))
            lat_grid = round(lat, 1)
            lon_grid = round(lon, 1)
            cache_key = f"{lat_grid}:{lon_grid}:{d_str}:{lvl}"

            if cache_key in self._mem_cache:
                continue

            group_key = (d_str, lvl)
            if group_key not in needed_by_group:
                needed_by_group[group_key] = []
            if not any(k == cache_key for _, _, k in needed_by_group[group_key]):
                needed_by_group[group_key].append((lat_grid, lon_grid, cache_key))

        if not needed_by_group:
            return

        # 3. Check local SQLite cache in bulk
        all_candidate_keys = [k for group in needed_by_group.values() for _, _, k in group]
        try:
            with sqlite3.connect(self.cache_db) as conn:
                for i in range(0, len(all_candidate_keys), 100):
                    chunk = all_candidate_keys[i : i + 100]
                    placeholders = ",".join("?" for _ in chunk)
                    for r in conn.execute(
                        f"SELECT cache_key, payload_json FROM era5_upper_air_cache WHERE cache_key IN ({placeholders})",
                        chunk,
                    ).fetchall():
                        self._mem_cache[r[0]] = json.loads(r[1])
        except Exception:
            pass

        # 4. Batch-fetch only uncached points in a single request (up to 10 points per request)
        for (d_str, lvl), pts in needed_by_group.items():
            uncached = [p for p in pts if p[2] not in self._mem_cache]
            if not uncached:
                continue

            dt_sample = pd.to_datetime(d_str)
            for start_idx in range(0, len(uncached), 10):
                chunk = uncached[start_idx : start_idx + 10]
                chunk_lats = [c[0] for c in chunk]
                chunk_lons = [c[1] for c in chunk]
                try:
                    if dt_sample.year >= 2021:
                        resp = requests.get(
                            self.OPEN_METEO_PRESSURE_URL,
                            params={
                                "latitude": chunk_lats,
                                "longitude": chunk_lons,
                                "start_date": d_str,
                                "end_date": d_str,
                                "hourly": f"temperature_{lvl}hPa,relative_humidity_{lvl}hPa,wind_speed_{lvl}hPa,wind_direction_{lvl}hPa",
                                "wind_speed_unit": "kn",
                            },
                            timeout=5,
                        )
                    else:
                        resp = requests.get(
                            self.OPEN_METEO_ARCHIVE_URL,
                            params={
                                "latitude": chunk_lats,
                                "longitude": chunk_lons,
                                "start_date": d_str,
                                "end_date": d_str,
                                "hourly": "temperature_2m,relative_humidity_2m,cloud_cover_high,wind_speed_100m,wind_direction_100m",
                                "wind_speed_unit": "kn",
                            },
                            timeout=5,
                        )
                    if resp.status_code == 200:
                        data = resp.json()
                        items = data if isinstance(data, list) else [data]
                        with sqlite3.connect(self.cache_db) as conn:
                            for idx_pt, item in enumerate(items):
                                if idx_pt < len(chunk):
                                    k_cache = chunk[idx_pt][2]
                                    h_payload = item.get("hourly", {})
                                    self._mem_cache[k_cache] = h_payload
                                    conn.execute(
                                        "INSERT OR REPLACE INTO era5_upper_air_cache (cache_key, payload_json) VALUES (?, ?)",
                                        (k_cache, json.dumps(h_payload)),
                                    )
                except Exception:
                    pass

    def get_upper_air_state(
        self,
        lat: float,
        lon: float,
        flight_level: float,
        timestamp: str | datetime | None = None,
    ) -> UpperAirPointState:
        """
        Evaluate upper-air wind vector, temperature, and RHi at `(lat, lon, flight_level, timestamp)`.
        Local-first: checks memory cache -> SQLite cache -> nearest synoptic checkpoint -> deterministic Rossby model.
        """
        p_hpa = fl_to_pressure_hpa(flight_level)

        if self.use_live_api and timestamp is not None:
            try:
                dt = pd.to_datetime(timestamp)
                d_str = dt.strftime("%Y-%m-%d")
                lvl = self._nearest_era5_pressure_level(p_hpa)
                lat_grid = round(lat, 1)
                lon_grid = round(lon, 1)
                cache_key = f"{lat_grid}:{lon_grid}:{d_str}:{lvl}"

                hourly_data: dict[str, Any] | None = self._mem_cache.get(cache_key)
                if hourly_data is None:
                    with sqlite3.connect(self.cache_db) as conn:
                        row = conn.execute(
                            "SELECT payload_json FROM era5_upper_air_cache WHERE cache_key = ?",
                            (cache_key,),
                        ).fetchone()
                        if row:
                            hourly_data = json.loads(row[0])
                            self._mem_cache[cache_key] = hourly_data

                # If exact key not cached, find nearest cached synoptic checkpoint on same date and level
                if hourly_data is None:
                    for k, val in self._mem_cache.items():
                        parts = k.split(":")
                        if len(parts) == 4 and parts[2] == d_str and parts[3] == str(lvl):
                            try:
                                c_lat, c_lon = float(parts[0]), float(parts[1])
                                if abs(c_lat - lat_grid) <= 2.5 and abs(c_lon - lon_grid) <= 3.5:
                                    hourly_data = val
                                    break
                            except Exception:
                                pass

                if hourly_data:
                    t_arr = hourly_data.get(f"temperature_{lvl}hPa") or []
                    rh_arr = hourly_data.get(f"relative_humidity_{lvl}hPa") or []
                    ws_arr = hourly_data.get(f"wind_speed_{lvl}hPa") or []
                    wd_arr = hourly_data.get(f"wind_direction_{lvl}hPa") or []
                    hr = min(dt.hour, max(0, len(t_arr) - 1))
                    if t_arr and hr < len(t_arr) and t_arr[hr] is not None and ws_arr[hr] is not None:
                        t_c = float(t_arr[hr])
                        rh_w = float(rh_arr[hr]) if rh_arr[hr] is not None else 45.0
                        w_spd = float(ws_arr[hr])
                        w_dir = float(wd_arr[hr]) if wd_arr[hr] is not None else 270.0
                        rh_i = rh_water_to_rh_ice(rh_w, t_c)
                        return UpperAirPointState(
                            latitude=round(lat, 4),
                            longitude=round(lon, 4),
                            flight_level=round(flight_level, 0),
                            pressure_hpa=p_hpa,
                            temperature_c=round(t_c, 1),
                            wind_speed_kts=round(w_spd, 1),
                            wind_dir_deg=round(w_dir, 1),
                            rh_water_pct=round(rh_w, 1),
                            rh_ice_pct=rh_i,
                            is_issr=bool(rh_i >= 100.0 and t_c <= -40.0),
                        )
                    # Handle pre-2021 ERA5 Archive response (`temperature_2m`, `cloud_cover_high`, `wind_speed_100m`)
                    t2m_arr = hourly_data.get("temperature_2m") or []
                    cc_high_arr = hourly_data.get("cloud_cover_high") or []
                    ws100_arr = hourly_data.get("wind_speed_100m") or []
                    wd100_arr = hourly_data.get("wind_direction_100m") or []
                    hr2 = min(dt.hour, max(0, len(t2m_arr) - 1))
                    if t2m_arr and hr2 < len(t2m_arr) and t2m_arr[hr2] is not None:
                        alt_km = max(0.0, float(flight_level) * 0.03048)
                        t2m = float(t2m_arr[hr2])
                        t_c = max(-62.0, t2m - 6.5 * min(11.0, alt_km))
                        cc_high = float(cc_high_arr[hr2] or 0.0) if hr2 < len(cc_high_arr) else 0.0
                        rh_w = min(95.0, max(20.0, 32.0 + 0.48 * cc_high))
                        ws100 = float(ws100_arr[hr2] or 15.0) if hr2 < len(ws100_arr) else 15.0
                        wd100 = float(wd100_arr[hr2] or 265.0) if hr2 < len(wd100_arr) else 265.0
                        w_spd = min(160.0, ws100 * (1.0 + 0.28 * alt_km))
                        rh_i = rh_water_to_rh_ice(rh_w, t_c)
                        return UpperAirPointState(
                            latitude=round(lat, 4),
                            longitude=round(lon, 4),
                            flight_level=round(flight_level, 0),
                            pressure_hpa=p_hpa,
                            temperature_c=round(t_c, 1),
                            wind_speed_kts=round(w_spd, 1),
                            wind_dir_deg=round(wd100, 1),
                            rh_water_pct=round(rh_w, 1),
                            rh_ice_pct=rh_i,
                            is_issr=bool(rh_i >= 100.0 and t_c <= -40.0),
                        )
            except Exception:
                pass

        # Deterministic Baroclinic Rossby Wave + Tropopause Physics Model
        alt_km = max(0.0, float(flight_level) * 0.03048)
        doy = 285
        hour_val = 12
        if timestamp is not None:
            try:
                dt = pd.to_datetime(timestamp)
                doy = int(dt.dayofyear)
                hour_val = int(dt.hour)
            except Exception:
                pass

        # 1. ISA Temperature + latitudinal & Rossby trough/ridge perturbation
        isa_t = 15.0 - 6.5 * min(11.0, alt_km)
        if alt_km > 11.0:
            isa_t = -56.5 + (alt_km - 11.0) * 0.5
        wave_phase = math.radians(lon * 3.5 + doy * 4.2 + hour_val * 2.5)
        lat_rad = math.radians(lat)
        temp_c = isa_t - 4.5 * math.sin(lat_rad - math.radians(45.0)) + 3.2 * math.cos(wave_phase)

        # 2. Polar / Subtropical Jet Stream speed & direction (peaks around 45°–55°N and FL330–FL370)
        alt_jet_factor = math.exp(-(((flight_level - 350.0) / 85.0) ** 2))
        lat_jet_factor = math.exp(-(((abs(lat) - 48.0) / 16.0) ** 2))
        seasonal_factor = 1.15 + 0.25 * math.cos(math.radians((doy - 15) * 360.0 / 365.0))
        wind_speed_kts = max(12.0, (25.0 + 78.0 * lat_jet_factor * alt_jet_factor) * seasonal_factor + 14.0 * math.sin(wave_phase))
        wind_dir_deg = (268.0 + 22.0 * math.sin(wave_phase)) % 360.0

        # 3. Upper-tropospheric moisture & Ice Supersaturation Layer (thin 2,000-4,000 ft moist layers near tropopause FL310-FL380)
        vertical_moist_band = math.exp(-(((flight_level - 345.0) / 28.0) ** 2))
        synoptic_moist_ridge = 0.5 + 0.5 * math.sin(wave_phase + math.radians(lat * 4.0))
        rh_w = min(96.0, max(18.0, 28.0 + 54.0 * vertical_moist_band * synoptic_moist_ridge))
        rh_i = rh_water_to_rh_ice(rh_w, temp_c)

        return UpperAirPointState(
            latitude=round(lat, 4),
            longitude=round(lon, 4),
            flight_level=round(flight_level, 0),
            pressure_hpa=p_hpa,
            temperature_c=round(temp_c, 1),
            wind_speed_kts=round(wind_speed_kts, 1),
            wind_dir_deg=round(wind_dir_deg, 1),
            rh_water_pct=round(rh_w, 1),
            rh_ice_pct=rh_i,
            is_issr=bool(rh_i >= 100.0 and temp_c <= -40.0),
        )

