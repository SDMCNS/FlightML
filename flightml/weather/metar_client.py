from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import pandas as pd
import requests

from flightml.config import WEATHER_CACHE_DIR
from flightml.weather.metar_parser import MetarObservation, compute_flight_category


class MetarClient:
    """
    Client for fetching, caching, and querying historical METAR observations
    using the Iowa Environmental Mesonet (IEM) ASOS / MADIS archive.
    """

    IEM_BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir or WEATHER_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, station: str, start_dt: datetime, end_dt: datetime) -> Path:
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")
        return self.cache_dir / f"{station.upper()}_{start_str}_{end_str}.parquet"

    def fetch_station_metars(
        self,
        station: str,
        start_time: datetime | str,
        end_time: datetime | str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch historical METAR data for a specific station and time window using the IEM ASOS API.
        """
        station = station.upper()
        if isinstance(start_time, str):
            start_dt = pd.to_datetime(start_time).to_pydatetime()
        else:
            start_dt = start_time

        if isinstance(end_time, str):
            end_dt = pd.to_datetime(end_time).to_pydatetime()
        else:
            end_dt = end_time

        # Ensure at least 1 day range for API sanity
        if (end_dt - start_dt).total_seconds() < 86400:
            query_end_dt = start_dt + timedelta(days=1)
        else:
            query_end_dt = end_dt

        cache_file = self._get_cache_path(station, start_dt, query_end_dt)
        if cache_file.exists() and not force_refresh:
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

        params = [
            ("station", station),
            ("data", "all"),
            ("year1", str(start_dt.year)),
            ("month1", str(start_dt.month)),
            ("day1", str(start_dt.day)),
            ("year2", str(query_end_dt.year)),
            ("month2", str(query_end_dt.month)),
            ("day2", str(query_end_dt.day)),
            ("tz", "Etc/UTC"),
            ("format", "onlycomma"),
            ("latlon", "yes"),
            ("elev", "yes"),
            ("missing", "null"),
            ("trace", "T"),
            ("direct", "no"),
            ("report_type", "3"),
            ("report_type", "4"),
        ]

        try:
            resp = requests.get(self.IEM_BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
            text = resp.text
        except Exception:
            return pd.DataFrame()

        clean_lines = [l for l in text.splitlines() if l and not l.startswith("#")]
        if len(clean_lines) <= 1:
            return pd.DataFrame()

        try:
            raw_df = pd.read_csv(io.StringIO("\n".join(clean_lines)), low_memory=False)
        except Exception:
            return pd.DataFrame()

        if raw_df.empty or "valid" not in raw_df.columns:
            return pd.DataFrame()

        rows = []
        for _, r in raw_df.iterrows():
            valid_str = str(r["valid"])
            try:
                obs_dt = datetime.strptime(valid_str, "%Y-%m-%d %H:%M")
            except Exception:
                continue

            # Wind
            drct = pd.to_numeric(r.get("drct"), errors="coerce")
            sknt = pd.to_numeric(r.get("sknt"), errors="coerce")
            gust = pd.to_numeric(r.get("gust"), errors="coerce")

            # Vis and clouds
            vsby = pd.to_numeric(r.get("vsby"), errors="coerce")
            ceilings = []
            for i in range(1, 5):
                cov = str(r.get(f"skyc{i}", "")).upper()
                alt = pd.to_numeric(r.get(f"skyl{i}"), errors="coerce")
                if cov in ("BKN", "OVC", "VV") and pd.notna(alt):
                    ceilings.append(float(alt))
            ceiling_ft = min(ceilings) if ceilings else None

            # Temperature F to C
            tmpf = pd.to_numeric(r.get("tmpf"), errors="coerce")
            dwpf = pd.to_numeric(r.get("dwpf"), errors="coerce")
            temp_c = (tmpf - 32.0) * 5.0 / 9.0 if pd.notna(tmpf) else None
            dew_c = (dwpf - 32.0) * 5.0 / 9.0 if pd.notna(dwpf) else None

            alti = pd.to_numeric(r.get("alti"), errors="coerce")
            raw_metar = str(r.get("metar", ""))
            wxcodes = str(r.get("wxcodes", ""))

            flight_cat = compute_flight_category(
                visibility_sm=float(vsby) if pd.notna(vsby) else 10.0,
                ceiling_ft=ceiling_ft,
            )

            rows.append({
                "station_id": station,
                "raw_metar": raw_metar,
                "timestamp": obs_dt.isoformat(),
                "dt": obs_dt,
                "wind_dir_deg": float(drct) if pd.notna(drct) else None,
                "wind_speed_kts": float(sknt) if pd.notna(sknt) else None,
                "wind_gust_kts": float(gust) if pd.notna(gust) else None,
                "visibility_sm": float(vsby) if pd.notna(vsby) else None,
                "ceiling_ft": ceiling_ft,
                "flight_category": flight_cat,
                "temp_c": round(temp_c, 1) if temp_c is not None else None,
                "dewpoint_c": round(dew_c, 1) if dew_c is not None else None,
                "altimeter_inhg": float(alti) if pd.notna(alti) else None,
                "wxcodes": wxcodes if wxcodes != "null" else "",
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        save_df = df.drop(columns=["dt"])
        save_df.to_parquet(cache_file, index=False)
        return df

    def get_closest_metar(
        self,
        station: str,
        target_time: datetime | str,
        max_delta_hours: float = 2.0,
    ) -> MetarObservation | None:
        """
        Query the closest recorded METAR observation for a given station and timestamp.
        Fetches on-demand from IEM if not already cached.
        """
        if isinstance(target_time, str):
            target_dt = pd.to_datetime(target_time).to_pydatetime()
        else:
            target_dt = target_time

        station = station.upper()
        # Query window: target +/- 1 day
        start_dt = target_dt - timedelta(days=1)
        end_dt = target_dt + timedelta(days=1)

        df = self.fetch_station_metars(station, start_dt, end_dt)
        if df.empty:
            return None

        df["dt"] = pd.to_datetime(df["timestamp"])
        df["time_diff"] = (df["dt"] - target_dt).abs()

        closest = df.sort_values("time_diff").iloc[0]
        if closest["time_diff"] > timedelta(hours=max_delta_hours):
            return None

        return MetarObservation(
            station_id=station,
            raw_metar=closest["raw_metar"],
            timestamp=closest["dt"].to_pydatetime(),
            wind_dir_deg=closest["wind_dir_deg"] if pd.notna(closest["wind_dir_deg"]) else None,
            wind_speed_kts=closest["wind_speed_kts"] if pd.notna(closest["wind_speed_kts"]) else None,
            wind_gust_kts=closest["wind_gust_kts"] if pd.notna(closest["wind_gust_kts"]) else None,
            visibility_sm=closest["visibility_sm"] if pd.notna(closest["visibility_sm"]) else None,
            ceiling_ft=closest["ceiling_ft"] if pd.notna(closest["ceiling_ft"]) else None,
            flight_category=closest["flight_category"],
            temp_c=closest["temp_c"] if pd.notna(closest["temp_c"]) else None,
            dewpoint_c=closest["dewpoint_c"] if pd.notna(closest["dewpoint_c"]) else None,
            altimeter_inhg=closest["altimeter_inhg"] if pd.notna(closest["altimeter_inhg"]) else None,
        )

    def fetch_delayed_flight_weather(self, flight: dict[str, Any]) -> dict[str, Any]:
        """
        On-demand weather retrieval for a flight experiencing delays.
        Fetches departure weather at ADEP and arrival weather at ADES.
        """
        adep = flight.get("adep")
        ades = flight.get("ades")
        off_block = flight.get("actual_off_block_time") or flight.get("filed_off_block_time")
        arrival = flight.get("actual_arrival_time") or flight.get("filed_arrival_time")

        dep_weather = self.get_closest_metar(adep, off_block) if adep and off_block else None
        arr_weather = self.get_closest_metar(ades, arrival) if ades and arrival else None

        return {
            "adep": adep,
            "dep_weather": dep_weather.to_dict() if dep_weather else None,
            "ades": ades,
            "arr_weather": arr_weather.to_dict() if arr_weather else None,
        }
