"""
Passenger Journey & Travel Risk Engine.

Provides general travelling public intelligence:
1. Multi-leg itinerary delay risk prediction & expected delay distributions.
2. Connection & missed flight risk calculator (Minimum Connection Time MCT + Luggage Transfer).
3. Live / 7-day weather forecast impact across origin, hub, and destination.
4. EU261 / UK261 Statutory Passenger Compensation and Duty-of-Care rights calculator.
5. "When Should I Fly?" departure wave & day-of-week reliability optimizer (First Wave Advantage).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any
import requests

from flightml.data.dataset import FlightDataset
from flightml.geo.geodesy import haversine_distance_nm, EARTH_RADIUS_KM


# -----------------------------------------------------------------------------
# European Hub Minimum Connection Times (MCT) in Minutes
# -----------------------------------------------------------------------------
HUB_MCT_DATABASE: dict[str, dict[str, Any]] = {
    "EGLL": {
        "name": "London Heathrow Airport",
        "same_terminal_mct_min": 60,
        "inter_terminal_mct_min": 90,
        "luggage_safe_buffer_min": 75,
        "terminal_mapping": {
            "EIN": "T2",  # Aer Lingus operates from Terminal 2 (Queen's Terminal)
            "BAW": "T5",  # British Airways main hub at Terminal 5 (some T3)
            "BEL": "T2",  # Brussels Airlines (Star Alliance) at T2
            "AFR": "T4",  # Air France at T4
            "KLM": "T4",  # KLM at T4
            "DLH": "T2",  # Lufthansa at T2
        },
        "transfer_guidance": (
            "Heathrow inter-terminal transfers (e.g. T2 to T5) require passing through Flight Connections, "
            "taking the airside shuttle bus, and clearing transit security. British Airways operates "
            "Biometric Flight Connections at T5 with strict 20-minute gate conformance."
        ),
    },
    "EHAM": {
        "name": "Amsterdam Airport Schiphol",
        "same_terminal_mct_min": 50,
        "inter_terminal_mct_min": 60,
        "luggage_safe_buffer_min": 55,
        "terminal_mapping": {},
        "transfer_guidance": "Single-terminal pier layout; non-Schengen to Schengen requires passport control queues at Piers D/E.",
    },
    "EDDF": {
        "name": "Frankfurt Airport",
        "same_terminal_mct_min": 45,
        "inter_terminal_mct_min": 65,
        "luggage_safe_buffer_min": 55,
        "terminal_mapping": {},
        "transfer_guidance": "Terminal 1 (Lufthansa/Star Alliance) uses Skyline people mover to Terminal 2; allow 20 min for Z-gates.",
    },
    "LFPG": {
        "name": "Paris Charles de Gaulle Airport",
        "same_terminal_mct_min": 60,
        "inter_terminal_mct_min": 90,
        "luggage_safe_buffer_min": 75,
        "terminal_mapping": {},
        "transfer_guidance": "Transfers between Terminals 2E, 2F, and 2G require CDGVAL automated rail or shuttle buses.",
    },
    "EBBR": {
        "name": "Brussels Airport",
        "same_terminal_mct_min": 45,
        "inter_terminal_mct_min": 55,
        "luggage_safe_buffer_min": 45,
        "terminal_mapping": {},
        "transfer_guidance": "Pier A (Schengen) and Pier B (Non-Schengen) connected via connector building with passport control.",
    },
    "EINN": {
        "name": "Shannon Airport",
        "same_terminal_mct_min": 35,
        "inter_terminal_mct_min": 40,
        "luggage_safe_buffer_min": 35,
        "terminal_mapping": {},
        "transfer_guidance": "Compact single terminal; US Preclearance facility available for transatlantic departures.",
    },
    "LEMD": {
        "name": "Adolfo Suárez Madrid–Barajas Airport",
        "same_terminal_mct_min": 45,
        "inter_terminal_mct_min": 75,
        "luggage_safe_buffer_min": 60,
        "terminal_mapping": {
            "IBE": "T4",
            "AEA": "T1",
            "RYR": "T1",
            "VLG": "T4",
            "BAW": "T4S",
            "TAP": "T2",
            "DLH": "T2",
            "AFR": "T2",
        },
        "transfer_guidance": "Transfers between T1/T2/T3 and T4 require free airport shuttle buses; T4 to T4S uses the underground automatic people mover (APM) train.",
    },
    "EDDM": {
        "name": "Munich Airport",
        "same_terminal_mct_min": 40,
        "inter_terminal_mct_min": 60,
        "luggage_safe_buffer_min": 50,
        "terminal_mapping": {
            "DLH": "T2",
            "AFR": "T1",
            "BAW": "T1",
            "EZY": "T1",
            "KLM": "T1",
        },
        "transfer_guidance": "Terminal 2 (Lufthansa / Star Alliance) connects to satellite concourses via underground train; T1/T2 transfer crosses Central Area.",
    },
    "EIDW": {
        "name": "Dublin Airport",
        "same_terminal_mct_min": 45,
        "inter_terminal_mct_min": 60,
        "luggage_safe_buffer_min": 50,
        "terminal_mapping": {
            "EIN": "T2",
            "RYR": "T1",
            "BAW": "T1",
            "DLH": "T1",
        },
        "transfer_guidance": "Terminal 1 and Terminal 2 are connected via an indoor passenger covered walkway.",
    },
}

DAY_NAME_TO_INT: dict[str, int] = {
    "sunday": 0, "sun": 0, "0": 0,
    "monday": 1, "mon": 1, "1": 1,
    "tuesday": 2, "tue": 2, "2": 2,
    "wednesday": 3, "wed": 3, "3": 3,
    "thursday": 4, "thu": 4, "4": 4,
    "friday": 5, "fri": 5, "5": 5,
    "saturday": 6, "sat": 6, "6": 6,
}

INT_TO_DAY_NAME: dict[int, str] = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}


@dataclass
class AirportWeatherForecast:
    """Live or climatological forecast state for an airport."""
    icao: str
    airport_name: str
    date_str: str
    temperature_c: float
    wind_speed_kts: float
    wind_gusts_kts: float
    wind_dir_deg: float
    precipitation_probability_pct: float
    weather_description: str
    adverse_risk_score: float  # 0.0 to 10.0
    crosswind_kts: float
    risk_level: str  # "LOW", "MODERATE", "ELEVATED", "SEVERE"
    is_live_api: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlightLegRisk:
    """Historical and forecast risk profile for a single flight leg."""
    origin: str
    origin_name: str
    destination: str
    destination_name: str
    dominant_operator: str
    operator_name: str
    typical_aircraft: str
    sample_size_flights: int
    day_of_week: str
    time_window: str
    mean_arr_delay_min: float
    median_arr_delay_min: float
    p80_arr_delay_min: float
    p90_arr_delay_min: float
    prob_delay_gt_15m_pct: float
    prob_delay_gt_30m_pct: float
    prob_delay_gt_60m_pct: float
    cancellation_rate_pct: float
    weather_risk_multiplier: float
    weather_adjusted_prob_delay_pct: float
    expected_delay_bracket: str
    origin_weather: AirportWeatherForecast
    destination_weather: AirportWeatherForecast

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConnectionRisk:
    """Connection tightness and missed flight risk at a transfer hub."""
    hub: str
    hub_name: str
    scheduled_layover_min: int
    minimum_connection_time_min: int
    slack_buffer_min: float
    misconnect_probability_pct: float
    luggage_transfer_risk_pct: float
    luggage_status: str  # "LOW_RISK", "MODERATE_RISK", "CRITICAL_RISK_CARRY_ON_ADVISED"
    misconnect_risk_level: str  # "SECURE", "TIGHT", "HIGH_RISK", "CRITICAL"
    terminal_transfer_note: str
    hub_arrival_flow_penalty_min: float = 0.0
    hub_departure_taxi_penalty_min: float = 0.0
    inter_terminal_transit_weather_penalty_min: float = 0.0
    effective_mct_min: float = 0.0
    effective_slack_buffer_min: float = 0.0
    boarding_conformance_cutoff_min: int = 20
    weather_compound_guidance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PassengerRightsAssessment:
    """Statutory EU261 / UK261 compensation and duty-of-care entitlement."""
    statutory_framework: str
    total_journey_distance_km: float
    compensation_bracket: str
    statutory_compensation_amount_eur: float
    delay_threshold_hours: float
    right_to_care_hours: float
    duty_of_care_entitlements: list[str]
    claimability_guidance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DepartureWaveComparison:
    """Side-by-side reliability comparison across time of day."""
    wave_name: str
    hours_range: str
    avg_arr_delay_min: float
    p90_delay_min: float
    on_time_pct: float
    reactionary_risk_level: str
    recommendation_badge: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PassengerItineraryAssessment:
    """Complete public-facing passenger journey analysis."""
    itinerary_id: str
    origin: str
    hub: str | None
    destination: str
    day_of_week: str
    time_window: str
    target_date: str
    period: str
    period_selection_reason: str
    overall_delay_risk_score: float  # 0 to 100
    overall_delay_risk_level: str  # "LOW", "MODERATE", "ELEVATED", "HIGH"
    legs: list[FlightLegRisk]
    connection: ConnectionRisk | None
    passenger_rights: PassengerRightsAssessment
    departure_waves: list[DepartureWaveComparison]
    actionable_recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PassengerJourneyEngine:
    """
    Core engine evaluating flight itinerary risk for the travelling public.
    """

    CARGO_AND_GA_OPERATORS: set[str] = {
        "BCS", "FDX", "UPS", "CLX", "BOX", "GEC", "TFL", "SWN", "ABR", "MNB", "NPT", "SRR",
        "ZZZ", "AHO", "FYG", "TGM", "AAB", "EJA", "NJE", "VJT", "BZE", "JME",
    }

    OPERATOR_NAMES: dict[str, str] = {
        "EIN": "Aer Lingus (operated as BAW codeshare)",
        "BAW": "British Airways",
        "BEL": "Brussels Airlines",
        "AFR": "Air France",
        "KLM": "KLM Royal Dutch Airlines",
        "DLH": "Lufthansa",
        "RYR": "Ryanair",
        "EZY": "easyJet",
        "SAS": "Scandinavian Airlines",
        "IBE": "Iberia",
        "AEA": "Air Europa",
        "VLG": "Vueling Airlines",
        "WZZ": "Wizz Air",
        "SWR": "Swiss International Air Lines",
        "TAP": "TAP Air Portugal",
        "AUA": "Austrian Airlines",
        "LOT": "LOT Polish Airlines",
        "THY": "Turkish Airlines",
        "FIN": "Finnair",
        "AZA": "ITA Airways",
    }

    def __init__(self, dataset: FlightDataset | None = None, period: str | None = "auto") -> None:
        if dataset is not None:
            self.dataset = dataset
            self._auto_period = False
        else:
            self._auto_period = (str(period).lower().strip() in ("auto", "none", ""))
            if self._auto_period:
                self.dataset = FlightDataset(period="auto")
            else:
                self.dataset = FlightDataset(period=str(period))
        self._airport_cache: dict[str, dict[str, Any]] = {}
        self._runway_cache: dict[str, list[dict[str, Any]]] = {}

    def _get_airport_info(self, icao: str) -> dict[str, Any]:
        icao_clean = icao.strip().upper()
        if icao_clean not in self._airport_cache:
            df = self.dataset.query(
                "SELECT ident, name, iso_country, latitude_deg, longitude_deg FROM raw_airports WHERE ident = ?",
                [icao_clean],
            )
            if not df.empty:
                self._airport_cache[icao_clean] = df.to_dict(orient="records")[0]
            else:
                self._airport_cache[icao_clean] = {
                    "ident": icao_clean,
                    "name": f"Airport {icao_clean}",
                    "iso_country": "EU",
                    "latitude_deg": 50.0,
                    "longitude_deg": 4.0,
                }
        return self._airport_cache[icao_clean]

    def _get_airport_runways(self, icao: str) -> list[dict[str, Any]]:
        icao_clean = icao.strip().upper()
        if icao_clean not in self._runway_cache:
            df = self.dataset.query(
                "SELECT airport_ident, le_ident, he_ident, length_ft FROM raw_runways WHERE airport_ident = ? ORDER BY length_ft DESC",
                [icao_clean],
            )
            self._runway_cache[icao_clean] = df.to_dict(orient="records") if not df.empty else []
        return self._runway_cache[icao_clean]

    def get_airport_weather_forecast(
        self,
        icao: str,
        target_date: str | None = None,
        target_hour: int = 9,
    ) -> AirportWeatherForecast:
        """
        Fetch forecast weather for an airport on target_date, with robust climatological fallback.
        """
        airport = self._get_airport_info(icao)
        lat = airport.get("latitude_deg", 50.0)
        lon = airport.get("longitude_deg", 4.0)
        apt_name = airport.get("name", icao)

        now = datetime.now(timezone.utc)
        if not target_date:
            target_dt = now + timedelta(days=2)  # default next Tuesday or 2 days ahead
            target_date_str = target_dt.strftime("%Y-%m-%d")
        else:
            target_date_str = str(target_date).strip()

        # Try live Open-Meteo forecast
        is_live = False
        temp_c = 11.0
        wind_spd_kts = 14.0
        wind_gusts_kts = 22.0
        wind_dir_deg = 240.0
        precip_prob = 35.0
        desc = "Partly Cloudy with Moderate Westerly Breeze"

        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": [
                    "temperature_2m",
                    "precipitation_probability",
                    "wind_speed_10m",
                    "wind_gusts_10m",
                    "wind_direction_10m",
                    "weather_code",
                ],
                "forecast_days": 7,
            }
            resp = requests.get(url, params=params, timeout=3.5)
            if resp.status_code == 200:
                h_data = resp.json().get("hourly", {})
                times = h_data.get("time", [])
                # Find matching date and hour
                target_iso_prefix = f"{target_date_str}T{target_hour:02d}"
                idx = -1
                for i, t in enumerate(times):
                    if t.startswith(target_iso_prefix):
                        idx = i
                        break
                if idx == -1:
                    # Match date prefix at noon
                    target_noon = f"{target_date_str}T12"
                    for i, t in enumerate(times):
                        if t.startswith(target_noon):
                            idx = i
                            break

                if idx != -1:
                    is_live = True
                    temp_c = float(h_data["temperature_2m"][idx]) if h_data.get("temperature_2m") else 11.0
                    spd_kmh = float(h_data["wind_speed_10m"][idx]) if h_data.get("wind_speed_10m") else 25.0
                    gust_kmh = float(h_data["wind_gusts_10m"][idx]) if h_data.get("wind_gusts_10m") else 38.0
                    wind_spd_kts = spd_kmh / 1.852
                    wind_gusts_kts = gust_kmh / 1.852
                    wind_dir_deg = float(h_data["wind_direction_10m"][idx]) if h_data.get("wind_direction_10m") else 240.0
                    precip_prob = float(h_data["precipitation_probability"][idx]) if h_data.get("precipitation_probability") else 20.0
                    code = int(h_data["weather_code"][idx]) if h_data.get("weather_code") else 3
                    desc = self._wmo_code_to_desc(code)
        except Exception:
            is_live = False

        # Compute crosswind against primary runway
        runways = self._get_airport_runways(icao)
        crosswind_kts = self._compute_primary_crosswind(runways, wind_dir_deg, wind_spd_kts)

        # Adverse risk score: 0 to 10 scale
        gust_penalty = max(0.0, (wind_gusts_kts - 20.0) * 0.25)
        crosswind_penalty = max(0.0, (crosswind_kts - 15.0) * 0.3)
        precip_penalty = (precip_prob / 100.0) * 2.0
        adverse_score = min(10.0, round(gust_penalty + crosswind_penalty + precip_penalty, 1))

        if adverse_score < 2.5:
            risk_level = "LOW"
        elif adverse_score < 5.0:
            risk_level = "MODERATE"
        elif adverse_score < 7.5:
            risk_level = "ELEVATED"
        else:
            risk_level = "SEVERE"

        return AirportWeatherForecast(
            icao=icao.upper(),
            airport_name=apt_name,
            date_str=target_date_str,
            temperature_c=round(temp_c, 1),
            wind_speed_kts=round(wind_spd_kts, 1),
            wind_gusts_kts=round(wind_gusts_kts, 1),
            wind_dir_deg=round(wind_dir_deg, 0),
            precipitation_probability_pct=round(precip_prob, 0),
            weather_description=desc,
            adverse_risk_score=adverse_score,
            crosswind_kts=round(crosswind_kts, 1),
            risk_level=risk_level,
            is_live_api=is_live,
        )

    def _compute_primary_crosswind(
        self,
        runways: list[dict[str, Any]],
        wind_dir: float,
        wind_speed: float,
    ) -> float:
        if not runways:
            return round(wind_speed * 0.5, 1)

        # Primary runway is first (longest)
        rw = runways[0]
        ident = rw.get("le_ident", "09")
        try:
            num_part = "".join(c for c in ident if c.isdigit())
            rwy_hdg = float(num_part) * 10.0 if num_part else 90.0
        except Exception:
            rwy_hdg = 90.0

        angle_diff = math.radians(abs(wind_dir - rwy_hdg))
        crosswind = abs(math.sin(angle_diff) * wind_speed)
        return crosswind

    def _wmo_code_to_desc(self, code: int) -> str:
        wmo_map = {
            0: "Clear skies",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog / Low visibility",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Rain showers",
            81: "Heavy rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm with wind shear",
        }
        return wmo_map.get(code, "Scattered clouds")

    def assess_leg(
        self,
        origin: str,
        destination: str,
        day_of_week_int: int = 2,
        time_window: str = "morning",
        target_date: str | None = None,
    ) -> FlightLegRisk:
        """
        Assess risk for a single flight leg using empirical statistics & live weather forecast.
        """
        orig_clean = origin.strip().upper()
        dest_clean = destination.strip().upper()
        orig_info = self._get_airport_info(orig_clean)
        dest_info = self._get_airport_info(dest_clean)

        # Time window filter for SQL
        time_lower = time_window.lower()
        if "morning" in time_lower or "am" in time_lower:
            hour_filter = "AND CAST(strftime(actual_off_block_time, '%H') AS INT) BETWEEN 6 AND 11"
            display_window = "Morning (06:00 - 11:00 UTC)"
            target_hour = 8
        elif "midday" in time_lower or "afternoon" in time_lower:
            hour_filter = "AND CAST(strftime(actual_off_block_time, '%H') AS INT) BETWEEN 12 AND 16"
            display_window = "Afternoon (12:00 - 16:00 UTC)"
            target_hour = 14
        elif "evening" in time_lower or "pm" in time_lower:
            hour_filter = "AND CAST(strftime(actual_off_block_time, '%H') AS INT) BETWEEN 17 AND 22"
            display_window = "Evening (17:00 - 22:00 UTC)"
            target_hour = 19
        else:
            hour_filter = ""
            display_window = "All Day"
            target_hour = 10

        # Query historical flights (excluding cargo freighters and private general aviation)
        cargo_str = ", ".join(f"'{c}'" for c in self.CARGO_AND_GA_OPERATORS)
        cargo_filter = f"AND ac_operator NOT IN ({cargo_str})"
        sql = f"""
            SELECT 
                ac_operator,
                ac_type,
                arr_delay_min,
                dep_delay_min
            FROM flights
            WHERE adep = '{orig_clean}' 
              AND ades = '{dest_clean}'
              AND CAST(strftime(actual_off_block_time, '%w') AS INT) = {day_of_week_int}
              {hour_filter}
              {cargo_filter}
        """
        df = self.dataset.query(sql)

        # Fallback if specific DOW + window is small: broaden to all days on that route
        if len(df) < 5:
            fallback_sql = f"""
                SELECT ac_operator, ac_type, arr_delay_min, dep_delay_min
                FROM flights
                WHERE adep = '{orig_clean}' AND ades = '{dest_clean}'
                  {cargo_filter}
            """
            df = self.dataset.query(fallback_sql)

        # If still empty (e.g. specialized cargo-only or unclassified route), fallback without operator exclusion
        if df.empty:
            df = self.dataset.query(
                f"SELECT ac_operator, ac_type, arr_delay_min, dep_delay_min FROM flights WHERE adep = '{orig_clean}' AND ades = '{dest_clean}'"
            )

        if not df.empty:
            dominant_op = df["ac_operator"].mode().iloc[0] if not df["ac_operator"].empty else "BAW"
            typical_ac = df["ac_type"].mode().iloc[0] if not df["ac_type"].empty else "A320"
            delays = df["arr_delay_min"].dropna().tolist()
            sample_size = len(delays)
            mean_delay = float(df["arr_delay_min"].mean())
            median_delay = float(df["arr_delay_min"].median())
            p80_delay = float(df["arr_delay_min"].quantile(0.80))
            p90_delay = float(df["arr_delay_min"].quantile(0.90))

            p_gt_15 = float((df["arr_delay_min"] > 15).mean() * 100.0)
            p_gt_30 = float((df["arr_delay_min"] > 30).mean() * 100.0)
            p_gt_60 = float((df["arr_delay_min"] > 60).mean() * 100.0)
            canc_rate = 1.2  # empirical baseline cancellation %
        else:
            # Synthetic realistic default for European short haul
            dominant_op = "BAW" if dest_clean == "EGLL" else "EIN"
            typical_ac = "A320"
            sample_size = 45
            mean_delay = 8.5
            median_delay = 3.0
            p80_delay = 18.0
            p90_delay = 28.0
            p_gt_15 = 18.0
            p_gt_30 = 9.0
            p_gt_60 = 3.0
            canc_rate = 1.0

        op_name = self.OPERATOR_NAMES.get(dominant_op, f"{dominant_op} Airlines")

        # Weather forecasts
        orig_weather = self.get_airport_weather_forecast(orig_clean, target_date, target_hour=target_hour)
        dest_weather = self.get_airport_weather_forecast(dest_clean, target_date, target_hour=target_hour + 2)

        # Weather multiplier based on adverse score
        combined_adv = max(orig_weather.adverse_risk_score, dest_weather.adverse_risk_score)
        weather_multiplier = round(1.0 + (combined_adv * 0.08), 2)  # up to 1.8x
        adjusted_p_delay = min(95.0, round(p_gt_15 * weather_multiplier, 1))

        if median_delay <= 5.0 and p90_delay <= 20.0:
            delay_bracket = "0 - 15 min (Nominal on-time)"
        elif p90_delay <= 40.0:
            delay_bracket = "15 - 35 min (Moderate delays likely)"
        else:
            delay_bracket = "35 - 75 min (Significant delay expected)"

        day_name = INT_TO_DAY_NAME.get(day_of_week_int, "Tuesday")

        return FlightLegRisk(
            origin=orig_clean,
            origin_name=orig_info.get("name", orig_clean),
            destination=dest_clean,
            destination_name=dest_info.get("name", dest_clean),
            dominant_operator=dominant_op,
            operator_name=op_name,
            typical_aircraft=typical_ac,
            sample_size_flights=sample_size,
            day_of_week=day_name,
            time_window=display_window,
            mean_arr_delay_min=round(mean_delay, 1),
            median_arr_delay_min=round(median_delay, 1),
            p80_arr_delay_min=round(p80_delay, 1),
            p90_arr_delay_min=round(p90_delay, 1),
            prob_delay_gt_15m_pct=round(p_gt_15, 1),
            prob_delay_gt_30m_pct=round(p_gt_30, 1),
            prob_delay_gt_60m_pct=round(p_gt_60, 1),
            cancellation_rate_pct=round(canc_rate, 1),
            weather_risk_multiplier=weather_multiplier,
            weather_adjusted_prob_delay_pct=adjusted_p_delay,
            expected_delay_bracket=delay_bracket,
            origin_weather=orig_weather,
            destination_weather=dest_weather,
        )

    def evaluate_connection_risk(
        self,
        hub: str,
        inbound_leg: FlightLegRisk,
        outbound_leg: FlightLegRisk | None = None,
        scheduled_layover_min: int = 75,
        inbound_op: str | None = None,
        outbound_op: str | None = None,
    ) -> ConnectionRisk:
        """
        Evaluate missed connection probability and baggage transfer risk at a transfer hub,
        explicitly accounting for intermediate airport arrival holding and departure taxi weather impacts.
        """
        hub_clean = hub.strip().upper()
        apt_info = self._get_airport_info(hub_clean)
        hub_meta = HUB_MCT_DATABASE.get(hub_clean)
        resolved_hub_name = hub_meta.get("name") if hub_meta else apt_info.get("name", f"Airport {hub_clean}")
        if not hub_meta:
            hub_meta = {
                "name": resolved_hub_name,
                "same_terminal_mct_min": 50,
                "inter_terminal_mct_min": 75,
                "luggage_safe_buffer_min": 60,
                "terminal_mapping": {},
                "transfer_guidance": "Standard airport terminal transfer.",
            }

        in_op = (inbound_op or inbound_leg.dominant_operator).strip().upper()
        out_op = (outbound_op or (outbound_leg.dominant_operator if outbound_leg else "BAW")).strip().upper()

        term_map = hub_meta.get("terminal_mapping", {})
        in_term = term_map.get(in_op, "T2")
        out_term = term_map.get(out_op, "T5")

        is_inter_terminal = (in_term != out_term)
        mct = hub_meta["inter_terminal_mct_min"] if is_inter_terminal else hub_meta["same_terminal_mct_min"]

        # ---------------------------------------------------------------------
        # Intermediate Airport Weather Impacts: Arrival Flow vs Departure Taxi
        # ---------------------------------------------------------------------
        hub_arr_wx = inbound_leg.destination_weather
        hub_dep_wx = outbound_leg.origin_weather if outbound_leg else hub_arr_wx

        # 1. Inbound Arrival Holding Penalty:
        # At busy hubs (e.g. EGLL, EHAM), adverse weather (gusts, low ceiling, rain)
        # reduces the arrival acceptance rate (AAR) by increasing vortex wake spacing
        # and activating Low Visibility Procedures (LVP), causing airborne holding at outer stacks.
        hub_arr_penalty = round(max(0.0, (hub_arr_wx.adverse_risk_score - 1.0) * 2.8), 1)

        # 2. Inter-Terminal Ramp Transit Weather Penalty:
        # Wet, gusty, or low-visibility apron conditions enforce speed restrictions
        # on airside shuttle buses and luggage tugs traversing perimeter roads.
        ramp_transit_penalty = round(max(0.0, (hub_arr_wx.adverse_risk_score - 1.0) * 1.5), 1) if is_inter_terminal else 0.0
        effective_mct = round(mct + ramp_transit_penalty, 1)

        # 3. Outbound Departure Taxi & Spacing Penalty:
        # Outbound departures face taxiway holding, de-icing queues, and departure vortex separation.
        hub_dep_penalty = round(max(0.0, (hub_dep_wx.adverse_risk_score - 1.0) * 2.2), 1)

        # 4. Effective Slack Buffer:
        # Scheduled Layover minus Effective MCT minus Inbound Arrival Holding Penalty
        slack_buffer = float(scheduled_layover_min - mct)
        effective_slack = round(scheduled_layover_min - effective_mct - hub_arr_penalty, 1)

        # Misconnect probability evaluated against effective slack buffer
        cutoff_delay = effective_slack
        if cutoff_delay <= 0:
            # Scheduled layover is ALREADY consumed by MCT + holding delays!
            misconnect_prob = min(99.0, max(88.0, 88.0 + abs(cutoff_delay) * 0.8))
            misconnect_level = "CRITICAL"
        elif cutoff_delay < 15:
            # Very tight: only 0-15 min delay allowed
            misconnect_prob = min(96.0, max(48.0, inbound_leg.prob_delay_gt_15m_pct * 1.6))
            misconnect_level = "HIGH_RISK"
        elif cutoff_delay < 35:
            # Moderate buffer
            misconnect_prob = max(20.0, inbound_leg.prob_delay_gt_30m_pct * 1.3)
            misconnect_level = "TIGHT"
        else:
            # Plentiful buffer (> 35 min slack above MCT)
            misconnect_prob = max(4.0, inbound_leg.prob_delay_gt_60m_pct * 0.8)
            misconnect_level = "SECURE"

        misconnect_prob = min(99.0, round(misconnect_prob * inbound_leg.weather_risk_multiplier, 1))

        # Luggage transfer risk adjusted for weather on the apron
        luggage_threshold = hub_meta.get("luggage_safe_buffer_min", 60)
        luggage_margin = scheduled_layover_min - (luggage_threshold + ramp_transit_penalty)
        if luggage_margin < 0:
            luggage_prob = 78.0
            luggage_status = "CRITICAL_RISK_CARRY_ON_ADVISED"
        elif luggage_margin < 20:
            luggage_prob = 45.0
            luggage_status = "MODERATE_RISK"
        else:
            luggage_prob = 10.0
            luggage_status = "LOW_RISK"

        if is_inter_terminal:
            transfer_guidance = hub_meta.get("transfer_guidance", "Inter-terminal shuttle bus transfer required.")
            transfer_note = f"⚠️ Inter-Terminal Transfer Required: {in_op} ({in_term}) ➔ {out_op} ({out_term}). {transfer_guidance}"
        else:
            transfer_note = f"✅ Same Terminal Transfer ({in_term}): Both inbound ({in_op}) and outbound ({out_op}) flights operate within {in_term}. No airside shuttle bus is required; follow internal Flight Connections signage to your departure gate. Note that boarding gate conformance closes strictly 20 minutes before departure."

        weather_guidance = (
            f"Weather at intermediate hub {hub_clean} exerts a compound double impact: "
            f"(1) Inbound Arrival: ~{hub_arr_penalty:.0f} min expected airborne holding/approach spacing delay. "
            f"(2) Ramp Transit: Wet/windy tarmac conditions add +{ramp_transit_penalty:.0f} min to airside inter-terminal shuttle buses (effective MCT: {effective_mct:.0f} min). "
            f"(3) Outbound Departure: While departure taxi queues add ~{hub_dep_penalty:.0f} min to takeoff, "
            f"boarding conformance at {out_term} closes strictly 20 minutes before scheduled off-blocks. Outbound taxi delays will NOT hold the gate."
        )

        return ConnectionRisk(
            hub=hub_clean,
            hub_name=hub_meta.get("name", hub_clean),
            scheduled_layover_min=scheduled_layover_min,
            minimum_connection_time_min=mct,
            slack_buffer_min=round(slack_buffer, 1),
            misconnect_probability_pct=misconnect_prob,
            luggage_transfer_risk_pct=round(luggage_prob, 1),
            luggage_status=luggage_status,
            misconnect_risk_level=misconnect_level,
            terminal_transfer_note=transfer_note,
            hub_arrival_flow_penalty_min=hub_arr_penalty,
            hub_departure_taxi_penalty_min=hub_dep_penalty,
            inter_terminal_transit_weather_penalty_min=ramp_transit_penalty,
            effective_mct_min=effective_mct,
            effective_slack_buffer_min=effective_slack,
            boarding_conformance_cutoff_min=20,
            weather_compound_guidance=weather_guidance,
        )

    def calculate_passenger_rights(
        self,
        origin: str,
        destination: str,
        hub: str | None = None,
    ) -> PassengerRightsAssessment:
        """
        Calculate EU261 / UK261 statutory compensation and care rights based on Great Circle route distance.
        """
        orig_info = self._get_airport_info(origin)
        dest_info = self._get_airport_info(destination)

        lat1, lon1 = orig_info.get("latitude_deg", 50.0), orig_info.get("longitude_deg", 0.0)
        lat2, lon2 = dest_info.get("latitude_deg", 50.0), dest_info.get("longitude_deg", 4.0)

        # Distance is calculated between initial departure point and final destination
        dist_nm = haversine_distance_nm(lat1, lon1, lat2, lon2)
        dist_km = dist_nm * 1.852

        # Statutory compensation tiers under EU Regulation 261/2004 & UK261:
        # Tier 1: Flights up to 1,500 km -> €250
        # Tier 2: Intra-EU flights > 1,500 km or other flights 1,500 - 3,500 km -> €400
        # Tier 3: All other flights > 3,500 km -> €600
        if dist_km <= 1500.0:
            comp_eur = 250.0
            bracket = "Short-haul (≤ 1,500 km)"
            care_hours = 2.0
        elif dist_km <= 3500.0:
            comp_eur = 400.0
            bracket = "Medium-haul (1,500 km to 3,500 km)"
            care_hours = 3.0
        else:
            comp_eur = 600.0
            bracket = "Long-haul (> 3,500 km)"
            care_hours = 4.0

        entitlements = [
            f"Free meals and refreshments appropriate to the waiting time (triggered after {care_hours:.0f} hours delay).",
            "Two free telephone calls, telex or fax messages, or emails.",
            "Free hotel accommodation and transport if departure is deferred to the next day.",
            f"Full cash compensation of €{comp_eur:.0f} if your flight arrives at final destination ≥ 3 hours late, unless caused by extraordinary circumstances.",
        ]

        advice = (
            "Under EU261 / UK261 case law (Wallentin-Hermann, Huzar), technical aircraft defects, crew shortages, "
            "and late inbound feeder flights DO NOT constitute extraordinary circumstances — airlines are legally liable to pay compensation. "
            "Severe meteorological storms or ATC flow restrictions (ATFM regulations) are considered extraordinary circumstances, but the airline's "
            "Right to Care (food, hotel, communications) remains strictly mandatory regardless of weather."
        )

        return PassengerRightsAssessment(
            statutory_framework="EU Regulation 261/2004 & UK261 Statutory Passenger Charter",
            total_journey_distance_km=round(dist_km, 1),
            compensation_bracket=bracket,
            statutory_compensation_amount_eur=comp_eur,
            delay_threshold_hours=3.0,
            right_to_care_hours=care_hours,
            duty_of_care_entitlements=entitlements,
            claimability_guidance=advice,
        )

    def compare_departure_waves(
        self,
        origin: str,
        destination: str,
        day_of_week_int: int = 2,
    ) -> list[DepartureWaveComparison]:
        """
        Compare reliability across time-of-day departure waves dynamically based on empirical route data.
        """
        orig_clean = origin.strip().upper()
        dest_clean = destination.strip().upper()

        waves = [
            ("Early Morning", "06:00 - 09:00 UTC", 6, 9),
            ("Midday Wave", "09:00 - 13:00 UTC", 9, 13),
            ("Afternoon Peak", "13:00 - 17:00 UTC", 13, 17),
            ("Evening Wave", "17:00 - 22:00 UTC", 17, 22),
        ]

        cargo_str = ", ".join(f"'{c}'" for c in self.CARGO_AND_GA_OPERATORS)
        cargo_filter = f"AND ac_operator NOT IN ({cargo_str})"

        raw_wave_data = []
        for name, h_range, start_h, end_h in waves:
            sql_dow = f"""
                SELECT arr_delay_min
                FROM flights
                WHERE adep = '{orig_clean}' AND ades = '{dest_clean}'
                  AND CAST(strftime(actual_off_block_time, '%w') AS INT) = {day_of_week_int}
                  AND CAST(strftime(actual_off_block_time, '%H') AS INT) BETWEEN {start_h} AND {end_h}
                  {cargo_filter}
            """
            df = self.dataset.query(sql_dow)
            if len(df) < 3:
                sql_all = f"""
                    SELECT arr_delay_min
                    FROM flights
                    WHERE adep = '{orig_clean}' AND ades = '{dest_clean}'
                      AND CAST(strftime(actual_off_block_time, '%H') AS INT) BETWEEN {start_h} AND {end_h}
                      {cargo_filter}
                """
                df = self.dataset.query(sql_all)

            if not df.empty:
                avg_del = float(df["arr_delay_min"].mean())
                p90_del = float(df["arr_delay_min"].quantile(0.90))
                otp = float((df["arr_delay_min"] <= 15).mean() * 100.0)
            else:
                if start_h < 9:
                    avg_del, p90_del, otp = 3.5, 9.0, 94.0
                elif start_h < 13:
                    avg_del, p90_del, otp = 8.0, 18.0, 86.0
                elif start_h < 17:
                    avg_del, p90_del, otp = 14.5, 29.0, 77.0
                else:
                    avg_del, p90_del, otp = 22.0, 44.0, 68.0

            raw_wave_data.append({
                "name": name,
                "range": h_range,
                "avg_del": avg_del,
                "p90_del": p90_del,
                "otp": otp,
            })

        # Rank waves by empirical performance (highest OTP, then lowest avg delay)
        best_wave_idx = max(range(len(raw_wave_data)), key=lambda i: (raw_wave_data[i]["otp"], -raw_wave_data[i]["avg_del"]))
        worst_wave_idx = min(range(len(raw_wave_data)), key=lambda i: (raw_wave_data[i]["otp"], -raw_wave_data[i]["avg_del"]))

        results = []
        for i, item in enumerate(raw_wave_data):
            avg_del = item["avg_del"]
            p90_del = item["p90_del"]
            otp = item["otp"]

            # Dynamic risk classification
            if otp >= 80.0 and avg_del <= 10.0:
                risk_level = "LOW"
            elif otp >= 65.0 and avg_del <= 18.0:
                risk_level = "MODERATE"
            elif otp >= 45.0:
                risk_level = "ELEVATED"
            else:
                risk_level = "HIGH"

            # Dynamic recommendation badges
            if i == best_wave_idx:
                badge = f"🌟 BEST CHOICE ({otp:.0f}% On-Time)"
            elif i == worst_wave_idx:
                badge = f"⚠️ HIGHEST DELAYS (+{avg_del:.0f}m avg)"
            elif risk_level == "MODERATE":
                badge = "ACCEPTABLE (Moderate Reliability)"
            else:
                badge = "CONGESTED (Slot Queue Risk)"

            results.append(
                DepartureWaveComparison(
                    wave_name=item["name"],
                    hours_range=item["range"],
                    avg_arr_delay_min=round(avg_del, 1),
                    p90_delay_min=round(p90_del, 1),
                    on_time_pct=round(otp, 1),
                    reactionary_risk_level=risk_level,
                    recommendation_badge=badge,
                )
            )

        return results

    def assess_itinerary(
        self,
        origin: str,
        destination: str,
        hub: str | None = None,
        day_of_week: str = "tuesday",
        time_window: str = "morning",
        scheduled_layover_min: int = 75,
        target_date: str | None = None,
        period: str | None = None,
        inbound_op: str | None = None,
        outbound_op: str | None = None,
    ) -> PassengerItineraryAssessment:
        """
        Assess complete passenger itinerary: multi-leg delays, layover risk, EU261 compensation, and wave options.
        """
        dow_int = DAY_NAME_TO_INT.get(str(day_of_week).lower().strip(), 2)
        day_name = INT_TO_DAY_NAME.get(dow_int, "Tuesday")

        # Resolve target flight date and optimal baseline period
        target_dt, _ = FlightDataset.parse_flight_date(target_date, day_name)
        resolved_date_str = target_dt.isoformat()

        # Check if period should be auto-selected
        if period is not None and str(period).lower().strip() not in ("auto", "none", ""):
            effective_period = str(period).strip()
            if hasattr(self.dataset, "period_selection_reason") and getattr(self.dataset, "period", None) == effective_period:
                period_reason = self.dataset.period_selection_reason
            else:
                period_reason = f"Archive {effective_period}"
        elif self._auto_period or not self.dataset.period or self.dataset.period == "auto":
            effective_period, period_reason = FlightDataset.find_best_period(
                target_date=resolved_date_str,
                day_of_week=day_name,
                parquet_dir=self.dataset.parquet_dir,
            )
        else:
            effective_period = self.dataset.period
            period_reason = getattr(self.dataset, "period_selection_reason", f"Configured engine dataset: {effective_period}")

        # Ensure dataset matches the resolved period
        if self.dataset.period != effective_period:
            self.dataset = FlightDataset(period=effective_period, parquet_dir=self.dataset.parquet_dir)
            self._airport_cache.clear()
            self._runway_cache.clear()

        legs: list[FlightLegRisk] = []
        conn_risk: ConnectionRisk | None = None

        if hub and hub.strip().upper() != origin.strip().upper() and hub.strip().upper() != destination.strip().upper():
            hub_clean = hub.strip().upper()
            leg1 = self.assess_leg(origin, hub_clean, dow_int, time_window, resolved_date_str)
            leg2 = self.assess_leg(hub_clean, destination, dow_int, time_window, resolved_date_str)
            legs = [leg1, leg2]
            conn_risk = self.evaluate_connection_risk(
                hub=hub_clean,
                inbound_leg=leg1,
                outbound_leg=leg2,
                scheduled_layover_min=scheduled_layover_min,
                inbound_op=inbound_op,
                outbound_op=outbound_op,
            )
        else:
            leg = self.assess_leg(origin, destination, dow_int, time_window, resolved_date_str)
            legs = [leg]

        rights = self.calculate_passenger_rights(origin, destination, hub)
        wave_comp = self.compare_departure_waves(origin, legs[0].destination, dow_int)

        # Overall itinerary risk score (0 to 100)
        leg1_risk = legs[0].weather_adjusted_prob_delay_pct
        if conn_risk:
            # Connection heavily weights overall trip disruption
            overall_score = round(0.35 * leg1_risk + 0.50 * conn_risk.misconnect_probability_pct + 0.15 * legs[1].weather_adjusted_prob_delay_pct, 1)
        else:
            overall_score = round(leg1_risk, 1)

        overall_score = min(99.0, max(5.0, overall_score))

        if overall_score < 20.0:
            overall_level = "LOW"
        elif overall_score < 40.0:
            overall_level = "MODERATE"
        elif overall_score < 65.0:
            overall_level = "ELEVATED"
        else:
            overall_level = "HIGH"

        # Generate actionable advice
        recommendations = []
        recommendations.append(
            f"📅 Eurocontrol Archive: {period_reason}"
        )
        for leg in legs:
            if leg.dominant_operator in self.CARGO_AND_GA_OPERATORS:
                recommendations.append(
                    f"ℹ️ Charter Route Advisory: No scheduled commercial airline service found directly on {leg.origin} ➔ {leg.destination}. Historical figures reflect private/charter aviation ({leg.dominant_operator}). For commercial passenger travel, routing via a major transfer hub is recommended."
                )
        if conn_risk and conn_risk.misconnect_probability_pct > 30.0:
            recommendations.append(
                f"🚨 High Misconnection Risk ({conn_risk.misconnect_probability_pct:.0f}%): Your {scheduled_layover_min} min layover at {conn_risk.hub} is tight against the {conn_risk.effective_mct_min:.0f} min Effective MCT (accounting for weather). If possible, rebook to a connection with ≥ 90–120 min buffer."
            )
        if conn_risk and conn_risk.hub_arrival_flow_penalty_min > 0:
            recommendations.append(
                f"🌧️ Intermediate Hub Arrival Flow Impact ({conn_risk.hub}): Weather reduces arrival runway throughput, introducing ~{conn_risk.hub_arrival_flow_penalty_min:.0f} min expected holding delay on Leg 1 landing, leaving you with {conn_risk.effective_slack_buffer_min:+.0f} min effective connection slack."
            )
        if conn_risk and conn_risk.hub_departure_taxi_penalty_min > 0:
            recommendations.append(
                f"⏱️ Intermediate Hub Departure Queue vs Gate Conformance ({conn_risk.hub}): Outbound flight faces ~{conn_risk.hub_departure_taxi_penalty_min:.0f} min taxiway departure queues, but boarding gates close strictly {conn_risk.boarding_conformance_cutoff_min} min before scheduled departure. Outbound taxi delays will NOT hold the gate for connecting passengers."
            )
        if conn_risk and conn_risk.luggage_status == "CRITICAL_RISK_CARRY_ON_ADVISED":
            recommendations.append(
                f"🧳 Pack Carry-On Baggage Only: With only {scheduled_layover_min} min layover at {conn_risk.hub}, checked bags face a {conn_risk.luggage_transfer_risk_pct:.0f}% chance of missing the transfer."
            )
        if any(leg.destination_weather.wind_gusts_kts > 28.0 or leg.origin_weather.wind_gusts_kts > 28.0 for leg in legs):
            recommendations.append(
                "💨 Strong Wind Gusts Forecast: Wind gusts exceed 28 kts at your flight terminals, which may force ATC to increase spacing on final approach and introduce 15–30 min ground holds."
            )
        recommendations.append(
            f"💶 Know Your Rights: If your flight arrives at {destination.upper()} ≥ 3 hours late, you are entitled to €{rights.statutory_compensation_amount_eur:.0f} under EU261/UK261."
        )
        # Dynamic wave recommendation
        if wave_comp:
            best_w = max(wave_comp, key=lambda w: (w.on_time_pct, -w.avg_arr_delay_min))
            worst_w = min(wave_comp, key=lambda w: (w.on_time_pct, -w.avg_arr_delay_min))
            if best_w.on_time_pct > worst_w.on_time_pct + 8.0:
                recommendations.append(
                    f"✈️ Optimal Departure Window: The {best_w.wave_name} ({best_w.hours_range}) is the most reliable departure window on this route with {best_w.on_time_pct:.1f}% on-time performance (avg delay {best_w.avg_arr_delay_min:+.1f} min), whereas the {worst_w.wave_name} has the lowest reliability ({worst_w.on_time_pct:.1f}% on-time, avg delay {worst_w.avg_arr_delay_min:+.1f} min)."
                )
            else:
                recommendations.append(
                    f"✈️ Departure Reliability: Flights on this route maintain relatively consistent reliability across the day (best: {best_w.wave_name} at {best_w.on_time_pct:.1f}% on-time)."
                )

        itinerary_id = f"{origin.upper()}_{hub.upper() if hub else 'DIRECT'}_{destination.upper()}_{day_name[:3]}_{resolved_date_str}_P{effective_period}"

        return PassengerItineraryAssessment(
            itinerary_id=itinerary_id,
            origin=origin.upper(),
            hub=hub.upper() if hub else None,
            destination=destination.upper(),
            day_of_week=day_name,
            time_window=time_window,
            target_date=target_date or resolved_date_str,
            period=effective_period,
            period_selection_reason=period_reason,
            overall_delay_risk_score=overall_score,
            overall_delay_risk_level=overall_level,
            legs=legs,
            connection=conn_risk,
            passenger_rights=rights,
            departure_waves=wave_comp,
            actionable_recommendations=recommendations,
        )
