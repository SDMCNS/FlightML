from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class MetarObservation:
    station_id: str
    raw_metar: str
    timestamp: datetime | None
    wind_dir_deg: float | None
    wind_speed_kts: float | None
    wind_gust_kts: float | None
    visibility_sm: float | None
    ceiling_ft: float | None  # Lowest BKN, OVC, or VV
    flight_category: str  # VFR, MVFR, IFR, LIFR, or UNKNOWN
    temp_c: float | None
    dewpoint_c: float | None
    altimeter_inhg: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "raw_metar": self.raw_metar,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "wind_dir_deg": self.wind_dir_deg,
            "wind_speed_kts": self.wind_speed_kts,
            "wind_gust_kts": self.wind_gust_kts,
            "visibility_sm": self.visibility_sm,
            "ceiling_ft": self.ceiling_ft,
            "flight_category": self.flight_category,
            "temp_c": self.temp_c,
            "dewpoint_c": self.dewpoint_c,
            "altimeter_inhg": self.altimeter_inhg,
        }


def compute_flight_category(visibility_sm: float | None, ceiling_ft: float | None) -> str:
    """
    FAA/ICAO Flight Category Determination:
    - LIFR (Low IFR): Ceiling < 500 ft and/or Visibility < 1 mile
    - IFR: Ceiling 500 to < 1,000 ft and/or Visibility 1 to < 3 miles
    - MVFR (Marginal VFR): Ceiling 1,000 to 3,000 ft and/or Visibility 3 to 5 miles
    - VFR: Ceiling > 3,000 ft and Visibility > 5 miles (no ceiling also counts as VFR)
    """
    vis = visibility_sm if visibility_sm is not None else 10.0
    ceil = ceiling_ft if ceiling_ft is not None else 99999.0

    if ceil < 500 or vis < 1.0:
        return "LIFR"
    if ceil < 1000 or vis < 3.0:
        return "IFR"
    if ceil <= 3000 or vis <= 5.0:
        return "MVFR"
    return "VFR"


def parse_metar_line(raw_text: str, default_station: str = "") -> MetarObservation:
    """
    Parse a standard METAR report string into structured observation attributes.
    """
    raw_clean = raw_text.strip()
    tokens = raw_clean.split()

    station = default_station
    timestamp = None
    wind_dir = None
    wind_speed = None
    wind_gust = None
    visibility_sm = None
    ceiling_ft = None
    temp_c = None
    dewpoint_c = None
    altimeter_inhg = None

    # Check for station
    if tokens and re.match(r"^[A-Z0-9]{4}$", tokens[0]):
        station = tokens[0]

    # Regex patterns
    wind_pattern = re.compile(r"^(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT$")
    vis_sm_pattern = re.compile(r"^(\d{1,2}(?:\s+\d/\d)?|\d/\d)SM$")
    vis_meters_pattern = re.compile(r"^(\d{4})$")
    cloud_pattern = re.compile(r"^(FEW|SCT|BKN|OVC|VV)(\d{3})(?:CB|TCU)?$")
    temp_pattern = re.compile(r"^(M?\d{2})/(M?\d{2})$")
    altimeter_a_pattern = re.compile(r"^A(\d{4})$")
    altimeter_q_pattern = re.compile(r"^Q(\d{4})$")

    ceilings = []

    for i, token in enumerate(tokens):
        # Wind
        m_wind = wind_pattern.match(token)
        if m_wind and wind_speed is None:
            dir_str, spd_str, gust_str = m_wind.groups()
            wind_dir = float(dir_str) if dir_str != "VRB" else None
            wind_speed = float(spd_str)
            wind_gust = float(gust_str) if gust_str else None
            continue

        # Visibility SM
        m_vis_sm = vis_sm_pattern.match(token)
        if m_vis_sm and visibility_sm is None:
            v_str = m_vis_sm.group(1)
            try:
                if "/" in v_str:
                    parts = v_str.split()
                    if len(parts) == 2:
                        frac_parts = parts[1].split("/")
                        visibility_sm = float(parts[0]) + float(frac_parts[0]) / float(frac_parts[1])
                    else:
                        frac_parts = parts[0].split("/")
                        visibility_sm = float(frac_parts[0]) / float(frac_parts[1])
                else:
                    visibility_sm = float(v_str)
            except Exception:
                visibility_sm = 10.0
            continue

        # Visibility in meters (European standard, e.g. 9999 = 10km+)
        m_vis_m = vis_meters_pattern.match(token)
        if m_vis_m and visibility_sm is None and i > 1:
            meters = float(m_vis_m.group(1))
            visibility_sm = meters * 0.000621371
            continue

        # Clouds & Ceiling
        m_cloud = cloud_pattern.match(token)
        if m_cloud:
            cover, height_str = m_cloud.groups()
            height_ft = float(height_str) * 100.0
            if cover in ("BKN", "OVC", "VV"):
                ceilings.append(height_ft)
            continue

        # Temperature / Dewpoint
        m_temp = temp_pattern.match(token)
        if m_temp and temp_c is None:
            t_str, d_str = m_temp.groups()
            temp_c = float(t_str.replace("M", "-"))
            dewpoint_c = float(d_str.replace("M", "-"))
            continue

        # Altimeter
        m_alt_a = altimeter_a_pattern.match(token)
        if m_alt_a:
            altimeter_inhg = float(m_alt_a.group(1)) / 100.0
            continue

        m_alt_q = altimeter_q_pattern.match(token)
        if m_alt_q:
            # QNH in hPa -> inHg
            hpa = float(m_alt_q.group(1))
            altimeter_inhg = hpa * 0.029530
            continue

    if ceilings:
        ceiling_ft = min(ceilings)

    flight_cat = compute_flight_category(visibility_sm, ceiling_ft)

    return MetarObservation(
        station_id=station,
        raw_metar=raw_clean,
        timestamp=timestamp,
        wind_dir_deg=wind_dir,
        wind_speed_kts=wind_speed,
        wind_gust_kts=wind_gust,
        visibility_sm=visibility_sm,
        ceiling_ft=ceiling_ft,
        flight_category=flight_cat,
        temp_c=temp_c,
        dewpoint_c=dewpoint_c,
        altimeter_inhg=altimeter_inhg,
    )

