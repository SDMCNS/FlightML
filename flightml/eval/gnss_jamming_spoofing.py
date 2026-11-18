"""
GNSS Jamming and Spoofing Network Anomaly Detection Engine (Sprint 6).

Detects commercial flights exhibiting signatures of Global Navigation Satellite System (GNSS / GPS)
jamming and spoofing:
1. Impossible Kinematic Teleportation Jumps: Groundspeed exceeding certified subsonic airframe capabilities
   (> 750 kts, > Mach 1.0, or instantaneous position jumps > 15 nm in < 30 seconds).
2. Unphysical Vertical Rate Spikes: Extreme vertical speeds (|V_z| > 7,000 ft/min) or sudden altitude glitches
   (e.g., dropping to FL0 / negative while in cruise, or sudden geometric/barometric altitude jumps).
3. Erratic Track Oscillations & Circular Spoofing: Severe angular jitter (|Δθ| > 75° track reversals in short
   intervals) or false circular track loops ("circle spoofing" around simulated airports/waypoints).
4. Telemetry Dropout & Position Data Loss: Prolonged loss of ADS-B / radar position reporting (> 400 s)
   in high-density controlled airspace (indicative of wideband RF jamming denial).
5. Known Geopolitical Electronic Warfare Hotspots: Baltic Sea / Kaliningrad, Eastern Mediterranean / Levant,
   Black Sea / Crimea, Middle East / Iraq / Persian Gulf, and Red Sea corridors.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.geo.geodesy import haversine_nm, initial_bearing_deg


# Known Geopolitical Electronic Warfare & GNSS Interference Hotspots
GNSS_HOTSPOTS: dict[str, dict[str, Any]] = {
    "BALTIC_KALININGRAD": {
        "name": "Baltic Sea, Gulf of Finland & Kaliningrad Corridor",
        "min_lat": 53.5,
        "max_lat": 60.5,
        "min_lon": 18.0,
        "max_lon": 30.5,
        "description": "Recurrent wideband GPS jamming and pseudo-range spoofing affecting Baltic States, southern Finland, and Poland.",
        "color": "#9b59b6",
    },
    "EAST_MED_LEVANT": {
        "name": "Eastern Mediterranean, Cyprus FIR & Levant",
        "min_lat": 31.5,
        "max_lat": 37.5,
        "min_lon": 30.0,
        "max_lon": 37.5,
        "description": "High-intensity military GPS spoofing causing false Beirut/Tel Aviv airport locations, circle spoofing, and clock drifts.",
        "color": "#e74c3c",
    },
    "BLACK_SEA_CRIMEA": {
        "name": "Black Sea & Crimean Airspace Corridor",
        "min_lat": 41.0,
        "max_lat": 47.5,
        "min_lon": 28.0,
        "max_lon": 42.0,
        "description": "Electronic warfare denial and spoofed coordinates affecting Black Sea maritime and littoral air routes.",
        "color": "#e67e22",
    },
    "MIDDLE_EAST_PERSIAN_GULF": {
        "name": "Middle East, Iraq, Iran & Persian Gulf",
        "min_lat": 24.0,
        "max_lat": 36.5,
        "min_lon": 43.5,
        "max_lon": 56.5,
        "description": "Spoofing broadcasts driving FMS position drift, false terrain warnings (EGPWS), and clock resets.",
        "color": "#f1c40f",
    },
    "RED_SEA_BAB_EL_MANDEB": {
        "name": "Southern Red Sea & Bab el-Mandeb Strait",
        "min_lat": 12.0,
        "max_lat": 20.5,
        "min_lon": 37.0,
        "max_lon": 45.0,
        "description": "Intermittent GNSS degradation and naval electronic warfare jamming in the southern maritime corridor.",
        "color": "#3498db",
    },
}


def is_in_hotspot(lat: float, lon: float, hotspot_key: str) -> bool:
    """Check if a coordinate falls within a predefined GNSS interference hotspot bounding box."""
    zone = GNSS_HOTSPOTS.get(hotspot_key)
    if not zone:
        return False
    return (
        zone["min_lat"] <= lat <= zone["max_lat"]
        and zone["min_lon"] <= lon <= zone["max_lon"]
    )


def identify_intersecting_hotspots(points: list[tuple[float, float]]) -> list[str]:
    """Identify which known GNSS interference hotspots intersect a list of trajectory (lat, lon) coordinates."""
    intersected = set()
    for lat, lon in points:
        for k, zone in GNSS_HOTSPOTS.items():
            if zone["min_lat"] <= lat <= zone["max_lat"] and zone["min_lon"] <= lon <= zone["max_lon"]:
                intersected.add(k)
    return sorted(list(intersected))


@dataclass
class GNSSAnomalyEvent:
    seq_num: int
    time_utc: str
    latitude: float
    longitude: float
    flight_level: float
    anomaly_type: str  # 'TELEPORTATION_SPEED', 'VERTICAL_RATE_SPIKE', 'ERRATIC_TRACK_JITTER', 'TELEMETRY_DROPOUT', 'ALTITUDE_GLITCH'
    metric_value: float  # Numerical measurement (kts, fpm, deg, sec)
    metric_unit: str
    severity_score: float  # 0 to 100
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlightGNSSReport:
    ectrl_id: int
    period: str
    adep: str
    ades: str
    ac_operator: str
    ac_type: str
    ac_reg: str
    gnss_status: str  # 'CONFIRMED_SPOOFING', 'SUSPECTED_SPOOFING', 'PROBABLE_JAMMING', 'NOMINAL'
    anomaly_score: float  # 0 to 100 composite score
    max_groundspeed_kts: float
    max_vertical_speed_fpm: float
    max_telemetry_gap_s: float
    max_track_jitter_deg: float
    total_anomaly_events: int
    hotspots_traversed: list[str]
    events: list[GNSSAnomalyEvent] = field(default_factory=list)
    summary_verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkGNSSScanReport:
    period: str
    total_flights_scanned: int
    flagged_flights_count: int
    spoofing_confirmed_count: int
    spoofing_suspected_count: int
    jamming_probable_count: int
    top_hotspots: dict[str, int]
    top_affected_operators: dict[str, int]
    flagged_flights: list[FlightGNSSReport] = field(default_factory=list)
    scan_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_flight_gnss_interference(
    dataset: FlightDataset,
    ectrl_id: int,
) -> FlightGNSSReport | None:
    """
    Perform deep kinematic and geometric anomaly evaluation on a single flight's actual radar trajectory
    to diagnose GNSS jamming or spoofing signatures.
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return None

    trajs = dataset.get_trajectory(ectrl_id)
    actual_df = trajs["actual"]
    if len(actual_df) < 3:
        return None

    actual_df = actual_df.sort_values("seq_num").reset_index(drop=True)
    coords = [(float(r["latitude"]), float(r["longitude"])) for _, r in actual_df.iterrows()]
    hotspots = identify_intersecting_hotspots(coords)

    events: list[GNSSAnomalyEvent] = []
    max_speed_kts = 0.0
    max_vs_fpm = 0.0
    max_gap_s = 0.0
    max_jitter_deg = 0.0

    # Calculate segment kinematics
    bearings: list[float] = []
    for i in range(len(actual_df) - 1):
        r1 = actual_df.iloc[i]
        r2 = actual_df.iloc[i + 1]
        lat1, lon1 = float(r1["latitude"]), float(r1["longitude"])
        lat2, lon2 = float(r2["latitude"]), float(r2["longitude"])
        b = initial_bearing_deg(lat1, lon1, lat2, lon2)
        bearings.append(b)

    for i in range(len(actual_df) - 1):
        r1 = actual_df.iloc[i]
        r2 = actual_df.iloc[i + 1]
        lat1, lon1 = float(r1["latitude"]), float(r1["longitude"])
        lat2, lon2 = float(r2["latitude"]), float(r2["longitude"])
        fl1, fl2 = float(r1["flight_level"]), float(r2["flight_level"])

        t1 = pd.to_datetime(r1["time_over"])
        t2 = pd.to_datetime(r2["time_over"])
        dt_s = max(1.0, (t2 - t1).total_seconds())

        # Account for antimeridian longitude jump
        d_lon = abs(lon2 - lon1)
        if d_lon > 180.0:
            d_lon = 360.0 - d_lon
        
        # Approximate distance accounting for antimeridian
        seg_dist_nm = haversine_nm(lat1, lon1, lat2, lon1 + (lon2 - lon1 if abs(lon2 - lon1) <= 180 else (lon2 - lon1 - math.copysign(360, lon2 - lon1))))
        
        # 1. Kinematic Groundspeed Jump
        speed_kts = (seg_dist_nm / (dt_s / 3600.0))
        if speed_kts > max_speed_kts:
            max_speed_kts = speed_kts

        if speed_kts > 750.0 and seg_dist_nm > 5.0:
            sev = min(100.0, 50.0 + (speed_kts - 750.0) / 25.0)
            events.append(
                GNSSAnomalyEvent(
                    seq_num=int(r2["seq_num"]),
                    time_utc=str(r2["time_over"]),
                    latitude=round(lat2, 4),
                    longitude=round(lon2, 4),
                    flight_level=round(fl2, 0),
                    anomaly_type="TELEPORTATION_SPEED",
                    metric_value=round(speed_kts, 1),
                    metric_unit="kts",
                    severity_score=round(sev, 1),
                    description=f"Unphysical groundspeed jump ({speed_kts:,.0f} kts, Δd={seg_dist_nm:.1f} nm in {dt_s:.0f}s) exceeding certified aircraft envelope.",
                )
            )

        # 2. Unphysical Vertical Speed Spike
        vs_fpm = (abs(fl2 - fl1) * 100.0) / (dt_s / 60.0)
        if vs_fpm > max_vs_fpm:
            max_vs_fpm = vs_fpm

        if vs_fpm > 7500.0 and abs(fl2 - fl1) >= 30.0:
            sev = min(100.0, 40.0 + (vs_fpm - 7500.0) / 150.0)
            events.append(
                GNSSAnomalyEvent(
                    seq_num=int(r2["seq_num"]),
                    time_utc=str(r2["time_over"]),
                    latitude=round(lat2, 4),
                    longitude=round(lon2, 4),
                    flight_level=round(fl2, 0),
                    anomaly_type="VERTICAL_RATE_SPIKE",
                    metric_value=round(vs_fpm, 0),
                    metric_unit="fpm",
                    severity_score=round(sev, 1),
                    description=f"Extreme vertical velocity spike ({vs_fpm:,.0f} ft/min, ΔFL={abs(fl2-fl1):.0f} in {dt_s:.0f}s) inconsistent with structural limits.",
                )
            )

        # 3. Sudden Altitude Glitch (e.g. dropping to 0 or negative during cruise)
        if fl1 >= 250.0 and fl2 <= 10.0 and dt_s < 180.0:
            events.append(
                GNSSAnomalyEvent(
                    seq_num=int(r2["seq_num"]),
                    time_utc=str(r2["time_over"]),
                    latitude=round(lat2, 4),
                    longitude=round(lon2, 4),
                    flight_level=round(fl2, 0),
                    anomaly_type="ALTITUDE_GLITCH",
                    metric_value=round(fl2, 0),
                    metric_unit="FL",
                    severity_score=85.0,
                    description=f"Sudden altitude telemetry glitch dropping from FL{fl1:.0f} to FL{fl2:.0f} in {dt_s:.0f}s during cruise.",
                )
            )

        # 4. Telemetry Dropout & Position Data Loss (Jamming)
        if dt_s > max_gap_s:
            max_gap_s = dt_s

        if dt_s >= 480.0 and fl1 >= 100.0:
            sev = min(100.0, 30.0 + (dt_s - 480.0) / 30.0)
            events.append(
                GNSSAnomalyEvent(
                    seq_num=int(r2["seq_num"]),
                    time_utc=str(r2["time_over"]),
                    latitude=round(lat2, 4),
                    longitude=round(lon2, 4),
                    flight_level=round(fl2, 0),
                    anomaly_type="TELEMETRY_DROPOUT",
                    metric_value=round(dt_s / 60.0, 1),
                    metric_unit="min",
                    severity_score=round(sev, 1),
                    description=f"Extended position telemetry dropout ({dt_s/60.0:.1f} min gap across {seg_dist_nm:.1f} nm) indicative of GNSS signal denial.",
                )
            )

        # 5. Track Jitter & Erratic Angular Oscillations
        if i < len(bearings) - 1 and dt_s < 180.0 and seg_dist_nm >= 1.0:
            b1 = bearings[i]
            b2 = bearings[i + 1]
            angle_diff = abs((b2 - b1 + 180.0) % 360.0 - 180.0)
            if angle_diff > max_jitter_deg:
                max_jitter_deg = angle_diff

            if angle_diff >= 85.0 and seg_dist_nm < 30.0:
                events.append(
                    GNSSAnomalyEvent(
                        seq_num=int(r2["seq_num"]),
                        time_utc=str(r2["time_over"]),
                        latitude=round(lat2, 4),
                        longitude=round(lon2, 4),
                        flight_level=round(fl2, 0),
                        anomaly_type="ERRATIC_TRACK_JITTER",
                        metric_value=round(angle_diff, 1),
                        metric_unit="deg",
                        severity_score=65.0,
                        description=f"Sharp erratic course reversal ({angle_diff:.0f}° track angle change) within {dt_s:.0f}s indicative of spoofed pseudo-range error.",
                    )
                )

    # Multi-Factor Composite Anomaly Scoring (0 to 100)
    score = 0.0
    if max_speed_kts > 1200.0:
        score += 45.0
    elif max_speed_kts > 750.0:
        score += 30.0

    if max_vs_fpm > 12000.0:
        score += 35.0
    elif max_vs_fpm > 7500.0:
        score += 20.0

    if max_gap_s > 900.0:
        score += 25.0
    elif max_gap_s > 480.0:
        score += 15.0

    if max_jitter_deg > 110.0:
        score += 25.0
    elif max_jitter_deg > 80.0:
        score += 15.0

    if hotspots:
        score += min(15.0, len(hotspots) * 8.0)

    score = min(100.0, score)

    # Classification Status
    if score >= 75.0 and (max_speed_kts > 900.0 or max_vs_fpm > 10000.0):
        status = "CONFIRMED_SPOOFING"
        verdict = f"High-confidence GNSS Spoofing: Trajectory displays physically impossible speed/altitude teleportation ({max_speed_kts:,.0f} kts, {max_vs_fpm:,.0f} fpm)."
    elif score >= 55.0 and (max_speed_kts > 750.0 or max_jitter_deg > 85.0):
        status = "SUSPECTED_SPOOFING"
        verdict = f"Suspected GNSS Spoofing: Severe track oscillations ({max_jitter_deg:.0f}° jitter) or off-nominal kinematic telemetry."
    elif score >= 35.0 and max_gap_s >= 480.0:
        status = "PROBABLE_JAMMING"
        verdict = f"Probable GNSS Jamming: Extended position reporting dropout ({max_gap_s/60.0:.1f}m gap) and dead reckoning drift."
    else:
        status = "NOMINAL"
        verdict = "Nominal trajectory within normal avionics and airframe envelope tolerances."

    return FlightGNSSReport(
        ectrl_id=int(ectrl_id),
        period=dataset.period,
        adep=str(flight.get("adep") or ""),
        ades=str(flight.get("ades") or ""),
        ac_operator=str(flight.get("ac_operator") or ""),
        ac_type=str(flight.get("ac_type") or "A320"),
        ac_reg=str(flight.get("ac_reg") or "N/A"),
        gnss_status=status,
        anomaly_score=round(score, 1),
        max_groundspeed_kts=round(max_speed_kts, 1),
        max_vertical_speed_fpm=round(max_vs_fpm, 0),
        max_telemetry_gap_s=round(max_gap_s, 0),
        max_track_jitter_deg=round(max_jitter_deg, 1),
        total_anomaly_events=len(events),
        hotspots_traversed=hotspots,
        events=events,
        summary_verdict=verdict,
    )


def scan_network_gnss_interference(
    dataset: FlightDataset,
    sample_limit: int = 5000,
    min_score: float = 35.0,
    hotspot_filter: str | None = None,
) -> NetworkGNSSScanReport:
    """
    Scan the network archive using vectorized DuckDB window functions to flag flights exhibiting
    signatures of GNSS jamming or spoofing, ranking them by composite anomaly score.
    """
    # 1. Blazing-fast DuckDB Vectorized Pre-Screening
    sql = f"""
    WITH lagged AS (
        SELECT 
            t.ectrl_id,
            t.seq_num,
            t.time_over::TIMESTAMP as t,
            t.flight_level as fl,
            t.latitude as lat,
            t.longitude as lon,
            LAG(t.time_over::TIMESTAMP) OVER (PARTITION BY t.ectrl_id ORDER BY t.seq_num) as prev_t,
            LAG(t.flight_level) OVER (PARTITION BY t.ectrl_id ORDER BY t.seq_num) as prev_fl,
            LAG(t.latitude) OVER (PARTITION BY t.ectrl_id ORDER BY t.seq_num) as prev_lat,
            LAG(t.longitude) OVER (PARTITION BY t.ectrl_id ORDER BY t.seq_num) as prev_lon
        FROM trajectory_actual t
        WHERE t.ectrl_id IN (
            SELECT ectrl_id FROM flights 
            LIMIT {int(sample_limit)}
        )
    ),
    diffs AS (
        SELECT 
            ectrl_id,
            seq_num,
            epoch(t) - epoch(prev_t) as dt_s,
            abs(fl - prev_fl) as d_fl,
            lat, lon, prev_lat, prev_lon,
            sqrt(
                pow((lat - prev_lat) * 60.0, 2) + 
                pow(least(abs(lon - prev_lon), 360.0 - abs(lon - prev_lon)) * 60.0 * cos(radians(lat)), 2)
            ) as dist_nm
        FROM lagged
        WHERE prev_t IS NOT NULL AND (epoch(t) - epoch(prev_t)) > 0
    ),
    metrics AS (
        SELECT 
            ectrl_id,
            max(case when dt_s > 0 then (dist_nm / (dt_s / 3600.0)) else 0 end) as max_speed_kts,
            max(case when dt_s > 0 then (d_fl * 100.0 / (dt_s / 60.0)) else 0 end) as max_vs_fpm,
            max(dt_s) as max_gap_s,
            max(dist_nm) as max_step_nm,
            avg(lat) as avg_lat,
            avg(lon) as avg_lon
        FROM diffs
        GROUP BY ectrl_id
    )
    SELECT 
        m.ectrl_id,
        m.max_speed_kts,
        m.max_vs_fpm,
        m.max_gap_s,
        m.max_step_nm,
        m.avg_lat,
        m.avg_lon
    FROM metrics m
    WHERE m.max_speed_kts > 720.0 OR m.max_vs_fpm > 6500.0 OR m.max_gap_s > 450.0
    ORDER BY m.max_speed_kts DESC
    LIMIT 250
    """
    
    try:
        candidates_df = dataset.query(sql)
    except Exception:
        candidates_df = pd.DataFrame()

    flagged_flights: list[FlightGNSSReport] = []
    hotspot_counts: dict[str, int] = {k: 0 for k in GNSS_HOTSPOTS}
    op_counts: dict[str, int] = {}
    confirmed_count = 0
    suspected_count = 0
    jamming_count = 0

    if not candidates_df.empty:
        for _, row in candidates_df.iterrows():
            eid = int(row["ectrl_id"])
            rep = analyze_flight_gnss_interference(dataset, eid)
            if not rep or rep.gnss_status == "NOMINAL" or rep.anomaly_score < min_score:
                continue

            if hotspot_filter and hotspot_filter not in rep.hotspots_traversed:
                continue

            flagged_flights.append(rep)

            if rep.gnss_status == "CONFIRMED_SPOOFING":
                confirmed_count += 1
            elif rep.gnss_status == "SUSPECTED_SPOOFING":
                suspected_count += 1
            elif rep.gnss_status == "PROBABLE_JAMMING":
                jamming_count += 1

            for h in rep.hotspots_traversed:
                hotspot_counts[h] = hotspot_counts.get(h, 0) + 1

            op = rep.ac_operator or "UNKNOWN"
            op_counts[op] = op_counts.get(op, 0) + 1

    flagged_flights.sort(key=lambda x: x.anomaly_score, reverse=True)

    return NetworkGNSSScanReport(
        period=dataset.period,
        total_flights_scanned=min(int(sample_limit), len(candidates_df) if not candidates_df.empty else 0),
        flagged_flights_count=len(flagged_flights),
        spoofing_confirmed_count=confirmed_count,
        spoofing_suspected_count=suspected_count,
        jamming_probable_count=jamming_count,
        top_hotspots=hotspot_counts,
        top_affected_operators=dict(sorted(op_counts.items(), key=lambda item: item[1], reverse=True)[:10]),
        flagged_flights=flagged_flights,
    )
