"""
GNSS Jamming and Spoofing Network Anomaly Detection Engine (Sprint 6).

Detects commercial flights exhibiting signatures of Global Navigation Satellite System (GNSS / GPS)
jamming and spoofing across European and geopolitical electronic warfare airspace corridors:
1. Impossible Kinematic Teleportation Jumps: Groundspeed exceeding certified subsonic airframe capabilities
   (> 750 kts, > Mach 1.0, or instantaneous acceleration surges > 300 kts in < 60s).
2. Unphysical Vertical Rate Spikes: Extreme vertical speeds (|V_z| > 6,000 ft/min) or sudden altitude glitches
   (e.g., dropping to FL0 / negative while in cruise, or sudden stepped altitude level jumps).
3. Erratic Track Oscillations & Circular Spoofing: Severe angular jitter (|Δθ| > 65° track reversals in short
   intervals outside holding stacks) or false circular track loops ("circle spoofing").
4. Jumping Positions & Speed Breaches on Terminal Approach: High-speed violations (> 320 kts below FL100) or
   spatial position teleportation near airports during final approach alignment.
5. Known Geopolitical Electronic Warfare Hotspots: Baltic Sea / Kaliningrad, Eastern Mediterranean / Levant,
   Black Sea / Crimea, Middle East / Iraq / Persian Gulf, and Red Sea corridors.

Note: Telemetry dropouts are explicitly excluded as an anomaly indicator because Eurocontrol DDR2
archives reflect route waypoints and FIR boundary traversals rather than uniform 1Hz periodic ADS-B broadcasts.
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
    anomaly_type: str  # 'SPEED_TELEPORTATION', 'VERTICAL_RATE_SPIKE', 'ERRATIC_TRACK_JITTER', 'APPROACH_POSITION_JUMP', 'ALTITUDE_GLITCH'
    metric_value: float  # Numerical measurement (kts, fpm, deg)
    metric_unit: str
    severity_score: float  # 0 to 100
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GNSSAnomalySegment:
    seq_start: int
    seq_end: int
    start_time_utc: str
    end_time_utc: str
    flight_level_start: float
    flight_level_end: float
    distance_nm: float
    duration_s: float
    anomaly_type: str  # 'SPEED_TELEPORTATION', 'VERTICAL_RATE_SPIKE', 'ERRATIC_TRACK_JITTER', 'APPROACH_POSITION_JUMP', 'ALTITUDE_GLITCH'
    metric_value: float
    metric_unit: str
    severity_score: float  # 0 to 100
    description: str
    start_lat: float = 0.0
    start_lon: float = 0.0
    end_lat: float = 0.0
    end_lon: float = 0.0
    cum_dist_start_nm: float = 0.0
    cum_dist_end_nm: float = 0.0

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
    gnss_status: str  # 'CONFIRMED_SPOOFING', 'SUSPECTED_SPOOFING', 'APPROACH_JUMP', 'NOMINAL'
    anomaly_score: float  # 0 to 100 composite score
    max_groundspeed_kts: float
    max_vertical_speed_fpm: float
    max_telemetry_gap_s: float  # Maintained for telemetry inspection
    max_track_jitter_deg: float
    total_anomaly_events: int
    hotspots_traversed: list[str]
    events: list[GNSSAnomalyEvent] = field(default_factory=list)
    anomalous_segments: list[GNSSAnomalySegment] = field(default_factory=list)
    annotated_profile: list[dict[str, Any]] = field(default_factory=list)
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
    jamming_probable_count: int  # Approach anomaly count
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
    Perform deep kinematic and aerodynamic envelope anomaly evaluation on a single flight's actual radar trajectory
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
    anomalous_segments: list[GNSSAnomalySegment] = []
    annotated_profile: list[dict[str, Any]] = []

    max_speed_kts = 0.0
    max_vs_fpm = 0.0
    max_gap_s = 0.0
    max_jitter_deg = 0.0

    # Pre-calculate forward azimuths (bearings)
    bearings: list[float] = []
    for i in range(len(actual_df) - 1):
        r1 = actual_df.iloc[i]
        r2 = actual_df.iloc[i + 1]
        lat1, lon1 = float(r1["latitude"]), float(r1["longitude"])
        lat2, lon2 = float(r2["latitude"]), float(r2["longitude"])
        b = initial_bearing_deg(lat1, lon1, lat2, lon2)
        bearings.append(b)

    # Initialize first point of annotated profile
    r0 = actual_df.iloc[0]
    lat0, lon0 = float(r0["latitude"]), float(r0["longitude"])
    fl0 = float(r0["flight_level"])
    annotated_profile.append({
        "seq_num": int(r0["seq_num"]),
        "time_over": str(r0["time_over"]),
        "latitude": round(lat0, 5),
        "longitude": round(lon0, 5),
        "flight_level": round(fl0, 1),
        "cum_dist_nm": 0.0,
        "speed_kts": 0.0,
        "vs_fpm": 0.0,
        "track_angle": round(bearings[0], 1) if bearings else 0.0,
        "is_anomaly": False,
        "anomaly_type": None,
        "severity_score": 0.0,
        "description": "",
    })

    cum_dist_nm = 0.0
    prev_speed_kts: float | None = None
    has_approach_jump = False
    has_glitch = False
    n_pts = len(actual_df)

    for i in range(n_pts - 1):
        r1 = actual_df.iloc[i]
        r2 = actual_df.iloc[i + 1]
        lat1, lon1 = float(r1["latitude"]), float(r1["longitude"])
        lat2, lon2 = float(r2["latitude"]), float(r2["longitude"])
        fl1, fl2 = float(r1["flight_level"]), float(r2["flight_level"])

        t1 = pd.to_datetime(r1["time_over"])
        t2 = pd.to_datetime(r2["time_over"])
        dt_s = max(1.0, (t2 - t1).total_seconds())

        if dt_s > max_gap_s:
            max_gap_s = dt_s

        # Accounting for antimeridian traversal
        d_lon = abs(lon2 - lon1)
        if d_lon > 180.0:
            d_lon = 360.0 - d_lon

        seg_dist_nm = haversine_nm(
            lat1, lon1, lat2,
            lon1 + (lon2 - lon1 if abs(lon2 - lon1) <= 180 else (lon2 - lon1 - math.copysign(360, lon2 - lon1)))
        )

        cum_start = cum_dist_nm
        cum_end = cum_dist_nm + seg_dist_nm
        cum_dist_nm = cum_end

        # 1. Kinematic Groundspeed
        speed_kts = (seg_dist_nm / (dt_s / 3600.0))
        if speed_kts > max_speed_kts:
            max_speed_kts = speed_kts

        # 2. Vertical Speed
        vs_fpm = (abs(fl2 - fl1) * 100.0) / (dt_s / 60.0)
        if vs_fpm > max_vs_fpm:
            max_vs_fpm = vs_fpm

        # 3. Track Angle Jitter
        b_curr = bearings[i]
        angle_diff = 0.0
        if i < len(bearings) - 1:
            b_next = bearings[i + 1]
            angle_diff = abs((b_next - b_curr + 180.0) % 360.0 - 180.0)
            if angle_diff > max_jitter_deg and seg_dist_nm >= 0.5:
                max_jitter_deg = angle_diff

        # Terminal approach condition: low flight level and proximity to departure/arrival ends
        is_terminal_phase = (fl1 <= 120.0 or fl2 <= 120.0) and (i <= 4 or i >= n_pts - 6)

        segment_anomalies: list[tuple[str, float, str, float, str]] = []

        # Indicator A: Speed jumps beyond aircraft capability
        if (speed_kts > 750.0 and seg_dist_nm > 4.0) or (speed_kts > 900.0 and seg_dist_nm > 1.5):
            sev = min(100.0, 50.0 + (speed_kts - 750.0) / 20.0)
            desc = f"Unphysical groundspeed jump ({speed_kts:,.0f} kts, Δd={seg_dist_nm:.1f} nm in {dt_s:.0f}s) exceeding certified airframe Mach envelope."
            segment_anomalies.append(("SPEED_TELEPORTATION", speed_kts, "kts", sev, desc))
        elif prev_speed_kts is not None and (speed_kts - prev_speed_kts > 320.0) and dt_s < 75.0 and seg_dist_nm > 2.0:
            sev = 75.0
            desc = f"Kinematic acceleration jump: Speed surged from {prev_speed_kts:,.0f} kts to {speed_kts:,.0f} kts (ΔV={speed_kts - prev_speed_kts:,.0f} kts in {dt_s:.0f}s)."
            segment_anomalies.append(("SPEED_TELEPORTATION", speed_kts, "kts", sev, desc))

        # Indicator B: Altitude gains / losses outside possible profile
        if vs_fpm > 6000.0 and abs(fl2 - fl1) >= 20.0 and dt_s < 120.0:
            sev = min(100.0, 45.0 + (vs_fpm - 6000.0) / 120.0)
            desc = f"Unphysical vertical velocity breach ({vs_fpm:,.0f} ft/min, ΔFL={abs(fl2-fl1):.0f} in {dt_s:.0f}s) exceeding aerodynamic/structural limits."
            segment_anomalies.append(("VERTICAL_RATE_SPIKE", vs_fpm, "fpm", sev, desc))

        # Sudden Altitude Glitch (dropping to 0 or stepped teleportation)
        if fl1 >= 200.0 and fl2 <= 10.0 and dt_s < 180.0:
            has_glitch = True
            desc = f"Sudden altitude telemetry glitch dropping from FL{fl1:.0f} to FL{fl2:.0f} in {dt_s:.0f}s during cruise."
            segment_anomalies.append(("ALTITUDE_GLITCH", fl2, "FL", 90.0, desc))
        elif abs(fl2 - fl1) >= 40.0 and dt_s <= 35.0:
            has_glitch = True
            desc = f"Stepped altitude teleportation: Instant jump of ΔFL={abs(fl2-fl1):.0f} in {dt_s:.0f}s."
            segment_anomalies.append(("ALTITUDE_GLITCH", abs(fl2 - fl1), "FL", 85.0, desc))

        # Indicator C: Erratic track jitter & circular spoofing
        if angle_diff >= 65.0 and seg_dist_nm < 25.0 and dt_s < 150.0 and not is_terminal_phase:
            sev = min(95.0, 45.0 + (angle_diff - 65.0) * 1.5)
            desc = f"Erratic course reversal / track jitter ({angle_diff:.0f}° track angle flip in {dt_s:.0f}s) indicative of spoofed pseudo-range error."
            segment_anomalies.append(("ERRATIC_TRACK_JITTER", angle_diff, "deg", sev, desc))

        # Indicator D: Jumping positions on approach / terminal phase
        if is_terminal_phase:
            if speed_kts > 320.0 and (fl1 < 100.0 or fl2 < 100.0) and seg_dist_nm > 2.0:
                has_approach_jump = True
                desc = f"Terminal approach speed breach: Groundspeed {speed_kts:,.0f} kts below FL100 violates terminal envelope (ICAO 250 kts limit)."
                segment_anomalies.append(("APPROACH_POSITION_JUMP", speed_kts, "kts", 75.0, desc))
            elif seg_dist_nm > 8.0 and dt_s < 30.0:
                has_approach_jump = True
                desc = f"Spatial position jump on approach: {seg_dist_nm:.1f} nm teleportation in {dt_s:.0f}s near airport."
                segment_anomalies.append(("APPROACH_POSITION_JUMP", seg_dist_nm, "nm", 80.0, desc))
            elif angle_diff >= 65.0 and (fl1 < 80.0 or fl2 < 80.0) and dt_s < 90.0:
                has_approach_jump = True
                desc = f"Erratic track jitter on final approach ({angle_diff:.0f}° track flip below FL80)."
                segment_anomalies.append(("APPROACH_POSITION_JUMP", angle_diff, "deg", 70.0, desc))

        # Record segment anomalies and waypoint profile
        pt_is_anom = False
        pt_anom_type = None
        pt_sev = 0.0
        pt_desc = ""

        for a_type, m_val, m_unit, sev, desc in segment_anomalies:
            pt_is_anom = True
            pt_anom_type = a_type
            pt_sev = max(pt_sev, sev)
            pt_desc = desc

            events.append(
                GNSSAnomalyEvent(
                    seq_num=int(r2["seq_num"]),
                    time_utc=str(r2["time_over"]),
                    latitude=round(lat2, 4),
                    longitude=round(lon2, 4),
                    flight_level=round(fl2, 0),
                    anomaly_type=a_type,
                    metric_value=round(m_val, 1),
                    metric_unit=m_unit,
                    severity_score=round(sev, 1),
                    description=desc,
                )
            )

            anomalous_segments.append(
                GNSSAnomalySegment(
                    seq_start=int(r1["seq_num"]),
                    seq_end=int(r2["seq_num"]),
                    start_time_utc=str(r1["time_over"]),
                    end_time_utc=str(r2["time_over"]),
                    flight_level_start=round(fl1, 1),
                    flight_level_end=round(fl2, 1),
                    distance_nm=round(seg_dist_nm, 2),
                    duration_s=round(dt_s, 1),
                    anomaly_type=a_type,
                    metric_value=round(m_val, 1),
                    metric_unit=m_unit,
                    severity_score=round(sev, 1),
                    description=desc,
                    start_lat=round(lat1, 5),
                    start_lon=round(lon1, 5),
                    end_lat=round(lat2, 5),
                    end_lon=round(lon2, 5),
                    cum_dist_start_nm=round(cum_start, 2),
                    cum_dist_end_nm=round(cum_end, 2),
                )
            )

        annotated_profile.append({
            "seq_num": int(r2["seq_num"]),
            "time_over": str(r2["time_over"]),
            "latitude": round(lat2, 5),
            "longitude": round(lon2, 5),
            "flight_level": round(fl2, 1),
            "cum_dist_nm": round(cum_end, 1),
            "speed_kts": round(speed_kts, 1),
            "vs_fpm": round(vs_fpm, 0),
            "track_angle": round(b_curr, 1),
            "is_anomaly": pt_is_anom,
            "anomaly_type": pt_anom_type,
            "severity_score": round(pt_sev, 1),
            "description": pt_desc,
        })

        prev_speed_kts = speed_kts

    # Multi-Factor Composite Anomaly Scoring (0 to 100)
    # Strictly driven by physical aerodynamic and kinematic envelope breaches
    score = 0.0
    if max_speed_kts > 1200.0:
        score += 45.0
    elif max_speed_kts > 750.0:
        score += 35.0
    elif max_speed_kts > 680.0:
        score += 20.0

    if max_vs_fpm > 10000.0:
        score += 40.0
    elif max_vs_fpm > 6000.0:
        score += 25.0

    if has_glitch:
        score += 30.0

    if max_jitter_deg > 100.0:
        score += 30.0
    elif max_jitter_deg > 65.0:
        score += 20.0

    if has_approach_jump:
        score += 25.0

    if hotspots:
        score += min(15.0, len(hotspots) * 8.0)

    score = min(100.0, score)

    # Classification Status
    if score >= 65.0 and (max_speed_kts > 750.0 or max_vs_fpm > 7500.0 or has_glitch):
        status = "CONFIRMED_SPOOFING"
        verdict = f"High-confidence GNSS Spoofing: Trajectory displays physically impossible speed/altitude teleportation ({max_speed_kts:,.0f} kts, {max_vs_fpm:,.0f} fpm)."
    elif score >= 45.0 and (max_speed_kts > 680.0 or max_jitter_deg >= 65.0 or max_vs_fpm > 6000.0):
        status = "SUSPECTED_SPOOFING"
        verdict = f"Suspected GNSS Spoofing: Severe track oscillations ({max_jitter_deg:.0f}° jitter) or off-nominal kinematic telemetry."
    elif score >= 35.0 and has_approach_jump:
        status = "SUSPECTED_SPOOFING"
        verdict = "Terminal Approach Anomaly: Jumping positions or excessive speed on approach below FL100."
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
        anomalous_segments=anomalous_segments,
        annotated_profile=annotated_profile,
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
    signatures of GNSS spoofing, speed teleportation, or unphysical vertical spikes.
    Note: Position reporting gaps are excluded from candidate filtering.
    """
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
    WHERE m.max_speed_kts > 680.0 OR m.max_vs_fpm > 5500.0
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
            else:
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
