"""
Unit & Integration Tests for GNSS Jamming & Spoofing Detection Suite:
- Multi-factor kinematic and aerodynamic anomaly detection (speed teleportation, vertical spikes, erratic track oscillations, approach position jumps)
- Verification that telemetry dropouts do NOT trigger false positives
- Highlighted anomalous segments and annotated vertical profiles
- Interactive Leaflet/Folium network map and flight-specific route map builders
- Typer CLI commands (`gnss-scan`, `gnss-inspect`)
- FastMCP tools
"""

from __future__ import annotations

import json
import pytest
from typer.testing import CliRunner

from flightml.cli import app as cli_app
from flightml.data.dataset import FlightDataset
from flightml.eval.gnss_jamming_spoofing import (
    GNSS_HOTSPOTS,
    FlightGNSSReport,
    GNSSAnomalyEvent,
    GNSSAnomalySegment,
    NetworkGNSSScanReport,
    analyze_flight_gnss_interference,
    scan_network_gnss_interference,
)
from flightml.mcp.server import create_mcp_server
from flightml.viz.map_builder import (
    create_gnss_interference_network_map,
    create_gnss_flight_route_map,
)


def test_gnss_hotspots_definitions():
    """Verify electronic warfare hotspot bounding boxes are well-formed."""
    assert len(GNSS_HOTSPOTS) >= 5
    for key, info in GNSS_HOTSPOTS.items():
        assert "name" in info
        assert "min_lat" in info and "max_lat" in info
        assert "min_lon" in info and "max_lon" in info
        assert info["min_lat"] < info["max_lat"]
        assert info["min_lon"] < info["max_lon"]


def test_gnss_single_flight_analysis():
    """Verify single flight GNSS anomaly analysis returns a valid FlightGNSSReport with segments and profile."""
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute("SELECT ectrl_id FROM flights LIMIT 1").fetchone()
    assert row is not None
    fid = int(row[0])

    report = analyze_flight_gnss_interference(ds, ectrl_id=fid)
    assert report is not None
    assert isinstance(report, FlightGNSSReport)
    assert report.ectrl_id == fid
    assert report.total_anomaly_events >= 0
    assert report.gnss_status in ["NOMINAL", "SUSPECTED_SPOOFING", "CONFIRMED_SPOOFING", "APPROACH_JUMP"]
    assert report.anomaly_score >= 0.0
    assert isinstance(report.events, list)
    assert isinstance(report.anomalous_segments, list)
    assert isinstance(report.annotated_profile, list)
    assert len(report.annotated_profile) >= 3

    # Verify annotated vertical profile structure
    p0 = report.annotated_profile[0]
    assert "seq_num" in p0
    assert "cum_dist_nm" in p0
    assert "flight_level" in p0
    assert "speed_kts" in p0
    assert "vs_fpm" in p0
    assert "track_angle" in p0
    assert "is_anomaly" in p0


def test_no_telemetry_dropout_false_positives():
    """Verify that reporting gaps between waypoints in DDR2 do NOT produce TELEMETRY_DROPOUT events."""
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute("SELECT ectrl_id FROM flights LIMIT 1").fetchone()
    assert row is not None
    fid = int(row[0])

    report = analyze_flight_gnss_interference(ds, ectrl_id=fid)
    assert report is not None
    for ev in report.events:
        assert ev.anomaly_type != "TELEMETRY_DROPOUT", "Telemetry dropouts should not be used as an indicator"


def test_gnss_network_scan_and_ranking():
    """Verify vectorized network scan discovers and scores flights."""
    ds = FlightDataset(period="202510")
    scan = scan_network_gnss_interference(ds, sample_limit=150, min_score=20.0)

    assert isinstance(scan, NetworkGNSSScanReport)
    assert scan.period == "202510"
    assert scan.total_flights_scanned <= 150
    assert scan.flagged_flights_count == len(scan.flagged_flights)
    assert isinstance(scan.top_hotspots, dict)
    assert isinstance(scan.top_affected_operators, dict)

    if scan.flagged_flights:
        top_flight = scan.flagged_flights[0]
        assert top_flight.anomaly_score >= 20.0
        assert len(top_flight.events) > 0


def test_gnss_network_map_rendering():
    """Verify Leaflet/Folium network map renders properly with hotspots and tracks."""
    ds = FlightDataset(period="202510")
    scan = scan_network_gnss_interference(ds, sample_limit=100, min_score=15.0)

    top_fid = scan.flagged_flights[0].ectrl_id if scan.flagged_flights else None

    # Generate Folium map
    folium_map = create_gnss_interference_network_map(
        dataset=ds,
        scan_report=scan,
        selected_ectrl_id=top_fid,
        max_flights_rendered=10,
    )
    assert folium_map is not None
    html_repr = folium_map.get_root().render()
    assert "Humanitarian OpenStreetMap" in html_repr or "openstreetmap" in html_repr.lower() or "leaflet" in html_repr.lower()
    assert "GNSS Interference" in html_repr or "Spoofing" in html_repr


def test_gnss_flight_route_map_rendering():
    """Verify dedicated flight route map renders with contrail-style multi-layer highlighted segments."""
    ds = FlightDataset(period="202510")
    row = ds.cursor.execute("SELECT ectrl_id FROM flights LIMIT 1").fetchone()
    assert row is not None
    fid = int(row[0])

    report = analyze_flight_gnss_interference(ds, ectrl_id=fid)
    assert report is not None

    flight_map = create_gnss_flight_route_map(dataset=ds, gnss_report=report)
    assert flight_map is not None
    html_repr = flight_map.get_root().render()
    assert "Humanitarian OpenStreetMap" in html_repr or "openstreetmap" in html_repr.lower() or "leaflet" in html_repr.lower()
    assert str(fid) in html_repr


def test_gnss_cli_commands():
    """Verify Typer CLI commands `gnss-scan` and `gnss-inspect`."""
    runner = CliRunner()

    # 1. Test `gnss-scan` with JSON output
    res_scan = runner.invoke(cli_app, ["gnss-scan", "--period", "202510", "--limit", "30", "--min-score", "10", "--json"])
    assert res_scan.exit_code == 0
    data = json.loads(res_scan.output)
    assert data["period"] == "202510"
    assert "flagged_flights_count" in data
    assert "flagged_flights" in data

    # 2. Test `gnss-inspect` on first flagged flight or default flight
    fid = data["flagged_flights"][0]["ectrl_id"] if data["flagged_flights"] else None
    if not fid:
        ds = FlightDataset(period="202510")
        row = ds.cursor.execute("SELECT ectrl_id FROM flights LIMIT 1").fetchone()
        fid = int(row[0])

    res_inspect = runner.invoke(cli_app, ["gnss-inspect", str(fid), "--period", "202510"])
    assert res_inspect.exit_code == 0
    assert "GNSS Anomaly Inspection" in res_inspect.output or "Flight #" in res_inspect.output


@pytest.mark.asyncio
async def test_gnss_mcp_tools():
    """Verify FastMCP server exposes `scan_gnss_jamming_and_spoofing` and `inspect_flight_gnss_interference` tools."""
    mcp = create_mcp_server()
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]

    assert "scan_gnss_jamming_and_spoofing" in tool_names
    assert "inspect_flight_gnss_interference" in tool_names


def test_terminal_proximity_50nm_filtering():
    """Verify that within 50nm of ADEP/ADES, only unrealistic altitudes and rates of climb/descent trigger anomalies."""
    import pandas as pd
    from unittest.mock import MagicMock

    ds = MagicMock(spec=FlightDataset)
    ds.period = "202402"
    ds.get_flight.return_value = {
        "ectrl_id": 999001,
        "adep": "EGLL",
        "ades": "LFPG",
        "ac_operator": "TEST",
        "ac_type": "A320",
        "ac_reg": "G-TEST",
    }
    # Mock raw_airports query returning Heathrow and Charles de Gaulle coordinates
    ds.query.return_value = pd.DataFrame([
        {"ident": "EGLL", "latitude_deg": 51.4700, "longitude_deg": -0.4543},
        {"ident": "LFPG", "latitude_deg": 49.0097, "longitude_deg": 2.5479},
    ])

    # Trajectory within 15 nm of Heathrow undergoing sharp 90-degree vectoring turn at normal climb rate
    actual_terminal_turn = pd.DataFrame([
        {"seq_num": 1, "time_over": "2024-02-15 10:00:00", "latitude": 51.4700, "longitude": -0.4543, "flight_level": 10.0},
        {"seq_num": 2, "time_over": "2024-02-15 10:02:00", "latitude": 51.5200, "longitude": -0.4543, "flight_level": 30.0},
        # Sharp 90-degree heading change while climbing normally within 15 nm of EGLL
        {"seq_num": 3, "time_over": "2024-02-15 10:04:00", "latitude": 51.5200, "longitude": -0.3000, "flight_level": 50.0},
        {"seq_num": 4, "time_over": "2024-02-15 10:06:00", "latitude": 51.5200, "longitude": -0.1500, "flight_level": 70.0},
    ])
    ds.get_trajectory.return_value = {"actual": actual_terminal_turn, "planned": pd.DataFrame()}

    report = analyze_flight_gnss_interference(ds, ectrl_id=999001)
    assert report is not None
    # Sharp turn under radar vectoring within 50nm must NOT be flagged as ERRATIC_TRACK_JITTER
    jitter_events = [e for e in report.events if e.anomaly_type == "ERRATIC_TRACK_JITTER"]
    assert len(jitter_events) == 0, "Sharp turns within 50nm of ADEP must not trigger track jitter anomalies"
    assert report.gnss_status == "NOMINAL"

    # Now test an unphysical vertical rate spike within 50nm of ADEP
    actual_terminal_spike = pd.DataFrame([
        {"seq_num": 1, "time_over": "2024-02-15 10:00:00", "latitude": 51.4700, "longitude": -0.4543, "flight_level": 10.0},
        # Extreme vertical rate spike: Climbing 8,000 ft in 30 seconds = 16,000 fpm!
        {"seq_num": 2, "time_over": "2024-02-15 10:00:30", "latitude": 51.4800, "longitude": -0.4543, "flight_level": 90.0},
        {"seq_num": 3, "time_over": "2024-02-15 10:02:00", "latitude": 51.5200, "longitude": -0.4000, "flight_level": 90.0},
    ])
    ds.get_trajectory.return_value = {"actual": actual_terminal_spike, "planned": pd.DataFrame()}
    rep_spike = analyze_flight_gnss_interference(ds, ectrl_id=999001)
    assert rep_spike is not None
    vs_events = [e for e in rep_spike.events if e.anomaly_type == "VERTICAL_RATE_SPIKE"]
    assert len(vs_events) > 0, "Unrealistic rate of climb within 50nm MUST be flagged as an anomaly"


def test_confirmation_score_threshold_75():
    """Verify that CONFIRMED_SPOOFING status strictly requires an anomaly score of 75.0 or above."""
    import pandas as pd
    from unittest.mock import MagicMock

    ds = MagicMock(spec=FlightDataset)
    ds.period = "202402"
    ds.get_flight.return_value = {
        "ectrl_id": 999002,
        "adep": "EGLL",
        "ades": "LFPG",
        "ac_operator": "TEST",
        "ac_type": "A320",
        "ac_reg": "G-TEST",
    }
    ds.query.return_value = pd.DataFrame([
        {"ident": "EGLL", "latitude_deg": 51.4700, "longitude_deg": -0.4543},
        {"ident": "LFPG", "latitude_deg": 49.0097, "longitude_deg": 2.5479},
    ])

    # Trajectory with moderate en-route track oscillation yielding score in the 50-74 range
    actual_moderate = pd.DataFrame([
        {"seq_num": 1, "time_over": "2024-02-15 10:00:00", "latitude": 55.0000, "longitude": 10.0000, "flight_level": 350.0},
        {"seq_num": 2, "time_over": "2024-02-15 10:02:00", "latitude": 55.1000, "longitude": 10.0000, "flight_level": 350.0},
        {"seq_num": 3, "time_over": "2024-02-15 10:03:00", "latitude": 55.1000, "longitude": 10.1500, "flight_level": 350.0},
        {"seq_num": 4, "time_over": "2024-02-15 10:04:00", "latitude": 55.2000, "longitude": 10.1500, "flight_level": 350.0},
    ])
    ds.get_trajectory.return_value = {"actual": actual_moderate, "planned": pd.DataFrame()}
    rep = analyze_flight_gnss_interference(ds, ectrl_id=999002)
    assert rep is not None
    if rep.anomaly_score < 75.0:
        assert rep.gnss_status != "CONFIRMED_SPOOFING", "Scores below 75 must not be marked CONFIRMED_SPOOFING"


def test_scan_all_flights_in_period():
    """Verify that sample_limit=None includes all flights in the period for assessment."""
    ds = FlightDataset(period="202510")
    total_in_period = int(ds.query("SELECT count(*) as c FROM flights").iloc[0, 0])

    scan = scan_network_gnss_interference(ds, sample_limit=None, min_score=75.0)
    assert isinstance(scan, NetworkGNSSScanReport)
    assert scan.total_flights_scanned == total_in_period, f"Expected {total_in_period} scanned, got {scan.total_flights_scanned}"
    for f in scan.flagged_flights:
        assert f.anomaly_score >= 75.0
