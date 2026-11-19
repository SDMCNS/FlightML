"""
FlightML Pan-European ECAC Surveillance Radar Resilience & Rationalization Pipeline.

Executes a sequenced, self-reinforcing evaluation of terrestrial primary and secondary
surveillance radar (PSR/SSR/Mode-S) infrastructure across European FIRs:
1. Scans our radar dataset (data/radars.json) and identifies FIRs that contain disclosed radar stations.
   FIRs without disclosed radars are excluded (as their radar stations have not been disclosed).
2. Ranks qualifying sovereign ECAC FIRs in descending order of historical flight traffic volume.
3. Sequentially evaluates permutations for each FIR, validating against real Eurocontrol traffic (100% census),
   calculating tactical non-radar separation delays, detour fuel burn, and downstream rotational knock-on cascades.
4. Cascades confirmed active radar heads from busier FIRs to provide line-of-sight border surveillance in adjacent FIRs.
5. Generates standalone HTML reports for each FIR and maintains a persistent outcomes dataset
   (data/reports/radar_resilience/ecac_radar_network_evaluations.jsonl / .parquet).
6. Compiles a comprehensive Pan-European ECAC Radar Summary Report with interactive continental Leaflet map,
   aggregated CBA figures, and cross-border synchronization metrics.
"""

from __future__ import annotations

import datetime
import html
import json
import os
from pathlib import Path
from typing import Any, Callable

import folium
import numpy as np
import pandas as pd
from shapely.geometry import Point

from flightml.data.dataset import FlightDataset
from flightml.eval.ecac_station_rationalization import (
    ECAC_FIR_REGISTRY,
    ECACStation,
    get_fir_geometry,
    load_ecac_stations,
    RADAR_ANNUAL_COST_SAVINGS_EUR,
)
from flightml.eval.radar_permutations import (
    simulate_radar_permutations,
    generate_radar_resilience_html_report,
    RadarFIRResilienceAnalysis,
)
from flightml.eval.dme_learning import RadarSelfReinforcingLearner
from flightml.viz.html_exporter import _folium_map_to_iframe

RADAR_OUTCOMES_JSONL_PATH = Path("data/reports/radar_resilience/ecac_radar_network_evaluations.jsonl")
RADAR_OUTCOMES_PARQUET_PATH = Path("data/reports/radar_resilience/ecac_radar_network_evaluations.parquet")


def get_ranked_radar_firs(
    dataset: FlightDataset,
    period: str = "202507",
    radars_path: str = "data/radars.json",
) -> list[dict[str, Any]]:
    """
    Identify and rank only those sovereign ECAC FIRs that contain disclosed radar stations
    according to our dataset (data/radars.json), sorted in descending order of historical flight traffic volume.
    FIRs without disclosed radars in the dataset are strictly excluded.
    """
    if not os.path.exists(radars_path):
        return []

    with open(radars_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])

    fir_candidates: list[dict[str, Any]] = []

    for fid, name in ECAC_FIR_REGISTRY.items():
        if fid == "ALL":
            continue
        geom = get_fir_geometry(dataset, fid)
        if not geom:
            continue
        poly, b_coords, bbox = geom

        # Count actual disclosed radars located inside this FIR polygon
        disclosed_radars: list[str] = []
        for feat in features:
            coords = feat.get("geometry", {}).get("coordinates", [0.0, 0.0])
            lon, lat = float(coords[0]), float(coords[1])
            if bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]:
                if poly.contains(Point(lon, lat)):
                    r_name = feat.get("properties", {}).get("name") or feat.get("properties", {}).get("id") or "Radar"
                    disclosed_radars.append(r_name)

        # STRICT CONSTRAINT: Only assess FIRs that contain radar stations according to our dataset
        if not disclosed_radars:
            continue

        moves = 0
        try:
            if dataset.has_fir_traversals():
                cnt_row = dataset.cursor.execute(
                    "SELECT COUNT(*) FROM flight_firs_actual WHERE fir_id = ?", [fid]
                ).fetchone()
                moves = int(cnt_row[0]) if cnt_row and cnt_row[0] is not None else 0
        except Exception:
            moves = 0

        if moves == 0:
            try:
                cnt_row = dataset.cursor.execute(
                    """
                    SELECT COUNT(*) FROM flights
                    WHERE (adep_lat BETWEEN ? AND ? AND adep_lon BETWEEN ? AND ?)
                       OR (ades_lat BETWEEN ? AND ? AND ades_lon BETWEEN ? AND ?)
                    """,
                    [bbox[0], bbox[1], bbox[2], bbox[3], bbox[0], bbox[1], bbox[2], bbox[3]],
                ).fetchone()
                moves = int(cnt_row[0]) if cnt_row and cnt_row[0] is not None else len(disclosed_radars) * 2000
            except Exception:
                moves = len(disclosed_radars) * 2000

        fir_candidates.append({
            "fir_id": fid,
            "fir_name": name,
            "traffic_moves": moves,
            "radar_count": len(disclosed_radars),
            "radar_sample_names": disclosed_radars[:5],
        })

    # Sort descending by historical flight volume
    fir_candidates.sort(key=lambda x: x["traffic_moves"], reverse=True)
    return fir_candidates


def record_radar_network_evaluation(record: dict[str, Any], path: Path = RADAR_OUTCOMES_JSONL_PATH) -> None:
    """Append a completed Radar FIR evaluation outcome to the persistent JSONL dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    try:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(RADAR_OUTCOMES_PARQUET_PATH, index=False)
    except Exception:
        pass


def load_radar_network_evaluations(path: Path = RADAR_OUTCOMES_JSONL_PATH) -> pd.DataFrame:
    """Load persistent radar network evaluation outcomes."""
    if not path.exists():
        return pd.DataFrame()
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def create_ecac_radar_network_map(
    dataset: FlightDataset,
    harmonized_network_state: dict[str, Any],
) -> folium.Map:
    """
    Build an interactive continental Folium Leaflet map of Europe displaying:
    - ECAC sovereign FIR boundaries
    - Disclosed radar stations color-coded by posture:
      Green: Active Core Radars
      Red: Standby / Rationalized Radars
    - Zero external API key dependencies (Esri World Light Gray Canvas & Humanitarian OSM).
    """
    m = folium.Map(
        location=[50.5, 10.0],
        zoom_start=5,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Esri Light Gray Canvas (Continental Default)",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri World Street Map",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; National Geographic, Esri, DeLorme, NAVTEQ",
        name="Esri Topographic",
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, Humanitarian OpenStreetMap Team",
        name="OpenStreetMap Humanitarian (HOT)",
        control=True,
    ).add_to(m)

    # 1. FIR Boundaries Layer
    fir_layer = folium.FeatureGroup(name="ECAC Sovereign FIR Boundaries", show=True)
    for fid, name in ECAC_FIR_REGISTRY.items():
        if fid == "ALL":
            continue
        geom = get_fir_geometry(dataset, fid)
        if not geom:
            continue
        _, b_coords, _ = geom
        if b_coords:
            folium.Polygon(
                locations=b_coords,
                color="#1e3a8a",
                weight=1.8,
                dash_array="5, 5",
                fill=False,
                tooltip=f"<b>{html.escape(name)}</b>",
            ).add_to(fir_layer)
    fir_layer.add_to(m)

    # 2. Harmonized Ground Radars Layer
    active_group = folium.FeatureGroup(name="Active Core Radars", show=True)
    standby_group = folium.FeatureGroup(name="Standby / Rationalized Radars", show=True)
    canopy_group = folium.FeatureGroup(name="Active Radar Service Canopy (200 NM)", show=False)

    for sid, info in harmonized_network_state.items():
        is_standby = info.get("recommended_posture") == "STANDBY"
        color = "#ef4444" if is_standby else "#10b981"
        fill_color = "#f87171" if is_standby else "#34d399"
        target_group = standby_group if is_standby else active_group

        folium.CircleMarker(
            location=[info["latitude"], info["longitude"]],
            radius=7 if not is_standby else 5,
            color=color,
            weight=2,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.9 if not is_standby else 0.7,
            tooltip=(
                f"📡 <b>{html.escape(info.get('name', sid))} [{sid}]</b><br>"
                f"Home FIR: <b>{info.get('home_fir', 'ECAC')}</b><br>"
                f"Status: <b>{info.get('recommended_posture', 'ACTIVE')}</b><br>"
                f"Coordinates: {info['latitude']:.3f}°N, {info['longitude']:.3f}°E<br>"
                f"Standard: {info.get('last_assessed_standard', 'VIEW_3')}"
            ),
        ).add_to(target_group)

        if not is_standby:
            folium.Circle(
                location=[info["latitude"], info["longitude"]],
                radius=float(info.get("nominal_range_nm", 200.0)) * 1852.0,
                color="#10b981",
                weight=1,
                fill=True,
                fill_color="#34d399",
                fill_opacity=0.03,
                tooltip=f"📡 {html.escape(info.get('name', sid))} (200 NM Service Canopy)",
            ).add_to(canopy_group)

    canopy_group.add_to(m)
    active_group.add_to(m)
    standby_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def generate_ecac_radar_summary_report(
    dataset: FlightDataset,
    period: str,
    target_standard: str,
    fir_analyses: list[RadarFIRResilienceAnalysis],
    fir_report_paths: dict[str, str],
    output_path: Path | str | None = None,
) -> Path:
    """
    Generate a Pan-European ECAC Radar Summary HTML Report with continental map,
    comparative rankings, aggregated CBA figures, and cross-border cascading metrics.
    """
    if output_path is None:
        out_dir = Path("data/reports/radar_resilience")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"ecac_radar_summary_report_{period}_{target_standard}.html"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Maintain strictly one version: prune older timestamped reports
    try:
        prefix = f"ecac_radar_summary_report_{period}_{target_standard}"
        for old_f in output_path.parent.glob(f"{prefix}*.html"):
            if old_f.resolve() != output_path.resolve():
                old_f.unlink(missing_ok=True)
    except Exception:
        pass

    # Load all continental radar infrastructure
    all_continental_stations, _ = load_ecac_stations(dataset, station_type="RADAR")

    # Match stations to their true sovereign home FIRs
    fir_polys = []
    for fid in ECAC_FIR_REGISTRY:
        if fid == "ALL":
            continue
        geom = get_fir_geometry(dataset, fid)
        if geom and geom[0]:
            fir_polys.append((fid, geom[0]))

    station_home_fir: dict[str, str] = {}
    for s in all_continental_stations:
        pt = Point(s.longitude, s.latitude)
        for fid, poly in fir_polys:
            if poly.contains(pt):
                station_home_fir[s.id] = fid
                break
        if s.id not in station_home_fir:
            station_home_fir[s.id] = getattr(s, "country", "ECAC") or "ECAC"

    # Identify decommissioned vs active stations across evaluated FIRs
    evaluated_decommissioned_ids: set[str] = set()
    evaluated_active_ids: set[str] = set()
    for a in fir_analyses:
        opt = a.optimal_permutation
        if opt:
            evaluated_decommissioned_ids.update(opt.disabled_station_ids)
            evaluated_active_ids.update(opt.active_station_ids)

    # Build comprehensive continental harmonized network state across all European radar heads
    harmonized_network_state: dict[str, Any] = {}
    for s in all_continental_stations:
        is_standby = s.id in evaluated_decommissioned_ids
        posture = "STANDBY" if is_standby else "ACTIVE"
        h_fir = station_home_fir.get(s.id, "ECAC")
        harmonized_network_state[s.id] = {
            "name": s.name,
            "home_fir": h_fir,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "elevation_ft": s.elevation_ft,
            "nominal_range_nm": getattr(s, "nominal_range_nm", 200.0),
            "last_assessed_standard": target_standard,
            "recommended_posture": posture,
        }

    m = create_ecac_radar_network_map(dataset, harmonized_network_state)
    map_iframe_html = _folium_map_to_iframe(
        m,
        height=580,
        title=f"Pan-European Radar Surveillance Network Resilience ({target_standard})",
    )

    # Aggregated KPI Metrics
    total_baseline_radars = sum(a.baseline_station_count for a in fir_analyses)
    total_optimal_active = sum(a.optimal_permutation.active_count if a.optimal_permutation else a.baseline_station_count for a in fir_analyses)
    total_decommissioned = total_baseline_radars - total_optimal_active
    total_annual_savings_eur = total_decommissioned * RADAR_ANNUAL_COST_SAVINGS_EUR
    total_traffic_evaluated = sum(a.traffic_month_total for a in fir_analyses)

    total_primary_delay_min = sum(a.traffic_simulation.total_primary_delay_min if a.traffic_simulation else 0.0 for a in fir_analyses)
    total_reactionary_delay_min = sum(a.traffic_simulation.total_reactionary_delay_min if a.traffic_simulation else 0.0 for a in fir_analyses)
    total_cancellation_risks = sum(a.traffic_simulation.cancellation_risk_count if a.traffic_simulation else 0 for a in fir_analyses)
    total_cascaded_links = sum(len(a.cascaded_adjacent_stations) for a in fir_analyses if getattr(a, "cascaded_adjacent_stations", None))

    # FIR Table Rows
    table_rows = []
    for idx, a in enumerate(fir_analyses, start=1):
        opt = a.optimal_permutation
        act_cnt = opt.active_count if opt else a.baseline_station_count
        dis_cnt = a.baseline_station_count - act_cnt
        red_pct = (dis_cnt / max(1, a.baseline_station_count)) * 100.0
        cov_pct = opt.coverage_coverable_pct if opt else 100.0
        deg_apts = opt.degraded_airports_count if opt else 0
        annual_sav = dis_cnt * RADAR_ANNUAL_COST_SAVINGS_EUR
        sim = a.traffic_simulation
        pri_d = f"+{sim.total_primary_delay_min:,.0f}m" if sim else "0m"
        react_d = f"+{sim.total_reactionary_delay_min:,.0f}m" if sim else "0m"
        canc_str = f"<span class='badge badge-danger'>{sim.cancellation_risk_count}</span>" if sim and sim.cancellation_risk_count > 0 else "<span class='badge badge-success'>0</span>"
        status_badge = '<span class="badge badge-success">Optimal</span>' if opt and opt.is_valid else '<span class="badge badge-warning">Caution</span>'

        rep_filename = fir_report_paths.get(a.fir_id, "#")
        report_link = f'<a href="{html.escape(rep_filename)}" style="color: #2563eb; font-weight: 600; text-decoration: underline;" target="_blank">View Report &rarr;</a>' if rep_filename != "#" else "N/A"

        casc_cnt = len(a.cascaded_adjacent_stations) if getattr(a, "cascaded_adjacent_stations", None) else 0
        casc_str = f"<span style='color: #0284c7; font-weight: 700;'>+{casc_cnt}</span>" if casc_cnt > 0 else "<span style='color: #94a3b8;'>0</span>"

        table_rows.append(
            f'<tr>\n'
            f'  <td style="text-align: center; font-weight: 700;">#{idx}</td>\n'
            f'  <td><b>{html.escape(a.fir_name)}</b> (<code>{a.fir_id}</code>)</td>\n'
            f'  <td style="text-align: right;">{a.traffic_month_total:,}</td>\n'
            f'  <td style="text-align: right;"><b>{act_cnt}</b> / {a.baseline_station_count}</td>\n'
            f'  <td style="text-align: center;">{casc_str}</td>\n'
            f'  <td style="text-align: right; color: #16a34a; font-weight: 700;">{red_pct:.1f}%</td>\n'
            f'  <td style="text-align: right;">{cov_pct:.1f}%</td>\n'
            f'  <td style="text-align: center;">{"<span style=\'color: #dc2626; font-weight: 700;\'>" + str(deg_apts) + "</span>" if deg_apts > 0 else "<span style=\'color: #16a34a;\'>0</span>"}</td>\n'
            f'  <td style="text-align: right; color: #16a34a; font-weight: 700;">€{annual_sav:,.0f} / yr</td>\n'
            f'  <td style="text-align: right;">{pri_d}</td>\n'
            f'  <td style="text-align: right;">{react_d}</td>\n'
            f'  <td style="text-align: center;">{canc_str}</td>\n'
            f'  <td style="text-align: center;">{status_badge}</td>\n'
            f'  <td style="text-align: center;">{report_link}</td>\n'
            f'</tr>'
        )

    firs_table_html = "\n".join(table_rows)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="referrer" content="no-referrer-when-downgrade">
  <title>Pan-European ECAC Radar Network Resilience &amp; Rationalization Summary — {period}</title>
  <style>
    :root {{
      --primary: #0f172a;
      --accent: #2563eb;
      --success: #16a34a;
      --warning: #d97706;
      --danger: #dc2626;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text: #1e293b;
      --text-muted: #64748b;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.55;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    .header-card {{
      background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
      color: white;
      border-radius: 12px;
      padding: 28px 32px;
      margin-bottom: 24px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    }}
    .header-card h1 {{
      margin: 0 0 8px 0;
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.5px;
    }}
    .header-card .subtitle {{
      color: #93c5fd;
      font-size: 15px;
      margin-bottom: 16px;
    }}
    .header-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      font-size: 13px;
      color: #e0e7ff;
    }}
    .header-meta span {{
      background: rgba(255,255,255,0.12);
      padding: 4px 12px;
      border-radius: 6px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }}
    .kpi-label {{
      font-size: 11.5px;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--text-muted);
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-size: 26px;
      font-weight: 800;
      color: var(--primary);
      margin-bottom: 4px;
    }}
    .kpi-desc {{
      font-size: 12px;
      color: var(--text-muted);
    }}
    .section-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }}
    .section-card h2 {{
      margin: 0 0 16px 0;
      font-size: 18px;
      font-weight: 700;
      color: var(--primary);
      border-bottom: 2px solid var(--border);
      padding-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 12px;
    }}
    th {{
      background-color: #f1f5f9;
      color: #334155;
      text-align: left;
      padding: 10px 12px;
      font-weight: 700;
      border-bottom: 2px solid var(--border);
    }}
    td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      vertical-align: middle;
    }}
    tr:hover {{
      background-color: #f8fafc;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .badge-success {{ background-color: #dcfce7; color: #15803d; }}
    .badge-warning {{ background-color: #fef3c7; color: #b45309; }}
    .badge-danger {{ background-color: #fee2e2; color: #b91c1c; }}
  </style>
</head>
<body>
<div class="container">

  <!-- Header Card -->
  <div class="header-card">
    <h1>🇪🇺 Pan-European ECAC Radar Network Resilience &amp; Rationalization Summary</h1>
    <div class="subtitle">Sequenced Multi-FIR Continental Assessment across Disclosed Radar Stations with Cross-Border Cascading</div>
    <div class="header-meta">
      <span><b>Data Cycle:</b> {period}</span>
      <span><b>Surveillance Standard:</b> <code>{target_standard}</code></span>
      <span><b>FIRs Assessed:</b> {len(fir_analyses)} sovereign FIRs with disclosed radar stations</span>
      <span><b>Total Monthly Traffic:</b> {total_traffic_evaluated:,} flights</span>
      <span><b>Generated:</b> {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}</span>
    </div>
  </div>

  <!-- Continental Executive KPI Grid -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Aggregated Annual OPEX Savings</div>
      <div class="kpi-value" style="color: #16a34a;">€{total_annual_savings_eur/1e6:,.2f}M / yr</div>
      <div class="kpi-desc">Across {total_decommissioned} decommissioned/standby radar heads (€85k/yr each)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Continental Equipment Reduction</div>
      <div class="kpi-value">{round((total_decommissioned / max(1, total_baseline_radars))*100, 1)}%</div>
      <div class="kpi-desc">{total_optimal_active:,} active core radars of {total_baseline_radars:,} baseline stations</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Cross-Border Active Cascades</div>
      <div class="kpi-value" style="color: #0284c7;">{total_cascaded_links:,}</div>
      <div class="kpi-desc">Active radar heads compensating surveillance across neighbor borders</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Downstream Reactionary Delay</div>
      <div class="kpi-value" style="color: #d97706;">+{total_reactionary_delay_min:,.0f} min</div>
      <div class="kpi-desc">Primary: +{total_primary_delay_min:,.0f} min across entire continental network</div>
    </div>
  </div>

  <!-- Interactive Map Section -->
  <div class="section-card">
    <h2>🗺️ Interactive Pan-European Harmonized Radar Resilience Map</h2>
    <p style="font-size: 13.5px; color: #475569; margin-top: 0;">
      Interactive Leaflet OpenStreetMap displaying all evaluated sovereign ECAC FIR boundaries and disclosed radar stations.
      Green beacons indicate retained active core stations; red beacons represent synchronized standby/rationalized stations.
    </p>
    {map_iframe_html}
  </div>

  <!-- FIR Comparative Ranking Table -->
  <div class="section-card">
    <h2>📊 Sovereign ECAC FIRs Ranked by Traffic Volume &amp; Radar Rationalization Outcomes</h2>
    <p style="font-size: 13.5px; color: #475569;">
      Evaluated in descending order of historical flight traffic density. Only FIRs that contain radar stations according to our dataset are assessed:
    </p>
    <div style="overflow-x: auto;">
      <table>
        <thead>
          <tr>
            <th style="text-align: center;">Rank</th>
            <th>FIR Name &amp; Code</th>
            <th style="text-align: right;">Monthly Traffic</th>
            <th style="text-align: right;">Active / Base Radars</th>
            <th style="text-align: center;">Cascaded Infill</th>
            <th style="text-align: right;">Cut %</th>
            <th style="text-align: right;">Coverage %</th>
            <th style="text-align: center;">Degraded Apt</th>
            <th style="text-align: right;">Annual OPEX Savings</th>
            <th style="text-align: right;">Primary Delay</th>
            <th style="text-align: right;">Reactionary Delay</th>
            <th style="text-align: center;">Canc. Risk</th>
            <th style="text-align: center;">Status</th>
            <th style="text-align: center;">FIR Report</th>
          </tr>
        </thead>
        <tbody>
          {firs_table_html}
        </tbody>
      </table>
    </div>
  </div>

  <div style="text-align: center; font-size: 12px; color: #94a3b8; margin-top: 32px;">
    FlightML Aeronautical Intelligence Engine • Generated automatically from Eurocontrol historical trajectory logs and ICAO Doc 9684 standards.
  </div>

</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    return output_path


def run_ecac_radar_pipeline(
    dataset: FlightDataset,
    period: str = "202507",
    target_standard: str = "VIEW_3",
    max_firs: int | None = None,
    max_permutations_per_fir: int = 5,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """
    Execute the sequenced Pan-European ECAC Radar resilience evaluation pipeline:
    1. Scans our dataset for FIRs containing disclosed radar stations.
    2. Ranks qualifying FIRs in descending order of traffic moves.
    3. Runs radar resilience simulation for each FIR, cascading active neighbor radars.
    4. Exports individual FIR reports and logs outcomes to the persistent dataset.
    5. Generates the unified Pan-European ECAC Radar Summary Report.
    """
    ranked_firs = get_ranked_radar_firs(dataset, period=period)
    if max_firs is not None and max_firs > 0:
        ranked_firs = ranked_firs[:max_firs]

    total_firs = len(ranked_firs)
    fir_analyses: list[RadarFIRResilienceAnalysis] = []
    fir_report_paths: dict[str, str] = {}
    cascaded_active_station_ids: set[str] = set()

    for i, fir_item in enumerate(ranked_firs, start=1):
        fid = fir_item["fir_id"]
        fname = fir_item["fir_name"]
        if progress_callback:
            progress_callback(i, total_firs, f"{fname} [{fid}]")

        try:
            analysis = simulate_radar_permutations(
                dataset=dataset,
                fir_id=fid,
                period=period,
                target_standard=target_standard,
                max_permutations=max_permutations_per_fir,
                external_active_station_ids=cascaded_active_station_ids,
                enable_cross_border_cascading=True,
            )
            # Add confirmed active radar heads from this FIR into the cascaded pool
            opt = analysis.optimal_permutation
            if opt:
                cascaded_active_station_ids.update(opt.active_station_ids)
            elif analysis.permutations:
                cascaded_active_station_ids.update(analysis.permutations[0].active_station_ids)

            rep_path = generate_radar_resilience_html_report(analysis, dataset=dataset)
            fir_analyses.append(analysis)
            fir_report_paths[fid] = rep_path.name

            # Log to persistent dataset
            sim = analysis.traffic_simulation
            act_cnt = opt.active_count if opt else analysis.baseline_station_count
            dis_cnt = analysis.baseline_station_count - act_cnt
            red_pct = round((dis_cnt / max(1, analysis.baseline_station_count)) * 100.0, 1)

            rec = {
                "period": period,
                "target_standard": target_standard,
                "fir_id": fid,
                "fir_name": fname,
                "rank_by_traffic": i,
                "traffic_monthly_moves": fir_item["traffic_moves"],
                "baseline_radars": analysis.baseline_station_count,
                "optimal_active_radars": act_cnt,
                "decommissioned_radars": dis_cnt,
                "equipment_reduction_pct": red_pct,
                "optimal_active_station_ids": list(opt.active_station_ids) if opt else [],
                "decommissioned_station_ids": list(opt.disabled_station_ids) if opt else [],
                "all_station_ids": [u.station_id for u in analysis.station_utilization] if getattr(analysis, "station_utilization", None) else [],
                "cascaded_neighbor_radars_count": len(analysis.cascaded_adjacent_stations) if analysis.cascaded_adjacent_stations else 0,
                "coverage_compliance_pct": opt.coverage_coverable_pct if opt else 100.0,
                "degraded_airports_count": opt.degraded_airports_count if opt else 0,
                "daily_net_savings_eur": abs(opt.windows["24h"].net_economic_impact_eur) if opt and opt.windows["24h"].net_economic_impact_eur < 0 else 0.0,
                "annual_savings_eur": dis_cnt * RADAR_ANNUAL_COST_SAVINGS_EUR,
                "flights_simulated": sim.total_flights_simulated if sim else 0,
                "flights_directly_impacted": sim.directly_impacted_flights if sim else 0,
                "primary_delay_minutes": sim.total_primary_delay_min if sim else 0.0,
                "detour_fuel_kg": sim.total_extra_fuel_kg if sim else 0.0,
                "detour_co2_kg": sim.total_extra_co2_kg if sim else 0.0,
                "downstream_flights_delayed": sim.downstream_flights_delayed if sim else 0,
                "reactionary_delay_minutes": sim.total_reactionary_delay_min if sim else 0.0,
                "reactionary_multiplier": sim.reactionary_multiplier if sim else 0.0,
                "cancellation_risks_count": sim.cancellation_risk_count if sim else 0,
                "curfew_breaches_count": 0,
                "crew_fdp_timeouts_count": sim.crew_fdp_timeouts_count if sim else 0,
                "compensated_cells_count": (analysis.cns_joint_summary.get("regimes", {}).get("surveillance_compensated", {}).get("cell_count", 0)) if getattr(analysis, "cns_joint_summary", None) else 0,
                "compound_deficit_cells_count": (analysis.cns_joint_summary.get("regimes", {}).get("compound_deficit", {}).get("cell_count", 0)) if getattr(analysis, "cns_joint_summary", None) else 0,
                "traffic_compensated_daily": (analysis.cns_joint_summary.get("regimes", {}).get("surveillance_compensated", {}).get("daily_flights", 0.0)) if getattr(analysis, "cns_joint_summary", None) else 0.0,
                "fir_report_path": rep_path.name,
                "evaluated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            try:
                record_radar_network_evaluation(rec)
            except Exception:
                pass

            # Persist structured FIR configuration dataset for experimentation & dashboard loading
            try:
                cfg_dir = Path("data/reports/radar_resilience/configurations")
                cfg_dir.mkdir(parents=True, exist_ok=True)
                cfg_file = cfg_dir / f"{fid}_{target_standard}_{period}.json"
                cfg_data = {
                    "fir_id": fid,
                    "fir_name": fname,
                    "period": period,
                    "system_type": "RADAR",
                    "target_standard": target_standard,
                    "baseline_station_count": analysis.baseline_station_count,
                    "optimal_permutation_id": opt.permutation_id if opt else None,
                    "optimal_permutation_name": opt.name if opt else None,
                    "optimal_active_station_ids": list(opt.active_station_ids) if opt else [],
                    "decommissioned_station_ids": list(opt.disabled_station_ids) if opt else [],
                    "all_stations": [
                        {
                            "id": u.station_id,
                            "name": u.station_name,
                            "role": u.airspace_role,
                            "lat": u.latitude,
                            "lon": u.longitude,
                            "elevation_ft": u.elevation_ft,
                            "is_active_in_optimal": (u.station_id in opt.active_station_ids) if opt else True,
                        }
                        for u in (analysis.station_utilization or [])
                    ],
                    "coverage_compliance_pct": opt.coverage_coverable_pct if opt else 100.0,
                    "degraded_airports_count": opt.degraded_airports_count if opt else 0,
                    "degraded_airports_idents": opt.degraded_airports_idents if opt else [],
                    "annual_savings_eur": dis_cnt * RADAR_ANNUAL_COST_SAVINGS_EUR,
                    "daily_net_savings_eur": abs(opt.windows["24h"].net_economic_impact_eur) if opt and opt.windows["24h"].net_economic_impact_eur < 0 else 0.0,
                    "critical_spofs": analysis.critical_stations or [],
                }
                with open(cfg_file, "w", encoding="utf-8") as cf:
                    json.dump(cfg_data, cf, indent=2)
            except Exception:
                pass
        except Exception as ex:
            import traceback
            traceback.print_exc()

    summary_report_path = generate_ecac_radar_summary_report(
        dataset=dataset,
        period=period,
        target_standard=target_standard,
        fir_analyses=fir_analyses,
        fir_report_paths=fir_report_paths,
    )

    return {
        "period": period,
        "target_standard": target_standard,
        "total_firs_evaluated": len(fir_analyses),
        "summary_report_path": summary_report_path,
        "fir_report_paths": fir_report_paths,
        "fir_analyses": fir_analyses,
    }
