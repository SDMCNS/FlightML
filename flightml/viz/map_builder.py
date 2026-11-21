from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any
import folium
import pandas as pd

from flightml.geo.geodesy import (
    circular_mean_longitude,
    circular_mean_coords,
    split_polyline_at_antimeridian,
)
from flightml.geo.trajectory import analyze_trajectory


def _is_finite_coord(val: Any) -> bool:
    """Check if a coordinate or numeric value is non-null and a finite float."""
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def create_safe_polyline(locations: Any, **kwargs: Any) -> folium.PolyLine:
    """
    Create a folium.PolyLine that automatically splits trajectory points at the
    International Date Line (antimeridian) and polar convergence limits to prevent
    erroneous cross-globe horizontal polyline deformation in Leaflet/Folium.
    """
    clean_pts: list[list[float]] = []
    for p in locations:
        if isinstance(p, (list, tuple)) and len(p) >= 2 and _is_finite_coord(p[0]) and _is_finite_coord(p[1]):
            clean_pts.append([float(p[0]), float(p[1])])
    segments = split_polyline_at_antimeridian(clean_pts)
    if not segments:
        return folium.PolyLine(locations=[], **kwargs)
    loc_param = segments if len(segments) > 1 else segments[0]
    return folium.PolyLine(locations=loc_param, **kwargs)


def create_trajectory_map(
    flight_info: dict[str, Any],
    actual_df: pd.DataFrame,
    planned_df: pd.DataFrame,
    output_html_path: Path | str | None = None,
    active_navaids: list[Any] | None = None,
    show_navaids: bool = True,
    contrail_segments: list[Any] | None = None,
    show_contrails: bool = True,
    ideal_df: pd.DataFrame | None = None,
) -> folium.Map:
    """
    Generate an interactive Leaflet/Folium map visualizing:
    - Planned 4D flight path (dashed blue)
    - Actual radar trajectory (solid purple)
    - Detected airborne holding stacks (orange/red circles)
    - Likely used radio navaids along the trajectory as color-coded dots (DME, VOR, ILS/LOC, NDB)
    - Departure (ADEP) & Destination (ADES) airport pins
    - Zero-API-key, completely unblocked open tile layers (OSM France HOT, OpenTopoMap, USGS Topo)
    """
    ectrl_id = flight_info.get("ectrl_id", 0)
    adep = flight_info.get("adep", "ADEP")
    ades = flight_info.get("ades", "ADES")
    op = flight_info.get("ac_operator", "OP")
    ac = flight_info.get("ac_type", "AC")

    # Center map on mid-point
    center_lat, center_lon = 50.0, 10.0
    if not actual_df.empty and "latitude" in actual_df.columns and "longitude" in actual_df.columns:
        valid_act = actual_df.dropna(subset=["latitude", "longitude"])
        if not valid_act.empty:
            c_lat = float(valid_act["latitude"].mean())
            c_lon = circular_mean_longitude(valid_act["longitude"].tolist())
            if _is_finite_coord(c_lat) and _is_finite_coord(c_lon):
                center_lat, center_lon = c_lat, c_lon
    elif not planned_df.empty and "latitude" in planned_df.columns and "longitude" in planned_df.columns:
        valid_plan = planned_df.dropna(subset=["latitude", "longitude"])
        if not valid_plan.empty:
            c_lat = float(valid_plan["latitude"].mean())
            c_lon = circular_mean_longitude(valid_plan["longitude"].tolist())
            if _is_finite_coord(c_lat) and _is_finite_coord(c_lon):
                center_lat, center_lon = c_lat, c_lon

    # Initialize map with no default tile layer to prevent API key or blocked tile warnings
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles=None,
    )

    # 1. Primary Basemap: Humanitarian OpenStreetMap Team (HOT) hosted by OSM France
    # Completely unblocked for local files (file://) and applications, zero API keys, no referer enforcement
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, Tiles style by Humanitarian OpenStreetMap Team hosted by OpenStreetMap France",
        name="Humanitarian (Unblocked)",
        subdomains="abc",
        show=True,
        control=True,
    ).add_to(m)

    # 2. Standard OpenStreetMap France mirror (unblocked)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap France & OpenStreetMap contributors",
        name="OSM France (Unblocked)",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    # 3. USGS Topographic Map (US Government Public Domain, zero restrictions, unblocked)
    folium.TileLayer(
        tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
        attr="USGS The National Map (Public Domain)",
        name="USGS Topo (Unblocked)",
        show=False,
        control=True,
    ).add_to(m)

    # 4. OpenTopoMap (Unblocked)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap",
        name="OpenTopoMap",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    metrics = analyze_trajectory(actual_df, planned_df)

    # Add planned trajectory line
    if not planned_df.empty:
        planned_coords = list(zip(planned_df["latitude"], planned_df["longitude"]))
        create_safe_polyline(
            planned_coords,
            color="#2980b9",
            weight=3,
            dash_array="6, 6",
            opacity=0.85,
            tooltip=f"Filed Flight Plan ({len(planned_df)} waypoints)",
        ).add_to(m)

    # Add actual trajectory line
    if not actual_df.empty:
        actual_coords = list(zip(actual_df["latitude"], actual_df["longitude"]))
        create_safe_polyline(
            actual_coords,
            color="#8e44ad",
            weight=4,
            opacity=0.95,
            tooltip=f"Actual Flown Track ({metrics.actual_distance_nm:.1f} nm)",
        ).add_to(m)

    # Add ideal trajectory line (ATC Digital Twin)
    if ideal_df is not None and not ideal_df.empty:
        ideal_coords = list(zip(ideal_df["latitude"], ideal_df["longitude"]))
        create_safe_polyline(
            ideal_coords,
            color="#10b981",
            weight=4,
            opacity=0.95,
            tooltip=f"✨ ATC Digital Twin: Ideal Trajectory ({len(ideal_df)} waypoints, 0 min holding)",
        ).add_to(m)

        # Mark ADEP
        first_pt = actual_df.iloc[0]
        folium.Marker(
            [first_pt["latitude"], first_pt["longitude"]],
            tooltip=f"Departure: {adep} (Off-Block Delay: {flight_info.get('dep_delay_min', 0)} min)",
            icon=folium.Icon(color="green", icon="plane-departure", prefix="fa"),
        ).add_to(m)

        # Mark ADES
        last_pt = actual_df.iloc[-1]
        folium.Marker(
            [last_pt["latitude"], last_pt["longitude"]],
            tooltip=f"Destination: {ades} (Arrival Delay: {flight_info.get('arr_delay_min', 0)} min)",
            icon=folium.Icon(color="red", icon="plane-arrival", prefix="fa"),
        ).add_to(m)

    # Add holding stack markers
    for idx, h in enumerate(metrics.holding_events):
        folium.Circle(
            location=[h.center_lat, h.center_lon],
            radius=12000,  # ~6.5 nm radius
            color="#e74c3c",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.35,
            popup=folium.Popup(
                f"""
                <div style="font-family: sans-serif; min-width: 180px;">
                    <h4 style="margin: 0; color: #c0392b;">Holding Stack #{idx + 1}</h4>
                    <p style="margin: 4px 0;"><b>Duration:</b> {h.duration_min:.1f} min</p>
                    <p style="margin: 4px 0;"><b>Altitude:</b> FL{h.flight_level:.0f}</p>
                    <p style="margin: 4px 0;"><b>Time:</b> {h.start_time} to {h.end_time}</p>
                </div>
                """,
                max_width=250,
            ),
            tooltip=f"Holding Stack: {h.duration_min:.1f} min at FL{h.flight_level:.0f}",
        ).add_to(m)

    # Add likely used route navaids as colored dots along the trajectory
    if active_navaids and show_navaids:
        nav_group = folium.FeatureGroup(name="📡 Active Route Navaids (VOR-DME / DME / VOR / ILS / NDB)", show=True)
        for nav in active_navaids:
            nd = nav.to_dict() if hasattr(nav, "to_dict") else (nav if isinstance(nav, dict) else nav.__dict__)
            n_ident = str(nd.get("ident", "NAV"))
            n_name = str(nd.get("name", ""))
            n_type = str(nd.get("nav_type", "DME"))
            n_freq = nd.get("frequency_mhz")
            n_lat = float(nd.get("latitude", 0.0))
            n_lon = float(nd.get("longitude", 0.0))
            n_elev = float(nd.get("elevation_ft", 0.0))
            n_rng = float(nd.get("published_range_nm", 40.0))
            n_dist = float(nd.get("min_distance_to_track_nm", 0.0))
            n_fl = float(nd.get("aircraft_fl_at_closest", 0.0))
            t_lat = float(nd.get("closest_track_lat", n_lat))
            t_lon = float(nd.get("closest_track_lon", n_lon))
            n_rlos = float(nd.get("rlos_horizon_nm", 150.0))
            n_role = str(nd.get("fms_navigation_role", ""))
            n_col = str(nd.get("marker_color", "#00a8ff"))
            if n_type in ("ILS_LOC", "LOC", "DME_ILS"):
                n_col = "#ff2e93"  # High-contrast magenta for ILS/LOC dots

            freq_str = f"{n_freq:.2f} MHz" if n_freq is not None else "UHF Ch"

            # Subtle dashed radio reception sightline to closest trajectory point
            create_safe_polyline(
                locations=[[n_lat, n_lon], [t_lat, t_lon]],
                color=n_col,
                weight=1.5,
                dash_array="3, 5",
                opacity=0.65,
                tooltip=f"{n_ident} ({n_type}) slant reception: {n_dist:.1f} nm at FL{n_fl:.0f}",
            ).add_to(nav_group)

            # Colored dot for the navaid station
            folium.CircleMarker(
                location=[n_lat, n_lon],
                radius=7.0 if n_type in ("VOR-DME", "ILS_LOC") else 6.5,
                color="#1e272e",
                weight=1.5,
                fill=True,
                fill_color=n_col,
                fill_opacity=0.95,
                tooltip=f"📡 {n_ident} ({n_type} | {freq_str}) — {n_dist:.1f} nm from track @ FL{n_fl:.0f}",
                popup=folium.Popup(
                    f"""
                    <div style="font-family: sans-serif; min-width: 230px; font-size: 12.5px;">
                        <b style="color: {n_col}; font-size: 14px;">📡 {n_ident} — {n_name}</b><br>
                        <hr style="margin: 4px 0;">
                        <b>Navaid Type:</b> {n_type} ({freq_str})<br>
                        <b>Station Elevation:</b> {n_elev:,.0f} ft MSL (Pub Range: {n_rng:.0f} nm)<br>
                        <b>Closest Track Pass:</b> {n_dist:.2f} nm @ FL{n_fl:.0f}<br>
                        <b>Radio Horizon (RLOS):</b> {n_rlos:.1f} nm<br>
                        <b>FMS Role:</b> {n_role}
                    </div>
                    """,
                    max_width=290,
                ),
            ).add_to(nav_group)

        nav_group.add_to(m)

    # Add persistent contrail (ISSR) segments if provided
    has_contrails = False
    if contrail_segments and show_contrails:
        c_group = folium.FeatureGroup(name="🧊 Persistent Contrail Cirrus (ISSR)", show=True)
        for seg in contrail_segments:
            s_dict = seg.to_dict() if hasattr(seg, "to_dict") else (seg if isinstance(seg, dict) else seg.__dict__)
            lat1 = float(s_dict.get("start_lat", 0.0))
            lon1 = float(s_dict.get("start_lon", 0.0))
            lat2 = float(s_dict.get("end_lat", 0.0))
            lon2 = float(s_dict.get("end_lon", 0.0))
            if not (_is_finite_coord(lat1) and _is_finite_coord(lon1) and _is_finite_coord(lat2) and _is_finite_coord(lon2)):
                continue
            if abs(lat1) < 0.1 and abs(lon1) < 0.1:
                continue

            has_contrails = True
            fl = float(s_dict.get("flight_level", 0.0))
            dist = float(s_dict.get("distance_nm", 0.0))
            dur = float(s_dict.get("duration_min", 0.0))
            t_c = float(s_dict.get("temperature_c", -45.0))
            rhi = float(s_dict.get("rh_ice_pct", 105.0))
            rec_fl = float(s_dict.get("recommended_alt_fl", fl))
            note = str(s_dict.get("alt_deviation_note", ""))

            # Casing
            create_safe_polyline(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="#0c2461",
                weight=8,
                opacity=0.75,
            ).add_to(c_group)

            # Glowing core
            create_safe_polyline(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="#00f0ff",
                weight=5,
                opacity=0.98,
                tooltip=f"🧊 Persistent Contrail: FL{fl:.0f} | {dist:.1f} nm ({dur:.1f}m) | RHi: {rhi:.0f}%",
                popup=folium.Popup(
                    f"""
                    <div style="font-family: sans-serif; min-width: 210px; font-size: 12.5px;">
                        <b style="color: #0984e3; font-size: 13.5px;">🧊 Persistent Contrail (ISSR)</b><br>
                        <hr style="margin: 4px 0;">
                        <b>Flight Level:</b> FL{fl:.0f}<br>
                        <b>Distance:</b> {dist:.1f} nm ({dur:.1f} min)<br>
                        <b>Ambient Temp:</b> {t_c:.1f}°C (Threshold T_SAC)<br>
                        <b>Ice Relative Humidity:</b> {rhi:.0f}% (Supersaturated)<br>
                        <b>Recommended Shift:</b> FL{rec_fl:.0f}<br>
                        <i style="color: #27ae60;">{note}</i>
                    </div>
                    """,
                    max_width=270,
                ),
            ).add_to(c_group)
        if has_contrails:
            c_group.add_to(m)

    # Add layer control to toggle basemaps and navaid overlay
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    # Add title header box (bottom-left to avoid layer control collision)
    nav_legend_line = (
        '<br><b>Navaids:</b> '
        '<span style="color: #00a8ff;">● VOR-DME</span> | '
        '<span style="color: #0097e6;">● DME</span> | '
        '<span style="color: #00b894;">● VOR</span> | '
        '<span style="color: #ff2e93;">● ILS</span> | '
        '<span style="color: #f39c12;">● NDB</span>'
        if (active_navaids and show_navaids)
        else ""
    )
    contrail_legend = ' | <span style="color: #00f0ff; font-weight: bold;">━ Contrail (ISSR)</span>' if has_contrails else ""
    extra_h = 20 if has_contrails else 0
    box_height = (138 if (active_navaids and show_navaids) else 115) + extra_h
    title_html = f"""
    <div style="position: fixed;
                bottom: 25px; left: 15px; width: 360px; height: {box_height}px;
                background-color: white; border:2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size:12.5px; font-family: sans-serif;
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size: 14.5px; color: #2c3e50;">✈️ Flight {ectrl_id} ({op} - {ac})</b><br>
        <b>Route:</b> {adep} &rarr; {ades}<br>
        <b>Distance:</b> Actual {metrics.actual_distance_nm:.0f} nm (Plan: {metrics.planned_distance_nm:.0f} nm)<br>
        <b>Efficiency Ratio:</b> {metrics.horizontal_extension_ratio:.3f}x | <b>Holdings:</b> {len(metrics.holding_events)}<br>
        <span style="color: #2980b9;">-- Filed Plan</span> | <span style="color: #8e44ad;">— Actual Track</span> | <span style="color: #e74c3c;">⭕ Holding</span>{contrail_legend}
        {nav_legend_line}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    if output_html_path:
        out_p = Path(output_html_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_p))

    return m


def create_contrail_horizontal_map(
    flight_info: dict[str, Any],
    actual_df: pd.DataFrame,
    contrail_segments: list[Any] | None = None,
    eco_route_waypoints: list[Any] | None = None,
    output_html_path: Path | str | None = None,
) -> folium.Map:
    """
    Generate an interactive Leaflet/Folium map visualizing the horizontal elements
    of the flown route with persistent contrail (ISSR) forming segments imposed on top.
    """
    ectrl_id = flight_info.get("ectrl_id", 0)
    adep = flight_info.get("adep", "ADEP")
    ades = flight_info.get("ades", "ADES")
    op = flight_info.get("ac_operator", "OP")
    ac = flight_info.get("ac_type", "AC")

    center_lat, center_lon = 50.0, 10.0
    if not actual_df.empty and "latitude" in actual_df.columns and "longitude" in actual_df.columns:
        valid_act = actual_df.dropna(subset=["latitude", "longitude"])
        if not valid_act.empty:
            c_lat = float(valid_act["latitude"].mean())
            c_lon = circular_mean_longitude(valid_act["longitude"].tolist())
            if _is_finite_coord(c_lat) and _is_finite_coord(c_lon):
                center_lat, center_lon = c_lat, c_lon

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, Tiles style by Humanitarian OpenStreetMap Team hosted by OpenStreetMap France",
        name="Humanitarian (Unblocked)",
        subdomains="abc",
        show=True,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
        attr="USGS The National Map (Public Domain)",
        name="USGS Topo (Unblocked)",
        show=False,
        control=True,
    ).add_to(m)

    # 1. Base Flown Trajectory
    if not actual_df.empty and "latitude" in actual_df.columns and "longitude" in actual_df.columns:
        valid_pts = actual_df.dropna(subset=["latitude", "longitude"])
        actual_coords = list(zip(valid_pts["latitude"], valid_pts["longitude"]))
        create_safe_polyline(
            actual_coords,
            color="#8e44ad",
            weight=4,
            opacity=0.65,
            tooltip=f"Flown Radar Route: Flight #{ectrl_id} ({adep} ➔ {ades})",
        ).add_to(m)

        # Mark ADEP
        first_pt = valid_pts.iloc[0]
        folium.Marker(
            [first_pt["latitude"], first_pt["longitude"]],
            tooltip=f"Departure: {adep}",
            icon=folium.Icon(color="green", icon="plane-departure", prefix="fa"),
        ).add_to(m)

        # Mark ADES
        last_pt = valid_pts.iloc[-1]
        folium.Marker(
            [last_pt["latitude"], last_pt["longitude"]],
            tooltip=f"Destination: {ades}",
            icon=folium.Icon(color="red", icon="plane-arrival", prefix="fa"),
        ).add_to(m)

    # 2. Imposed Persistent Contrail (ISSR) Forming Segments
    c_segs = contrail_segments or []
    tot_c_dist = 0.0
    if c_segs:
        contrail_group = folium.FeatureGroup(name="🧊 Persistent Contrail Cirrus (ISSR)", show=True)
        for idx, seg in enumerate(c_segs):
            s_dict = seg.to_dict() if hasattr(seg, "to_dict") else (seg if isinstance(seg, dict) else seg.__dict__)
            lat1 = float(s_dict.get("start_lat", 0.0))
            lon1 = float(s_dict.get("start_lon", 0.0))
            lat2 = float(s_dict.get("end_lat", 0.0))
            lon2 = float(s_dict.get("end_lon", 0.0))
            if not (_is_finite_coord(lat1) and _is_finite_coord(lon1) and _is_finite_coord(lat2) and _is_finite_coord(lon2)):
                continue
            if abs(lat1) < 0.1 and abs(lon1) < 0.1:
                continue

            fl = float(s_dict.get("flight_level", 0.0))
            dist = float(s_dict.get("distance_nm", 0.0))
            dur = float(s_dict.get("duration_min", 0.0))
            tot_c_dist += dist
            t_c = float(s_dict.get("temperature_c", -45.0))
            rhi = float(s_dict.get("rh_ice_pct", 105.0))
            rec_fl = float(s_dict.get("recommended_alt_fl", fl))
            note = str(s_dict.get("alt_deviation_note", ""))

            # Casing line (navy)
            create_safe_polyline(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="#0c2461",
                weight=8,
                opacity=0.75,
            ).add_to(contrail_group)

            # Glowing core line (neon cyan)
            create_safe_polyline(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="#00f0ff",
                weight=5,
                opacity=0.98,
                tooltip=f"🧊 Contrail Forming: FL{fl:.0f} | {dist:.1f} nm ({dur:.1f}m) | RHi: {rhi:.0f}% | Temp: {t_c:.1f}°C",
                popup=folium.Popup(
                    f"""
                    <div style="font-family: sans-serif; min-width: 230px; font-size: 12.5px;">
                        <b style="color: #0984e3; font-size: 14px;">🧊 Persistent Contrail Cirrus (ISSR)</b><br>
                        <hr style="margin: 4px 0;">
                        <b>Flight Level:</b> FL{fl:.0f}<br>
                        <b>Flown Distance in ISSR:</b> {dist:.1f} nm ({dur:.1f} min)<br>
                        <b>Ambient Temp:</b> {t_c:.1f}°C (Schmidt-Appleman T_SAC met)<br>
                        <b>Ice Relative Humidity:</b> {rhi:.0f}% (Supersaturated RHi &ge; 100%)<br>
                        <hr style="margin: 4px 0;">
                        <b>Tactical Avoidance:</b> <b style="color: #27ae60;">FL{rec_fl:.0f}</b><br>
                        <span style="color: #27ae60; font-size: 11.5px;">{note}</span>
                    </div>
                    """,
                    max_width=290,
                ),
            ).add_to(contrail_group)

            # Start of segment marker if first in cluster or significant
            if idx == 0 or idx == len(c_segs) - 1:
                folium.CircleMarker(
                    location=[lat1, lon1],
                    radius=5.5,
                    color="#00f0ff",
                    fill=True,
                    fill_color="#0984e3",
                    fill_opacity=1.0,
                    tooltip=f"ISSR Boundary Point: FL{fl:.0f}",
                ).add_to(contrail_group)

        contrail_group.add_to(m)

    # 3. Alternative Eco-Optimal Route
    has_eco = False
    if eco_route_waypoints:
        has_eco = True
        eco_group = folium.FeatureGroup(name="🌿 Alternative Eco-Optimal Route (Minimal Contrails & Balanced Emissions)", show=True)
        eco_coords = []
        for wp in eco_route_waypoints:
            w_dict = wp.to_dict() if hasattr(wp, "to_dict") else (wp if isinstance(wp, dict) else wp.__dict__)
            w_lat = float(w_dict.get("latitude", 0.0))
            w_lon = float(w_dict.get("longitude", 0.0))
            if _is_finite_coord(w_lat) and _is_finite_coord(w_lon) and abs(w_lat) > 0.1:
                eco_coords.append([w_lat, w_lon])
                fl_diff = float(w_dict.get("fl_delta_vs_flown", 0.0))
                w_fl = float(w_dict.get("flight_level", 0.0))
                reason = str(w_dict.get("deviation_reason", ""))
                # If there was a tactical altitude deviation, put a marker on the map
                if abs(fl_diff) >= 15.0:
                    folium.CircleMarker(
                        location=[w_lat, w_lon],
                        radius=6.0,
                        color="#27ae60",
                        weight=2,
                        fill=True,
                        fill_color="#2ecc71",
                        fill_opacity=0.95,
                        tooltip=f"🌿 Eco Tactical Shift: FL{w_fl:.0f} ({fl_diff:+.0f} FL) — {reason}",
                        popup=folium.Popup(
                            f"""
                            <div style="font-family: sans-serif; min-width: 220px; font-size: 12.5px;">
                                <b style="color: #27ae60; font-size: 13.5px;">🌿 Eco-Optimal Maneuver</b><br>
                                <hr style="margin: 4px 0;">
                                <b>Eco Flight Level:</b> FL{w_fl:.0f} (Flown: {w_fl - fl_diff:.0f})<br>
                                <b>Altitude Shift:</b> {fl_diff:+.0f} FL<br>
                                <b>Advisory:</b> {reason}
                            </div>
                            """,
                            max_width=270,
                        ),
                    ).add_to(eco_group)

        if eco_coords:
            create_safe_polyline(
                eco_coords,
                color="#2ecc71",
                weight=4.5,
                dash_array="8, 6",
                opacity=0.95,
                tooltip="🌿 Alternative Eco-Optimal Trajectory (Balanced Fuel, NOx & Contrail Avoidance)",
            ).add_to(eco_group)
            eco_group.add_to(m)

    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    c_status = f'<span style="color: #00f0ff; font-weight: bold;">{tot_c_dist:.1f} nm in ISSR ({len(c_segs)} segments)</span>' if c_segs else '<span style="color: #27ae60; font-weight: bold;">Clean Air (0 nm contrail)</span>'
    eco_legend_line = '<br><span style="color: #2ecc71; font-weight: bold;">-- Alternative Eco-Optimal Route</span>' if has_eco else ""
    info_box_height = 138 if has_eco else 118
    info_html = f"""
    <div style="position: fixed;
                bottom: 25px; left: 15px; width: 360px; height: {info_box_height}px;
                background-color: white; border:2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size:12.5px; font-family: sans-serif;
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size: 14px; color: #2c3e50;">🌬️ Contrail Climate &amp; Eco Trajectory — #{ectrl_id}</b><br>
        <b>Route:</b> {adep} &rarr; {ades} ({op} - {ac})<br>
        <b>Contrail Status:</b> {c_status}<br>
        <span style="color: #8e44ad;">— Flown Route</span> | <span style="color: #00f0ff; font-weight: bold;">━ Contrail (ISSR)</span>{eco_legend_line}<br>
        <span style="font-size: 11px; color: #7f8c8d;">Balanced: Fuel + CO₂ + Contrail Cirrus + NOₓ</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    if output_html_path:
        out_p = Path(output_html_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_p))

    return m


def create_grid_saturation_map(
    grid_data: dict[str, Any],
    proximity_events: list[Any] | None = None,
) -> folium.Map:
    """
    Render an interactive Folium map of 5°x5° European airspace squares colored by average hourly
    flight density and dynamic complexity for the selected hour of the day (`00:00` - `23:00` UTC),
    optionally overlaying detected 4D co-altitude aircraft proximity pairs.
    """
    m = folium.Map(
        location=[48.5, 10.0],
        zoom_start=4,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, HOT style hosted by OpenStreetMap France",
        name="Humanitarian (Unblocked)",
        subdomains="abc",
        show=True,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap France & OpenStreetMap contributors",
        name="OSM France (Unblocked)",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    cells = grid_data.get("cells") or []
    hr = int(grid_data.get("hour_utc", 10))
    period = str(grid_data.get("period", ""))
    grid_size = float(grid_data.get("grid_size_deg", 5.0))

    status_colors = {
        "CRITICAL_SATURATION": ("#c0392b", 0.48),
        "HIGH_DENSITY": ("#e67e22", 0.40),
        "MODERATE_FLOW": ("#f1c40f", 0.32),
        "LOW_DENSITY": ("#2ecc71", 0.22),
    }

    line_weight = 1 if grid_size <= 0.5 else (1.4 if grid_size <= 1.0 else 2)

    for c in cells:
        lat0 = float(c["cell_lat"])
        lon0 = float(c["cell_lon"])
        lat1 = lat0 + grid_size
        lon1 = lon0 + grid_size
        status = str(c.get("saturation_status", "LOW_DENSITY"))
        color, fill_op = status_colors.get(status, ("#2ecc71", 0.22))
        avg_f = float(c.get("avg_flights_per_hour", 0.0))
        peak_f = int(c.get("peak_flights_per_hour", 0))
        vert_pct = float(c.get("vertical_transition_pct", 0.0))
        comp_score = float(c.get("dynamic_complexity_score", 0.0))
        avg_fl = float(c.get("avg_flight_level", 320.0))
        sector_lbl = str(c.get("grid_sector", f"{lat0:.1f}°N, {lon0:.1f}°E"))

        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 220px; font-size: 12.5px;">
            <h4 style="margin: 0 0 6px 0; color: {color};">{grid_size:g}°×{grid_size:g}° Cell: {sector_lbl}</h4>
            <b>Hour Window:</b> {hr:02d}:00 – {hr:02d}:59 UTC<br>
            <b>Avg Flights / Hr:</b> {avg_f:.1f} flts/hr (Peak: {peak_f}/hr)<br>
            <b>Dynamic Complexity Index:</b> {comp_score:.1f}<br>
            <b>Climb / Descent Share:</b> {vert_pct:.1f}%<br>
            <b>Mean Flight Level:</b> FL{avg_fl:.0f}<br>
            <b>Status:</b> {status.replace('_', ' ')}
        </div>
        """

        folium.Rectangle(
            bounds=[[lat0, lon0], [lat1, lon1]],
            color=color,
            weight=line_weight,
            fill=True,
            fill_color=color,
            fill_opacity=fill_op,
            tooltip=(
                f"{grid_size:g}°×{grid_size:g}° {sector_lbl} | Avg: {avg_f:.1f} flts/hr (Peak: {peak_f}) | "
                f"Climb/Desc: {vert_pct:.1f}% | Complexity: {comp_score:.1f}"
            ),
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(m)

        # Render inline badge inside medium/high density squares when cell is sufficiently large
        show_badge = (grid_size >= 2.0 and avg_f >= 15.0) or (grid_size >= 1.0 and avg_f >= 30.0)
        if show_badge:
            folium.Marker(
                location=[lat0 + grid_size * 0.5, lon0 + grid_size * 0.5],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-family: sans-serif; font-size: 10.5px; font-weight: bold; '
                        f'color: #ffffff; background: rgba(44,62,80,0.78); padding: 1px 4px; '
                        f'border-radius: 3px; text-align: center; white-space: nowrap; '
                        f'transform: translate(-50%, -50%);">{avg_f:.0f}/hr</div>'
                    )
                ),
                tooltip=f"{sector_lbl}: {avg_f:.1f} avg flts/hr",
            ).add_to(m)

    # Overlay 4D co-altitude proximity conflict pairs if supplied
    if proximity_events:
        for ev in proximity_events:
            d = ev.to_dict() if hasattr(ev, "to_dict") else (ev if isinstance(ev, dict) else ev.__dict__)
            lat = float(d["latitude"])
            lon = float(d["longitude"])
            sev = str(d.get("severity", ""))
            is_los = "LOSS_OF_SEPARATION" in sev
            pin_col = "red" if is_los else "orange"
            f1 = f"{d['flight_1_id']} ({d.get('operator_1', '')} {d.get('ac_type_1', '')})"
            f2 = f"{d['flight_2_id']} ({d.get('operator_2', '')} {d.get('ac_type_2', '')})"
            h_nm = float(d.get("horizontal_sep_nm", 0.0))
            v_ft = float(d.get("vertical_sep_ft", 0.0))
            fl1 = float(d.get("fl_1", 0.0))
            fl2 = float(d.get("fl_2", 0.0))

            folium.CircleMarker(
                location=[lat, lon],
                radius=8 if is_los else 6,
                color="#900C3F" if is_los else "#d35400",
                fill=True,
                fill_color="#ff0033" if is_los else "#f39c12",
                fill_opacity=0.9,
                tooltip=(
                    f"⚠️ Proximity Pair: {f1} ⇄ {f2} | "
                    f"Horiz: {h_nm:.2f} nm | Vert: {v_ft:.0f} ft (FL{fl1:.0f}/FL{fl2:.0f})"
                ),
                popup=folium.Popup(
                    f"""
                    <div style="font-family: sans-serif; min-width: 220px; font-size: 12.5px;">
                        <b style="color: {'#c0392b' if is_los else '#d35400'};">{sev}</b><br>
                        <b>Flight 1:</b> {f1} [{d.get('route_1', '')}] @ FL{fl1:.0f}<br>
                        <b>Flight 2:</b> {f2} [{d.get('route_2', '')}] @ FL{fl2:.0f}<br>
                        <b>Separation:</b> {h_nm:.2f} nm horiz / {v_ft:.0f} ft vert<br>
                        <b>Timestamp:</b> {d.get('timestamp_utc', '')}
                    </div>
                    """,
                    max_width=290,
                ),
            ).add_to(m)

    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 15px; width: 350px;
                background-color: white; border: 2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size: 12px; font-family: sans-serif;
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size: 13.5px; color: #2c3e50;">🗺️ {grid_size:g}°×{grid_size:g}° Hourly Airspace Saturation ({hr:02d}:00 UTC, {period})</b><br>
        <span style="color: #c0392b;">■ Critical Saturation</span> |
        <span style="color: #e67e22;">■ High Density</span> |
        <span style="color: #f1c40f;">■ Moderate</span> |
        <span style="color: #2ecc71;">■ Low</span>
        {"<br><span style='color: #ff0033;'>● Co-Altitude Proximity Alert (&le;5 nm / &le;1,000 ft)</span>" if proximity_events else ""}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def create_conflict_map(
    proximity_events: list[Any],
    hour_utc: int = 10,
    date_str: str = "",
) -> folium.Map:
    """
    Render a clean interactive Folium map showing ONLY the detected 4D co-altitude aircraft
    proximity/conflict pairs (without 5°x5° grid squares), centered on the detected conflicts.
    """
    lats: list[float] = []
    lons: list[float] = []
    parsed_events: list[dict[str, Any]] = []
    for ev in proximity_events or []:
        d = ev.to_dict() if hasattr(ev, "to_dict") else (ev if isinstance(ev, dict) else ev.__dict__)
        parsed_events.append(d)
        lat = d.get("latitude")
        lon = d.get("longitude")
        if _is_finite_coord(lat) and _is_finite_coord(lon):
            lats.append(float(lat))
            lons.append(float(lon))

    if lats and lons:
        c_lat = sum(lats) / len(lats)
        c_lon = sum(lons) / len(lons)
        center_lat = float(c_lat) if _is_finite_coord(c_lat) else 48.5
        center_lon = float(c_lon) if _is_finite_coord(c_lon) else 10.0
    else:
        center_lat, center_lon = 48.5, 10.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, HOT style hosted by OpenStreetMap France",
        name="Humanitarian (Unblocked)",
        subdomains="abc",
        show=True,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap France & OpenStreetMap contributors",
        name="OSM France (Unblocked)",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    los_count = 0
    for d in parsed_events:
        lat = float(d["latitude"])
        lon = float(d["longitude"])
        sev = str(d.get("severity", ""))
        is_los = "LOSS_OF_SEPARATION" in sev
        if is_los:
            los_count += 1
        f1 = f"{d['flight_1_id']} ({d.get('operator_1', '')} {d.get('ac_type_1', '')})"
        f2 = f"{d['flight_2_id']} ({d.get('operator_2', '')} {d.get('ac_type_2', '')})"
        h_nm = float(d.get("horizontal_sep_nm", 0.0))
        v_ft = float(d.get("vertical_sep_ft", 0.0))
        fl1 = float(d.get("fl_1", 0.0))
        fl2 = float(d.get("fl_2", 0.0))

        # Outer halo ring for clear visual prominence
        folium.CircleMarker(
            location=[lat, lon],
            radius=13 if is_los else 10,
            color="#ff0033" if is_los else "#f39c12",
            weight=1.5,
            fill=True,
            fill_color="#ff0033" if is_los else "#f39c12",
            fill_opacity=0.22,
        ).add_to(m)

        # Core conflict marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=7.5 if is_los else 6.0,
            color="#900C3F" if is_los else "#d35400",
            weight=2.0,
            fill=True,
            fill_color="#ff0033" if is_los else "#f39c12",
            fill_opacity=0.95,
            tooltip=(
                f"⚠️ {sev}: {f1} ⇄ {f2} | "
                f"Horiz: {h_nm:.2f} nm | Vert: {v_ft:.0f} ft (FL{fl1:.0f}/FL{fl2:.0f})"
            ),
            popup=folium.Popup(
                f"""
                <div style="font-family: sans-serif; min-width: 230px; font-size: 12.5px;">
                    <b style="color: {'#c0392b' if is_los else '#d35400'}; font-size: 13.5px;">⚠️ {sev}</b><br>
                    <hr style="margin: 4px 0;">
                    <b>Flight 1:</b> {f1} [{d.get('route_1', '')}] @ FL{fl1:.0f}<br>
                    <b>Flight 2:</b> {f2} [{d.get('route_2', '')}] @ FL{fl2:.0f}<br>
                    <b>Separation:</b> {h_nm:.2f} nm horiz / {v_ft:.0f} ft vert<br>
                    <b>Coordinates:</b> {lat:.3f}°, {lon:.3f}°<br>
                    <b>Timestamp:</b> {d.get('timestamp_utc', '')}
                </div>
                """,
                max_width=300,
            ),
        ).add_to(m)

    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 15px; width: 380px;
                background-color: white; border: 2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size: 12px; font-family: sans-serif;
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size: 13.5px; color: #2c3e50;">⚠️ 4D Co-Altitude Conflict Map ({date_str} {hour_utc:02d}:00 UTC)</b><br>
        <b>Detected Pairs:</b> {len(parsed_events)} total ({los_count} Loss-of-Sep &lt;3 nm)<br>
        <span style="color: #ff0033;">● Loss-of-Separation Alert (&lt;3.0 nm &amp; &le;1,000 ft)</span><br>
        <span style="color: #f39c12;">● High-Proximity Pair (3.0–5.0 nm &amp; &le;1,000 ft)</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def create_world_navaids_map(
    navaids_df: pd.DataFrame,
    center_lat: float = 30.0,
    center_lon: float = 10.0,
    zoom_start: int = 2,
) -> folium.Map:
    """
    Render an interactive Canvas-accelerated Folium world map displaying all filtered global
    radio navigation aids (VOR-DME / VORTAC, DME / TACAN, VOR, ILS / LOC, NDB) as color-coded dots.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=None,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, HOT style hosted by OpenStreetMap France",
        name="Humanitarian (Unblocked)",
        subdomains="abc",
        show=True,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap France & OpenStreetMap contributors",
        name="OSM France (Unblocked)",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap",
        name="OpenTopoMap",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    type_colors = {
        "VOR-DME": "#00a8ff",
        "DME": "#0097e6",
        "VOR": "#00b894",
        "ILS_LOC": "#ff2e93",
        "NDB": "#f39c12",
    }

    if navaids_df is not None and not navaids_df.empty:
        for _, row in navaids_df.iterrows():
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            ident = str(row.get("ident", "NAV"))
            name = str(row.get("name", ""))
            ntype = str(row.get("nav_type", "DME"))
            raw_t = str(row.get("raw_type", ntype))
            freq = row.get("frequency_mhz")
            freq_str = f"{float(freq):.2f} MHz" if pd.notna(freq) and freq is not None else "UHF Ch"
            rng = float(row.get("range_nm", 40.0)) if pd.notna(row.get("range_nm")) else 40.0
            elev = float(row.get("elevation_ft", 0.0)) if pd.notna(row.get("elevation_ft")) else 0.0
            col = type_colors.get(ntype, "#00a8ff")

            folium.CircleMarker(
                location=[lat, lon],
                radius=5.0 if ntype in ("VOR-DME", "ILS_LOC") else 4.2,
                color="#1e272e",
                weight=1.0,
                fill=True,
                fill_color=col,
                fill_opacity=0.88,
                tooltip=f"📡 {ident} — {name} [{raw_t} | {freq_str} | Rng {rng:.0f} nm | Elev {elev:,.0f} ft]",
            ).add_to(m)

    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    total_shown = 0 if navaids_df is None else len(navaids_df)
    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 15px; width: 395px;
                background-color: white; border: 2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size: 12px; font-family: sans-serif;
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size: 13.5px; color: #2c3e50;">🌍 Global Radio Navaids Atlas ({total_shown:,} stations shown)</b><br>
        <span style="color: #00a8ff;">● VOR-DME / VORTAC</span> |
        <span style="color: #0097e6;">● DME / TACAN</span> |
        <span style="color: #00b894;">● VOR</span> |
        <span style="color: #ff2e93;">● ILS / LOC</span> |
        <span style="color: #f39c12;">● NDB</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def create_flight_fir_charge_map(
    dataset: Any,
    ectrl_id: int,
    crco_report: Any | None = None,
    output_html_path: Path | str | None = None,
) -> folium.Map:
    """
    Generate an interactive Leaflet/Folium map visualizing a flight's trajectory
    underlaid with the boundary polygons of every Flight Information Region (FIR) traversed,
    color-coded with Eurocontrol CRCO national charging rates and fee metrics.
    """
    from flightml.eval.crco_economics import compute_flight_crco_charges

    flight = dataset.get_flight(ectrl_id) or {}
    adep = flight.get("adep", "ADEP")
    ades = flight.get("ades", "ADES")

    trajs = dataset.get_trajectory(ectrl_id)
    actual_df = trajs.get("actual", pd.DataFrame())
    planned_df = trajs.get("planned", pd.DataFrame())

    if crco_report is None:
        crco_report = compute_flight_crco_charges(dataset, ectrl_id)

    # Center map on mid-point
    center_lat, center_lon = 50.0, 10.0
    if not actual_df.empty and "latitude" in actual_df.columns and "longitude" in actual_df.columns:
        valid_act = actual_df.dropna(subset=["latitude", "longitude"])
        if not valid_act.empty:
            c_lat = float(valid_act["latitude"].mean())
            c_lon = circular_mean_longitude(valid_act["longitude"].tolist())
            if _is_finite_coord(c_lat) and _is_finite_coord(c_lon):
                center_lat, center_lon = c_lat, c_lon
    elif not planned_df.empty and "latitude" in planned_df.columns and "longitude" in planned_df.columns:
        valid_plan = planned_df.dropna(subset=["latitude", "longitude"])
        if not valid_plan.empty:
            c_lat = float(valid_plan["latitude"].mean())
            c_lon = circular_mean_longitude(valid_plan["longitude"].tolist())
            if _is_finite_coord(c_lat) and _is_finite_coord(c_lon):
                center_lat, center_lon = c_lat, c_lon

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles=None,
    )

    # 1. Humanitarian OpenStreetMap Team (HOT) hosted by OSM France (Unblocked)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, Humanitarian OpenStreetMap Team",
        name="Humanitarian (Unblocked)",
        subdomains="abc",
        show=True,
        control=True,
    ).add_to(m)

    # 2. OSM France mirror (Unblocked)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap France & OpenStreetMap contributors",
        name="OSM France (Unblocked)",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    # 3. USGS Topographic Map (Unblocked)
    folium.TileLayer(
        tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
        attr="USGS The National Map (Public Domain)",
        name="USGS Topo (Unblocked)",
        show=False,
        control=True,
    ).add_to(m)

    # 4. OpenTopoMap (Unblocked)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap",
        name="OpenTopoMap",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    # Collect FIR IDs from CRCO report
    fir_breakdown = crco_report.fir_breakdown if crco_report else []
    fir_ids = [b.fir_id for b in fir_breakdown]
    query_fir_ids = list(set(fir_ids + [f.replace("FIR", "UIR") for f in fir_ids if f.endswith("FIR")]))
    fir_boundaries = dataset.get_fir_boundaries(query_fir_ids)

    fir_palette = [
        "#2980b9", "#27ae60", "#8e44ad", "#d35400", "#16a085",
        "#c0392b", "#f39c12", "#2c3e50", "#1abc9c", "#7f8c8d"
    ]

    fir_group = folium.FeatureGroup(name="🗺️ Traversed FIR Airspaces & CRCO Zones", show=True)

    for idx, b in enumerate(fir_breakdown):
        color = fir_palette[idx % len(fir_palette)]
        coords = fir_boundaries.get(b.fir_id)
        if not coords and b.fir_id.endswith("FIR"):
            coords = fir_boundaries.get(b.fir_id[:-3] + "UIR")
        if not coords and len(b.fir_id) > 4:
            coords = fir_boundaries.get(b.fir_id[:4])

        if coords:
            poly_locs = [[lat, lon] for lat, lon in coords]
            popup_html = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; min-width: 240px; font-size: 13px;">
                <div style="background-color: {color}; color: white; padding: 6px 10px; border-radius: 4px; font-weight: bold; margin-bottom: 8px;">
                    {b.fir_id} — {b.country_name}
                </div>
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <tr><td style="color: #7f8c8d; padding: 3px 0;">Unit Rate:</td><td style="text-align: right; font-weight: bold; color: #27ae60;">€{b.unit_rate_eur:.2f} / SU</td></tr>
                    <tr><td style="color: #7f8c8d; padding: 3px 0;">Traversed Distance:</td><td style="text-align: right; font-weight: bold;">{b.estimated_distance_nm:.1f} nm ({b.estimated_distance_km:.1f} km)</td></tr>
                    <tr><td style="color: #7f8c8d; padding: 3px 0;">Airspace Dwell:</td><td style="text-align: right; font-weight: bold;">{b.dwell_min:.1f} min</td></tr>
                    <tr><td style="color: #7f8c8d; padding: 3px 0;">Distance Factor:</td><td style="text-align: right;">{b.distance_factor:.3f}</td></tr>
                    <tr><td style="color: #7f8c8d; padding: 3px 0;">Weight Factor:</td><td style="text-align: right;">{b.weight_factor:.3f}</td></tr>
                    <tr><td style="color: #7f8c8d; padding: 3px 0;">Service Units (SU):</td><td style="text-align: right; font-weight: bold;">{b.service_units:.3f} SU</td></tr>
                    <tr style="border-top: 1px solid #eee;"><td style="color: #2c3e50; padding: 6px 0; font-weight: bold;">En-Route Charge:</td><td style="text-align: right; font-weight: bold; color: #2980b9; font-size: 13px;">€{b.charge_eur:,.2f} EUR</td></tr>
                </table>
            </div>
            """
            folium.Polygon(
                locations=poly_locs,
                color=color,
                weight=2.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.20,
                tooltip=f"<b>{b.fir_id}</b> ({b.country_name}) — Unit Rate: €{b.unit_rate_eur:.2f}/SU | Charge: €{b.charge_eur:,.2f} EUR",
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(fir_group)

    fir_group.add_to(m)

    # Add planned trajectory line
    if not planned_df.empty:
        planned_coords = list(zip(planned_df["latitude"], planned_df["longitude"]))
        create_safe_polyline(
            planned_coords,
            color="#2980b9",
            weight=3,
            dash_array="6, 6",
            opacity=0.85,
            tooltip=f"Filed Flight Plan ({len(planned_df)} waypoints)",
        ).add_to(m)

    # Add actual trajectory line
    if not actual_df.empty:
        actual_coords = list(zip(actual_df["latitude"], actual_df["longitude"]))
        act_dist_val = crco_report.actual_distance_nm if crco_report else 0.0
        create_safe_polyline(
            actual_coords,
            color="#8e44ad",
            weight=4,
            opacity=0.95,
            tooltip=f"Actual Flown Track ({act_dist_val:.1f} nm)",
        ).add_to(m)

        # Mark Departure (ADEP)
        first_pt = actual_df.iloc[0]
        folium.Marker(
            [first_pt["latitude"], first_pt["longitude"]],
            tooltip=f"Departure: {adep} (Off-Block Delay: {flight.get('dep_delay_min', 0)} min)",
            icon=folium.Icon(color="green", icon="plane-departure", prefix="fa"),
        ).add_to(m)

        # Mark Destination (ADES)
        last_pt = actual_df.iloc[-1]
        folium.Marker(
            [last_pt["latitude"], last_pt["longitude"]],
            tooltip=f"Destination: {ades} (Arrival Delay: {flight.get('arr_delay_min', 0)} min)",
            icon=folium.Icon(color="red", icon="plane-arrival", prefix="fa"),
        ).add_to(m)

    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    # Floating Summary Card in bottom-left
    if crco_report:
        badges = []
        for idx, b in enumerate(fir_breakdown):
            col = fir_palette[idx % len(fir_palette)]
            badges.append(
                f'<span style="background-color: {col}; color: white; padding: 2px 6px; border-radius: 3px; font-weight: 500; margin-right: 4px; display: inline-block; margin-bottom: 2px;">{b.fir_id} (€{b.unit_rate_eur:.0f})</span>'
            )
        badge_str = "".join(badges)
        legend_html = f"""
        <div style="position: fixed; bottom: 20px; left: 15px; max-width: 420px;
                    background-color: white; border: 2px solid #bdc3c7; border-radius: 8px;
                    z-index: 9999; font-size: 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    padding: 10px 14px; box-shadow: 2px 2px 8px rgba(0,0,0,0.22);">
            <div style="font-size: 13.5px; font-weight: bold; color: #2c3e50; margin-bottom: 4px;">
                💶 Eurocontrol CRCO FIR Airspace Traversal Map
            </div>
            <div style="color: #555; margin-bottom: 6px;">
                <b>Total En-Route Fee:</b> <span style="color: #27ae60; font-weight: bold; font-size: 13px;">€{crco_report.actual_crco_charge_eur:,.2f} EUR</span> |
                <b>Total SUs:</b> <b>{crco_report.total_service_units:.2f} SU</b> |
                <b>Distance:</b> <b>{crco_report.actual_distance_nm:.0f} nm</b>
            </div>
            <div style="margin-top: 4px; font-size: 11.5px; line-height: 1.4;">
                <b>Traversed FIR Charging Zones:</b><br>{badge_str}
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    if output_html_path:
        out_p = Path(output_html_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_p))

    return m


def create_fra_boundary_map(
    boundary_pair: str,
    worst_flights: list[dict[str, Any]],
    fir_polygons: dict[str, list[tuple[float, float]]] | None = None,
    output_html_path: Path | str | None = None,
) -> folium.Map:
    """
    Generate an interactive Leaflet/Folium map visualizing the worst flights across a FIR boundary:
    - Shaded bounding FIR airspace sectors (semi-transparent polygons with distinct borders).
    - Actual radar flight paths (solid color-coded lines, highlighting severe dog-leg detours).
    - Direct Free Route Airspace (FRA) benchmark paths (dashed emerald green geodesic chords).
    - Departure (ADEP) and Arrival (ADES) airport circle pins.
    - Interactive tooltips and popups detailing detour NM, excess fuel burn (kg), and penalty (€ EUR).
    - Floating legend card and Folium LayerControl.
    """
    parts = [p.strip() for p in boundary_pair.split("<->")]
    fir_a = parts[0] if len(parts) >= 1 else "FIR-A"
    fir_b = parts[1] if len(parts) >= 2 else "FIR-B"

    # Compute bounding center
    all_lats: list[float] = []
    all_lons: list[float] = []

    if fir_polygons:
        for pts in fir_polygons.values():
            for pt in pts:
                if len(pt) >= 2 and _is_finite_coord(pt[0]) and _is_finite_coord(pt[1]):
                    all_lats.append(float(pt[0]))
                    all_lons.append(float(pt[1]))

    for f in worst_flights:
        for pt in f.get("trajectory_points", []):
            lat = pt.get("latitude")
            lon = pt.get("longitude")
            if _is_finite_coord(lat) and _is_finite_coord(lon):
                all_lats.append(float(lat))
                all_lons.append(float(lon))
        ad_lat, ad_lon = f.get("adep_lat"), f.get("adep_lon")
        if _is_finite_coord(ad_lat) and _is_finite_coord(ad_lon):
            all_lats.append(float(ad_lat))
            all_lons.append(float(ad_lon))
        as_lat, as_lon = f.get("ades_lat"), f.get("ades_lon")
        if _is_finite_coord(as_lat) and _is_finite_coord(as_lon):
            all_lats.append(float(as_lat))
            all_lons.append(float(as_lon))

    if all_lats and all_lons:
        c_lat = sum(all_lats) / len(all_lats)
        c_lon = circular_mean_longitude(all_lons)
        center_lat = float(c_lat) if _is_finite_coord(c_lat) else 48.5
        center_lon = float(c_lon) if _is_finite_coord(c_lon) else 4.0
    else:
        center_lat, center_lon = 48.5, 4.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles=None,
    )

    # 1. Open tile layers
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, Tiles style by Humanitarian OpenStreetMap Team hosted by OpenStreetMap France",
        name="Humanitarian (Unblocked)",
        subdomains="abc",
        show=True,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap France & OpenStreetMap contributors",
        name="OSM France (Unblocked)",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
        attr="USGS The National Map (Public Domain)",
        name="USGS Topo (Unblocked)",
        show=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: &copy; OpenStreetMap contributors, SRTM | Map style: &copy; OpenTopoMap",
        name="OpenTopoMap",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    # 2. Airspace FIR Sectors
    fir_group = folium.FeatureGroup(name="Airspace FIR Boundaries", show=True).add_to(m)
    if fir_polygons:
        colors = {fir_a: ("#2980b9", "#3498db"), fir_b: ("#d35400", "#e67e22")}
        for fid, poly in fir_polygons.items():
            border_c, fill_c = colors.get(fid, ("#7f8c8d", "#bdc3c7"))
            clean_poly = [
                (float(lat), float(lon))
                for lat, lon in poly
                if _is_finite_coord(lat) and _is_finite_coord(lon)
            ]
            if len(clean_poly) >= 3:
                folium.Polygon(
                    locations=clean_poly,
                    color=border_c,
                    weight=2.5,
                    fill=True,
                    fill_color=fill_c,
                    fill_opacity=0.12,
                    dash_array="5, 5",
                    tooltip=f"FIR Airspace Sector: {fid}",
                    popup=folium.Popup(f"<b>FIR Airspace Sector</b><br>{fid}", max_width=250),
                ).add_to(fir_group)

    # 3. Direct Free Route Airspace Paths (Geodesic Chords)
    fra_group = folium.FeatureGroup(name="Direct Free Route (FRA) Chords", show=True).add_to(m)
    for f in worst_flights:
        fra_pts = f.get("fra_direct_path", [])
        ad_lbl = f.get("adep_label", f.get("adep", "ADEP"))
        as_lbl = f.get("ades_label", f.get("ades", "ADES"))
        clean_fra_pts = [
            (float(p[0]), float(p[1]))
            for p in fra_pts
            if len(p) >= 2 and _is_finite_coord(p[0]) and _is_finite_coord(p[1])
        ]
        if len(clean_fra_pts) >= 2:
            create_safe_polyline(
                locations=clean_fra_pts,
                color="#27ae60",
                weight=2.5,
                dash_array="7, 7",
                opacity=0.85,
                tooltip=f"Direct FRA Benchmark: {ad_lbl} ➔ {as_lbl} ({f.get('great_circle_nm', 0):.0f} NM)",
            ).add_to(fra_group)

    # 4. Actual Radar Trajectories (Worst Inefficient Flights)
    traj_group = folium.FeatureGroup(name="Actual Radar Trajectories (Severe Detour)", show=True).add_to(m)
    airports_group = folium.FeatureGroup(name="Airports & ECAC Boundary Points", show=True).add_to(m)

    palette = [
        "#e74c3c", "#9b59b6", "#e67e22", "#16a085", "#2980b9",
        "#d35400", "#c0392b", "#8e44ad", "#1abc9c", "#34495e"
    ]

    total_detour_nm = sum(f.get("detour_nm", 0.0) for f in worst_flights)
    total_fuel_tonnes = sum(f.get("excess_fuel_tonnes", 0.0) for f in worst_flights)
    total_cost_eur = sum(f.get("excess_cost_eur", 0.0) for f in worst_flights)
    total_co2_tonnes = sum(f.get("excess_co2_tonnes", 0.0) for f in worst_flights)

    for idx, f in enumerate(worst_flights):
        col = palette[idx % len(palette)]
        rank = f.get("rank", idx + 1)
        fid = f["ectrl_id"]
        adep = f["adep"]
        ades = f["ades"]
        adep_lbl = f.get("adep_label", adep)
        ades_lbl = f.get("ades_label", ades)
        is_adep_apt = f.get("is_adep_ecac", True)
        is_ades_apt = f.get("is_ades_ecac", True)
        op = f["ac_operator"]
        ac = f["ac_type"]
        reg = f.get("ac_reg", "")
        reg_str = f" ({reg})" if reg and reg != "nan" else ""
        detour = f["detour_nm"]
        detour_pct = f["detour_pct"]
        act_dist = f["actual_distance_nm"]
        gc_dist = f["great_circle_nm"]
        cost = f["excess_cost_eur"]
        fuel_t = f["excess_fuel_tonnes"]
        co2_t = f["excess_co2_tonnes"]

        pts = [
            (float(pt["latitude"]), float(pt["longitude"]))
            for pt in f.get("trajectory_points", [])
            if _is_finite_coord(pt.get("latitude")) and _is_finite_coord(pt.get("longitude"))
        ]
        if len(pts) >= 2:
            tooltip_txt = (
                f"#{rank} {adep_lbl} ➔ {ades_lbl} | Flight {fid} [{op} {ac}] | "
                f"Detour: +{detour:.1f} NM (+{detour_pct:.1f}%) | "
                f"Waste: €{cost:,.0f} EUR"
            )
            popup_html = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; min-width: 240px;">
                <div style="font-size: 13.5px; font-weight: bold; color: {col}; margin-bottom: 4px;">
                    Flight {fid} (#{rank})
                </div>
                <b>Route:</b> {adep_lbl} &rarr; {ades_lbl}<br>
                <b>Operator:</b> {op} | <b>Aircraft:</b> {ac}{reg_str}<br>
                <hr style="margin: 5px 0; border: 0; border-top: 1px solid #e0e0e0;">
                <b>Actual Distance:</b> <b>{act_dist:.0f} NM</b><br>
                <b>Direct Geodesic:</b> <b>{gc_dist:.0f} NM</b><br>
                <span style="color: #c0392b; font-weight: bold;">Routing Detour: +{detour:.1f} NM (+{detour_pct:.1f}%)</span><br>
                <b>Excess Fuel:</b> <b>{fuel_t:.2f} tonnes</b> ({f.get('excess_fuel_kg', 0):.0f} kg)<br>
                <b>Operating Cost Penalty:</b> <span style="color: #c0392b; font-weight: bold;">€{cost:,.2f} EUR</span><br>
                <b>Excess CO₂:</b> <b>{co2_t:.2f} tonnes</b>
            </div>
            """
            create_safe_polyline(
                locations=pts,
                color=col,
                weight=3.5,
                opacity=0.9,
                tooltip=tooltip_txt,
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(traj_group)

        # Mark ADEP & ADES (or ECAC entry/exit points)
        adep_lat, adep_lon = f.get("adep_lat"), f.get("adep_lon")
        if _is_finite_coord(adep_lat) and _is_finite_coord(adep_lon):
            orig_tooltip = f"Departure: {adep_lbl} (Flight {fid})" if is_adep_apt else f"ECAC Entry Point: {adep_lbl} (Flight {fid})"
            folium.CircleMarker(
                location=[float(adep_lat), float(adep_lon)],
                radius=5,
                color="#27ae60",
                fill=True,
                fill_color="#2ecc71",
                fill_opacity=0.9,
                tooltip=orig_tooltip,
            ).add_to(airports_group)

        ades_lat, ades_lon = f.get("ades_lat"), f.get("ades_lon")
        if _is_finite_coord(ades_lat) and _is_finite_coord(ades_lon):
            dest_tooltip = f"Destination: {ades_lbl} (Flight {fid})" if is_ades_apt else f"ECAC Exit Point: {ades_lbl} (Flight {fid})"
            folium.CircleMarker(
                location=[float(ades_lat), float(ades_lon)],
                radius=5,
                color="#c0392b",
                fill=True,
                fill_color="#e74c3c",
                fill_opacity=0.9,
                tooltip=dest_tooltip,
            ).add_to(airports_group)

    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    # Floating Summary Card in bottom-left
    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 15px; max-width: 440px;
                background-color: white; border: 2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size: 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 10px 14px; box-shadow: 2px 2px 8px rgba(0,0,0,0.22);">
        <div style="font-size: 13.5px; font-weight: bold; color: #2c3e50; margin-bottom: 4px;">
            🗺️ Top 10 Inefficient Flights vs. FRA: <code>{boundary_pair}</code>
        </div>
        <div style="color: #444; margin-bottom: 6px; font-size: 11.5px;">
            <b>Total Detour (Top 10):</b> <span style="color: #c0392b; font-weight: bold;">+{total_detour_nm:,.0f} NM</span> |
            <b>Excess Fuel:</b> <b>{total_fuel_tonnes:.1f} t</b> |
            <b>Cost Penalty:</b> <span style="color: #c0392b; font-weight: bold;">€{total_cost_eur:,.0f} EUR</span>
        </div>
        <div style="margin-top: 4px; font-size: 11px; line-height: 1.4; color: #555;">
            <span style="color: #27ae60; font-weight: bold;">&#9644;&#9644; Dashed Green:</span> Direct Geodesic Free Route Airspace (FRA) Benchmark<br>
            <span style="color: #e74c3c; font-weight: bold;">&#9644;&#9644; Solid Colors:</span> Actual Dog-Legged Radar Trajectory Flown<br>
            <span style="color: #2980b9; font-weight: bold;">&#9644;&#9644; Shaded Polygons:</span> Bordering FIR Sectors (<code>{fir_a}</code> & <code>{fir_b}</code>)
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if output_html_path:
        out_p = Path(output_html_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_p))

    return m


def create_gnss_interference_network_map(
    dataset: Any,
    scan_report: Any,
    selected_ectrl_id: int | None = None,
    max_flights_rendered: int = 40,
    output_html_path: Path | str | None = None,
) -> folium.Map:
    """
    Generate an interactive Leaflet/Folium network map visualizing:
    - Known Geopolitical Electronic Warfare & GNSS Interference Hotspots (Baltic, East Med, Black Sea, Middle East, Red Sea).
    - Trajectories of flights exhibiting confirmed spoofing, suspected spoofing, and signal jamming.
    - Exact geographic anomaly pins where impossible teleportation jumps, vertical spikes, and dropouts occurred.
    - High-contrast highlight of an individually inspected flight.
    """
    from flightml.eval.gnss_jamming_spoofing import GNSS_HOTSPOTS

    m = folium.Map(
        location=[46.5, 24.0],
        zoom_start=4,
        tiles=None,
    )

    # 1. Humanitarian OpenStreetMap Basemap (Unblocked & referer-free)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, Humanitarian OpenStreetMap Team",
        name="Humanitarian (Unblocked)",
        subdomains="abc",
        show=True,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap France & OpenStreetMap contributors",
        name="OSM France (Unblocked)",
        subdomains="abc",
        show=False,
        control=True,
    ).add_to(m)

    # 2. Electronic Warfare & GNSS Hotspot Polygons Layer
    hotspots_layer = folium.FeatureGroup(name="⚠️ GNSS Conflict Hotspots", show=True).add_to(m)
    for h_key, zone in GNSS_HOTSPOTS.items():
        bounds = [
            [zone["min_lat"], zone["min_lon"]],
            [zone["max_lat"], zone["max_lon"]],
        ]
        color = zone.get("color", "#e74c3c")
        name = zone.get("name", h_key)
        desc = zone.get("description", "")

        poly_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; min-width: 250px;">
            <div style="font-size: 13.5px; font-weight: bold; color: {color}; margin-bottom: 4px;">
                ⚠️ {html.escape(name)}
            </div>
            <div style="color: #444; line-height: 1.4; margin-bottom: 6px;">
                {html.escape(desc)}
            </div>
            <div style="font-size: 11px; color: #7f8c8d;">
                <b>Coordinates:</b> Lat {zone['min_lat']}°–{zone['max_lat']}°N | Lon {zone['min_lon']}°–{zone['max_lon']}°E
            </div>
        </div>
        """
        folium.Rectangle(
            bounds=bounds,
            color=color,
            weight=2,
            dash_array="6, 6",
            fill=True,
            fill_color=color,
            fill_opacity=0.12,
            tooltip=f"⚠️ {name}",
            popup=folium.Popup(poly_html, max_width=320),
        ).add_to(hotspots_layer)

    # 3. Trajectory Layers by Anomaly Severity
    fg_confirmed = folium.FeatureGroup(name="🔴 Confirmed Spoofing (Speed/Altitude Teleport)", show=True).add_to(m)
    fg_suspected = folium.FeatureGroup(name="🟠 Suspected Spoofing (Track Jitter & Loops)", show=True).add_to(m)
    fg_jamming = folium.FeatureGroup(name="🟣 Probable Jamming (Position Dropouts)", show=True).add_to(m)
    fg_events = folium.FeatureGroup(name="📍 Anomaly Incident Trigger Pins", show=True).add_to(m)
    fg_highlight = folium.FeatureGroup(name="🎯 Inspected Flight Focus", show=True).add_to(m)

    flagged_flights = getattr(scan_report, "flagged_flights", []) or []
    rendered_count = 0

    for rep in flagged_flights:
        eid = rep.ectrl_id
        is_selected = (selected_ectrl_id is not None and int(selected_ectrl_id) == eid)

        # Skip if beyond display budget unless it is the selected flight
        if rendered_count >= max_flights_rendered and not is_selected:
            continue

        trajs = dataset.get_trajectory(eid)
        act_df = trajs.get("actual", pd.DataFrame())
        if act_df.empty or len(act_df) < 2:
            continue

        rendered_count += 1
        pts = [
            (float(r["latitude"]), float(r["longitude"]))
            for _, r in act_df.iterrows()
            if _is_finite_coord(r.get("latitude")) and _is_finite_coord(r.get("longitude"))
        ]
        if len(pts) < 2:
            continue

        # Color & Target Layer
        if rep.gnss_status == "CONFIRMED_SPOOFING":
            track_color = "#e74c3c"
            target_fg = fg_confirmed
            weight = 3.0
            dash = None
        elif rep.gnss_status == "SUSPECTED_SPOOFING":
            track_color = "#f39c12"
            target_fg = fg_suspected
            weight = 2.4
            dash = None
        else:
            track_color = "#8e44ad"
            target_fg = fg_jamming
            weight = 2.2
            dash = "5, 5"

        popup_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; min-width: 250px;">
            <div style="font-size: 14px; font-weight: bold; color: {track_color}; margin-bottom: 4px;">
                Flight #{eid} &nbsp;[{rep.ac_operator} | {rep.ac_type}]
            </div>
            <div style="color: #2c3e50; font-weight: 600; margin-bottom: 4px;">
                <code>{rep.adep}</code> ➔ <code>{rep.ades}</code> &nbsp;|&nbsp; Reg: <code>{rep.ac_reg}</code>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 6px 8px; margin-bottom: 6px;">
                <b>GNSS Status:</b> <span style="color: {track_color}; font-weight: bold;">{rep.gnss_status}</span><br/>
                <b>Anomaly Score:</b> <b>{rep.anomaly_score:.1f} / 100</b><br/>
                <b>Max Speed:</b> <span style="color: #c0392b;">{rep.max_groundspeed_kts:,.0f} kts</span><br/>
                <b>Max Vert Rate:</b> <span style="color: #c0392b;">{rep.max_vertical_speed_fpm:,.0f} fpm</span><br/>
                <b>Telemetry Gap:</b> {rep.max_telemetry_gap_s/60.0:.1f} min<br/>
                <b>Track Jitter:</b> {rep.max_track_jitter_deg:.0f}°
            </div>
            <div style="font-size: 11px; color: #555;">
                <b>Hotspots:</b> {', '.join(rep.hotspots_traversed) if rep.hotspots_traversed else 'Open Airspace'}
            </div>
        </div>
        """

        # Draw trajectory polyline
        create_safe_polyline(
            locations=pts,
            color=track_color,
            weight=weight,
            dash_array=dash,
            opacity=0.82,
            tooltip=f"#{eid} ({rep.adep}➔{rep.ades}) | {rep.gnss_status} ({rep.anomaly_score:.0f}/100)",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(target_fg)

        # Departure and Arrival Markers
        folium.CircleMarker(
            location=pts[0],
            radius=4,
            color="#27ae60",
            fill=True,
            fill_color="#27ae60",
            fill_opacity=0.9,
            tooltip=f"ADEP: {rep.adep} (#{eid})",
        ).add_to(target_fg)
        folium.CircleMarker(
            location=pts[-1],
            radius=4,
            color="#c0392b",
            fill=True,
            fill_color="#c0392b",
            fill_opacity=0.9,
            tooltip=f"ADES: {rep.ades} (#{eid})",
        ).add_to(target_fg)

        # Highlight Selected Flight if active with contrail-style multi-layer glowing segments
        if is_selected:
            # Base flown route
            create_safe_polyline(
                locations=pts,
                color="#8e44ad",
                weight=4.0,
                opacity=0.7,
                tooltip=f"🎯 FOCUS: Flight #{eid} ({rep.adep}➔{rep.ades})",
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(fg_highlight)

            # Imposed anomalous segments (casing + neon core + endpoint boundary markers)
            anom_segs = getattr(rep, "anomalous_segments", []) or []
            for seg in anom_segs:
                s_dict = seg.to_dict() if hasattr(seg, "to_dict") else (seg if isinstance(seg, dict) else seg.__dict__)
                lat1 = float(s_dict.get("start_lat", 0.0))
                lon1 = float(s_dict.get("start_lon", 0.0))
                lat2 = float(s_dict.get("end_lat", 0.0))
                lon2 = float(s_dict.get("end_lon", 0.0))
                if not (_is_finite_coord(lat1) and _is_finite_coord(lon1) and _is_finite_coord(lat2) and _is_finite_coord(lon2)):
                    continue

                a_type = str(s_dict.get("anomaly_type", "ANOMALY"))
                m_val = float(s_dict.get("metric_value", 0.0))
                m_unit = str(s_dict.get("metric_unit", ""))
                fl_st = float(s_dict.get("flight_level_start", 0.0))
                fl_en = float(s_dict.get("flight_level_end", 0.0))
                d_nm = float(s_dict.get("distance_nm", 0.0))
                d_s = float(s_dict.get("duration_s", 0.0))
                desc_txt = str(s_dict.get("description", ""))

                if a_type in ["SPEED_TELEPORTATION", "ALTITUDE_GLITCH"]:
                    core_col = "#ff0055"
                    fill_col = "#c0392b"
                elif a_type == "VERTICAL_RATE_SPIKE":
                    core_col = "#ff4757"
                    fill_col = "#d35400"
                elif a_type == "APPROACH_POSITION_JUMP":
                    core_col = "#e056fd"
                    fill_col = "#8e44ad"
                else:
                    core_col = "#ffa502"
                    fill_col = "#d35400"

                # 1. Dark casing polyline
                create_safe_polyline(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color="#2d0c3a",
                    weight=8,
                    opacity=0.80,
                ).add_to(fg_highlight)

                # 2. Glowing core line
                create_safe_polyline(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color=core_col,
                    weight=5.5,
                    opacity=0.98,
                    tooltip=f"⚡ {a_type}: {m_val:,.0f} {m_unit} | FL{fl_st:.0f}➔FL{fl_en:.0f} ({d_nm:.1f}nm)",
                    popup=folium.Popup(
                        f"""
                        <div style="font-family: sans-serif; min-width: 230px; font-size: 12px;">
                            <b style="color: {core_col}; font-size: 13.5px;">⚡ GNSS Telemetry Breach: {html.escape(a_type)}</b><br>
                            <hr style="margin: 4px 0;">
                            <b>Observed Metric:</b> {m_val:,.1f} {html.escape(m_unit)}<br>
                            <b>Altitude:</b> FL{fl_st:.0f} ➔ FL{fl_en:.0f}<br>
                            <b>Segment Duration:</b> {d_s:.0f}s ({d_nm:.1f} nm)<br>
                            <hr style="margin: 4px 0;">
                            <span style="color: #444; font-size: 11px;">{html.escape(desc_txt)}</span>
                        </div>
                        """,
                        max_width=290,
                    ),
                ).add_to(fg_highlight)

                # 3. Incident boundary markers
                folium.CircleMarker(
                    location=[lat1, lon1],
                    radius=5.5,
                    color=core_col,
                    fill=True,
                    fill_color=fill_col,
                    fill_opacity=1.0,
                    tooltip=f"Incident Point: FL{fl_st:.0f}",
                ).add_to(fg_highlight)

        # Plot individual Anomaly Trigger Pins (top events)
        for ev in rep.events[:4]:
            if _is_finite_coord(ev.latitude) and _is_finite_coord(ev.longitude):
                if ev.anomaly_type in ["SPEED_TELEPORTATION", "TELEPORTATION_SPEED"]:
                    p_col = "#e74c3c"
                    p_lbl = f"⚡ Teleport: {ev.metric_value:,.0f} kts"
                elif ev.anomaly_type == "VERTICAL_RATE_SPIKE":
                    p_col = "#d35400"
                    p_lbl = f"↕️ Vert Spike: {ev.metric_value:,.0f} fpm"
                elif ev.anomaly_type == "APPROACH_POSITION_JUMP":
                    p_col = "#e056fd"
                    p_lbl = f"🛬 Approach Jump: {ev.metric_value:,.1f} {ev.metric_unit}"
                elif ev.anomaly_type == "ALTITUDE_GLITCH":
                    p_col = "#c0392b"
                    p_lbl = f"⚠️ Alt Glitch: {ev.metric_value:,.0f} FL"
                else:
                    p_col = "#f39c12"
                    p_lbl = f"🔀 Track Jitter: {ev.metric_value:.0f}°"

                ev_popup = f"""
                <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 11.5px; min-width: 220px;">
                    <div style="font-weight: bold; color: {p_col}; font-size: 12.5px; margin-bottom: 2px;">
                        {html.escape(p_lbl)}
                    </div>
                    <div><b>Flight:</b> #{eid} ({rep.ac_operator}) | FL{ev.flight_level:.0f}</div>
                    <div><b>Time:</b> {ev.time_utc}</div>
                    <div style="margin-top: 4px; color: #444; line-height: 1.3;">{html.escape(ev.description)}</div>
                </div>
                """
                folium.CircleMarker(
                    location=[ev.latitude, ev.longitude],
                    radius=5,
                    color=p_col,
                    fill=True,
                    fill_color=p_col,
                    fill_opacity=0.85,
                    weight=1.5,
                    tooltip=f"#{eid}: {p_lbl}",
                    popup=folium.Popup(ev_popup, max_width=280),
                ).add_to(fg_events)

    # 4. Floating Legend & Statistics Overlay
    tot_flagged = getattr(scan_report, "flagged_flights_count", len(flagged_flights))
    conf_c = getattr(scan_report, "spoofing_confirmed_count", 0)
    susp_c = getattr(scan_report, "spoofing_suspected_count", 0)
    appr_c = getattr(scan_report, "jamming_probable_count", 0)

    legend_html = f"""
    <div style="position: fixed; bottom: 20px; left: 15px; max-width: 440px;
                background-color: white; border: 2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size: 12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 10px 14px; box-shadow: 2px 2px 8px rgba(0,0,0,0.22);">
        <div style="font-size: 13.5px; font-weight: bold; color: #2c3e50; margin-bottom: 4px;">
            🛰️ GNSS Jamming &amp; Spoofing Network Anomaly Overview
        </div>
        <div style="color: #444; margin-bottom: 6px; font-size: 11.5px;">
            <b>Flagged Flights:</b> <span style="color: #c0392b; font-weight: bold;">{tot_flagged:,}</span> |
            <b>Confirmed Spoofing:</b> <span style="color: #e74c3c; font-weight: bold;">{conf_c}</span> |
            <b>Suspected:</b> <span style="color: #f39c12; font-weight: bold;">{susp_c}</span>
        </div>
        <div style="font-size: 11px; line-height: 1.45; color: #555;">
            <span style="color: #ff0055; font-weight: bold;">&#9644;&#9644; Neon Red:</span> Speed Teleportation (&gt;750 kts) &amp; Altitude Glitches<br/>
            <span style="color: #ff4757; font-weight: bold;">&#9644;&#9644; Coral:</span> Unphysical Vertical Spikes (&gt;6,000 ft/min)<br/>
            <span style="color: #ffa502; font-weight: bold;">&#9644;&#9644; Amber:</span> Erratic Track Jitter &amp; False Circle Loops (&gt;65° flip)<br/>
            <span style="color: #e056fd; font-weight: bold;">&#9644;&#9644; Magenta:</span> Terminal Approach Position Jumps (&gt;320 kts below FL100)<br/>
            <span style="color: #8e44ad; font-weight: bold;">&#9644;&#9644; Purple:</span> Flown Radar Trajectory (Base Profile)<br/>
            <span style="color: #7f8c8d; font-weight: bold;">&#9633; Shaded Zones:</span> Known EW / GNSS Interference Conflict Hotspots
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    if output_html_path:
        out_p = Path(output_html_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_p))

    return m


def create_gnss_flight_route_map(
    dataset: Any,
    gnss_report: Any,
    output_html_path: Path | str | None = None,
) -> folium.Map:
    """
    Generate an interactive Leaflet/Folium map for an individual flight exhibiting GNSS anomalies,
    using the exact same multi-layer highlighted segment approach as the contrail ISSR map:
    - Flown radar trajectory (solid purple)
    - Imposed anomalous segments (dark casing + glowing neon core line: red/amber/magenta)
    - Incident origin/boundary circle markers
    - Electronic warfare hotspot boundary rectangles (if traversed)
    - Interactive hover tooltips and rich HTML diagnostic popups
    """
    from flightml.eval.gnss_jamming_spoofing import GNSS_HOTSPOTS

    eid = gnss_report.ectrl_id
    trajs = dataset.get_trajectory(eid)
    actual_df = trajs.get("actual", pd.DataFrame())

    if actual_df.empty or len(actual_df) < 2:
        # Fallback basic map
        return folium.Map(location=[50.0, 10.0], zoom_start=4)

    pts = [
        [float(r["latitude"]), float(r["longitude"])]
        for _, r in actual_df.iterrows()
        if _is_finite_coord(r.get("latitude")) and _is_finite_coord(r.get("longitude"))
    ]

    c_lat, c_lon = circular_mean_coords(pts)

    m = folium.Map(
        location=[c_lat, c_lon],
        zoom_start=5,
        tiles=None,
    )

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, Humanitarian OpenStreetMap Team",
        name="Humanitarian (Unblocked)",
        subdomains="abc",
        show=True,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        name="Satellite Imagery",
        show=False,
        control=True,
    ).add_to(m)

    # 1. Hotspots Layer
    hotspots_layer = folium.FeatureGroup(name="⚠️ Electronic Warfare Hotspot Zones", show=True).add_to(m)
    for k in getattr(gnss_report, "hotspots_traversed", []):
        zone = GNSS_HOTSPOTS.get(k)
        if zone:
            folium.Rectangle(
                bounds=[[zone["min_lat"], zone["min_lon"]], [zone["max_lat"], zone["max_lon"]]],
                color=zone.get("color", "#e74c3c"),
                weight=2,
                dash_array="6, 6",
                fill=True,
                fill_color=zone.get("color", "#e74c3c"),
                fill_opacity=0.15,
                tooltip=f"⚠️ {zone['name']}",
            ).add_to(hotspots_layer)

    # 2. Flown Route (Purple base line)
    route_group = folium.FeatureGroup(name=f"✈️ Flown Route (#{eid})", show=True).add_to(m)
    create_safe_polyline(
        locations=pts,
        color="#8e44ad",
        weight=4,
        opacity=0.85,
        tooltip=f"Flight #{eid}: {gnss_report.adep} ➔ {gnss_report.ades}",
    ).add_to(route_group)

    # ADEP / ADES markers
    folium.Marker(
        location=pts[0],
        tooltip=f"Departure: {gnss_report.adep}",
        icon=folium.Icon(color="green", icon="plane-departure", prefix="fa"),
    ).add_to(route_group)
    folium.Marker(
        location=pts[-1],
        tooltip=f"Destination: {gnss_report.ades}",
        icon=folium.Icon(color="red", icon="plane-arrival", prefix="fa"),
    ).add_to(route_group)

    # 3. Imposed Anomalous Telemetry Segments (Exact Contrail Highlighting Approach)
    anom_group = folium.FeatureGroup(name="⚡ Highlighted GNSS Anomaly Segments", show=True).add_to(m)
    anom_segs = getattr(gnss_report, "anomalous_segments", []) or []

    for idx, seg in enumerate(anom_segs):
        s_dict = seg.to_dict() if hasattr(seg, "to_dict") else (seg if isinstance(seg, dict) else seg.__dict__)
        lat1 = float(s_dict.get("start_lat", 0.0))
        lon1 = float(s_dict.get("start_lon", 0.0))
        lat2 = float(s_dict.get("end_lat", 0.0))
        lon2 = float(s_dict.get("end_lon", 0.0))
        if not (_is_finite_coord(lat1) and _is_finite_coord(lon1) and _is_finite_coord(lat2) and _is_finite_coord(lon2)):
            continue

        a_type = str(s_dict.get("anomaly_type", "ANOMALY"))
        m_val = float(s_dict.get("metric_value", 0.0))
        m_unit = str(s_dict.get("metric_unit", ""))
        fl_st = float(s_dict.get("flight_level_start", 0.0))
        fl_en = float(s_dict.get("flight_level_end", 0.0))
        dist_nm = float(s_dict.get("distance_nm", 0.0))
        dur_s = float(s_dict.get("duration_s", 0.0))
        desc = str(s_dict.get("description", ""))

        if a_type in ["SPEED_TELEPORTATION", "ALTITUDE_GLITCH"]:
            core_color = "#ff0055"
            fill_color = "#c0392b"
        elif a_type == "VERTICAL_RATE_SPIKE":
            core_color = "#ff4757"
            fill_color = "#d35400"
        elif a_type == "APPROACH_POSITION_JUMP":
            core_color = "#e056fd"
            fill_color = "#8e44ad"
        else:
            core_color = "#ffa502"
            fill_color = "#d35400"

        # Casing line (thick dark)
        create_safe_polyline(
            locations=[[lat1, lon1], [lat2, lon2]],
            color="#2d0c3a",
            weight=8,
            opacity=0.75,
        ).add_to(anom_group)

        # Glowing core line (neon)
        create_safe_polyline(
            locations=[[lat1, lon1], [lat2, lon2]],
            color=core_color,
            weight=5.5,
            opacity=0.98,
            tooltip=f"⚡ {a_type}: {m_val:,.0f} {m_unit} | FL{fl_st:.0f}➔FL{fl_en:.0f} ({dist_nm:.1f}nm in {dur_s:.0f}s)",
            popup=folium.Popup(
                f"""
                <div style="font-family: sans-serif; min-width: 230px; font-size: 12px;">
                    <b style="color: {core_color}; font-size: 13.5px;">⚡ GNSS Telemetry Breach: {html.escape(a_type)}</b><br>
                    <hr style="margin: 4px 0;">
                    <b>Observed Metric:</b> {m_val:,.1f} {html.escape(m_unit)}<br>
                    <b>Altitude:</b> FL{fl_st:.0f} ➔ FL{fl_en:.0f}<br>
                    <b>Segment Duration:</b> {dur_s:.0f}s ({dist_nm:.1f} nm)<br>
                    <hr style="margin: 4px 0;">
                    <span style="color: #444; font-size: 11px;">{html.escape(desc)}</span>
                </div>
                """,
                max_width=290,
            ),
        ).add_to(anom_group)

        # Segment boundary incident marker
        folium.CircleMarker(
            location=[lat1, lon1],
            radius=6.0,
            color=core_color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=1.0,
            tooltip=f"Incident Origin: FL{fl_st:.0f}",
        ).add_to(anom_group)

    # 4. Floating Legend & Header Box
    status_label = gnss_report.gnss_status
    status_color = "#ff0055" if status_label == "CONFIRMED_SPOOFING" else "#ffa502" if status_label == "SUSPECTED_SPOOFING" else "#27ae60"

    info_html = f"""
    <div style="position: fixed;
                bottom: 25px; left: 15px; width: 380px; height: 135px;
                background-color: white; border:2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size:12px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size: 13.5px; color: #2c3e50;">🛰️ GNSS Anomaly Inspector — #{eid}</b><br>
        <b>Route:</b> {gnss_report.adep} &rarr; {gnss_report.ades} ({gnss_report.ac_operator} - {gnss_report.ac_type})<br>
        <b>Verdict:</b> <span style="color: {status_color}; font-weight: bold;">{status_label}</span> (Score: {gnss_report.anomaly_score:.0f}/100)<br>
        <hr style="margin: 4px 0;">
        <span style="color: #8e44ad;">— Flown Route</span> |
        <span style="color: #ff0055; font-weight: bold;">━ Teleport / Glitch</span> |
        <span style="color: #ffa502; font-weight: bold;">━ Track Jitter</span><br>
        <span style="font-size: 11px; color: #7f8c8d;">Indicators: Speed Teleport, Vertical Spike, Approach Jump, Track Flip</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    if output_html_path:
        out_p = Path(output_html_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_p))

    return m





def create_ecac_coverage_map(
    coverage_report: dict[str, Any],
    selected_fl: int = 200,
    show_cells: bool = True,
    show_stations: bool = True,
    show_gap_fillers: bool = True,
    show_hubs: bool = True,
    output_html_path: Path | str | None = None,
) -> folium.Map:
    """
    Generate an interactive Leaflet/Folium coverage map for ECAC airspace:
    - 10x10 NM cell grid coverage at FL50, FL100, FL200, or FL300
    - Multi-tier color coding: Dual+ (green), Single (amber), Outage (red), Uncoverable Gap (slate)
    - Active stations (blue) vs Standby/Deactivated stations (gray)
    - Proposed synthetic gap-filling facilities (purple/magenta)
    - High-density hubs (gold) with FL10 terminal coverage status
    """
    target_fir = coverage_report.get("target_fir", "ALL")
    is_single_fir = target_fir not in (None, "ALL")
    map_center = coverage_report.get("fir_center", [50.0, 10.0])
    zoom_start = 6 if is_single_fir else 4

    m = folium.Map(
        location=map_center,
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Highlight single FIR boundary polygon if applicable
    fir_boundary = coverage_report.get("fir_boundary_coords")
    if is_single_fir and fir_boundary:
        fir_name = coverage_report.get("fir_name", target_fir)
        folium.PolyLine(
            locations=fir_boundary,
            color="#d97706",
            weight=3.5,
            opacity=0.9,
            dash_array="6, 6",
            tooltip=f"FIR Boundary: {fir_name}",
        ).add_to(m)

    fl_key = f"fl{selected_fl}"
    fl_data = coverage_report.get("coverage_by_fl", {}).get(fl_key, {})
    station_type = coverage_report.get("station_type", "RADAR")
    active_view = coverage_report.get("active_view", "VIEW_1")

    # 1. Coverage Grid Layer (10 NM Grid Squares)
    if show_cells and fl_data and "cells" in fl_data:
        cell_group = folium.FeatureGroup(name=f"10 NM Grid Squares (FL{selected_fl})", show=True).add_to(m)
        for cell in fl_data["cells"]:
            status = cell.get("status", "OUTAGE")
            cnt = cell.get("count", 0)
            lat, lon = cell["lat"], cell["lon"]
            geom_tip = cell.get("geom_tip", "")
            d_nm = cell.get("cell_size_nm", 10.0)
            dlat = (d_nm / 60.0) / 2.0
            dlon = (d_nm / (60.0 * max(0.1, math.cos(math.radians(lat))))) / 2.0

            if status in ("MET", "DUAL_OR_MORE"):
                color = "#10b981"
                tip = f"🟢 MET: Performance Standard Achieved | {cnt} {station_type}s | {geom_tip}"
                opac = 0.55
            elif status in ("AT_RISK", "SINGLE"):
                color = "#f59e0b"
                tip = f"🟠 AT RISK / MARGINAL: {cnt} {station_type}s | {geom_tip}"
                opac = 0.65
            elif status in ("CANNOT_BE_MET", "OUTAGE"):
                color = "#ef4444"
                tip = f"🔴 CANNOT BE MET: 0 active {station_type}s at FL{selected_fl}"
                opac = 0.75
            else:  # UNCOVERABLE
                color = "#94a3b8"
                tip = f"⚪ Dataset Gap: Uncoverable with current inventory at FL{selected_fl}"
                opac = 0.25

            folium.Rectangle(
                bounds=[[lat - dlat, lon - dlon], [lat + dlat, lon + dlon]],
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=opac,
                weight=0.5,
                tooltip=tip,
            ).add_to(cell_group)

    # 2. Stations Layer (Active vs Deactivated)
    if show_stations and "stations" in coverage_report:
        sta_active_group = folium.FeatureGroup(name=f"Active {station_type}s", show=True).add_to(m)
        sta_inact_group = folium.FeatureGroup(name=f"Deactivated (Standby) {station_type}s", show=True).add_to(m)

        for s in coverage_report["stations"]:
            is_act = s.get("is_active", True)
            lat, lon = s["latitude"], s["longitude"]
            s_id = s.get("id", "")
            s_name = s.get("name", s_id)
            near_apt = s.get("nearest_airport")
            apt_txt = f" (Airport: {near_apt})" if near_apt else ""

            if is_act:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5.5,
                    color="#1d4ed8",
                    fill=True,
                    fill_color="#3b82f6",
                    fill_opacity=0.9,
                    weight=1.5,
                    tooltip=f"ACTIVE {station_type}: {s_id} - {s_name}{apt_txt}",
                ).add_to(sta_active_group)
            else:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4.0,
                    color="#475569",
                    fill=True,
                    fill_color="#94a3b8",
                    fill_opacity=0.6,
                    weight=1.0,
                    tooltip=f"STANDBY {station_type}: {s_id} - {s_name} [Deactivated for Rationalization]",
                ).add_to(sta_inact_group)

    # 3. Suggested Adjacent FIR Cross-Border Infill Stations
    if show_gap_fillers and coverage_report.get("adjacent_fir_infill_suggestions"):
        adj_group = folium.FeatureGroup(name="Suggested Adjacent FIR Cross-Border Infill", show=True).add_to(m)
        for s in coverage_report["adjacent_fir_infill_suggestions"]:
            folium.Marker(
                location=[s["latitude"], s["longitude"]],
                icon=folium.Icon(color="darkblue", icon="share-alt", prefix="fa"),
                tooltip=f"ADJACENT FIR INFILL: {s['id']} - {s['name']} ({s['country']}) | Covers {s['cells_covered_count']} internal gap cells | €0 Capital Cost",
            ).add_to(adj_group)

    # 4. Proposed Synthetic Gap-Filler Stations (Strictly on Land)
    if show_gap_fillers and "synthetic_gap_fillers" in coverage_report:
        gap_group = folium.FeatureGroup(name="Proposed Gap-Filler Infill Facilities (Land)", show=True).add_to(m)
        for g in coverage_report["synthetic_gap_fillers"]:
            folium.Marker(
                location=[g["latitude"], g["longitude"]],
                icon=folium.Icon(color="purple", icon="star", prefix="fa"),
                tooltip=f"PROPOSED INFILL: {g['id']} - {g['name']} | Covers {g['cells_covered_count']} gap cells | {g['region_note']}",
            ).add_to(gap_group)

    # 4. High-Density Hubs
    if show_hubs and "airports" in coverage_report:
        hub_group = folium.FeatureGroup(name="High-Density Hubs (>=500 mov/day)", show=True).add_to(m)
        for a in coverage_report["airports"]:
            if not a.get("is_high_density", False):
                continue
            c10 = a.get("coverage_count_fl10", 0)
            icon_color = "green" if c10 >= 2 else "orange" if c10 == 1 else "red"
            folium.Marker(
                location=[a["latitude"], a["longitude"]],
                icon=folium.Icon(color=icon_color, icon="plane", prefix="fa"),
                tooltip=f"HUB {a['ident']} ({a['name']}): {a['daily_movements']:.0f} mov/day | Surface/FL10 {station_type}s: {c10}",
            ).add_to(hub_group)

    # 5. Legend & KPI Box
    single_pct = fl_data.get("single_coverage_pct", 0.0)
    dual_pct = fl_data.get("dual_coverage_pct", 0.0)
    act_sta = coverage_report.get("active_stations_count", 0)
    tot_sta = coverage_report.get("total_stations_count", 0)
    savings_eur = coverage_report.get("annual_cost_savings_eur", 0.0)
    gap_count = len(coverage_report.get("synthetic_gap_fillers", []))

    met_pct = fl_data.get("met_pct", dual_pct)
    at_risk_pct = fl_data.get("at_risk_pct", single_pct)
    outage_pct = fl_data.get("outage_pct", 0.0)

    legend_html = f"""
    <div style="position: fixed;
                bottom: 25px; left: 15px; width: 400px; height: 195px;
                background-color: white; border: 2px solid #94a3b8; border-radius: 8px;
                z-index: 9999; font-size: 11px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 10px; box-shadow: 2px 2px 8px rgba(0,0,0,0.25);">
        <b style="font-size: 12.5px; color: #1e293b;">🇪🇺 ECAC {station_type} Rationalization Map — FL{selected_fl}</b><br>
        <b>Standard / View:</b> <span style="color: #2563eb; font-weight: bold;">{active_view}</span> | <b>Active:</b> {act_sta}/{tot_sta}<br>
        <b>10 NM Performance:</b> <span style="color: #10b981; font-weight: bold;">Met: {met_pct:.1f}%</span> |
        <span style="color: #f59e0b; font-weight: bold;">At Risk: {at_risk_pct:.1f}%</span> |
        <span style="color: #ef4444; font-weight: bold;">Outage: {outage_pct:.1f}%</span><br>
        <b>Annual OPEX Savings:</b> <span style="color: #059669; font-weight: bold;">€{savings_eur:,.0f}/yr</span><br>
        <hr style="margin: 4px 0;">
        <span style="color: #10b981; font-size: 14px;">■</span> <b>Green:</b> Performance Met (Valid DME Pair 30°-150°, Range OK)<br>
        <span style="color: #f59e0b; font-size: 14px;">■</span> <b>Orange:</b> At Risk / Marginal (Single DME / No N-1 Redundancy)<br>
        <span style="color: #ef4444; font-size: 14px;">■</span> <b>Red:</b> Outage / Cannot be Met &nbsp;&nbsp;
        <span style="color: #94a3b8; font-size: 14px;">■</span> <b>Gray:</b> Uncoverable Baseline Gap<br>
        <span style="color: #3b82f6;">●</span> Active Station &nbsp;&nbsp;
        <span style="color: #64748b;">●</span> Standby Station &nbsp;&nbsp;
        <span style="color: #1e40af;">●</span> Adjacent FIR Infill &nbsp;&nbsp;
        <span style="color: #9333ea;">★</span> Proposed Land Infill
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    if output_html_path:
        out_p = Path(output_html_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_p))

    return m
