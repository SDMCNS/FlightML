from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import folium
import pandas as pd

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


def create_trajectory_map(
    flight_info: dict[str, Any],
    actual_df: pd.DataFrame,
    planned_df: pd.DataFrame,
    output_html_path: Path | str | None = None,
    active_navaids: list[Any] | None = None,
    show_navaids: bool = True,
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
            c_lon = float(valid_act["longitude"].mean())
            if _is_finite_coord(c_lat) and _is_finite_coord(c_lon):
                center_lat, center_lon = c_lat, c_lon
    elif not planned_df.empty and "latitude" in planned_df.columns and "longitude" in planned_df.columns:
        valid_plan = planned_df.dropna(subset=["latitude", "longitude"])
        if not valid_plan.empty:
            c_lat = float(valid_plan["latitude"].mean())
            c_lon = float(valid_plan["longitude"].mean())
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
        folium.PolyLine(
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
        folium.PolyLine(
            actual_coords,
            color="#8e44ad",
            weight=4,
            opacity=0.95,
            tooltip=f"Actual Flown Track ({metrics.actual_distance_nm:.1f} nm)",
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
            folium.PolyLine(
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
    box_height = 138 if (active_navaids and show_navaids) else 115
    title_html = f"""
    <div style="position: fixed;
                bottom: 25px; left: 15px; width: 355px; height: {box_height}px;
                background-color: white; border:2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size:12.5px; font-family: sans-serif;
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size: 14.5px; color: #2c3e50;">✈️ Flight {ectrl_id} ({op} - {ac})</b><br>
        <b>Route:</b> {adep} &rarr; {ades}<br>
        <b>Distance:</b> Actual {metrics.actual_distance_nm:.0f} nm (Plan: {metrics.planned_distance_nm:.0f} nm)<br>
        <b>Efficiency Ratio:</b> {metrics.horizontal_extension_ratio:.3f}x | <b>Holdings:</b> {len(metrics.holding_events)}<br>
        <span style="color: #2980b9;">-- Filed Plan</span> | <span style="color: #8e44ad;">— Actual Track</span> | <span style="color: #e74c3c;">⭕ Holding</span>
        {nav_legend_line}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

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
            c_lon = float(valid_act["longitude"].mean())
            if _is_finite_coord(c_lat) and _is_finite_coord(c_lon):
                center_lat, center_lon = c_lat, c_lon
    elif not planned_df.empty and "latitude" in planned_df.columns and "longitude" in planned_df.columns:
        valid_plan = planned_df.dropna(subset=["latitude", "longitude"])
        if not valid_plan.empty:
            c_lat = float(valid_plan["latitude"].mean())
            c_lon = float(valid_plan["longitude"].mean())
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
        folium.PolyLine(
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
        folium.PolyLine(
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
        c_lon = sum(all_lons) / len(all_lons)
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
        clean_fra_pts = [
            (float(p[0]), float(p[1]))
            for p in fra_pts
            if len(p) >= 2 and _is_finite_coord(p[0]) and _is_finite_coord(p[1])
        ]
        if len(clean_fra_pts) >= 2:
            folium.PolyLine(
                locations=clean_fra_pts,
                color="#27ae60",
                weight=2.5,
                dash_array="7, 7",
                opacity=0.85,
                tooltip=f"Direct FRA Benchmark: {f.get('adep', 'ADEP')} ➔ {f.get('ades', 'ADES')} ({f.get('great_circle_nm', 0):.0f} NM)",
            ).add_to(fra_group)

    # 4. Actual Radar Trajectories (Worst Inefficient Flights)
    traj_group = folium.FeatureGroup(name="Actual Radar Trajectories (Severe Detour)", show=True).add_to(m)
    airports_group = folium.FeatureGroup(name="Airports (ADEP / ADES)", show=True).add_to(m)

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
                f"#{rank} {adep} ➔ {ades} | Flight {fid} [{op} {ac}] | "
                f"Detour: +{detour:.1f} NM (+{detour_pct:.1f}%) | "
                f"Waste: €{cost:,.0f} EUR"
            )
            popup_html = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; min-width: 220px;">
                <div style="font-size: 13.5px; font-weight: bold; color: {col}; margin-bottom: 4px;">
                    Flight {fid} (#{rank})
                </div>
                <b>Route:</b> {adep} &rarr; {ades}<br>
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
            folium.PolyLine(
                locations=pts,
                color=col,
                weight=3.5,
                opacity=0.9,
                tooltip=tooltip_txt,
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(traj_group)

        # Mark ADEP & ADES
        adep_lat, adep_lon = f.get("adep_lat"), f.get("adep_lon")
        if _is_finite_coord(adep_lat) and _is_finite_coord(adep_lon):
            folium.CircleMarker(
                location=[float(adep_lat), float(adep_lon)],
                radius=5,
                color="#27ae60",
                fill=True,
                fill_color="#2ecc71",
                fill_opacity=0.9,
                tooltip=f"Departure: {adep} (Flight {fid})",
            ).add_to(airports_group)

        ades_lat, ades_lon = f.get("ades_lat"), f.get("ades_lon")
        if _is_finite_coord(ades_lat) and _is_finite_coord(ades_lon):
            folium.CircleMarker(
                location=[float(ades_lat), float(ades_lon)],
                radius=5,
                color="#c0392b",
                fill=True,
                fill_color="#e74c3c",
                fill_opacity=0.9,
                tooltip=f"Destination: {ades} (Flight {fid})",
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


