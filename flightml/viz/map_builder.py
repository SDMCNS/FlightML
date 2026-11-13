from __future__ import annotations

from pathlib import Path
from typing import Any
import folium
import pandas as pd

from flightml.geo.trajectory import analyze_trajectory


def create_trajectory_map(
    flight_info: dict[str, Any],
    actual_df: pd.DataFrame,
    planned_df: pd.DataFrame,
    output_html_path: Path | str | None = None,
) -> folium.Map:
    """
    Generate an interactive Leaflet/Folium map visualizing:
    - Planned 4D flight path (dashed blue)
    - Actual radar trajectory (solid purple)
    - Detected airborne holding stacks (orange/red circles)
    - Departure (ADEP) & Destination (ADES) airport pins
    - Zero-API-key, completely unblocked open tile layers (OSM France HOT, OpenTopoMap, USGS Topo)
    """
    ectrl_id = flight_info.get("ectrl_id", 0)
    adep = flight_info.get("adep", "ADEP")
    ades = flight_info.get("ades", "ADES")
    op = flight_info.get("ac_operator", "OP")
    ac = flight_info.get("ac_type", "AC")

    # Center map on mid-point
    if not actual_df.empty:
        center_lat = float(actual_df["latitude"].mean())
        center_lon = float(actual_df["longitude"].mean())
    elif not planned_df.empty:
        center_lat = float(planned_df["latitude"].mean())
        center_lon = float(planned_df["longitude"].mean())
    else:
        center_lat, center_lon = 50.0, 10.0

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

    # Add layer control to toggle basemaps
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    # Add title header box (bottom-left to avoid layer control collision)
    title_html = f"""
    <div style="position: fixed;
                bottom: 25px; left: 15px; width: 340px; height: 115px;
                background-color: white; border:2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size:13px; font-family: sans-serif;
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size: 15px; color: #2c3e50;">✈️ Flight {ectrl_id} ({op} - {ac})</b><br>
        <b>Route:</b> {adep} &rarr; {ades}<br>
        <b>Distance:</b> Actual {metrics.actual_distance_nm:.0f} nm (Plan: {metrics.planned_distance_nm:.0f} nm)<br>
        <b>Efficiency Ratio:</b> {metrics.horizontal_extension_ratio:.3f}x | <b>Holdings:</b> {len(metrics.holding_events)}<br>
        <span style="color: #2980b9;">-- Filed Plan</span> | <span style="color: #8e44ad;">— Actual Track</span> | <span style="color: #e74c3c;">⭕ Holding</span>
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

    status_colors = {
        "CRITICAL_SATURATION": ("#c0392b", 0.48),
        "HIGH_DENSITY": ("#e67e22", 0.40),
        "MODERATE_FLOW": ("#f1c40f", 0.32),
        "LOW_DENSITY": ("#2ecc71", 0.22),
    }

    for c in cells:
        lat0 = float(c["cell_lat"])
        lon0 = float(c["cell_lon"])
        lat1 = lat0 + 5.0
        lon1 = lon0 + 5.0
        status = str(c.get("saturation_status", "LOW_DENSITY"))
        color, fill_op = status_colors.get(status, ("#2ecc71", 0.22))
        avg_f = float(c.get("avg_flights_per_hour", 0.0))
        peak_f = int(c.get("peak_flights_per_hour", 0))
        vert_pct = float(c.get("vertical_transition_pct", 0.0))
        comp_score = float(c.get("dynamic_complexity_score", 0.0))
        avg_fl = float(c.get("avg_flight_level", 320.0))
        sector_lbl = str(c.get("grid_sector", f"{lat0:.0f}°N, {lon0:.0f}°E"))

        popup_html = f"""
        <div style="font-family: sans-serif; min-width: 220px; font-size: 12.5px;">
            <h4 style="margin: 0 0 6px 0; color: {color};">5°×5° Cell: {sector_lbl}</h4>
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
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=fill_op,
            tooltip=(
                f"{sector_lbl} | Avg: {avg_f:.1f} flts/hr (Peak: {peak_f}) | "
                f"Climb/Desc: {vert_pct:.1f}% | Complexity: {comp_score:.1f}"
            ),
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(m)

        # Render inline badge inside medium/high density squares
        if avg_f >= 15.0:
            folium.Marker(
                location=[lat0 + 2.5, lon0 + 2.5],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-family: sans-serif; font-size: 11px; font-weight: bold; '
                        f'color: #ffffff; background: rgba(44,62,80,0.78); padding: 2px 5px; '
                        f'border-radius: 4px; text-align: center; white-space: nowrap; '
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
        <b style="font-size: 13.5px; color: #2c3e50;">🗺️ 5°×5° Hourly Airspace Saturation ({hr:02d}:00 UTC, {period})</b><br>
        <span style="color: #c0392b;">■ Critical Saturation</span> |
        <span style="color: #e67e22;">■ High Density</span> |
        <span style="color: #f1c40f;">■ Moderate</span> |
        <span style="color: #2ecc71;">■ Low</span>
        {"<br><span style='color: #ff0033;'>● Co-Altitude Proximity Alert (&le;5 nm / &le;1,000 ft)</span>" if proximity_events else ""}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m
