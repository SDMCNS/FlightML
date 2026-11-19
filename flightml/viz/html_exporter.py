from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import io
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import folium

from flightml.data.dataset import FlightDataset
from flightml.eval.aircraft_engine_perf import resolve_aircraft_engine_perf
from flightml.eval.cdo_analyzer import analyze_cco_climb, analyze_cdo_descent
from flightml.eval.contrail_climate import analyze_flight_wind_and_contrails
from flightml.eval.crco_economics import compute_flight_crco_charges
from flightml.eval.diagnostics import diagnose_flight_bottlenecks
from flightml.eval.eco_route_optimizer import optimize_eco_minimal_contrail_trajectory
from flightml.eval.flight_notes import get_notes_manager
from flightml.eval.flight_twins import compare_twin_flights
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
from flightml.eval.macro_comparison import compare_aviation_periods
from flightml.eval.pru_kpis import analyze_network_level_capping, compute_flight_pru_kpis
from flightml.eval.recurring_series import analyze_recurring_flight_series
from flightml.eval.surface_taxi import (
    analyze_airport_taxi_out_performance,
    analyze_flight_taxi_out,
)
from flightml.features.aua_saturation import calculate_aua_hourly_saturation
from flightml.features.fir_saturation import (
    calculate_fir_hourly_saturation,
    calculate_flight_fir_dwell_inefficiency,
)
from flightml.geo.airac_diff import compare_airac_cycles
from flightml.geo.conflict_scanner import (
    compute_hourly_grid_saturation,
    scan_airspace_proximity_events,
)
from flightml.geo.fra_analyzer import analyze_flight_fra_adherence
from flightml.geo.fra_boundary_engine import (
    analyze_fir_boundary_inefficiencies,
    calculate_fra_long_term_forecast,
    get_worst_flights_for_boundary,
    interpolate_great_circle,
)
from flightml.geo.rnp_navaid_analyzer import (
    analyze_flight_navaids_and_rnp,
    query_unified_navaids,
)
from flightml.geo.trajectory import analyze_trajectory
from flightml.models.airline_stress import stress_test_airline
from flightml.sim.gdp_optimizer import optimize_ground_delay_program
from flightml.sim.tail_swap_optimizer import optimize_hub_tail_swaps
from flightml.sim.what_if import simulate_turnaround_mitigation
from flightml.eval.ecac_station_rationalization import get_ecac_rationalization_report
from flightml.viz.map_builder import (
    create_contrail_horizontal_map,
    create_ecac_coverage_map,
    create_flight_fir_charge_map,
    create_trajectory_map,
)
from flightml.weather.metar_client import MetarClient
from flightml.weather.runway_safety import assess_flight_runway_safety


REPORT_PAGE_KEYS = [
    "single_flight",
    "system_summary",
    "what_if_scenarios",
    "maps_navigation",
    "general_economics",
    "longitudinal_trends",
    "airspace_overview",
    "fra_boundaries",
    "trajectory_rnp",
    "global_navaids",
    "ecac_dme_rationalization",
    "ecac_radar_rationalization",
    "dme_resilience",
    "flight_performance_eco",
    "vertical_cdo_cco",
    "delay_diagnostics",
    "fuel_carbon",
    "corridor_twin_benchmarks",
    "what_if_stress_tests",
    "macro_reference",
    "full_executive_bundle",
]

REPORT_PAGE_TITLES: dict[str, str] = {
    # Master Architectural Categories
    "single_flight": "✈️ Single Flight & Aircraft Analysis Report",
    "system_summary": "🌐 System-Level Summary & Analysis Report",
    "longitudinal_trends": "📈 Longitudinal & Predictive Trends Report",
    "what_if_scenarios": "🔮 What-If Scenarios & Operations Research Report",
    "maps_navigation": "🗺️ Information & Aeronautical Maps Report",
    "general_economics": "💶 General Aviation & Route Economics Report (€)",
    # Detailed sub-reports & bundle
    "airspace_overview": "🌐 Airspace & Sector Saturation Report",
    "fra_boundaries": "🌐 Free Route Airspace (FRA) Boundary Analysis & Forecaster Report",
    "trajectory_rnp": "🗺️ 4D Trajectory, Active Navaids & RNP Report",
    "global_navaids": "📡 Global Radio Navaids World Map Report",
    "ecac_dme_rationalization": "📻 ECAC DME-DME RNAV Backup & Rationalization Report",
    "ecac_radar_rationalization": "🛰️ ECAC Radar Surveillance (SSR/Mode S) Rationalization Report",
    "dme_resilience": "📉 DME Network Resilience, Outage Permutations & Operational Costs Report",
    "flight_performance_eco": "📐 Flight Performance, Delay & Eco (€) Report",
    "vertical_cdo_cco": "📉 Vertical Profile & CDO/CCO Analysis Report",
    "delay_diagnostics": "🩺 Delay Diagnostics & Runway Safety Report",
    "fuel_carbon": "🌱 Fuel, Carbon & Surface LTO Report",
    "corridor_twin_benchmarks": "📊 Corridor & Twin-Flight Benchmarks Report",
    "what_if_stress_tests": "🔮 What-If Simulations & Stress Tests Report",
    "macro_reference": "📈 Macro Evolution & Methodology Reference Report",
    "full_executive_bundle": "📦 Complete All-Pages Executive Intelligence Report",
}


def _fig_to_base64_img(fig: plt.Figure, alt_text: str = "Chart") -> str:
    """Encode a Matplotlib Figure into a self-contained HTML <img> tag with base64 PNG data."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    safe_alt = html.escape(alt_text)
    return (
        f'<div class="chart-card">'
        f'<img src="data:image/png;base64,{b64}" alt="{safe_alt}" class="report-img" />'
        f'<div class="chart-caption">{safe_alt}</div>'
        f'</div>'
    )


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 35) -> str:
    """Render a Pandas DataFrame as a styled HTML table."""
    if df is None or df.empty:
        return '<p class="muted-note">No records available for the current filter selection.</p>'
    sub = df.head(max_rows).copy()
    return f'<div class="table-wrap">{sub.to_html(index=False, border=0, classes="styled-table")}</div>'


def _kpi_grid_html(metrics: list[tuple[str, str, str]]) -> str:
    """Render a responsive row of KPI cards: (label, value, subtext)."""
    cards = []
    for lbl, val, sub in metrics:
        sub_html = f'<div class="kpi-sub">{html.escape(str(sub))}</div>' if sub else ""
        cards.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{html.escape(str(lbl))}</div>'
            f'<div class="kpi-value">{html.escape(str(val))}</div>'
            f'{sub_html}'
            f'</div>'
        )
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def _folium_map_to_iframe(m: Any, height: int = 520, title: str = "Interactive Map") -> str:
    """
    Render a Folium Leaflet Map into an isolated, self-contained iframe srcdoc container.
    Guarantees full Esri Light Gray / OpenStreetMap Humanitarian tiles render properly in standalone HTML
    exports without CSS/JS DOM collisions, and prevents 'access blocked' referrer errors in local file:/// contexts.
    """
    if m is None:
        return f'<div class="chart-card"><p class="muted-note">Map visualization not available for {html.escape(title)}.</p></div>'
    try:
        rendered_html = m.get_root().render()

        # Sanitize any legacy tile URLs that trigger 403 or 'access blocked' watermarks on local file:// or null referrer
        rendered_html = (
            rendered_html.replace(
                "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            )
            .replace(
                "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            )
            .replace(
                "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            )
        )

        # Inject referrer-friendly meta tag into iframe head
        if "<head>" in rendered_html:
            rendered_html = rendered_html.replace(
                "<head>",
                '<head>\n    <meta name="referrer" content="no-referrer-when-downgrade">',
                1,
            )

        escaped_srcdoc = html.escape(rendered_html)
        safe_title = html.escape(title)
        return (
            f'<div class="chart-card" style="padding: 8px; margin: 18px 0; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px;">\n'
            f'  <div class="chart-caption" style="text-align: left; padding: 6px 12px 10px 12px; font-size: 13.5px; font-weight: 700; color: #0c2461;">'
            f'    🗺️ {safe_title} — Interactive Basemap (Pan, Zoom, Layer Control &amp; Hover Inspect)'
            f'  </div>\n'
            f'  <iframe srcdoc="{escaped_srcdoc}" '
            f'style="width: 100%; height: {height}px; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);" '
            f'loading="lazy" title="{safe_title}"></iframe>\n'
            f'</div>'
        )
    except Exception as exc:
        return f'<div class="chart-card"><p class="muted-note">Failed to render interactive map {html.escape(title)}: {html.escape(str(exc))}</p></div>'


# ---------------------------------------------------------------------------
# Section 1: Airspace & Sector Saturation
# ---------------------------------------------------------------------------
def _build_airspace_overview_section(ds: FlightDataset, hour_utc: int = 10) -> str:
    stats = ds.get_summary()
    grid_data = compute_hourly_grid_saturation(ds, hour_utc=hour_utc)

    kpis = [
        ("Total Processed Flights", f"{stats['total_flights']:,}", f"Period {ds.period}"),
        ("Aircraft Operators", f"{stats['total_operators']:,}", f"{stats['unique_dep_airports']:,} Departure Airports"),
        ("Mean Arrival Delay", f"{stats['avg_arr_delay_min']} min", "Network-wide average"),
        ("Active 5°×5° Squares", f"{grid_data['total_active_squares']}", f"At {hour_utc:02d}:00 UTC"),
        ("Critical / High Squares", f"{grid_data['critical_squares_count']} / {grid_data['high_density_squares_count']}", f"Peak {grid_data['max_avg_flights_per_hour']:.1f} flts/hr"),
    ]

    # Plot 1: 5°×5° European Airspace Hourly Saturation Map + Top Hubs Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax1.set_facecolor("#eef5f9")
    status_colors = {
        "CRITICAL_SATURATION": "#c0392b",
        "HIGH_DENSITY": "#e67e22",
        "MODERATE_FLOW": "#f1c40f",
        "LOW_DENSITY": "#2ecc71",
    }
    cells = grid_data.get("cells") or []
    for c in cells:
        lat0 = float(c["cell_lat"])
        lon0 = float(c["cell_lon"])
        st_lbl = str(c.get("saturation_status", "LOW_DENSITY"))
        col = status_colors.get(st_lbl, "#2ecc71")
        avg_f = float(c.get("avg_flights_per_hour", 0.0))
        rect = plt.Rectangle((lon0, lat0), 5.0, 5.0, facecolor=col, edgecolor="#2c3e50", alpha=0.62, linewidth=0.9)
        ax1.add_patch(rect)
        if avg_f >= 10.0:
            ax1.text(
                lon0 + 2.5,
                lat0 + 2.5,
                f"{avg_f:.0f}/h",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold",
                color="#1e272e",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
    ax1.set_xlim(-25, 45)
    ax1.set_ylim(30, 70)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.set_title(f"5°×5° Airspace Hourly Saturation ({hour_utc:02d}:00 UTC, {ds.period})", fontweight="bold")

    top_apt = ds.get_top_airports(10)
    if not top_apt.empty:
        apt_sub = top_apt.iloc[::-1]
        lbl_col = "ident" if "ident" in apt_sub.columns else apt_sub.columns[0]
        cnt_col = "departure_count" if "departure_count" in apt_sub.columns else apt_sub.columns[-2]
        ax2.barh(apt_sub[lbl_col].astype(str), apt_sub[cnt_col].astype(float), color="#2980b9", alpha=0.88)
        ax2.set_xlabel("Total Departures")
        ax2.set_title("Top 10 Busiest Departure Hubs", fontweight="bold")
        ax2.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    chart_html = _fig_to_base64_img(
        fig,
        f"Figure 1: 5°×5° European Airspace Hourly Traffic Density Map ({hour_utc:02d}:00 UTC) & Top Departure Hubs ({ds.period})",
    )

    sq_df = pd.DataFrame(grid_data.get("squares") or [])
    if not sq_df.empty:
        sq_df = sq_df[
            [
                "cell_label",
                "saturation_status",
                "avg_flights_per_hour",
                "peak_flights_per_hour",
                "vertical_transition_pct",
                "avg_flight_level",
                "dynamic_complexity_score",
            ]
        ].rename(
            columns={
                "cell_label": "5°×5° Square",
                "saturation_status": "Status",
                "avg_flights_per_hour": "Avg Flights/Hr",
                "peak_flights_per_hour": "Peak Flights/Hr",
                "vertical_transition_pct": "Climb/Desc (%)",
                "avg_flight_level": "Mean FL",
                "dynamic_complexity_score": "Complexity Index",
            }
        )

    fir_html = ""
    if ds.has_fir_traversals():
        sat_data = calculate_fir_hourly_saturation(ds, threshold_flights_per_hr=40, top_n=10)
        if sat_data.get("top_saturated_firs"):
            fir_html += "<h3>En-Route FIR Sector Hourly Saturation</h3>" + _df_to_html_table(
                pd.DataFrame(sat_data["top_saturated_firs"])
            )
        if ds.has_aua_traversals():
            aua_sat = calculate_aua_hourly_saturation(ds, threshold_flights_per_hr=30, top_n=10)
            if aua_sat.get("top_saturated_auas"):
                fir_html += "<h3>ATC Unit Airspaces (AUA: TMA / CTA / UTA) Saturation</h3>" + _df_to_html_table(
                    pd.DataFrame(aua_sat["top_saturated_auas"])
                )
        taxi_res = analyze_airport_taxi_out_performance(ds, min_flights=50, top_n=10)
        if taxi_res.get("airports"):
            fir_html += "<h3>Airport Surface Movement & ICAO KPI02 Additional Taxi-Out Time (ATOT)</h3>" + _df_to_html_table(
                pd.DataFrame(taxi_res["airports"])
            )

    return f"""
    <section class="report-section">
        <h2>🌐 Network Status & 5°×5° Airspace Saturation ({html.escape(ds.period)})</h2>
        {_kpi_grid_html(kpis)}
        {chart_html}
        <h3>5°×5° Airspace Square Saturation & Complexity Directory ({hour_utc:02d}:00–{hour_utc:02d}:59 UTC)</h3>
        {_df_to_html_table(sq_df, max_rows=20)}
        <h3>Busiest Hubs & Most Delayed Corridors</h3>
        <div class="two-col">
            <div><h4>Top Departure Hubs</h4>{_df_to_html_table(top_apt, max_rows=10)}</div>
            <div><h4>Worst Delayed Corridors</h4>{_df_to_html_table(ds.get_top_delayed_routes(min_flights=50, limit=10), max_rows=10)}</div>
        </div>
        {fir_html}
    </section>
    """


# ---------------------------------------------------------------------------
# Section 2: 4D Trajectory, Active Route Navaids & Anticipated RNP/RNAV
# ---------------------------------------------------------------------------
def _build_trajectory_rnp_section(
    ds: FlightDataset,
    flight_id: int,
    show_navaids: bool = True,
    nav_type_filter: list[str] | None = None,
) -> str:
    flight = ds.get_flight(int(flight_id))
    if not flight:
        return f'<section class="report-section"><h2>🗺️ 4D Trajectory & RNP</h2><p>Flight {flight_id} not found.</p></section>'

    trajs = ds.get_trajectory(int(flight_id))
    actual_df = trajs["actual"]
    planned_df = trajs["planned"]
    metrics = analyze_trajectory(actual_df, planned_df)
    nav_rep = analyze_flight_navaids_and_rnp(ds, int(flight_id))

    active_navaids = []
    if nav_rep and nav_rep.active_navaids and show_navaids:
        if nav_type_filter:
            active_navaids = [n for n in nav_rep.active_navaids if n.nav_type in nav_type_filter]
        else:
            active_navaids = list(nav_rep.active_navaids)

    kpis = [
        ("Flight & Route", f"#{flight['ectrl_id']} ({flight['adep']} ➔ {flight['ades']})", f"{flight['ac_operator']} | {flight['ac_type']}"),
        ("Flown vs Filed Distance", f"{metrics.actual_distance_nm:.0f} nm", f"Plan: {metrics.planned_distance_nm:.0f} nm ({metrics.horizontal_extension_ratio:.3f}x)"),
        ("Holding Duration", f"{metrics.total_holding_duration_min:.1f} min", f"{len(metrics.holding_events)} holding stack(s)"),
        ("Active Route Navaids", f"{len(active_navaids)} shown", f"Total corridor: {nav_rep.total_active_navaids if nav_rep else 0} stations"),
        ("Anticipated RNP / RNAV", f"Best RNP {nav_rep.best_rnp_nm:.1f} nm" if nav_rep else "N/A", f"En-route RNAV {nav_rep.mean_enroute_rnp_nm:.1f} nm ({nav_rep.dme_dme_coverage_pct:.1f}% Multi-DME)" if nav_rep else ""),
    ]

    # Plot 1: Static 2D Map of Planned vs Actual Trajectory + Colored Navaids + Sightlines
    fig1, ax_map = plt.subplots(figsize=(12.5, 5.5))
    ax_map.set_facecolor("#f0f6fa")
    if not planned_df.empty:
        ax_map.plot(
            planned_df["longitude"],
            planned_df["latitude"],
            linestyle="--",
            color="#2980b9",
            linewidth=2.0,
            label=f"Filed Plan ({len(planned_df)} pts)",
        )
    if not actual_df.empty:
        ax_map.plot(
            actual_df["longitude"],
            actual_df["latitude"],
            linestyle="-",
            color="#8e44ad",
            linewidth=2.8,
            label=f"Actual Flown Track ({metrics.actual_distance_nm:.0f} nm)",
        )
        ax_map.scatter(
            [actual_df.iloc[0]["longitude"]],
            [actual_df.iloc[0]["latitude"]],
            color="#27ae60",
            s=95,
            zorder=6,
            label=f"ADEP ({flight['adep']})",
        )
        ax_map.scatter(
            [actual_df.iloc[-1]["longitude"]],
            [actual_df.iloc[-1]["latitude"]],
            color="#c0392b",
            s=95,
            zorder=6,
            label=f"ADES ({flight['ades']})",
        )

    if active_navaids and show_navaids:
        nav_colors = {
            "VOR-DME": "#00a8ff",
            "DME": "#0097e6",
            "VOR": "#00b894",
            "ILS_LOC": "#ff2e93",
            "NDB": "#f39c12",
        }
        plotted_types: set[str] = set()
        for n in active_navaids:
            col = nav_colors.get(n.nav_type, "#00a8ff")
            ax_map.plot(
                [n.longitude, n.closest_track_lon],
                [n.latitude, n.closest_track_lat],
                linestyle=":",
                color=col,
                linewidth=0.9,
                alpha=0.65,
            )
            lbl = f"Navaid: {n.nav_type}" if n.nav_type not in plotted_types else None
            plotted_types.add(n.nav_type)
            ax_map.scatter(
                [n.longitude],
                [n.latitude],
                color=col,
                edgecolors="#1e272e",
                s=42,
                linewidths=0.8,
                zorder=5,
                label=lbl,
            )
            ax_map.annotate(
                n.ident,
                (n.longitude, n.latitude),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6.5,
                color="#1e272e",
            )

    ax_map.grid(True, linestyle="--", alpha=0.45)
    ax_map.set_xlabel("Longitude (°E)")
    ax_map.set_ylabel("Latitude (°N)")
    ax_map.set_title(
        f"4D Flight Trajectory & Active Route Navaids — Flight #{flight['ectrl_id']} ({flight['adep']} ➔ {flight['ades']})",
        fontweight="bold",
    )
    ax_map.legend(loc="best", fontsize=8.5, framealpha=0.9)
    fig1.tight_layout()
    map_img_html = _fig_to_base64_img(
        fig1,
        f"Figure 1: 4D Trajectory & Active Route Navaid Sightlines for Flight #{flight['ectrl_id']} ({flight['adep']} ➔ {flight['ades']})",
    )

    # Plot 2: Vertical Profile (FL) vs Anticipated RNP/RNAV (nm) + DMEs in RLOS
    prof_img_html = ""
    nav_tables_html = ""
    if nav_rep and nav_rep.profile_points:
        prof_df = pd.DataFrame([p.to_dict() for p in nav_rep.profile_points])
        fig2, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(12.5, 5.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
        )
        ax_top.fill_between(
            prof_df["cumulative_dist_nm"],
            prof_df["flight_level"],
            color="#8e44ad",
            alpha=0.22,
        )
        ax_top.plot(
            prof_df["cumulative_dist_nm"],
            prof_df["flight_level"],
            color="#8e44ad",
            linewidth=2.4,
            label="Flight Level (FL, 100s ft)",
        )
        ax_top.set_ylabel("Flight Level (FL)", color="#8e44ad", fontweight="bold")
        ax_top.tick_params(axis="y", labelcolor="#8e44ad")
        ax_top.grid(True, linestyle="--", alpha=0.4)

        ax_rnp = ax_top.twinx()
        ax_rnp.step(
            prof_df["cumulative_dist_nm"],
            prof_df["anticipated_rnp_nm"],
            where="post",
            color="#d35400",
            linewidth=2.2,
            linestyle="--",
            label="Anticipated RNP / RNAV (nm)",
        )
        ax_rnp.scatter(
            prof_df["cumulative_dist_nm"],
            prof_df["anticipated_rnp_nm"],
            color="#e67e22",
            edgecolors="#1e272e",
            s=36,
            zorder=5,
        )
        ax_rnp.set_ylabel("Anticipated RNP / RNAV (nm — lower is tighter)", color="#d35400", fontweight="bold")
        ax_rnp.tick_params(axis="y", labelcolor="#d35400")
        ax_top.set_title(
            "Vertical Flight Profile (FL, Purple) vs. Anticipated RNP/RNAV Capability (nm, Orange Step Curve)",
            fontweight="bold",
        )

        ax_bot.bar(
            prof_df["cumulative_dist_nm"],
            prof_df["dme_in_rlos_count"],
            width=max(8.0, float(prof_df["cumulative_dist_nm"].max() or 100.0) / max(len(prof_df), 10) * 0.7),
            color="#00a8ff",
            alpha=0.78,
            label="DMEs in Radio Horizon (RLOS)",
        )
        ax_bot.set_ylabel("DMEs in RLOS", fontweight="bold")
        ax_bot.set_xlabel("Cumulative Flown Distance Along Route (nm)", fontweight="bold")
        ax_bot.grid(True, linestyle="--", alpha=0.4)
        fig2.tight_layout()
        prof_img_html = _fig_to_base64_img(
            fig2,
            f"Figure 2: Vertical Profile (FL) vs. Anticipated RNP/RNAV Capability (nm) & Multi-DME RLOS Redundancy",
        )

        nav_rows_df = pd.DataFrame(
            [
                {
                    "Ident": n.ident,
                    "Station Name": n.name,
                    "Type": n.nav_type,
                    "Freq (MHz)": f"{n.frequency_mhz:.2f}" if n.frequency_mhz is not None else "UHF",
                    "Closest Pass": f"{n.min_distance_to_track_nm:.1f} nm",
                    "Acft FL": f"FL{n.aircraft_fl_at_closest:.0f}",
                    "RLOS Horizon": f"{n.rlos_horizon_nm:.0f} nm",
                    "FMS Role": n.fms_navigation_role,
                }
                for n in active_navaids
            ]
        )
        rnp_rows_df = pd.DataFrame(
            [
                {
                    "Dist (nm)": f"{p.cumulative_dist_nm:.0f} nm",
                    "FL": f"FL{p.flight_level:.0f}",
                    "RNP (nm)": f"{p.anticipated_rnp_nm:.1f} nm",
                    "PBN Specification": p.pbn_specification,
                    "DME / VOR in RLOS": f"{p.dme_in_rlos_count} DME / {p.vor_in_rlos_count} VOR",
                    "Primary Tuned": p.primary_navaids_tuned,
                    "Why Available": p.availability_reason,
                }
                for p in nav_rep.profile_points
            ]
        )
        nav_tables_html = f"""
        <p class="summary-callout"><b>Navigation &amp; PBN Summary:</b> {html.escape(nav_rep.overall_navigation_summary)}</p>
        <div class="two-col">
            <div><h3>Likely Used Route Navaids ({len(active_navaids)} Stations)</h3>{_df_to_html_table(nav_rows_df, max_rows=30)}</div>
            <div><h3>Anticipated RNP/RNAV Availability &amp; 'Why' Along Route</h3>{_df_to_html_table(rnp_rows_df, max_rows=30)}</div>
        </div>
        """

    pru_html = ""
    pru_sc = compute_flight_pru_kpis(ds, int(flight_id))
    if pru_sc:
        pru_html = "<h3>ICAO GANP &amp; Eurocontrol PRU Horizontal &amp; Terminal KPIs</h3>" + _kpi_grid_html(
            [
                ("KPI04 (KEP Filed)", f"{pru_sc.kpi04_kep_pct:.2f}%", f"{pru_sc.enroute_filed_dist_nm:.0f} nm (>40nm)"),
                ("KPI05 (KEA Actual)", f"{pru_sc.kpi05_kea_pct:.2f}%", pru_sc.enroute_routing_verdict),
                ("Tactical KEA - KEP", f"{pru_sc.tactical_kea_minus_kep_pct:+.2f}%", f"{pru_sc.tactical_distance_delta_nm:+.1f} nm"),
                ("KPI08 ASMA (40nm)", f"{pru_sc.asma_40nm_actual_min:.1f} min", f"+{pru_sc.kpi08_asma_40nm_additional_min:.1f}m additional"),
                ("KPI19 Cruise FL", f"FL{pru_sc.max_actual_fl:.0f} (Req FL{pru_sc.requested_fl:.0f})", "CAPPED" if pru_sc.is_level_capped else "OPTIMAL"),
            ]
        )

    return f"""
    <section class="report-section">
        <h2>🗺️ 4D Trajectory, Active Route Navaids &amp; RNP/RNAV Capability — Flight #{flight['ectrl_id']}</h2>
        {_kpi_grid_html(kpis)}
        {map_img_html}
        {prof_img_html}
        {nav_tables_html}
        {pru_html}
    </section>
    """


# ---------------------------------------------------------------------------
# Section 3: Global Radio Navaids World Map
# ---------------------------------------------------------------------------
def _build_global_navaids_section(
    ds: FlightDataset,
    nav_types: list[str] | None = None,
    region_name: str = "🌍 Entire World",
    bounds: tuple[float, float, float, float] = (-85.0, 85.0, -180.0, 180.0),
    search_query: str | None = None,
    limit: int = 2500,
) -> str:
    selected_types = nav_types if nav_types is not None else ["VOR-DME", "DME", "VOR", "ILS_LOC", "NDB"]
    min_lat, max_lat, min_lon, max_lon = bounds
    df = query_unified_navaids(
        ds,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        nav_types=selected_types if selected_types else ["NONE"],
        search_query=search_query,
        limit=limit,
    )

    vordme_cnt = int((df["nav_type"] == "VOR-DME").sum()) if not df.empty else 0
    dme_cnt = int((df["nav_type"] == "DME").sum()) if not df.empty else 0
    vor_cnt = int((df["nav_type"] == "VOR").sum()) if not df.empty else 0
    ils_cnt = int((df["nav_type"] == "ILS_LOC").sum()) if not df.empty else 0
    ndb_cnt = int((df["nav_type"] == "NDB").sum()) if not df.empty else 0

    kpis = [
        ("Stations Rendered", f"{len(df):,}", region_name),
        ("VOR-DME / VORTAC", f"{vordme_cnt:,}", "Co-located Rho-Theta"),
        ("DME / TACAN", f"{dme_cnt:,}", "Slant-Range Transponders"),
        ("VOR", f"{vor_cnt:,}", "VHF Omnidirectional Range"),
        ("ILS / LOC & NDB", f"{ils_cnt:,} ILS / {ndb_cnt:,} NDB", "Approach & LF/MF Beacons"),
    ]

    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax_map.set_facecolor("#eaf3f8")
    type_colors = {
        "VOR-DME": "#00a8ff",
        "DME": "#0097e6",
        "VOR": "#00b894",
        "ILS_LOC": "#ff2e93",
        "NDB": "#f39c12",
    }
    if not df.empty:
        for ntype, col in type_colors.items():
            sub = df[df["nav_type"] == ntype]
            if not sub.empty:
                ax_map.scatter(
                    sub["longitude"],
                    sub["latitude"],
                    s=14,
                    color=col,
                    alpha=0.82,
                    label=f"{ntype} ({len(sub):,})",
                )
    region_clean = region_name.encode("ascii", "ignore").decode("ascii").strip() or "Global"
    ax_map.set_xlim(min_lon, max_lon)
    ax_map.set_ylim(min_lat, max_lat)
    ax_map.grid(True, linestyle="--", alpha=0.4)
    ax_map.set_xlabel("Longitude (°E)")
    ax_map.set_ylabel("Latitude (°N)")
    ax_map.set_title(f"Global Radio Navaids Atlas — {region_clean}", fontweight="bold")
    ax_map.legend(loc="lower left", fontsize=8)

    counts_map = {
        "VOR-DME": vordme_cnt,
        "DME": dme_cnt,
        "VOR": vor_cnt,
        "ILS_LOC": ils_cnt,
        "NDB": ndb_cnt,
    }
    ax_bar.bar(
        list(counts_map.keys()),
        list(counts_map.values()),
        color=[type_colors[k] for k in counts_map],
        edgecolor="#1e272e",
    )
    ax_bar.set_ylabel("Station Count")
    ax_bar.set_title("Rendered Navaids by Facility Category", fontweight="bold")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    chart_html = _fig_to_base64_img(fig, f"Figure 1: Global Radio Navaids Distribution Map & Category Breakdown ({region_name})")

    return f"""
    <section class="report-section">
        <h2>📡 Global Radio Navaids World Map &amp; Directory ({html.escape(region_name)})</h2>
        {_kpi_grid_html(kpis)}
        {chart_html}
        <h3>Filtered Global Navaids Station Sample (Top 35 of {len(df):,} Stations)</h3>
        {_df_to_html_table(df, max_rows=35)}
    </section>
    """


# ---------------------------------------------------------------------------
# Section 3B: ECAC CNS Infrastructure Rationalization (DME & Radar)
# ---------------------------------------------------------------------------
def _build_ecac_rationalization_section(
    ds: FlightDataset,
    station_type: str = "DME",
    active_view: str = "VIEW_1",
    target_fir: str = "ALL",
    selected_fl: int = 200,
    custom_inactive_ids: tuple[str, ...] | None = None,
    custom_active_ids: tuple[str, ...] | None = None,
) -> str:
    try:
        rep = get_ecac_rationalization_report(
            ds,
            station_type=station_type,
            active_view=active_view,
            target_fir=target_fir,
            custom_inactive_ids=custom_inactive_ids,
            custom_active_ids=custom_active_ids,
        )
    except Exception as exc:
        return f'<section class="report-section"><h2>ECAC {html.escape(station_type)} Rationalization Report</h2><p class="muted-note">Failed to generate report: {html.escape(str(exc))}</p></section>'

    fir_name = rep.get("fir_name", "ECAC Airspace")
    icon = "📻" if station_type == "DME" else "🛰️"
    title_tech = "DME-DME RNAV / PBN Coverage Lab" if station_type == "DME" else "Primary & Secondary Radar (SSR/Mode S) Surveillance Lab"

    fl_data = rep["coverage_by_fl"].get(f"fl{selected_fl}", {})
    total_sta = max(1, rep["total_stations_count"])
    inact_pct = (rep["deactivated_stations_count"] / total_sta) * 100.0

    if station_type == "DME":
        met_p = fl_data.get("met_pct", fl_data.get("dual_coverage_pct", 0.0))
        risk_p = fl_data.get("at_risk_pct", fl_data.get("single_coverage_pct", 0.0))
        out_p = fl_data.get("outage_pct", 0.0)
        kpis = [
            (f"Active {station_type}s", f"{rep['active_stations_count']} / {rep['total_stations_count']}", f"{rep['deactivated_stations_count']} Standby ({inact_pct:.1f}%)"),
            ("Annual OPEX Savings", f"€{rep['annual_cost_savings_eur']:,.0f}/yr", "O&M Rationalization"),
            ("Annual Power Savings", f"{rep['annual_power_savings_mwh']:,.0f} MWh/yr", "Grid & Standby Reduction"),
            (f"FL{selected_fl} Standard Met", f"{met_p:.1f}% (Green)", f"At Risk: {risk_p:.1f}% | Outage: {out_p:.1f}%"),
            ("Hubs Dual Ground Integrity", f"{rep['high_density_dual_pct']:.1f}%", f"{rep['high_density_dual_target_met_count']} of {rep['high_density_airports_count']} Busy Hubs (≥500 mv/d)"),
        ]
    else:
        dual_p = fl_data.get("dual_coverage_pct", 0.0)
        sing_p = fl_data.get("single_coverage_pct", 0.0)
        kpis = [
            (f"Active {station_type}s", f"{rep['active_stations_count']} / {rep['total_stations_count']}", f"{rep['deactivated_stations_count']} Standby ({inact_pct:.1f}%)"),
            ("Annual OPEX Savings", f"€{rep['annual_cost_savings_eur']:,.0f}/yr", "O&M Rationalization"),
            ("Annual Power Savings", f"{rep['annual_power_savings_mwh']:,.0f} MWh/yr", "Grid & Standby Reduction"),
            (f"FL{selected_fl} Dual Coverage", f"{dual_p:.1f}%", f"Single: {sing_p:.1f}%"),
            ("Hubs Dual Ground Integrity", f"{rep['high_density_dual_pct']:.1f}%", f"{rep['high_density_dual_target_met_count']} of {rep['high_density_airports_count']} Busy Hubs (≥500 mv/d)"),
        ]

    # 1. Interactive Folium Map with Leaflet OpenStreetMap Tiles (Preserved)
    cov_map = create_ecac_coverage_map(
        rep,
        selected_fl=selected_fl,
        show_cells=True,
        show_stations=True,
        show_gap_fillers=True,
        show_hubs=True,
    )
    map_html = _folium_map_to_iframe(
        cov_map,
        height=560,
        title=f"{fir_name} {station_type} Coverage & Redundancy Map (FL{selected_fl})"
    )

    # 2. Executive Matplotlib Overview Chart
    fig, (ax_cov, ax_sta) = plt.subplots(1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.2, 1.0]})
    fl_levels = [50, 100, 200, 300]
    fl_labels = [f"FL{fl}" for fl in fl_levels]

    if station_type == "DME":
        met_vals = [rep["coverage_by_fl"].get(f"fl{fl}", {}).get("met_pct", 0.0) for fl in fl_levels]
        risk_vals = [rep["coverage_by_fl"].get(f"fl{fl}", {}).get("at_risk_pct", 0.0) for fl in fl_levels]
        out_vals = [rep["coverage_by_fl"].get(f"fl{fl}", {}).get("outage_pct", 0.0) for fl in fl_levels]
        ax_cov.bar(fl_labels, met_vals, label="Standard Met (Green)", color="#2ecc71", edgecolor="#27ae60")
        ax_cov.bar(fl_labels, risk_vals, bottom=met_vals, label="At Risk / Marginal (Orange)", color="#e67e22", edgecolor="#d35400")
        bottom_out = [m + r for m, r in zip(met_vals, risk_vals)]
        ax_cov.bar(fl_labels, out_vals, bottom=bottom_out, label="Outage / Gaps (Red)", color="#e74c3c", edgecolor="#c0392b")
        ax_cov.set_ylabel("Airspace Compliance (% of Grid Cells)")
        ax_cov.set_title(f"Navigation Performance ({active_view}) by Flight Level", fontweight="bold")
    else:
        dual_vals = [rep["coverage_by_fl"].get(f"fl{fl}", {}).get("dual_coverage_pct", 0.0) for fl in fl_levels]
        sing_vals = [rep["coverage_by_fl"].get(f"fl{fl}", {}).get("single_coverage_pct", 0.0) for fl in fl_levels]
        ax_cov.bar(fl_labels, dual_vals, label="Dual Coverage (Green)", color="#2ecc71", edgecolor="#27ae60")
        ax_cov.bar(fl_labels, sing_vals, bottom=dual_vals, label="Single Coverage (Orange)", color="#f39c12", edgecolor="#d68910")
        ax_cov.set_ylabel("Airspace Coverage (% of Grid Cells)")
        ax_cov.set_title("Radar Surveillance Coverage by Flight Level", fontweight="bold")

    ax_cov.set_ylim(0, 105)
    ax_cov.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_cov.legend(loc="lower right", fontsize=8.5)

    # Station Inventory Bar Chart
    sta_cats = ["Active", "Standby", "Total"]
    sta_vals = [rep["active_stations_count"], rep["deactivated_stations_count"], rep["total_stations_count"]]
    ax_sta.bar(sta_cats, sta_vals, color=["#3498db", "#95a5a6", "#2c3e50"], edgecolor="#1e272e")
    for i, v in enumerate(sta_vals):
        ax_sta.text(i, v + max(1, v * 0.02), f"{v:,}", ha="center", fontweight="bold", fontsize=9.5)
    ax_sta.set_ylabel("Number of Stations")
    ax_sta.set_title(f"Station Allocation ({rep['active_stations_count']}/{rep['total_stations_count']} Active)", fontweight="bold")
    ax_sta.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    chart_html = _fig_to_base64_img(fig, f"Figure: {station_type} Coverage Compliance & Inventory Allocation Overview")

    # 3. Tables for Infill Facilities & Station Sample
    adj_html = ""
    if rep.get("adjacent_fir_infill_suggestions"):
        adj_df = pd.DataFrame([
            {
                "Adjacent Station ID": s["id"],
                "Station Name": s["name"],
                "Country": s["country"],
                "Co-located Airport": s.get("nearest_airport") or "En-Route",
                "Internal Gap Cells Covered": s.get("cells_covered_count", 0),
                "Capital Equipment Cost": "€0 (Cross-Border Sharing)",
            }
            for s in rep["adjacent_fir_infill_suggestions"]
        ])
        adj_html = f"""
        <h3>🌐 Step 1: Recommended Adjacent FIR Cross-Border Infill Stations (€0 Capital Outlay)</h3>
        <p class="muted-note">Existing cross-border facilities in line-of-sight capable of bridging internal coverage shortfalls without new hardware purchases.</p>
        {_df_to_html_table(adj_df, max_rows=25)}
        """

    gap_html = ""
    if rep.get("synthetic_gap_fillers"):
        gap_df = pd.DataFrame([
            {
                "Proposed Facility ID": g["id"],
                "Name / Reference": g["name"],
                "Latitude": f"{g['latitude']:.3f}°N",
                "Longitude": f"{g['longitude']:.3f}°E",
                "Elevation": f"{g['elevation_ft']:.0f} ft",
                "Target Altitude": f"FL{g['target_flight_level']}",
                "Nominal Range": f"{g['service_range_nm']:.0f} NM",
                "Gap Cells Filled": g["cells_covered_count"],
                "Tactical Siting Rationale": g["region_note"],
            }
            for g in rep["synthetic_gap_fillers"]
        ])
        gap_html = f"""
        <h3>💡 Step 2: Proposed New Ground Stations (Strictly Land-Sited)</h3>
        <p class="muted-note">Optimal facility locations strictly verified on terrestrial land coordinates. Uncoverable deep oceanic airspace cells are not assigned impossible offshore stations.</p>
        {_df_to_html_table(gap_df, max_rows=25)}
        """

    sta_sample_df = pd.DataFrame([
        {
            "Station ID": s["id"],
            "Name": s["name"],
            "Country": s["country"],
            "Co-located Airport": s["nearest_airport"] if s["is_airport_station"] else "En-Route",
            "Status": "ACTIVE" if s["is_active"] else "STANDBY (Deactivated)",
            "Annual Savings (€)": f"€{s['annual_savings_eur']:,}/yr" if not s["is_active"] else "€0 (Active)",
        }
        for s in rep["stations"][:35]
    ])
    sta_table_html = f"""
    <h3>📋 {station_type} Station Inventory Sample (Top 35 of {rep['total_stations_count']} Facilities)</h3>
    {_df_to_html_table(sta_sample_df, max_rows=35)}
    """

    # 4. Methodology, Standards & Economics
    standards_rows = ""
    if station_type == "DME":
        standards_rows = """
        <h4>📡 Navigation Performance Standards (RNAV / RNP) Criteria</h4>
        <div class="table-wrap">
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Navigation Standard</th>
                    <th>Required Accuracy (TSE 95%)</th>
                    <th>Siting &amp; Geometry Criteria</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>RNAV 5</b> <i>(B-RNAV)</i></td>
                    <td>&plusmn;5.0 NM (En-Route)</td>
                    <td>&ge; 2 DMEs with inclusion cut angle 30&deg; &le; &Delta;&theta; &le; 150&deg;, slant range &le; 160 NM, GDOP &le; 2.83</td>
                </tr>
                <tr>
                    <td><b>RNAV 2</b></td>
                    <td>&plusmn;2.0 NM (Continental)</td>
                    <td>&ge; 2 DMEs with valid pair geometry within &le; 130 NM; tighter slant-range threshold</td>
                </tr>
                <tr>
                    <td><b>RNAV 1</b> <i>(P-RNAV)</i></td>
                    <td>&plusmn;1.0 NM (Terminal TMA)</td>
                    <td><b>N-1 Redundancy</b>: &ge; 3 DMEs with valid pair geometry within &le; 60 NM (no "Critical DME" single point of failure)</td>
                </tr>
                <tr>
                    <td><b>RNP 1</b> <i>(Reversionary)</i></td>
                    <td>&plusmn;1.0 NM + Alerting</td>
                    <td>High-integrity multi-DME (&ge; 3&ndash;4 DMEs with dual independent pairs &le; 40 NM)</td>
                </tr>
            </tbody>
        </table>
        </div>
        """

    econ_html = """
    <h4>💶 Eurocontrol &amp; ICAO Cost and Power Savings Engineering Rationale</h4>
    <div class="table-wrap">
    <table class="styled-table">
        <thead>
            <tr>
                <th>Cost / Energy Component</th>
                <th>Surveillance Radar (PSR/MSSR)</th>
                <th>Distance Measuring Equipment (DME)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><b>Annual OPEX Savings</b></td>
                <td><b>€85,000 / yr per head</b></td>
                <td><b>€45,000 / yr per station</b></td>
            </tr>
            <tr>
                <td>↳ Preventative &amp; Corrective Maintenance</td>
                <td>€42,000 / yr</td>
                <td>€16,000 / yr</td>
            </tr>
            <tr>
                <td>↳ Periodic Airborne Flight Calibration</td>
                <td>€22,000 / yr</td>
                <td>€18,000 / yr</td>
            </tr>
            <tr>
                <td>↳ Telecommunications &amp; Site Lease</td>
                <td>€13,000 / yr</td>
                <td>€8,000 / yr</td>
            </tr>
            <tr>
                <td>↳ Electrical Energy Utility Cost</td>
                <td>€8,000 / yr</td>
                <td>€3,000 / yr</td>
            </tr>
            <tr>
                <td><b>Continuous Power Draw / Annual Energy</b></td>
                <td><b>15.41 kW (135 MWh/year)</b></td>
                <td><b>2.85 kW (25 MWh/year)</b></td>
            </tr>
        </tbody>
    </table>
    </div>
    """

    return f"""
    <section class="report-section">
        <h2>{icon} ECAC {html.escape(title_tech)} — {html.escape(fir_name)}</h2>
        <p class="muted-note">Evaluated Airspace Domain: <b>{html.escape(fir_name)}</b> | Scenario / View: <b>{html.escape(active_view)}</b> | Altitude: <b>FL{selected_fl} ({selected_fl*100:,} ft)</b></p>
        {_kpi_grid_html(kpis)}
        {map_html}
        {chart_html}
        {adj_html}
        {gap_html}
        {sta_table_html}
        {standards_rows}
        {econ_html}
    </section>
    """


# ---------------------------------------------------------------------------
# Section 4: Flight Performance, Delay & Eco (€)
# ---------------------------------------------------------------------------
def _build_flight_performance_eco_section(ds: FlightDataset, flight_id: int) -> str:
    flight = ds.get_flight(int(flight_id))
    if not flight:
        return f'<section class="report-section"><h2>📐 Flight Performance &amp; Eco (€)</h2><p>Flight {flight_id} not found.</p></section>'

    cdo_rep = analyze_cdo_descent(ds, int(flight_id))
    cco_rep = analyze_cco_climb(ds, int(flight_id))
    impact = compute_flight_fuel_and_emissions(ds, int(flight_id))
    crco_rep = compute_flight_crco_charges(ds, int(flight_id))
    rwy_saf = assess_flight_runway_safety(ds, metar_client=None, ectrl_id=int(flight_id), fetch_live_metar=False)
    contrail_rep = analyze_flight_wind_and_contrails(ds, int(flight_id))
    taxi_rep = analyze_flight_taxi_out(ds, int(flight_id)) if ds.has_fir_traversals() else None

    kpis = [
        ("CDO Descent Score", f"{cdo_rep.cdo_compliance_score:.1f}%" if cdo_rep else "N/A", f"Excess descent fuel: {cdo_rep.excess_fuel_burned_kg:.1f} kg" if cdo_rep else ""),
        ("CCO Climb Score", f"{cco_rep.cco_compliance_score:.1f}%" if cco_rep else "N/A", f"Excess climb fuel: {cco_rep.excess_fuel_burned_kg:.1f} kg" if cdo_rep else ""),
        ("Ideal vs Flown FL", f"FL{impact.optimal_cruise_fl:.0f} (Opt) / FL{impact.actual_cruise_fl:.0f} (Act)", f"Engine: {impact.selected_engine} (BPR {impact.bypass_ratio:.1f}:1, PR {impact.pressure_ratio:.1f})"),
        ("Altitude Penalty / Savings", f"+{impact.excess_altitude_fuel_kg:,.1f} kg (€{impact.excess_altitude_fuel_cost_eur:,.2f})" if impact.excess_altitude_fuel_kg > 0 else "0 kg (Optimal)", f"Save up to {impact.potential_fuel_savings_kg:,.1f} kg / {impact.potential_fuel_savings_pct:.1f}% at FL{impact.optimal_cruise_fl:.0f}"),
        ("Total Excess Fuel & CO₂", f"{impact.total_excess_fuel_kg:,.1f} kg", f"{impact.excess_co2_kg:,.1f} kg CO₂ (€{impact.estimated_excess_fuel_cost_eur:,.2f})"),
        ("Eurocontrol CRCO Charge", f"€{crco_rep.actual_crco_charge_eur:,.2f}" if crco_rep else "N/A", f"Total Direct Cost: €{crco_rep.total_direct_route_cost_eur:,.2f}" if crco_rep else ""),
        ("Arrival Runway Safety", f"RWY {rwy_saf.recommended_runway} ({rwy_saf.safety_risk_level})" if rwy_saf else "N/A", f"Margin +{rwy_saf.landing_distance_safety_margin_m:,.0f} m ({rwy_saf.ils_precision_category})" if rwy_saf else ""),
    ]

    trajs = ds.get_trajectory(int(flight_id))
    actual_df = trajs["actual"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    if not actual_df.empty:
        t_series = pd.to_datetime(actual_df["time_over"])
        elapsed_min = (t_series - t_series.iloc[0]).dt.total_seconds() / 60.0
        ax1.fill_between(elapsed_min, actual_df["flight_level"], color="#2980b9", alpha=0.2)
        ax1.plot(elapsed_min, actual_df["flight_level"], color="#2980b9", linewidth=2.3, label="Actual Trajectory")
        ax1.axhline(impact.optimal_cruise_fl, color="#27ae60", linestyle="--", linewidth=1.8, label=f"Optimal FL{impact.optimal_cruise_fl:.0f} ({impact.selected_engine})")
        if impact.requested_fl > 0 and impact.requested_fl != impact.optimal_cruise_fl:
            ax1.axhline(impact.requested_fl, color="#e67e22", linestyle=":", linewidth=1.5, label=f"Requested FL{impact.requested_fl:.0f}")
        ax1.legend(loc="lower right", fontsize=8)
        ax1.set_xlabel("Elapsed Flight Time (min)")
        ax1.set_ylabel("Flight Level (FL)")
        ax1.set_title(f"Vertical Profile & Ideal Cruise Level — Flight #{flight_id}", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4)

    if crco_rep and crco_rep.fir_breakdown:
        fb_df = pd.DataFrame([b.__dict__ for b in crco_rep.fir_breakdown])
        fir_col = "fir_id" if "fir_id" in fb_df.columns else fb_df.columns[0]
        ax2.barh(fb_df[fir_col].astype(str), fb_df["charge_eur"].astype(float), color="#27ae60", alpha=0.88)
        ax2.set_xlabel("CRCO Navigation Charge (€ EUR)")
        ax2.set_title("Eurocontrol CRCO Charges by Member State / FIR (€)", fontweight="bold")
        ax2.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    chart_html = _fig_to_base64_img(fig, f"Figure 1: Vertical Climb/Descent Profile & Eurocontrol CRCO Route Charges (€ EUR) for Flight #{flight_id}")

    crco_tbl = _df_to_html_table(pd.DataFrame([b.__dict__ for b in crco_rep.fir_breakdown])) if (crco_rep and crco_rep.fir_breakdown) else ""
    taxi_html = ""
    if taxi_rep:
        taxi_html = "<h3>Departure Surface Movement (TAXI_OUT) &amp; ICAO EDB v32B LTO Emissions</h3>" + _kpi_grid_html(
            [
                ("Actual Taxi-Out", f"{taxi_rep.actual_taxi_out_min:.1f} min", f"UTOT p20: {taxi_rep.unimpeded_utot_min:.1f} min"),
                ("ICAO KPI02 ATOT", f"+{taxi_rep.additional_atot_min:.1f} min", taxi_rep.congestion_status),
                ("Excess Queue Fuel", f"{taxi_rep.excess_atot_fuel_kg:,.1f} kg", f"€{taxi_rep.excess_fuel_cost_eur:,.2f} EUR"),
                ("Excess LTO Pollutants", f"{taxi_rep.excess_nox_g:,.0f}g NOx / {taxi_rep.excess_co_g:,.0f}g CO", f"{taxi_rep.excess_nvpm_mg:,.0f} mg nvPM soot"),
            ]
        )

    contrail_html = ""
    if contrail_rep:
        contrail_html = "<h3>Upper-Air Winds &amp; Persistent Contrail (ISSR) Climate Impact</h3>" + _kpi_grid_html(
            [
                ("Mean GS vs TAS", f"{contrail_rep.avg_groundspeed_kts:.0f} kts GS", f"TAS: {contrail_rep.avg_true_airspeed_kts:.0f} kts"),
                ("Net Wind Component", f"{contrail_rep.avg_tailwind_component_kts:+.1f} kts", f"{contrail_rep.wind_time_savings_min:+.1f} min wind time"),
                ("Persistent Contrail (ISSR)", f"{contrail_rep.persistent_contrail_distance_nm:.1f} nm", f"{contrail_rep.persistent_contrail_pct:.1f}% of cruise"),
                ("Total Climate Multiplier", f"{contrail_rep.total_climate_co2e_multiplier:.2f}x CO₂e", f"+{contrail_rep.contrail_co2e_tons:.2f}t contrail CO₂e"),
            ]
        )

    return f"""
    <section class="report-section">
        <h2>📐 Flight Performance, Vertical Efficiency, Safety &amp; Eco Accounting (€) — Flight #{flight_id}</h2>
        {_kpi_grid_html(kpis)}
        {chart_html}
        {taxi_html}
        <h3>Eurocontrol CRCO FIR Navigation Charge Breakdown (€ EUR)</h3>
        {crco_tbl}
        {contrail_html}
    </section>
    """


# ---------------------------------------------------------------------------
# Section 5: Corridor & Twin-Flight Benchmarks
# ---------------------------------------------------------------------------
def _build_corridor_twin_section(
    ds: FlightDataset,
    twin_adep: str = "RKSI",
    twin_ades: str = "EGLL",
    twin_ac: str = "B77W",
    rec_adep: str = "EGLL",
    rec_ades: str = "KORD",
    rec_op: str = "BAW",
) -> str:
    twin_comp = compare_twin_flights(ds, adep=twin_adep, ades=twin_ades, ac_type=twin_ac)
    rec_rep = analyze_recurring_flight_series(ds, adep=rec_adep, ades=rec_ades, operator=rec_op if rec_op else None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8))
    if twin_comp:
        fast_t = ds.get_trajectory(int(twin_comp.fast_id))["actual"]
        slow_t = ds.get_trajectory(int(twin_comp.slow_id))["actual"]
        if not fast_t.empty:
            ax1.plot(fast_t["longitude"], fast_t["latitude"], color="#27ae60", linewidth=2.4, label=f"Efficient Twin #{twin_comp.fast_id} ({twin_comp.fast_actual_distance_nm:.0f} nm)")
        if not slow_t.empty:
            ax1.plot(slow_t["longitude"], slow_t["latitude"], color="#c0392b", linewidth=2.4, linestyle="--", label=f"Inefficient Twin #{twin_comp.slow_id} ({twin_comp.slow_actual_distance_nm:.0f} nm)")
        ax1.set_title(f"Twin-Flight Trajectory Comparison ({twin_adep} ➔ {twin_ades}, {twin_ac})", fontweight="bold")
        ax1.set_xlabel("Longitude (°E)")
        ax1.set_ylabel("Latitude (°N)")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="best", fontsize=8.5)
    else:
        ax1.text(0.5, 0.5, f"No Twin Match for {twin_adep}➔{twin_ades} ({twin_ac})", ha="center", va="center")

    if rec_rep and rec_rep.diurnal_banks:
        b_names = [b.bank_name.split(" ")[0] for b in rec_rep.diurnal_banks]
        b_arr = [b.avg_arr_delay_min for b in rec_rep.diurnal_banks]
        b_dep = [b.avg_dep_delay_min for b in rec_rep.diurnal_banks]
        x = range(len(b_names))
        ax2.bar([i - 0.18 for i in x], b_dep, width=0.35, color="#3498db", label="Avg Dep Delay (min)")
        ax2.bar([i + 0.18 for i in x], b_arr, width=0.35, color="#e67e22", label="Avg Arr Delay (min)")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(b_names)
        ax2.set_ylabel("Delay (min)")
        ax2.set_title(f"Diurnal Bank Delay Profile ({rec_adep} ➔ {rec_ades})", fontweight="bold")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax2.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    chart_html = _fig_to_base64_img(fig, f"Figure 1: Twin-Flight Lateral Route Disparity ({twin_adep}➔{twin_ades}) & Diurnal Bank Delay Escalation ({rec_adep}➔{rec_ades})")

    kpis = []
    if twin_comp:
        kpis.extend([
            ("Twin Distance Disparity", f"+{twin_comp.distance_penalty_nm:.1f} nm", f"#{twin_comp.fast_id} vs #{twin_comp.slow_id}"),
            ("Twin Air Time Disparity", f"+{twin_comp.air_time_penalty_min:.1f} min", f"{twin_adep} ➔ {twin_ades} ({twin_ac})"),
            ("Twin Excess Fuel & Cost", f"+{twin_comp.excess_fuel_burned_kg:,.1f} kg", f"€{twin_comp.excess_fuel_cost_eur:,.2f} EUR"),
        ])
    if rec_rep:
        kpis.extend([
            ("Recurring Series Flights", f"{rec_rep.total_flights} occurrences", f"{rec_adep} ➔ {rec_ades}"),
            ("Series Mean Arr Delay", f"{rec_rep.mean_arr_delay_min:+.1f} min", f"±{rec_rep.std_arr_delay_min:.1f}m std"),
        ])

    occ_html = ""
    if rec_rep and rec_rep.occurrences:
        occ_df = pd.DataFrame(
            [
                {
                    "Flight ID": o.ectrl_id,
                    "Date": o.date_str,
                    "Off-Block (UTC)": o.off_block_time,
                    "Bank": o.time_of_day_bank,
                    "Operator": o.ac_operator,
                    "Aircraft": o.ac_type,
                    "Tail": o.ac_reg,
                    "Distance (nm)": f"{o.actual_distance_nm:.0f}",
                    "Air Time (min)": f"{o.actual_air_time_min:.1f}",
                    "Arr Delay (min)": f"{o.arr_delay_min:+.1f}",
                }
                for o in rec_rep.occurrences
            ]
        )
        occ_html = f"<h3>Recurring Flight Series Occurrences ({rec_adep} ➔ {rec_ades})</h3>" + _df_to_html_table(occ_df, max_rows=25)

    return f"""
    <section class="report-section">
        <h2>📊 Corridor &amp; Twin-Flight Efficiency Benchmarks</h2>
        {_kpi_grid_html(kpis)}
        {chart_html}
        {occ_html}
    </section>
    """


# ---------------------------------------------------------------------------
# Section 6: What-If Simulations & Operations Research Stress Tests
# ---------------------------------------------------------------------------
def _build_what_if_stress_section(
    ds: FlightDataset,
    conflict_hour: int = 10,
    conflict_nm: float = 5.0,
    gdp_airport: str = "EGLL",
    gdp_aar: int = 25,
    gdp_date: str | None = None,
) -> str:
    ts_res = optimize_hub_tail_swaps(ds, operator=None, hub_airport=None)
    prox_rep = scan_airspace_proximity_events(ds, hour=conflict_hour, horiz_sep_nm=conflict_nm)
    gdp_res = optimize_ground_delay_program(ds, target_airport=gdp_airport, degraded_capacity_rate_per_hr=gdp_aar, program_date=gdp_date)

    # Multi-operator stress test + conflict map figure
    ops = ["RYR", "BAW", "DLH", "AFR"]
    shock_steps = [15, 30, 45, 60, 90]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0))
    for op in ops:
        s_res = stress_test_airline(ds, operator=op)
        c_map = {c["injected_shock_min"]: c["infection_rate_pct"] for c in s_res.fragility_curve}
        ax1.plot(shock_steps, [c_map.get(float(s), 0.0) for s in shock_steps], marker="o", linewidth=2.2, label=f"{op} (Grade {s_res.resilience_grade})")
    ax1.set_xlabel("Injected Operational Delay Shock (min)")
    ax1.set_ylabel("Fleet Rotation Infection Rate (%)")
    ax1.set_title("Scenario C: Multi-Operator Fleet Shock Fragility Curves", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="best", fontsize=8.5)

    ax2.set_facecolor("#eef5f9")
    if prox_rep.events:
        los_pts = [e for e in prox_rep.events if "LOSS_OF_SEPARATION" in e.severity]
        prox_pts = [e for e in prox_rep.events if "LOSS_OF_SEPARATION" not in e.severity]
        if prox_pts:
            ax2.scatter([e.longitude for e in prox_pts], [e.latitude for e in prox_pts], color="#f39c12", edgecolors="#d35400", s=55, label=f"High-Proximity 3–5nm ({len(prox_pts)})")
        if los_pts:
            ax2.scatter([e.longitude for e in los_pts], [e.latitude for e in los_pts], color="#ff0033", edgecolors="#900C3F", s=75, label=f"Loss of Sep <3nm ({len(los_pts)})")
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.set_title(f"Scenario E: 4D Co-Altitude Conflict Map ({prox_rep.date_str} {conflict_hour:02d}:00 UTC)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    if prox_rep.events:
        ax2.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    chart_html = _fig_to_base64_img(fig, f"Figure 1: Fleet Shock Fragility Curves & 4D Co-Altitude Airspace Conflict Map ({prox_rep.date_str} {conflict_hour:02d}:00 UTC)")

    kpis = [
        ("Tail-Swap Pairs Found", f"{ts_res.recommended_swaps_count} swaps", f"-{ts_res.total_delay_minutes_saved:,.0f} min / €{ts_res.estimated_cost_saved_eur:,.0f} saved"),
        ("4D Co-Altitude Conflicts", f"{prox_rep.events_detected} pairs", f"{prox_rep.loss_of_sep_alerts} Loss-of-Sep (<3 nm) at {conflict_hour:02d}:00 UTC"),
        ("GDP Regulated Arrivals", f"{gdp_res.total_flights_regulated} flights", f"{gdp_res.airport} (AAR Cap {gdp_res.degraded_capacity_rate_per_hr}/hr)"),
        ("GDP Holding Fuel Saved", f"{gdp_res.total_fuel_saved_kg:,.0f} kg Jet-A", f"€{gdp_res.total_fuel_cost_saved_eur:,.2f} EUR (-{gdp_res.total_co2_saved_kg:,.0f} kg CO₂)"),
    ]

    swap_df = pd.DataFrame(
        [
            {
                "Date": s.swap_date,
                "Hub": s.hub_airport,
                "Operator": s.operator,
                "Aircraft": s.ac_type,
                "Tail Swap Pair": f"{s.delayed_tail_reg} ⇄ {s.donor_tail_reg}",
                "Baseline Delay": f"{s.baseline_reactionary_delay_min:.0f} min",
                "Post-Swap Delay": f"{s.post_swap_reactionary_delay_min:.0f} min",
                "Net Saved": f"-{s.net_delay_saved_min:.0f} min",
                "Cost Saved (€)": f"€{s.estimated_cost_saved_eur:,.0f}",
            }
            for s in ts_res.swaps
        ]
    )
    prox_df = pd.DataFrame(
        [
            {
                "Timestamp (UTC)": e.timestamp_utc,
                "Severity": e.severity,
                "Aircraft Pair": f"{e.operator_1} #{e.flight_1_id} ({e.ac_type_1}) ⇄ {e.operator_2} #{e.flight_2_id} ({e.ac_type_2})",
                "Routes": f"{e.route_1} ⇄ {e.route_2}",
                "Horiz Sep": f"{e.horizontal_sep_nm:.2f} nm",
                "Vert Sep": f"{e.vertical_sep_ft:.0f} ft",
                "Flight Level": f"FL{e.fl_1:.0f} / FL{e.fl_2:.0f}",
                "Coordinates": f"({e.latitude:.2f}°, {e.longitude:.2f}°)",
            }
            for e in prox_rep.events
        ]
    )
    gdp_df = pd.DataFrame(
        [
            {
                "Flight ID": f"#{s.ectrl_id}",
                "Operator": s.ac_operator,
                "Origin": s.adep,
                "Hub": s.ades,
                "Aircraft": s.ac_type,
                "Filed ETA": s.original_arrival_time[-8:],
                "Assigned CTA Slot": s.assigned_slot_time[-8:],
                "Ground Delay": f"{s.ground_delay_assigned_min:.1f} min",
                "Fuel Saved": f"{s.fuel_saved_kg:,.1f} kg",
                "Cost Saved (€)": f"€{s.fuel_cost_saved_eur:,.2f}",
            }
            for s in (gdp_res.slot_assignments or [])[:25]
        ]
    )

    return f"""
    <section class="report-section">
        <h2>🔮 Counterfactual Simulations, Airspace Conflict Scanner &amp; Operations Research</h2>
        {_kpi_grid_html(kpis)}
        {chart_html}
        <h3>Scenario D: Prescriptive Network-Wide Hub Tail-Swap Recovery Pairs</h3>
        {_df_to_html_table(swap_df, max_rows=20)}
        <h3>Scenario E: 4D Co-Altitude Airspace Proximity &amp; Loss-of-Separation Pairs ({conflict_hour:02d}:00 UTC)</h3>
        {_df_to_html_table(prox_df, max_rows=25)}
        <h3>Scenario F: Ration-by-Schedule Ground Delay Program (GDP) Slot Assignments ({html.escape(gdp_res.airport)})</h3>
        {_df_to_html_table(gdp_df, max_rows=25)}
    </section>
    """


# ---------------------------------------------------------------------------
# Section 7: Macro Evolution & Methodology Reference Guide
# ---------------------------------------------------------------------------
def _build_macro_reference_section(ds: FlightDataset) -> str:
    comp = compare_aviation_periods(period1="201611", period2="202510")
    airac_rep = compare_airac_cycles("2509", "2510")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6))
    ax1.bar(["201611", "202510"], [comp.total_flights_p1, comp.total_flights_p2], color=["#34495e", "#2980b9"], width=0.45)
    ax1.set_ylabel("Total Flights")
    ax1.set_title(f"Traffic Volume Shift ({comp.traffic_growth_pct:+.1f}%)", fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2.bar(
        ["Arr Delay (201611)", "Arr Delay (202510)"],
        [comp.avg_arr_delay_p1, comp.avg_arr_delay_p2],
        color=["#7f8c8d", "#e67e22"],
        width=0.45,
    )
    ax2.set_ylabel("Minutes per Flight")
    ax2.set_title(f"Macro Arrival Delay Shift ({comp.delay_change_min:+.2f} min)", fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    chart_html = _fig_to_base64_img(fig, "Figure 1: Cross-Decade Traffic Growth & Delay Evolution (201611 vs. 202510)")

    rc = airac_rep.route_changes
    fc = airac_rep.fir_changes
    kpis = [
        ("Flight Volume Growth", f"{comp.traffic_growth_pct:+.1f}%", f"{comp.total_flights_p1:,} ➔ {comp.total_flights_p2:,}"),
        ("Arrival Delay Delta", f"{comp.delay_change_min:+.2f} min", f"{comp.avg_arr_delay_p1:.1f} ➔ {comp.avg_arr_delay_p2:.1f} min"),
        ("ATS Routes (201611 vs 202510)", f"{rc.total_routes_c1:,} ➔ {rc.total_routes_c2:,}", f"+{rc.added_routes_count:,} added / -{rc.deleted_routes_count:,} retired"),
        ("FIR Sectors (201611 vs 202510)", f"{fc.total_firs_c1:,} ➔ {fc.total_firs_c2:,}", f"{fc.modified_boundaries_count:,} boundary updates"),
    ]

    ref_df = pd.DataFrame(
        [
            {"Indicator": "KPI02 — Additional Taxi-Out Time (ATOT)", "Formula": "ATOT = max(0, Actual_Taxi_Out - UTOT_p20)", "Standard": "ICAO GANP / Eurocontrol PRU"},
            {"Indicator": "KPI04 / KPI05 — En-Route Extension (KEP / KEA)", "Formula": "(Dist_40nm - GCD_40nm) / GCD_40nm * 100", "Standard": "ICAO GANP / Eurocontrol PRU"},
            {"Indicator": "KPI08 — Additional ASMA Time (40 nm)", "Formula": "max(0, Actual_ASMA_40nm - Unimpeded_ASMA_p20)", "Standard": "ICAO GANP / Eurocontrol PRU"},
            {"Indicator": "KPI17 / KPI19 — CCO, CDO & Level-Capping", "Formula": "Level-off time (>20 sec, |VS|<300 fpm) & Req_FL - Max_Actual_FL", "Standard": "ICAO Doc 9931 / 9993"},
            {"Indicator": "Radio Line-of-Sight (RLOS) & RNP/RNAV", "Formula": "RLOS_nm = 1.23 * (sqrt(h_acft_ft) + sqrt(h_station_ft))", "Standard": "ICAO Doc 9613 (PBN Manual)"},
            {"Indicator": "Eurocontrol CRCO Route Charges (€)", "Formula": "Fee_EUR = (d_km / 100) * sqrt(MTOW_t / 50) * Unit_Rate_EUR", "Standard": "Eurocontrol CRCO Conditions"},
        ]
    )

    return f"""
    <section class="report-section">
        <h2>📈 Macro Period Evolution (201611 vs. 202510) &amp; Methodology Reference</h2>
        {_kpi_grid_html(kpis)}
        {chart_html}
        <h3>Core Aviation Formulas, Physics Models &amp; Regulatory Standards</h3>
        {_df_to_html_table(ref_df)}
    </section>
    """


# ---------------------------------------------------------------------------
# Section 8: Free Route Airspace (FRA) Boundary Bottlenecks & Long-Term Forecast
# ---------------------------------------------------------------------------
def _build_fra_boundary_section(
    ds: FlightDataset,
    boundary_pair: str | None = None,
    min_flights: int = 500,
    annual_growth_rate: float = 0.02,
    realization_rate: float = 0.85,
    jet_a_eur_per_kg: float = 0.6421,
    fuel_growth_opt: float = 0.015,
    fuel_growth_med: float = 0.035,
    fuel_growth_shock: float = 0.065,
) -> str:
    """Build the Free Route Airspace (FRA) boundary inefficiencies and long-term forecast section."""
    fra_rep = analyze_fir_boundary_inefficiencies(
        dataset=ds,
        min_flights=min_flights,
        limit=60,
        cross_border_only=True,
    )

    baseline_year = ds.period[:4] if (ds and ds.period and len(ds.period) >= 4) else "2025"
    forecast = calculate_fra_long_term_forecast(
        boundary_metrics=fra_rep.boundaries,
        dataset=ds,
        year_baseline=baseline_year,
        use_seasonal_synthesis=True,
        annual_growth_rate=annual_growth_rate,
        realization_rate=realization_rate,
        jet_a_eur_per_kg=jet_a_eur_per_kg,
        fuel_growth_optimistic=fuel_growth_opt,
        fuel_growth_median=fuel_growth_med,
        fuel_growth_shock=fuel_growth_shock,
    )

    primes = [b for b in fra_rep.boundaries if "PRIME" in b.fra_classification]
    likely = [b for b in fra_rep.boundaries if "LIKELY" in b.fra_classification]

    sb = forecast.get("seasonal_breakdown") or {}
    annual_base = sb.get("annual_blended") or {}
    annual_crossings = annual_base.get("annual_crossings") or (sum(b.total_crossings for b in primes) * 4)
    annual_fuel_t = annual_base.get("annual_fuel_saved_tonnes") or (fra_rep.total_network_excess_fuel_tonnes * 4 * realization_rate)
    annual_co2_t = annual_base.get("annual_co2_mitigated_tonnes") or (annual_fuel_t * 3.16)
    annual_cost_eur = annual_base.get("annual_cost_saved_eur") or (annual_fuel_t * 1000.0 * jet_a_eur_per_kg)
    net_flts = forecast.get("network_total_flights", 11124247)
    co2_mit_dev = forecast.get("co2_mitigation_deviation_pct", 0.0)
    detour_dev = forecast.get("detour_deviation_pct", 0.0)

    kpis = [
        ("Eurocontrol Baseline", f"{net_flts:,} Flights", f"Total European flights in {baseline_year} baseline (+4.0% YoY)"),
        ("Scoped Boundary Crossings", f"{annual_crossings:,} Crossings", f"{annual_crossings/1e6:.2f}M crossings generated across {baseline_year} flights ({len(primes)} prime bottlenecks)"),
        ("Cross-Border Boundaries", f"{fra_rep.cross_border_boundaries_count:,}", f"From {fra_rep.total_boundaries_analyzed:,} total pairs"),
        ("Prime Bottlenecks", f"{fra_rep.prime_candidate_count:,}", "Immediate FRA implementation priority"),
        ("Likely Operating FRA", f"{fra_rep.likely_fra_boundaries_count:,}", "Harmonized cross-border directness"),
        ("CO₂ Deviation Mitigated", f"-{co2_mit_dev:.2f}%", f"+{detour_dev:.2f}% excess detour deviation avoided"),
        ("Synthesized Fuel Saved", f"{annual_fuel_t:,.0f} tonnes/yr", f"{annual_co2_t:,.0f}t gross CO₂ mitigated annually"),
        ("Annual Baseline Value (€)", f"€{annual_cost_eur:,.0f} EUR", f"At baseline €{jet_a_eur_per_kg:.4f}/kg Jet-A"),
    ]

    # Plot 1: Prime Candidates Inefficiencies Bar Chart
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.0, 1.0]})
    display_primes = primes[:8] if primes else fra_rep.boundaries[:8]
    pairs = [p.boundary_pair for p in display_primes]
    detours = [p.mean_detour_nm for p in display_primes]
    costs = [p.excess_cost_eur / 1e6 for p in display_primes]

    ax1.barh(pairs[::-1], detours[::-1], color="#e74c3c", alpha=0.88)
    ax1.set_xlabel("Mean Inefficiency Detour (NM per Crossing)")
    ax1.set_title("Prime Candidate Inefficiency Detour (NM)", fontweight="bold")
    ax1.grid(True, axis="x", linestyle="--", alpha=0.4)

    ax2.barh(pairs[::-1], costs[::-1], color="#27ae60", alpha=0.88)
    ax2.set_xlabel("Annual Inefficiency Cost (€ Millions EUR)")
    ax2.set_title("Annual Economic Savings Opportunity (€M EUR)", fontweight="bold")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig1.tight_layout()
    chart1_html = _fig_to_base64_img(
        fig1,
        f"Figure 1: Prime Candidate FIR Boundary Routing Detours & Economic Inefficiency Penalties ({ds.period})",
    )

    # Plot 2: 20-Year Compounding Savings & Fuel Price Projections
    prog = forecast.get("yearly_progression") or []
    chart2_html = ""
    if prog:
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.2, 0.9]})
        years = [p["year"] for p in prog]
        ax3.plot(years, [p["cum_savings_constant_eur"] / 1e6 for p in prog], label=f"Constant Fuel (€{jet_a_eur_per_kg:.4f}/kg)", color="#7f8c8d", linewidth=2.0, linestyle="--")
        ax3.plot(years, [p["cum_savings_optimistic_eur"] / 1e6 for p in prog], label=f"Optimistic (+{fuel_growth_opt*100:.1f}%/yr)", color="#27ae60", linewidth=2.2)
        ax3.plot(years, [p["cum_savings_median_eur"] / 1e6 for p in prog], label=f"Median Expected (+{fuel_growth_med*100:.1f}%/yr)", color="#f39c12", linewidth=2.5)
        ax3.plot(years, [p["cum_savings_shock_eur"] / 1e6 for p in prog], label=f"Shock Pricing (+{fuel_growth_shock*100:.1f}%/yr)", color="#c0392b", linewidth=2.2)
        ax3.set_xlabel("Years Ahead (1 to 20)")
        ax3.set_ylabel("Cumulative Benefit (€ Millions EUR)")
        ax3.set_title("20-Year Cumulative FRA Benefits across Energy Scenarios", fontweight="bold")
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

        ax4.plot(years, [p["price_constant_eur"] for p in prog], label="Constant Baseline", color="#7f8c8d", linestyle="--", linewidth=1.8)
        ax4.plot(years, [p["price_optimistic_eur"] for p in prog], label=f"Optimistic (+{fuel_growth_opt*100:.1f}%/yr)", color="#27ae60", linewidth=2.0)
        ax4.plot(years, [p["price_median_eur"] for p in prog], label=f"Median (+{fuel_growth_med*100:.1f}%/yr)", color="#f39c12", linewidth=2.3)
        ax4.plot(years, [p["price_shock_eur"] for p in prog], label=f"Shock (+{fuel_growth_shock*100:.1f}%/yr)", color="#c0392b", linewidth=2.0)
        ax4.set_xlabel("Years Ahead (1 to 20)")
        ax4.set_ylabel("Jet-A Price (€ / kg)")
        ax4.set_title("Forward Fuel Price Escalation Paths (€/kg)", fontweight="bold")
        ax4.grid(True, linestyle="--", alpha=0.4)
        ax4.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        fig2.tight_layout()
        chart2_html = _fig_to_base64_img(
            fig2,
            "Figure 2: 20-Year Cumulative Economic Savings & Forward Fuel Price Escalation Regimes",
        )

    # Plot 3: Worst Flights Dog-Leg Radar Trajectory Visualizer Map
    chart3_html = ""
    target_pair = boundary_pair or (display_primes[0].boundary_pair if display_primes else None)
    if target_pair:
        worst_flights = get_worst_flights_for_boundary(ds, target_pair, limit=6, include_trajectories=True)
        if worst_flights:
            fig3, ax_map = plt.subplots(figsize=(13.0, 5.4))
            ax_map.set_facecolor("#eef5f9")
            palette = ["#e74c3c", "#8e44ad", "#d35400", "#16a085", "#2980b9", "#2c3e50"]
            for idx, flt in enumerate(worst_flights):
                traj = flt.get("trajectory") or []
                if not traj:
                    continue
                lats = [pt[0] for pt in traj]
                lons = [pt[1] for pt in traj]
                c = palette[idx % len(palette)]
                lbl = f"{flt.get('callsign', 'FLT')} (+{flt.get('excess_detour_nm', 0):.0f}nm | €{flt.get('excess_fuel_cost_eur', 0):.0f})"
                ax_map.plot(lons, lats, color=c, linewidth=2.2, alpha=0.85, label=lbl)
                gc = interpolate_great_circle(lats[0], lons[0], lats[-1], lons[-1], num_points=20)
                ax_map.plot([p[1] for p in gc], [p[0] for p in gc], color=c, linestyle="--", linewidth=1.0, alpha=0.55)
                ax_map.scatter([lons[0]], [lats[0]], color="#27ae60", s=40, zorder=5)
                ax_map.scatter([lons[-1]], [lats[-1]], color="#c0392b", s=40, zorder=5)
                ax_map.annotate(flt.get("adep", ""), (lons[0], lats[0]), fontsize=7.5, fontweight="bold", xytext=(3, 3), textcoords="offset points")
                ax_map.annotate(flt.get("ades", ""), (lons[-1], lats[-1]), fontsize=7.5, fontweight="bold", xytext=(3, 3), textcoords="offset points")

            ax_map.grid(True, linestyle="--", alpha=0.4)
            ax_map.set_xlabel("Longitude (°E)")
            ax_map.set_ylabel("Latitude (°N)")
            ax_map.set_title(
                f"Worst Detour Trajectories vs. Direct Chords Across Bottleneck Boundary: {target_pair}",
                fontweight="bold",
            )
            h, l = ax_map.get_legend_handles_labels()
            if h and l:
                ax_map.legend(loc="best", fontsize=8.0, framealpha=0.92)
            fig3.tight_layout()
            chart3_html = _fig_to_base64_img(
                fig3,
                f"Figure 3: Worst Detoured Radar Tracks vs. Direct Geodesic Lines across {target_pair}",
            )

    # Tables
    primes_df = pd.DataFrame(
        [
            {
                "FIR Pair": b.boundary_pair,
                "Countries": f"{b.country_a} ↔ {b.country_b}",
                "Crossings": f"{b.total_crossings:,}",
                "Mean Detour (NM)": f"{b.mean_detour_nm:.1f}",
                "Directness": f"{b.directness_index_pct:.1f}%",
                "Detour Dev (%)": f"+{b.detour_deviation_pct:.2f}%",
                "CO₂ Excess Dev (%)": f"+{b.co2_deviation_pct:.2f}%",
                "Fuel Penalty (t)": f"{b.excess_fuel_tonnes:.1f}",
                "CO₂ Penalty (t)": f"{b.excess_co2_tonnes:.1f}",
                "Excess Cost (€)": f"€{b.excess_cost_eur:,.0f}",
                "Priority Score": f"{b.fra_priority_score:.1f}",
            }
            for b in primes
        ]
    )

    likely_df = pd.DataFrame(
        [
            {
                "FIR Pair": b.boundary_pair,
                "Countries": f"{b.country_a} ↔ {b.country_b}",
                "Crossings": f"{b.total_crossings:,}",
                "Mean Detour (NM)": f"{b.mean_detour_nm:.1f}",
                "Directness": f"{b.directness_index_pct:.1f}%",
                "Detour Dev (%)": f"+{b.detour_deviation_pct:.2f}%",
                "Classification": "🟢 Likely Operating FRA",
            }
            for b in likely[:15]
        ]
    )

    hz_list = []
    for h_key, h_data in forecast.get("horizons", {}).items():
        cum_flts = h_data.get("cumulative_network_flights", 0)
        hz_list.append(
            {
                "Horizon": f"{h_data.get('horizon_years')} Year(s)",
                "Traffic Mult": f"{h_data.get('traffic_multiplier', 1.0):.3f}x",
                "European Flights": f"~{cum_flts/1e6:.1f}M flts" if cum_flts else "N/A",
                "Boundary Crossings": f"{h_data.get('cumulative_crossings', 0):,}",
                "Detour Avoided": f"{h_data.get('cumulative_detour_nm_avoided', 0):,.0f} NM",
                "Fuel Saved (t)": f"{h_data.get('cumulative_fuel_saved_tonnes', 0):,.0f}",
                "Gross CO₂ Mitigated": f"{h_data.get('cumulative_co2_mitigated_tonnes', 0):,.0f} t",
                "Net Lifecycle CO₂": f"{h_data.get('cumulative_net_lifecycle_co2_tonnes', 0):,.0f} t",
                "CO₂ Mit Dev (%)": f"-{h_data.get('co2_mitigation_deviation_pct', 0.0):.2f}%",
                "SAF Blend Share": f"{h_data.get('saf_mandate_pct', 0.0):.1f}% (-{h_data.get('lifecycle_carbon_intensity_reduction_pct', 0.0):.1f}% intensity)",
                "Constant Fuel (€)": f"€{h_data.get('savings_constant_eur', 0):,.0f}",
                "Optimistic (+1.5%/yr)": f"€{h_data.get('savings_optimistic_eur', 0):,.0f}",
                "Median (+3.5%/yr)": f"€{h_data.get('savings_median_eur', 0):,.0f}",
                "Shock (+6.5%/yr)": f"€{h_data.get('savings_shock_eur', 0):,.0f}",
            }
        )
    hz_df = pd.DataFrame(hz_list)

    month_df = pd.DataFrame(forecast.get("eurocontrol_monthly_breakdown", []))
    if not month_df.empty:
        month_df = month_df.rename(
            columns={
                "month_num": "Month #",
                "month_name": "Month",
                "official_flights": "Eurocontrol Flights",
                "seasonal_cluster": "Seasonal Synthesis Cluster",
                "anchor_represented": "Archive Anchor",
            }
        )

    ranked_df = pd.DataFrame(
        [
            {
                "Rank": b.priority_rank,
                "FIR Boundary Pair": b.boundary_pair,
                "Countries": f"{b.country_a} ↔ {b.country_b}",
                "Crossings": f"{b.total_crossings:,}",
                "Mean Detour (NM)": f"{b.mean_detour_nm:.1f}",
                "Directness": f"{b.directness_index_pct:.1f}%",
                "Detour Dev (%)": f"+{b.detour_deviation_pct:.2f}%",
                "Classification": b.fra_classification,
                "Priority Score": f"{b.fra_priority_score:.1f}",
            }
            for b in fra_rep.boundaries[:25]
        ]
    )

    mb_12m = forecast.get("monthly_breakdown", [])
    mb_12m_html = ""
    if mb_12m:
        season_col = "season" if "season" in mb_12m[0] else "season_cluster"
        mb_12m_df = pd.DataFrame(mb_12m)[
            ["month_name", season_col, "flights_eurocontrol", "flights_archive", "crossings_raw", "crossings_scaled", "fuel_saved_tonnes", "co2_mitigated_tonnes", "cost_saved_eur", "detour_deviation_pct"]
        ].rename(
            columns={
                "month_name": "Month",
                season_col: "Season",
                "flights_eurocontrol": "Eurocontrol Flights",
                "flights_archive": "Archive Flights",
                "crossings_raw": "Raw Crossings",
                "crossings_scaled": "Calibrated Crossings",
                "fuel_saved_tonnes": "Fuel Saved (t)",
                "co2_mitigated_tonnes": "CO₂ Mitigated (t)",
                "cost_saved_eur": "Operating Value (€ EUR)",
                "detour_deviation_pct": "Mean Detour (%)",
            }
        )
        mb_12m_html = f"""
        <h3>🗓️ Contiguous 12-Month Empirical Monthly Baseline (Nov 2024 – Oct 2025)</h3>
        {_df_to_html_table(mb_12m_df)}
        """

    return f"""
    <section class="report-section">
        <h2>🌐 Free Route Airspace (FRA) Boundary Efficiency Analysis &amp; Long-Term Forecaster</h2>
        <p>
            Deterministic evaluation of Flight Information Region (FIR) boundary interfaces across European airspace.
            Reconciles full 12-month Eurocontrol network traffic ({net_flts:,} flights in {baseline_year}) and tracks lateral routing
            detours, directness indices, and exact percentage CO₂ deviations.
        </p>
        {_kpi_grid_html(kpis)}
        {chart1_html}
        <h3>🚨 Prime Candidates for Immediate Cross-Border FRA Implementation</h3>
        {_df_to_html_table(primes_df)}
        {chart2_html}
        <h3>💰 Long-Term Compounding Benefit Horizons (1, 5, 20 Years) across Fuel Escalation Pathways</h3>
        <p class="muted-note">
            Calculated at {annual_growth_rate*100:.1f}% annual traffic growth and {realization_rate*100:.0f}% directness realization efficiency.
            Tracks both Gross Tailpipe CO₂ and Net Lifecycle CO₂ under EU ReFuelEU Aviation SAF blending mandates (2% in 2025 -> 42% in 2045).
            Energy pathways: Optimistic (+{fuel_growth_opt*100:.1f}%/yr), Median Expected ReFuelEU/ETS (+{fuel_growth_med*100:.1f}%/yr), and Shock Pricing (+{fuel_growth_shock*100:.1f}%/yr).
        </p>
        {_df_to_html_table(hz_df)}
        {mb_12m_html}
        <h3>📅 Official Eurocontrol Monthly Flight Statistics &amp; Seasonal Synthesis ({baseline_year})</h3>
        {_df_to_html_table(month_df) if not month_df.empty else ""}
        {chart3_html}
        <h3>🟢 Likely Operating Cross-Border FRA (Harmonized Directness)</h3>
        {_df_to_html_table(likely_df)}
        <h3>🏆 Top 25 FIR Boundary Inefficiency League Table</h3>
        {_df_to_html_table(ranked_df, max_rows=25)}
    </section>
    """


# ---------------------------------------------------------------------------
# Section: Single Flight Comprehensive Multi-View 4D Report Builder
# ---------------------------------------------------------------------------
def _build_flight_vertical_profile_chart(
    actual_df: pd.DataFrame,
    planned_df: pd.DataFrame,
    cco_rep: Any | None,
    cdo_rep: Any | None,
    contrail_rep: Any | None = None,
) -> str:
    """Generate a high-DPI Base64 figure of the vertical profile with level-offs and contrail bands."""
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    ax.set_facecolor("#f8fafd")

    if not planned_df.empty and "flight_level" in planned_df.columns:
        x_plan = range(len(planned_df))
        ax.plot(
            x_plan,
            planned_df["flight_level"],
            linestyle="--",
            color="#2980b9",
            linewidth=1.8,
            alpha=0.75,
            label=f"Filed Flight Plan ({len(planned_df)} pts)",
        )

    if not actual_df.empty and "flight_level" in actual_df.columns:
        x_act = range(len(actual_df))
        ax.plot(
            x_act,
            actual_df["flight_level"],
            color="#8e44ad",
            linewidth=2.4,
            label="Actual Flown Altitude (FL)",
        )
        ax.fill_between(x_act, actual_df["flight_level"], color="#8e44ad", alpha=0.08)

        # Highlight ISSR contrail forming segments in glowing cyan
        if contrail_rep and getattr(contrail_rep, "issr_segments", None):
            first_lbl = True
            for seg in contrail_rep.issr_segments:
                s_seq = getattr(seg, "seq_start", 0)
                e_seq = getattr(seg, "seq_end", 0)
                sub_pts = actual_df[(actual_df["seq_num"] >= s_seq) & (actual_df["seq_num"] <= e_seq)]
                if len(sub_pts) >= 2:
                    sub_indices = sub_pts.index
                    ax.plot(
                        sub_indices,
                        sub_pts["flight_level"],
                        color="#00bcd4",
                        linewidth=4.5,
                        label="Persistent Contrail (ISSR)" if first_lbl else "",
                        zorder=5,
                    )
                    first_lbl = False

    ax.set_ylabel("Flight Level (FL, 100s ft)", fontsize=11, fontweight="bold", color="#1e272e")
    ax.set_xlabel("Trajectory Waypoint Index Along Flown Route", fontsize=11, fontweight="bold", color="#1e272e")
    ax.grid(True, linestyle=":", alpha=0.6, color="#cbd5e1")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=9.5)
    ax.set_title("Vertical Profile: Radar Flown Profile, Filed Altitude & ISSR Contrail Traversal Zones", fontsize=11.5, fontweight="bold", color="#0c2461")
    return _fig_to_base64_img(fig, alt_text="Vertical Altitude Profile & Contrail ISSR Bands")


def generate_flight_comprehensive_html_report(
    ds: FlightDataset | None = None,
    flight_id: int | None = None,
    output_path: Path | str | None = None,
) -> str:
    """
    Generate an exhaustive, publication-grade standalone HTML report for a single flight covering all views
    and subtabs in the 4D / Single Flight category in logical order:
    1. Executive Flight Summary & Global Operational KPIs
    2. 4D Trajectory, Radar Tracking & Active Radio Navigation Map (with interactive OpenStreetMap tiles)
    3. Active Radio Navigation Aids & RNP / RNAV Corridor Capability
    4. Vertical Profile Efficiency: CCO & CDO Climb/Descent Performance
    5. Multi-Factor Delay Attribution & Runway Braking Safety Diagnostics
    6. Airport Surface Movement, Taxi-Out & ICAO LTO Emissions
    7. Eurocontrol CRCO Route Charges & Airspace Economics (with interactive FIR charge map)
    8. Upper-Air Winds & Equivalent Still Air Distance (ESAD)
    9. Persistent Contrail Cirrus Formation & Climate Radiative Forcing (with interactive contrail map)
    10. Alternative Eco-Optimal Trajectory: Multi-Objective Trade-Off & Envelope Check
    11. Persistent Flight Operational Notes & Qualitative Anomaly Explanations
    12. Appendix: Comprehensive Mathematical & Atmospheric Physics Methodology

    Guarantees all Leaflet map views render real, interactive OpenStreetMap tiles via isolated iframe containers.
    """
    if ds is None:
        ds = FlightDataset()

    if flight_id is None:
        sample_df = ds.query("SELECT ectrl_id FROM flights LIMIT 1")
        flight_id = int(sample_df.iloc[0]["ectrl_id"]) if not sample_df.empty else 202007026

    flight = ds.get_flight(int(flight_id))
    if not flight:
        return f"<html><body><h1>Flight #{flight_id} Not Found in Dataset ({ds.period})</h1></body></html>"

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    trajs = ds.get_trajectory(int(flight_id))
    actual_df = trajs["actual"].sort_values("seq_num").reset_index(drop=True) if not trajs["actual"].empty else pd.DataFrame()
    planned_df = trajs["planned"].sort_values("seq_num").reset_index(drop=True) if not trajs["planned"].empty else pd.DataFrame()

    traj_metrics = analyze_trajectory(actual_df, planned_df)
    pru = compute_flight_pru_kpis(ds, int(flight_id))
    nav_rep = analyze_flight_navaids_and_rnp(ds, int(flight_id))
    cco_rep = analyze_cco_climb(ds, int(flight_id))
    cdo_rep = analyze_cdo_descent(ds, int(flight_id))
    try:
        diag = diagnose_flight_bottlenecks(ds, MetarClient(), int(flight_id))
    except Exception:
        diag = None
    try:
        rwy = assess_flight_runway_safety(ds, MetarClient(), int(flight_id))
    except Exception:
        rwy = None
    ac_type = str(flight.get("ac_type") or "A320").upper()
    fuel_imp = compute_flight_fuel_and_emissions(ds, int(flight_id))
    eng_perf = resolve_aircraft_engine_perf(ds, ac_type)
    taxi_rep = analyze_flight_taxi_out(ds, int(flight_id))
    crco_rep = compute_flight_crco_charges(ds, int(flight_id))
    contrail_rep = analyze_flight_wind_and_contrails(ds, int(flight_id))
    eco_rep = optimize_eco_minimal_contrail_trajectory(ds, int(flight_id))

    notes_mgr = get_notes_manager()
    try:
        flight_note = notes_mgr.get_note(int(flight_id))
    except Exception:
        flight_note = None

    sections: list[str] = []

    # =========================================================================
    # 0. EXECUTIVE FLIGHT SUMMARY
    # =========================================================================
    adep = flight.get("adep", "ZZZZ")
    ades = flight.get("ades", "ZZZZ")
    ac_op = flight.get("ac_operator", "UNK")
    tail = flight.get("ac_id", "N/A")
    flown_dist = float(flight.get("actual_distance_nm") or traj_metrics.actual_distance_nm or 0.0)
    filed_dist = float(flight.get("filed_distance_nm") or traj_metrics.planned_distance_nm or flown_dist)
    gcd_dist = float(flight.get("gcd_nm") or 0.0)
    detour_pct = ((flown_dist - gcd_dist) / max(1.0, gcd_dist)) * 100.0 if gcd_dist > 0 else 0.0
    arr_delay = float(flight.get("arr_delay_min") or 0.0)
    delay_str = f"{arr_delay:+.0f} min" if arr_delay != 0 else "On-Time (0m)"

    as_flown_fuel = eco_rep.as_flown_fuel_kg if eco_rep else (flown_dist * eng_perf.nominal_fuel_flow_kg_hr / 450.0)

    exec_kpis = [
        ("Flown vs Filed Distance", f"{flown_dist:,.0f} nm", f"Plan: {filed_dist:,.0f} nm ({traj_metrics.horizontal_extension_ratio:.3f}x)"),
        ("Cruise Flight Level", f"FL{fuel_imp.actual_cruise_fl:.0f}" if fuel_imp else "N/A", f"Ideal: FL{fuel_imp.optimal_cruise_fl:.0f} (Δ {fuel_imp.fl_delta_actual_vs_optimal:+.0f} FL)" if fuel_imp else ""),
        ("Punctuality & Arrival Delay", delay_str, f"Dep delay: {float(flight.get('dep_delay_min') or 0.0):+.0f} min"),
        ("En-Route Fuel Burn", f"{as_flown_fuel:,.0f} kg", f"Excess: {fuel_imp.total_excess_fuel_kg:+,.0f} kg (€{fuel_imp.estimated_excess_fuel_cost_eur:+,.2f})" if fuel_imp else ""),
        ("Eurocontrol CRCO Fees", f"€{crco_rep.actual_crco_charge_eur:,.2f}" if crco_rep else "N/A", f"{len(crco_rep.fir_breakdown)} FIRs traversed" if crco_rep else ""),
        ("Net Climate Impact", f"{(contrail_rep.direct_co2_tons + contrail_rep.contrail_co2e_tons):.1f} t CO₂e" if contrail_rep else "N/A", f"Contrail Cirrus: +{contrail_rep.contrail_co2e_tons:.2f} t ({contrail_rep.total_climate_co2e_multiplier:.2f}x)" if contrail_rep else ""),
    ]

    sec_summary = f"""
    <section class="report-section">
        <h2>✈️ Executive Flight Summary &amp; Operational Telemetry</h2>
        <div class="summary-callout" style="margin-bottom: 16px;">
            <b>Flight:</b> #{flight_id} &nbsp;|&nbsp;
            <b>Callsign:</b> <code>{html.escape(str(flight.get('callsign') or 'N/A'))}</code> &nbsp;|&nbsp;
            <b>Operator:</b> {html.escape(ac_op)} &nbsp;|&nbsp;
            <b>Airframe:</b> {html.escape(ac_type)} ({html.escape(eng_perf.aircraft_name)}) &nbsp;|&nbsp;
            <b>Tail Registration:</b> <code>{html.escape(str(tail))}</code><br/>
            <b>City Pair:</b> <code>{html.escape(adep)}</code> ➔ <code>{html.escape(ades)}</code> &nbsp;|&nbsp;
            <b>Off-Block:</b> {html.escape(str(flight.get('off_block_date') or flight.get('entry_time') or 'N/A'))} &nbsp;|&nbsp;
            <b>Great Circle Distance:</b> {gcd_dist:,.0f} nm &nbsp;|&nbsp;
            <b>Detour vs. Great Circle:</b> {detour_pct:+.1f}% &nbsp;|&nbsp;
            <b>Engine Performance Model:</b> {html.escape(eng_perf.engine_type_class)} (BPR {eng_perf.bypass_ratio:.1f}:1, PR {eng_perf.pressure_ratio:.1f})
        </div>
        {_kpi_grid_html(exec_kpis)}
    </section>
    """
    sections.append(sec_summary)

    # =========================================================================
    # 1. 4D TRAJECTORY, RADAR TRACKING & ACTIVE RADIO NAVIGATION MAP
    # =========================================================================
    active_navs = list(nav_rep.active_navaids) if (nav_rep and nav_rep.active_navaids) else []
    contrail_segs = list(contrail_rep.issr_segments) if (contrail_rep and contrail_rep.issr_segments) else []

    try:
        traj_map = create_trajectory_map(
            flight_info=flight,
            actual_df=actual_df,
            planned_df=planned_df,
            active_navaids=active_navs,
            show_navaids=True,
            contrail_segments=contrail_segs,
            show_contrails=True,
        )
        traj_map_html = _folium_map_to_iframe(traj_map, height=530, title="4D Trajectory, Radar Track, Holding Stacks & Route Navaids")
    except Exception as exc:
        traj_map_html = f'<div class="chart-card"><p class="muted-note">Interactive map generation failed: {html.escape(str(exc))}</p></div>'

    traj_kpis = [
        ("Actual Radar Track", f"{traj_metrics.actual_distance_nm:,.0f} nm", f"{len(actual_df)} radar telemetry waypoints"),
        ("Filed Flight Plan", f"{traj_metrics.planned_distance_nm:,.0f} nm", f"{len(planned_df)} filed flight plan points"),
        ("Extension Detour Ratio", f"{traj_metrics.horizontal_extension_ratio:.3f}x", f"KEA: {pru.kpi05_kea_pct:.2f}% | KEP: {pru.kpi04_kep_pct:.2f}%"),
        ("Holding Stacks", f"{len(traj_metrics.holding_events)} stacks", f"Total holding: {traj_metrics.total_holding_duration_min:.1f} min"),
        ("Vertical Deviation", f"Max {traj_metrics.max_vertical_deviation_fl:.0f} FL", f"Mean Δ: {traj_metrics.mean_vertical_deviation_fl:.1f} FL vs plan"),
    ]

    sec_traj = f"""
    <section class="report-section">
        <h2>🗺️ View 1: 4D Trajectory, Radar Tracking &amp; Flight Geometry</h2>
        <p>
            Evaluates 4D actual radar trajectory versus filed flight plan geometry, airborne holding stacks,
            cross-track deviations, and Eurocontrol PRU horizontal flight efficiency indicators (KEA / KEP).
        </p>
        {_kpi_grid_html(traj_kpis)}
        {traj_map_html}
    </section>
    """
    sections.append(sec_traj)

    # =========================================================================
    # 2. ACTIVE RADIO NAVIGATION AIDS & RNP / RNAV CORRIDOR CAPABILITY
    # =========================================================================
    nav_kpis = [
        ("Multi-DME Coverage", f"{nav_rep.dme_dme_coverage_pct:.1f}%" if nav_rep else "N/A", "Terrestrial DME/DME triangulation"),
        ("Mean En-Route RNP", f"RNP {nav_rep.mean_enroute_rnp_nm:.1f} nm" if nav_rep else "N/A", f"Best Terminal RNP: {nav_rep.best_rnp_nm:.1f} nm" if nav_rep else ""),
        ("Active Corridor Stations", f"{len(active_navs)} stations" if nav_rep else "0", f"Total catalog: {nav_rep.total_active_navaids if nav_rep else 0}"),
        ("GNSS / Oceanic Share", f"{nav_rep.gnss_only_oceanic_pct:.1f}%" if nav_rep else "N/A", f"DME: {nav_rep.dme_stations_used} | VOR: {nav_rep.vor_stations_used}" if nav_rep else ""),
    ]

    nav_table_html = ""
    if active_navs:
        nav_rows = []
        for n in active_navs[:30]:
            nav_rows.append({
                "Ident": n.ident,
                "Name": n.name,
                "Type": n.nav_type,
                "Frequency (MHz)": getattr(n, "frequency_mhz", getattr(n, "frequency", "N/A")),
                "Latitude": round(n.latitude, 4),
                "Longitude": round(n.longitude, 4),
                "Range (nm)": round(getattr(n, "published_range_nm", 0.0), 1),
                "Dist to Track (nm)": round(getattr(n, "min_distance_to_track_nm", 0.0), 1),
                "Horizon (nm)": round(getattr(n, "rlos_horizon_nm", 0.0), 1),
            })
        nav_table_html = _df_to_html_table(pd.DataFrame(nav_rows), max_rows=25)

    sec_navaids = f"""
    <section class="report-section">
        <h2>📡 View 2: Active Radio Navigation Aids &amp; RNP / RNAV Corridor Capability</h2>
        <p>
            Quantifies terrestrial ground radio navigation reception along the trajectory (VOR, DME, TACAN, NDB)
            and verifies Performance-Based Navigation (PBN), Area Navigation (RNAV 1 / RNAV 5), and Required Navigation
            Performance (RNP 1 / RNP 2) sensor adequacy.
        </p>
        {_kpi_grid_html(nav_kpis)}
        <h3>Ground Radio Navigation Stations Intersecting Flight Path (Top 25)</h3>
        {nav_table_html if nav_table_html else '<p class="muted-note">Zero terrestrial radio navaids within line-of-sight reception radius (pure oceanic/GNSS airspace).</p>'}
    </section>
    """
    sections.append(sec_navaids)

    # =========================================================================
    # 3. VERTICAL PROFILE EFFICIENCY: CCO & CDO CLIMB/DESCENT PERFORMANCE
    # =========================================================================
    vert_chart_html = _build_flight_vertical_profile_chart(actual_df, planned_df, cco_rep, cdo_rep, contrail_rep)

    cdo_kpis = [
        ("Continuous Climb (CCO)", f"{cco_rep.cco_compliance_score:.1f}%" if cco_rep else "N/A", f"{cco_rep.inefficiency_events_count} climb level-offs (+{cco_rep.excess_fuel_burned_kg:.0f} kg fuel)" if cco_rep else ""),
        ("Continuous Descent (CDO)", f"{cdo_rep.cdo_compliance_score:.1f}%" if cdo_rep else "N/A", f"{cdo_rep.inefficiency_events_count} descent level-offs (+{cdo_rep.excess_fuel_burned_kg:.0f} kg fuel)" if cdo_rep else ""),
        ("Descent Inefficiency Time", f"{cdo_rep.total_level_or_climb_min:.1f} min" if cdo_rep else "N/A", f"TOD to Touchdown: {cdo_rep.total_descent_duration_min:.1f} min" if cdo_rep else ""),
        ("Level-Capping Inefficiency", f"FL{fuel_imp.actual_cruise_fl:.0f}" if fuel_imp else "N/A", f"Requested: FL{fuel_imp.requested_fl:.0f} | Ideal: FL{fuel_imp.optimal_cruise_fl:.0f}" if fuel_imp else ""),
    ]

    sec_vertical = f"""
    <section class="report-section">
        <h2>📉 View 3: Vertical Profile Efficiency: Continuous Climb (CCO) &amp; Continuous Descent (CDO)</h2>
        <p>
            Detects inefficient staircase descents, intermediate level-offs, altitude reversals, and en-route cruise
            level capping. Uninterrupted continuous climb (CCO) and clean 3° idle-thrust descents (CDO) significantly
            mitigate terminal airspace noise, fuel burn, and engine emissions.
        </p>
        {_kpi_grid_html(cdo_kpis)}
        {vert_chart_html}
    </section>
    """
    sections.append(sec_vertical)

    # =========================================================================
    # 4. MULTI-FACTOR DELAY ATTRIBUTION & RUNWAY SAFETY DIAGNOSTICS
    # =========================================================================
    delay_kpis = [
        ("Departure Delay", f"{float(flight.get('dep_delay_min') or 0.0):+.0f} min", f"Class: {flight.get('delay_class', 'ON_TIME')}"),
        ("Arrival Delay", f"{float(flight.get('arr_delay_min') or 0.0):+.0f} min", f"ATFM Delay: {float(flight.get('atfm_delay_min') or 0.0):.0f} min"),
        ("Primary Delay Driver", f"{diag.primary_bottleneck_factor if diag else 'NO_SIGNIFICANT_DELAY'}", f"Weather: {diag.weather_attribution}" if diag else ""),
        ("Arrival Runway Safety", f"{rwy.braking_action_code if rwy else 'GOOD'}", f"Surface: {rwy.runway_surface_condition} (LDA: {rwy.available_lda_m:.0f} m)" if rwy else ""),
    ]

    sec_delay = f"""
    <section class="report-section">
        <h2>🩺 View 4: Multi-Factor Delay Attribution &amp; Runway Braking Safety Diagnostics</h2>
        <p>
            Decomposes flight delay into root causes (rotational airline delays, European ATFM slot delays, airport
            surface bottlenecks, or reactive downstream propagation) and verifies destination runway braking friction safety.
        </p>
        {_kpi_grid_html(delay_kpis)}
    </section>
    """
    sections.append(sec_delay)

    # =========================================================================
    # 5. AIRPORT SURFACE MOVEMENT, TAXI-OUT & ICAO LTO EMISSIONS
    # =========================================================================
    taxi_kpis = [
        ("Taxi-Out Duration", f"{taxi_rep.actual_taxi_out_min:.1f} min" if taxi_rep else "N/A", f"Unimpeded UTOT: {taxi_rep.unimpeded_utot_min:.1f} min" if taxi_rep else ""),
        ("Additional ATOT Delay", f"{taxi_rep.additional_atot_min:+.1f} min" if taxi_rep else "N/A", f"Filed: {taxi_rep.filed_taxi_out_min:.1f} min" if taxi_rep else ""),
        ("Taxi Fuel Burn", f"{taxi_rep.total_taxi_fuel_kg:,.0f} kg" if taxi_rep else "N/A", f"Excess ATOT fuel: {taxi_rep.excess_atot_fuel_kg:+,.0f} kg" if taxi_rep else ""),
        ("Surface Taxi CO₂", f"{taxi_rep.excess_co2_kg:,.0f} kg CO₂" if taxi_rep else "N/A", f"Excess NOₓ: {taxi_rep.excess_nox_g/1000.0:,.1f} kg" if taxi_rep else ""),
    ]

    lto_rows = []
    if fuel_imp and getattr(fuel_imp, "lto_emissions_breakdown", None):
        lto_rows = [b.__dict__ for b in fuel_imp.lto_emissions_breakdown]
    lto_table_html = _df_to_html_table(pd.DataFrame(lto_rows)) if lto_rows else ""

    sec_taxi = f"""
    <section class="report-section">
        <h2>🌱 View 5: Airport Surface Movement, Taxi-Out &amp; ICAO LTO Emissions</h2>
        <p>
            Quantifies airport apron and taxiway surface movement against unimpeded benchmarks and evaluates certified
            ICAO Engine Emissions Databank (EDB) Landing/Take-Off (LTO) cycle emissions.
        </p>
        {_kpi_grid_html(taxi_kpis)}
        <h3>ICAO Engine Emissions Databank (EDB) Gaseous Emissions</h3>
        {lto_table_html if lto_table_html else '<p class="muted-note">Standard ICAO LTO emissions profile evaluated across Idle, Approach, Climb-Out, and Take-Off thrust ratings.</p>'}
    </section>
    """
    sections.append(sec_taxi)

    # =========================================================================
    # 6. EUROCONTROL CRCO ROUTE CHARGES & AIRSPACE ECONOMICS (€ EUR)
    # =========================================================================
    try:
        fir_map = create_flight_fir_charge_map(ds, int(flight_id), crco_report=crco_rep)
        fir_map_html = _folium_map_to_iframe(fir_map, height=500, title="Eurocontrol CRCO Airspace Traversal & National FIR Charging Map")
    except Exception as exc:
        fir_map_html = f'<div class="chart-card"><p class="muted-note">FIR Map not available: {html.escape(str(exc))}</p></div>'

    crco_kpis = [
        ("Actual CRCO Route Fee", f"€{crco_rep.actual_crco_charge_eur:,.2f}" if crco_rep else "N/A", "Statutory Eurocontrol Central Route Charges Office billing"),
        ("Traversed FIRs", f"{len(crco_rep.fir_breakdown)} FIRs" if crco_rep else "N/A", f"Charged distance: {crco_rep.actual_distance_km:,.0f} km ({crco_rep.actual_distance_nm:,.0f} nm)" if crco_rep else ""),
        ("CRCO Weight Factor", f"{crco_rep.crco_weight_factor:.3f}" if crco_rep else "N/A", f"Aircraft MTOW: {crco_rep.mtow_tons:.0f} t" if crco_rep else ""),
        ("Total Direct Route Cost", f"€{crco_rep.total_direct_route_cost_eur:,.2f}" if crco_rep else "N/A", f"Fuel: €{crco_rep.estimated_total_fuel_cost_eur:,.2f} | CRCO: €{crco_rep.actual_crco_charge_eur:,.2f}" if crco_rep else ""),
    ]

    fir_breakdown_html = ""
    if crco_rep and crco_rep.fir_breakdown:
        fir_rows = [b.__dict__ for b in crco_rep.fir_breakdown]
        fir_breakdown_html = _df_to_html_table(pd.DataFrame(fir_rows), max_rows=30)

    sec_crco = f"""
    <section class="report-section">
        <h2>💶 View 6: Eurocontrol CRCO Route Charges &amp; Airspace Economics (€ EUR)</h2>
        <p>
            Calculates route navigation charges billed by the Eurocontrol Central Route Charges Office (CRCO) across
            every national Flight Information Region (FIR) using statutory national unit rates (EUR/SU),
            entry-to-exit traversal distances, and aircraft Maximum Take-Off Weight (MTOW) scaling factors.
        </p>
        {_kpi_grid_html(crco_kpis)}
        {fir_map_html}
        <h3>National FIR Traversal Billing Breakdown</h3>
        {fir_breakdown_html}
    </section>
    """
    sections.append(sec_crco)

    # =========================================================================
    # 7. UPPER-AIR WINDS & EQUIVALENT STILL AIR DISTANCE (ESAD)
    # =========================================================================
    g_dist = getattr(contrail_rep, "ground_distance_nm", contrail_rep.cruise_distance_nm) if contrail_rep else flown_dist
    esad_val = getattr(contrail_rep, "esad_nm", g_dist) if contrail_rep else flown_dist
    air_benefit = getattr(contrail_rep, "wind_distance_benefit_nm", 0.0) if contrail_rep else 0.0
    wind_f_kg = getattr(contrail_rep, "wind_fuel_impact_kg", 0.0) if contrail_rep else 0.0
    wind_f_eur = getattr(contrail_rep, "wind_fuel_impact_eur", 0.0) if contrail_rep else 0.0
    wind_verdict = getattr(contrail_rep, "wind_efficiency_verdict", "NEUTRAL_WIND_PROFILE") if contrail_rep else "N/A"

    wind_kpis = [
        ("Avg GS vs TAS", f"{contrail_rep.avg_groundspeed_kts:.0f} kts GS" if contrail_rep else "N/A", f"TAS: {contrail_rep.avg_true_airspeed_kts:.0f} kts" if contrail_rep else ""),
        ("Net Upper-Air Wind", f"{contrail_rep.avg_tailwind_component_kts:+.1f} kts" if contrail_rep else "N/A", f"{contrail_rep.wind_time_savings_min:+.1f} min wind time" if contrail_rep else ""),
        ("Air Distance (ESAD)", f"{esad_val:,.0f} nm", f"{air_benefit:+.0f} nm air savings vs Ground: {g_dist:,.0f} nm"),
        ("Wind Fuel Impact", f"{wind_f_kg:+,.0f} kg Jet-A", f"€{wind_f_eur:+,.2f} EUR vs still-air baseline"),
    ]

    sec_winds = f"""
    <section class="report-section">
        <h2>🌬️ View 7: Upper-Air Winds &amp; Equivalent Still Air Distance (ESAD)</h2>
        <p>
            Decomposes True Airspeed (TAS) and Groundspeed (GS) along the 4D cruise track to separate deliberate
            jetstream-riding detours from ATC delays, calculating the Equivalent Still Air Distance (ESAD) and net fuel impact.
        </p>
        <div class="summary-callout" style="margin-bottom: 14px;">
            <b>💨 Dispatch Wind Routing Verdict:</b> {html.escape(wind_verdict)}
        </div>
        {_kpi_grid_html(wind_kpis)}
    </section>
    """
    sections.append(sec_winds)

    # =========================================================================
    # 8. PERSISTENT CONTRAIL CIRRUS & CLIMATE RADIATIVE FORCING
    # =========================================================================
    try:
        contrail_map = create_contrail_horizontal_map(
            flight_info=flight,
            actual_df=actual_df,
            contrail_segments=contrail_segs,
            eco_route_waypoints=eco_rep.eco_waypoints if eco_rep else None,
        )
        contrail_map_html = _folium_map_to_iframe(contrail_map, height=540, title="Horizontal Route: Flown Radar Track vs. Persistent Contrails vs. Eco-Optimal Route")
    except Exception as exc:
        contrail_map_html = f'<div class="chart-card"><p class="muted-note">Contrail Map not available: {html.escape(str(exc))}</p></div>'

    contrail_kpis = [
        ("Persistent Contrail (ISSR)", f"{contrail_rep.persistent_contrail_distance_nm:.1f} nm" if contrail_rep else "0.0 nm", f"{contrail_rep.persistent_contrail_pct:.1f}% of cruise ({contrail_rep.issr_segments_count} segments)" if contrail_rep else ""),
        ("Direct Engine CO₂", f"{contrail_rep.direct_co2_tons:.2f} t CO₂" if contrail_rep else "N/A", "Kerosene combustion tailpipe emissions"),
        ("Contrail Cirrus Forcing", f"+{contrail_rep.contrail_co2e_tons:.2f} t CO₂e" if contrail_rep else "0.00 t", "Calibrated 18.5 kg CO₂e / nm linear forcing"),
        ("Total Climate Multiplier", f"{contrail_rep.total_climate_co2e_multiplier:.2f}x CO₂e" if contrail_rep else "1.00x", f"Total footprint: {(contrail_rep.direct_co2_tons + contrail_rep.contrail_co2e_tons):.2f} t CO₂e" if contrail_rep else ""),
    ]

    issr_table_html = ""
    if contrail_segs:
        issr_rows = []
        for s in contrail_segs:
            issr_rows.append({
                "Seq Start-End": f"#{s.seq_start} ➔ #{s.seq_end}",
                "Time (UTC)": f"{s.start_time_utc} – {s.end_time_utc}",
                "Flight Level": f"FL{s.flight_level:.0f}",
                "Distance (nm)": f"{s.distance_nm:.1f} nm",
                "Duration": f"{s.duration_min:.1f} min",
                "Temp (°C)": f"{s.temperature_c:.1f}°C",
                "RH_ice (%)": f"{s.rh_ice_pct:.1f}%",
                "Tactical Recommendation": s.alt_deviation_note,
            })
        issr_table_html = _df_to_html_table(pd.DataFrame(issr_rows))

    sec_contrail = f"""
    <section class="report-section">
        <h2>🧊 View 8: Persistent Contrail Cirrus &amp; Climate Radiative Forcing</h2>
        <p>
            Evaluates the Schmidt-Appleman Criterion (SAC) and Ice-Supersaturated Regions (ISSR) using Murphy &amp; Koop (2005)
            water-to-ice thermodynamics. Trajectory segments meeting both $T \\le T_{{\\text{{SAC}}}}$ and $RH_{{\\text{{ice}}}} \\ge 100\\%$
            produce persistent contrail cirrus with calibrated GWP₁₀₀ radiative forcing ($18.5\\text{{ kg CO}}_2\\text{{e/nm}}$).
        </p>
        {_kpi_grid_html(contrail_kpis)}
        {contrail_map_html}
        <h3>Sequenced Ice-Supersaturated (ISSR) Contrail Segments</h3>
        {issr_table_html if issr_table_html else '<p class="muted-note">✅ Clean Air Corridor: Zero persistent contrails formed along this trajectory (ambient temperatures remained above Schmidt-Appleman threshold or air remained dry).</p>'}
    </section>
    """
    sections.append(sec_contrail)

    # =========================================================================
    # 9. ALTERNATIVE ECO-OPTIMAL TRAJECTORY: MULTI-OBJECTIVE TRADE-OFF
    # =========================================================================
    if eco_rep:
        f_sign = "+" if eco_rep.fuel_delta_kg > 0 else ""
        eco_kpis = [
            ("Net Climate Footprint", f"{eco_rep.eco_total_climate_co2e_kg / 1000.0:.2f} t CO₂e", f"{eco_rep.net_climate_delta_kg / 1000.0:+.2f} t ({eco_rep.net_climate_reduction_pct:-.1f}%) net reduction"),
            ("Contrail Cirrus Avoided", f"{eco_rep.eco_contrail_nm:.0f} nm in ISSR", f"-{eco_rep.contrail_avoided_nm:.0f} nm ({eco_rep.contrail_avoided_pct:.0f}% persistent contrails avoided)"),
            ("Fuel Burn Trade-Off", f"{eco_rep.eco_fuel_kg:,.0f} kg Jet-A", f"{f_sign}{eco_rep.fuel_delta_kg:,.0f} kg ({eco_rep.fuel_delta_pct:+.1f}%) | €{eco_rep.fuel_cost_delta_eur:+,.2f}"),
            ("Direct CO₂ / Cruise NOₓ", f"{eco_rep.eco_co2_kg / 1000.0:.2f} t CO₂", f"{eco_rep.eco_nox_kg:.1f} kg NOₓ (Flown: {eco_rep.as_flown_nox_kg:.1f} kg)"),
        ]

        tradeoff_rows = [
            {
                "Environmental Dimension": "Persistent Contrail Distance",
                "As-Flown Trajectory": f"{eco_rep.as_flown_contrail_nm:.1f} nm",
                "Eco-Optimal Trajectory": f"{eco_rep.eco_contrail_nm:.1f} nm",
                "Operational Delta": f"-{eco_rep.contrail_avoided_nm:.1f} nm (-{eco_rep.contrail_avoided_pct:.0f}%)",
            },
            {
                "Environmental Dimension": "Contrail Cirrus Radiative Forcing",
                "As-Flown Trajectory": f"{eco_rep.as_flown_contrail_co2e_kg / 1000.0:.2f} t CO₂e",
                "Eco-Optimal Trajectory": f"{eco_rep.eco_contrail_co2e_kg / 1000.0:.2f} t CO₂e",
                "Operational Delta": f"{(eco_rep.eco_contrail_co2e_kg - eco_rep.as_flown_contrail_co2e_kg) / 1000.0:+.2f} t CO₂e",
            },
            {
                "Environmental Dimension": "En-Route Kerosene Fuel Burn",
                "As-Flown Trajectory": f"{eco_rep.as_flown_fuel_kg:,.0f} kg",
                "Eco-Optimal Trajectory": f"{eco_rep.eco_fuel_kg:,.0f} kg",
                "Operational Delta": f"{f_sign}{eco_rep.fuel_delta_kg:,.0f} kg ({eco_rep.fuel_delta_pct:+.1f}%)",
            },
            {
                "Environmental Dimension": "Direct Tailpipe Engine CO₂",
                "As-Flown Trajectory": f"{eco_rep.as_flown_co2_kg / 1000.0:.2f} t CO₂",
                "Eco-Optimal Trajectory": f"{eco_rep.eco_co2_kg / 1000.0:.2f} t CO₂",
                "Operational Delta": f"{(eco_rep.eco_co2_kg - eco_rep.as_flown_co2_kg) / 1000.0:+.2f} t CO₂",
            },
            {
                "Environmental Dimension": "Upper-Troposphere Cruise NOₓ",
                "As-Flown Trajectory": f"{eco_rep.as_flown_nox_kg:.1f} kg NOₓ",
                "Eco-Optimal Trajectory": f"{eco_rep.eco_nox_kg:.1f} kg NOₓ",
                "Operational Delta": f"{eco_rep.nox_delta_kg:+.1f} kg NOₓ",
            },
            {
                "Environmental Dimension": "Net Global Climate Impact (CO₂e)",
                "As-Flown Trajectory": f"{eco_rep.as_flown_total_climate_co2e_kg / 1000.0:.2f} t CO₂e",
                "Eco-Optimal Trajectory": f"{eco_rep.eco_total_climate_co2e_kg / 1000.0:.2f} t CO₂e",
                "Operational Delta": f"{eco_rep.net_climate_delta_kg / 1000.0:+.2f} t CO₂e ({eco_rep.net_climate_reduction_pct:-.1f}%)",
            },
        ]
        tradeoff_table_html = _df_to_html_table(pd.DataFrame(tradeoff_rows))

        maneuvers_html = "".join([f"<li><code>{html.escape(m)}</code></li>" for m in eco_rep.tactical_maneuvers_summary])

        sec_eco = f"""
        <section class="report-section">
            <h2>🌿 View 9: Alternative Eco-Optimal Trajectory: Multi-Objective Trade-Off &amp; Feasibility</h2>
            <p>
                Synthesizes a minimal-contrail flight profile balancing fuel burn, direct CO₂, upper-air NOₓ, and persistent
                contrail cirrus warming while strictly operating within aircraft performance limits, operational ceilings,
                and ICAO RVSM cruising rules.
            </p>
            {_kpi_grid_html(eco_kpis)}
            <h3>Environmental Dimension Side-by-Side Comparison</h3>
            {tradeoff_table_html}
            <h3>Tactical Altitude Maneuvers &amp; Flight Envelope Certification</h3>
            <div class="summary-callout" style="margin-bottom: 12px;">
                <b>Flight Envelope Feasibility:</b> {html.escape(eco_rep.operational_feasibility_summary)}<br/>
                <b>Aircraft Service Ceiling:</b> FL{eco_rep.operational_ceiling_fl:.0f} &nbsp;|&nbsp;
                <b>Certified Envelope Adherence:</b> {'✅ FULLY CERTIFIED &amp; COMPLIANT' if eco_rep.is_within_aircraft_envelope else '⚠️ OUTSIDE FLIGHT ENVELOPE'}
            </div>
            <ul>{maneuvers_html}</ul>
        </section>
        """
        sections.append(sec_eco)

    # =========================================================================
    # 10. PERSISTENT FLIGHT OPERATIONAL NOTES & ANOMALY EXPLANATIONS
    # =========================================================================
    if flight_note:
        cits_html = ""
        if getattr(flight_note, "source_url", None):
            cits_html = f"<h4>Source Citations &amp; References</h4><ul><li><a href='{html.escape(flight_note.source_url)}' target='_blank'>{html.escape(flight_note.source_url)}</a></li></ul>"

        note_body = getattr(flight_note, "note", getattr(flight_note, "note_text", ""))
        sec_notes = f"""
        <section class="report-section">
            <h2>📝 View 10: Persistent Flight Operational Notes &amp; Qualitative Explanations</h2>
            <div class="summary-callout" style="margin-bottom: 14px;">
                <b>Title:</b> {html.escape(flight_note.title)} &nbsp;|&nbsp;
                <b>Category:</b> <code>{html.escape(flight_note.category)}</code> &nbsp;|&nbsp;
                <b>Author:</b> {html.escape(flight_note.author)} &nbsp;|&nbsp;
                <b>Last Updated:</b> {html.escape(flight_note.updated_at[:19])}
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px 22px; font-size: 14px; line-height: 1.6; white-space: pre-wrap;">{html.escape(note_body)}</div>
            {cits_html}
        </section>
        """
        sections.append(sec_notes)

    # =========================================================================
    # 11. APPENDIX: MATHEMATICAL & ATMOSPHERIC PHYSICS METHODOLOGY
    # =========================================================================
    sec_appendix = """
    <section class="report-section">
        <h2>📐 Appendix: Comprehensive Mathematical &amp; Atmospheric Physics Methodology</h2>
        <p>
            Formal mathematical equations, thermodynamic derivations, and international aviation standards implemented
            in FlightML for trajectory physics, airspace economics, and environmental footprint accounting.
        </p>

        <h3>1. Geodesic Trajectory Distance (Haversine Formula)</h3>
        <p>
            For two spherical coordinates $(\\phi_1, \\lambda_1)$ and $(\\phi_2, \\lambda_2)$ with Earth radius $R = 3440.065\\text{ nm}$ ($6371.0\\text{ km}$):
        </p>
        <div style="background: #f8fafc; border-left: 4px solid #1e3799; padding: 12px 18px; margin: 10px 0; font-family: monospace; font-size: 13.5px;">
            a = sin²(Δϕ / 2) + cos(ϕ₁) · cos(ϕ₂) · sin²(Δλ / 2)<br/>
            c = 2 · atan2(√a, √(1 - a))<br/>
            D_great_circle = R · c
        </div>

        <h3>2. Horizontal Flight Inefficiency (Eurocontrol PRU KEA &amp; KEP)</h3>
        <div style="background: #f8fafc; border-left: 4px solid #1e3799; padding: 12px 18px; margin: 10px 0; font-family: monospace; font-size: 13.5px;">
            KEA = ((D_actual_enroute - D_achieved_gcd) / D_achieved_gcd) · 100%<br/>
            KEP = ((D_filed_enroute - D_achieved_gcd) / D_achieved_gcd) · 100%<br/>
            Detour_Ratio = D_actual / D_filed
        </div>

        <h3>3. Equivalent Still Air Distance (ESAD) &amp; Wind Triangle</h3>
        <p>
            Evaluates aerodynamic distance traveled relative to the ambient air mass:
        </p>
        <div style="background: #f8fafc; border-left: 4px solid #1e3799; padding: 12px 18px; margin: 10px 0; font-family: monospace; font-size: 13.5px;">
            V_tailwind,i = V_wind,i · cos(θ_wind_toward - θ_track)<br/>
            TAS_i = GS_i - V_tailwind,i<br/>
            ESAD = Σ [ D_ground,i · (TAS_i / GS_i) ]<br/>
            ΔD_air = D_ground - ESAD &nbsp;&nbsp;(>0: air miles saved by tailwind; <0: headwind penalty)
        </div>

        <h3>4. ICAO Standard Atmosphere (ISA) &amp; Pressure Profiles</h3>
        <div style="background: #f8fafc; border-left: 4px solid #1e3799; padding: 12px 18px; margin: 10px 0; font-family: monospace; font-size: 13.5px;">
            Troposphere (h ≤ 11,000 m):<br/>
            &nbsp;&nbsp;T_ISA(h) = 288.15 - 0.0065 · h  [K]<br/>
            &nbsp;&nbsp;P(h) = 1013.25 · (1 - 0.0065 · h / 288.15)^5.25588  [hPa]<br/>
            Stratosphere Isothermal Layer (11,000 m &lt; h ≤ 20,000 m):<br/>
            &nbsp;&nbsp;T_ISA(h) = 216.65 K (-56.5°C)<br/>
            &nbsp;&nbsp;P(h) = 226.32 · exp(-0.000157688 · (h - 11000))  [hPa]
        </div>

        <h3>5. Murphy &amp; Koop (2005) Water Vapor Thermodynamics &amp; RH_ice</h3>
        <p>
            Converts relative humidity over liquid water (RH_water) to relative humidity over hexagonal ice (RH_ice):
        </p>
        <div style="background: #f8fafc; border-left: 4px solid #1e3799; padding: 12px 18px; margin: 10px 0; font-family: monospace; font-size: 13.5px;">
            e_sat_water / e_sat_ice = exp(0.00955 · (273.15 - T))  [for T in K]<br/>
            RH_ice = RH_water · (e_sat_water / e_sat_ice)
        </div>

        <h3>6. The Schmidt-Appleman Criterion (SAC) &amp; ISSR Conditions</h3>
        <p>
            Slope of the exhaust mixing line in the (T, e) plane:
        </p>
        <div style="background: #f8fafc; border-left: 4px solid #1e3799; padding: 12px 18px; margin: 10px 0; font-family: monospace; font-size: 13.5px;">
            G = (c_p · P · EI_H2O) / (ε · Q · (1 - η))<br/>
            &nbsp;&nbsp;where c_p = 1005 J/(kg·K), ε = 0.622, EI_H2O = 1.25 kg/kg, Q = 43.13×10⁶ J/kg, η = engine propulsion efficiency.<br/>
            Critical SAC threshold temperature:<br/>
            &nbsp;&nbsp;T_SAC(P, η) ≈ -41.5 + 2.8 · ln(P / 250.0) + 12.0 · (η - 0.33)  [°C]<br/>
            Persistent Contrail Cirrus Condition:<br/>
            &nbsp;&nbsp;T_ambient ≤ T_SAC(P, η)  AND  RH_ice ≥ 100%  AND  RH_water &lt; 100%
        </div>

        <h3>7. Persistent Contrail Cirrus Radiative Forcing (18.5 kg CO₂e / nm)</h3>
        <div style="background: #f8fafc; border-left: 4px solid #1e3799; padding: 12px 18px; margin: 10px 0; font-family: monospace; font-size: 13.5px;">
            Contrail_Cirrus_CO2e [t] = (D_persistent_ISSR [nm] · 18.5) / 1000<br/>
            Total_Climate_Multiplier = (E_direct_CO2 + E_contrail_CO2e) / E_direct_CO2
        </div>

        <h3>8. Multi-Objective Eco-Trajectory Optimization Cost Functional</h3>
        <div style="background: #f8fafc; border-left: 4px solid #1e3799; padding: 12px 18px; margin: 10px 0; font-family: monospace; font-size: 13.5px;">
            min J(x) = ΔE_direct_CO2 + ΔE_cirrus_CO2e + α_NOx · ΔE_NOx<br/>
            subject to:<br/>
            &nbsp;&nbsp;FL_i ≤ FL_certified_ceiling(ac_type)<br/>
            &nbsp;&nbsp;FL_i complies with ICAO RVSM Semicircular Rule (Odd Eastbound / Even Westbound)<br/>
            &nbsp;&nbsp;Buffet Margin ≥ 1.30g, Rate of Climb Margin ≥ 300 fpm
        </div>

        <h3>9. Eurocontrol CRCO Route Charging Formula</h3>
        <div style="background: #f8fafc; border-left: 4px solid #1e3799; padding: 12px 18px; margin: 10px 0; font-family: monospace; font-size: 13.5px;">
            Charge_FIR = d_FIR · p · U_FIR<br/>
            &nbsp;&nbsp;where d_FIR = traversal distance in FIR [km] / 100<br/>
            &nbsp;&nbsp;p = √(MTOW_metric_tons / 50.0) [Aircraft Weight Factor]<br/>
            &nbsp;&nbsp;U_FIR = National Unit Rate [EUR / Service Unit]
        </div>
    </section>
    """
    sections.append(sec_appendix)

    body_html = "\n".join(sections)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Comprehensive 4D Flight Intelligence Report — #{flight_id} ({adep} ➔ {ades})</title>
    <style>
        :root {{
            --bg: #f4f7fb;
            --card: #ffffff;
            --primary: #0c2461;
            --accent: #0984e3;
            --text: #1e272e;
            --muted: #576574;
            --border: #dfe6e9;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 24px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }}
        .report-container {{
            max-width: 1320px;
            margin: 0 auto;
        }}
        .report-header {{
            background: linear-gradient(135deg, #0c2461 0%, #1e3799 55%, #0984e3 100%);
            color: #ffffff;
            padding: 28px 34px;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(12, 36, 97, 0.18);
            margin-bottom: 24px;
        }}
        .report-header h1 {{
            margin: 0 0 10px 0;
            font-size: 26px;
            letter-spacing: -0.3px;
        }}
        .report-header .meta {{
            font-size: 13.5px;
            opacity: 0.94;
            line-height: 1.6;
        }}
        .report-section {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 26px 30px;
            margin-bottom: 24px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        }}
        .report-section h2 {{
            margin-top: 0;
            color: #0c2461;
            border-bottom: 2px solid #eef2f7;
            padding-bottom: 10px;
            font-size: 20px;
        }}
        .report-section h3 {{
            margin-top: 22px;
            color: #2c3e50;
            font-size: 16px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin: 16px 0 22px 0;
        }}
        .kpi-card {{
            background: #f8fafd;
            border: 1px solid #e2e8f0;
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            padding: 12px 15px;
        }}
        .kpi-label {{
            font-size: 11.5px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            color: var(--muted);
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 19px;
            font-weight: 700;
            color: #0c2461;
            margin: 4px 0 2px 0;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: var(--muted);
        }}
        .chart-card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            margin: 18px 0;
            text-align: center;
        }}
        .report-img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        .chart-caption {{
            margin-top: 8px;
            font-size: 12.5px;
            color: var(--muted);
            font-weight: 600;
        }}
        .summary-callout {{
            background: #eef6ff;
            border-left: 4px solid #0984e3;
            padding: 14px 18px;
            border-radius: 6px;
            font-size: 13.5px;
            line-height: 1.6;
        }}
        .table-wrap {{
            overflow-x: auto;
            margin: 10px 0 18px 0;
        }}
        table.styled-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
        }}
        table.styled-table th {{
            background: #0c2461;
            color: #ffffff;
            text-align: left;
            padding: 9px 11px;
            font-weight: 600;
        }}
        table.styled-table td {{
            padding: 8px 11px;
            border-bottom: 1px solid #edf2f7;
        }}
        table.styled-table tr:nth-child(even) {{
            background: #f8fafc;
        }}
        .muted-note {{
            color: var(--muted);
            font-style: italic;
        }}
        .report-footer {{
            text-align: center;
            color: var(--muted);
            font-size: 12.5px;
            margin-top: 24px;
            padding: 14px;
        }}
        @media print {{
            body {{ background: #ffffff; padding: 0; }}
            .report-section {{ page-break-inside: avoid; box-shadow: none; }}
            iframe {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <header class="report-header">
            <h1>✈️ Comprehensive 4D Flight Intelligence Report: #{flight_id} ({adep} ➔ {ades})</h1>
            <div class="meta">
                <b>FlightML Intelligence Platform</b> &nbsp;|&nbsp;
                <b>Cycle:</b> Eurocontrol {html.escape(ds.period)} &nbsp;|&nbsp;
                <b>Callsign:</b> <code>{html.escape(str(flight.get('callsign') or 'N/A'))}</code> &nbsp;|&nbsp;
                <b>Operator:</b> {html.escape(ac_op)} &nbsp;|&nbsp;
                <b>Airframe:</b> {html.escape(ac_type)} &nbsp;|&nbsp;
                <b>Pricing &amp; Currency:</b> Euros (€ EUR) &nbsp;|&nbsp;
                <b>Report Timestamp:</b> {now_str}
            </div>
        </header>
        {body_html}
        <footer class="report-footer">
            Generated by FlightML Comprehensive 4D Flight Intelligence Engine — Standalone document with embedded interactive OpenStreetMap Leaflet maps &amp; mathematical methodology appendix.
        </footer>
    </div>
</body>
</html>
"""
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(full_html, encoding="utf-8")

    return full_html


# ---------------------------------------------------------------------------
# Master Static HTML Report Generator
# ---------------------------------------------------------------------------
def generate_static_html_report(
    ds: FlightDataset | None = None,
    page_key: str = "trajectory_rnp",
    flight_id: int | None = None,
    hour_utc: int = 10,
    show_navaids: bool = True,
    nav_type_filter: list[str] | None = None,
    world_nav_types: list[str] | None = None,
    world_region_name: str = "🌍 Entire World",
    world_region_bounds: tuple[float, float, float, float] = (-85.0, 85.0, -180.0, 180.0),
    world_search_query: str | None = None,
    world_limit: int = 2500,
    twin_adep: str = "RKSI",
    twin_ades: str = "EGLL",
    twin_ac: str = "B77W",
    rec_adep: str = "EGLL",
    rec_ades: str = "KORD",
    rec_op: str = "BAW",
    conflict_hour: int = 10,
    conflict_nm: float = 5.0,
    gdp_airport: str = "EGLL",
    gdp_aar: int = 25,
    gdp_date: str | None = None,
    fra_boundary_pair: str | None = None,
    fra_min_flights: int = 500,
    fra_annual_growth: float = 0.02,
    fra_realization: float = 0.85,
    fra_fuel_price: float = 0.6421,
    fra_fuel_opt: float = 0.015,
    fra_fuel_med: float = 0.035,
    fra_fuel_shock: float = 0.065,
    dme_view: str = "VIEW_1",
    dme_fir: str = "ALL",
    dme_fl: int = 200,
    dme_inactive_ids: tuple[str, ...] | None = None,
    dme_active_ids: tuple[str, ...] | None = None,
    radar_view: str = "VIEW_1",
    radar_fir: str = "ALL",
    radar_fl: int = 200,
    radar_inactive_ids: tuple[str, ...] | None = None,
    radar_active_ids: tuple[str, ...] | None = None,
    output_path: Path | str | None = None,
) -> str:
    """
    Generate a complete, standalone, self-contained HTML report with embedded high-DPI Base64 PNG
    charts/maps and styled data tables for any selected dashboard page (or the full executive bundle).
    """
    if ds is None:
        ds = FlightDataset()

    if flight_id is None:
        sample_df = ds.query("SELECT ectrl_id FROM flights LIMIT 1")
        flight_id = int(sample_df.iloc[0]["ectrl_id"]) if not sample_df.empty else 202007026

    # If single_flight is requested, return the comprehensive 4D flight report with full interactive map tiles!
    if page_key == "single_flight":
        return generate_flight_comprehensive_html_report(ds, flight_id=int(flight_id), output_path=output_path)

    title = REPORT_PAGE_TITLES.get(page_key, "FlightML Operational Intelligence Report")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    sections: list[str] = []

    # 1. Single Flight Views
    if page_key in ("single_flight", "trajectory_rnp", "full_executive_bundle"):
        sections.append(
            _build_trajectory_rnp_section(
                ds,
                flight_id=int(flight_id),
                show_navaids=show_navaids,
                nav_type_filter=nav_type_filter,
            )
        )

    if page_key in (
        "single_flight",
        "flight_performance_eco",
        "vertical_cdo_cco",
        "delay_diagnostics",
        "fuel_carbon",
        "full_executive_bundle",
    ):
        sections.append(_build_flight_performance_eco_section(ds, flight_id=int(flight_id)))

    # 2. System-Level Airspace Views
    if page_key in ("system_summary", "airspace_overview", "full_executive_bundle"):
        sections.append(_build_airspace_overview_section(ds, hour_utc=hour_utc))

    # 3. Free Route Airspace (FRA) Boundary Views
    if page_key in ("system_summary", "fra_boundaries", "full_executive_bundle"):
        sections.append(
            _build_fra_boundary_section(
                ds,
                boundary_pair=fra_boundary_pair,
                min_flights=fra_min_flights,
                annual_growth_rate=fra_annual_growth,
                realization_rate=fra_realization,
                jet_a_eur_per_kg=fra_fuel_price,
                fuel_growth_opt=fra_fuel_opt,
                fuel_growth_med=fra_fuel_med,
                fuel_growth_shock=fra_fuel_shock,
            )
        )

    # 4. Macro Period Evolution & Longitudinal Trends
    if page_key in ("system_summary", "macro_reference", "longitudinal_trends", "full_executive_bundle"):
        sections.append(_build_macro_reference_section(ds))

    # 5. What-If Scenarios & Operations Research
    if page_key in ("what_if_scenarios", "what_if_stress_tests", "full_executive_bundle"):
        sections.append(
            _build_what_if_stress_section(
                ds,
                conflict_hour=conflict_hour,
                conflict_nm=conflict_nm,
                gdp_airport=gdp_airport,
                gdp_aar=gdp_aar,
                gdp_date=gdp_date,
            )
        )

    # 6. Aeronautical Maps & Navigation
    if page_key in ("maps_navigation", "global_navaids", "full_executive_bundle"):
        sections.append(
            _build_global_navaids_section(
                ds,
                nav_types=world_nav_types,
                region_name=world_region_name,
                bounds=world_region_bounds,
                search_query=world_search_query,
                limit=world_limit,
            )
        )

    # 7. Route & Corridor Economics
    if page_key in ("general_economics", "corridor_twin_benchmarks", "full_executive_bundle"):
        sections.append(
            _build_corridor_twin_section(
                ds,
                twin_adep=twin_adep,
                twin_ades=twin_ades,
                twin_ac=twin_ac,
                rec_adep=rec_adep,
                rec_ades=rec_ades,
                rec_op=rec_op,
            )
        )

    # 8. ECAC CNS Infrastructure Rationalization (DME & Radar)
    if page_key in ("maps_navigation", "ecac_dme_rationalization", "full_executive_bundle"):
        sections.append(
            _build_ecac_rationalization_section(
                ds,
                station_type="DME",
                active_view=dme_view,
                target_fir=dme_fir,
                selected_fl=dme_fl,
                custom_inactive_ids=dme_inactive_ids,
                custom_active_ids=dme_active_ids,
            )
        )

    if page_key in ("maps_navigation", "ecac_radar_rationalization", "full_executive_bundle"):
        sections.append(
            _build_ecac_rationalization_section(
                ds,
                station_type="RADAR",
                active_view=radar_view,
                target_fir=radar_fir,
                selected_fl=radar_fl,
                custom_inactive_ids=radar_inactive_ids,
                custom_active_ids=radar_active_ids,
            )
        )

    if page_key in ("dme_resilience",):
        from flightml.eval.dme_permutations import simulate_dme_permutations, create_dme_resilience_map
        fir_target = dme_fir if (dme_fir and dme_fir != "ALL") else "EGTTFIR"
        std_target = dme_view if dme_view in ("RNAV_5", "RNAV_2", "RNAV_1", "RNP_1") else "RNAV_5"
        sim_res = simulate_dme_permutations(ds, fir_id=fir_target, period=ds.period, target_standard=std_target, max_permutations=12)
        opt = sim_res.optimal_permutation or sim_res.permutations[0]
        res_map = create_dme_resilience_map(sim_res, dataset=ds)
        map_iframe = _folium_map_to_iframe(res_map, height=520, title=f"{fir_target} DME Resilience Map")

        perm_rows = []
        for r in sim_res.permutations:
            w24 = r.windows["24h"]
            def _fnet(v: float) -> str:
                return f"-€{abs(v):,.0f}" if v < 0 else (f"+€{v:,.0f}" if v > 0 else "€0")
            perm_rows.append(
                f"<tr><td><code>{html.escape(r.permutation_id)}</code></td>"
                f"<td><b>{html.escape(r.name)}</b></td>"
                f"<td>{r.active_count}/{sim_res.baseline_station_count}</td>"
                f"<td>{r.coverage_coverable_pct:.1f}%</td>"
                f"<td>{r.degraded_airports_count}</td>"
                f"<td>{_fnet(r.windows['1h'].net_economic_impact_eur)}</td>"
                f"<td>{_fnet(r.windows['6h'].net_economic_impact_eur)}</td>"
                f"<td><b>{_fnet(w24.net_economic_impact_eur)}</b></td>"
                f"<td>{html.escape(r.classification)}</td></tr>"
            )
        table_html = (
            '<div class="table-wrap"><table class="styled-table">'
            '<thead><tr><th>ID</th><th>Configuration</th><th>Active</th><th>Coverage</th><th>Degraded Apt</th><th>1h Net</th><th>6h Net</th><th>24h Net</th><th>Classification</th></tr></thead>'
            '<tbody>' + "".join(perm_rows) + '</tbody></table></div>'
        )

        sections.append(
            f'<section class="report-section">'
            f'<h2>📉 DME Network Resilience, Permutations &amp; Operational Delay Economics</h2>'
            f'<div class="kpi-grid">'
            f'  <div class="kpi-card"><div class="kpi-label">Optimal Equipment Posture</div><div class="kpi-value">{opt.active_count} / {sim_res.baseline_station_count} DMEs</div><div class="kpi-sub">Decommissions {opt.disabled_count} redundant beacons</div></div>'
            f'  <div class="kpi-card"><div class="kpi-label">Net Daily Benefit (24h)</div><div class="kpi-value" style="color: #16a34a;">€{abs(opt.windows["24h"].net_economic_impact_eur):,.0f}</div><div class="kpi-sub">Net savings after absorbing tactical delays</div></div>'
            f'  <div class="kpi-card"><div class="kpi-label">Airspace Coverage Compliance</div><div class="kpi-value">{opt.coverage_coverable_pct:.1f}%</div><div class="kpi-sub">{opt.cells_met} coverable cells meeting {std_target}</div></div>'
            f'</div>'
            f'{map_iframe}'
            f'<h3>📊 Evaluated Permutations &amp; Multi-Window Balances</h3>'
            f'{table_html}'
            f'</section>'
        )

    # Robust Safety Fallback: Never return an empty report body!
    if not sections:
        sections.append(
            f'<section class="report-section"><h2>📋 Operational Report Overview — {html.escape(title)}</h2>'
            f'<p>Report generated for period <b>{html.escape(ds.period)}</b>. Key view: <code>{html.escape(page_key)}</code>.</p></section>'
        )
        sections.append(_build_airspace_overview_section(ds, hour_utc=hour_utc))
        sections.append(
            _build_fra_boundary_section(
                ds,
                boundary_pair=fra_boundary_pair,
                min_flights=fra_min_flights,
                annual_growth_rate=fra_annual_growth,
                realization_rate=fra_realization,
                jet_a_eur_per_kg=fra_fuel_price,
                fuel_growth_opt=fra_fuel_opt,
                fuel_growth_med=fra_fuel_med,
                fuel_growth_shock=fra_fuel_shock,
            )
        )

    body_html = "\n".join(sections)

    doc_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{html.escape(title)} — FlightML ({html.escape(ds.period)})</title>
    <style>
        :root {{
            --bg: #f4f7fb;
            --card: #ffffff;
            --primary: #1e3799;
            --accent: #0984e3;
            --text: #1e272e;
            --muted: #576574;
            --border: #dfe6e9;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 24px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.5;
        }}
        .report-container {{
            max-width: 1280px;
            margin: 0 auto;
        }}
        .report-header {{
            background: linear-gradient(135deg, #0c2461 0%, #1e3799 60%, #0984e3 100%);
            color: #ffffff;
            padding: 26px 32px;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(12, 36, 97, 0.16);
            margin-bottom: 24px;
        }}
        .report-header h1 {{
            margin: 0 0 8px 0;
            font-size: 26px;
        }}
        .report-header .meta {{
            font-size: 13.5px;
            opacity: 0.92;
        }}
        .report-section {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        }}
        .report-section h2 {{
            margin-top: 0;
            color: #0c2461;
            border-bottom: 2px solid #eef2f7;
            padding-bottom: 10px;
        }}
        .report-section h3 {{
            margin-top: 22px;
            color: #2c3e50;
            font-size: 17px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 14px;
            margin: 16px 0 22px 0;
        }}
        .kpi-card {{
            background: #f8fafd;
            border: 1px solid #e2e8f0;
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            padding: 12px 15px;
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            color: var(--muted);
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 20px;
            font-weight: 700;
            color: #0c2461;
            margin: 4px 0 2px 0;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: var(--muted);
        }}
        .chart-card {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
            margin: 18px 0;
            text-align: center;
        }}
        .report-img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        .chart-caption {{
            margin-top: 8px;
            font-size: 12.5px;
            color: var(--muted);
            font-weight: 600;
        }}
        .summary-callout {{
            background: #eef6ff;
            border-left: 4px solid #0984e3;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 13.5px;
        }}
        .two-col {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
            gap: 20px;
        }}
        .table-wrap {{
            overflow-x: auto;
            margin: 10px 0 18px 0;
        }}
        table.styled-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
        }}
        table.styled-table th {{
            background: #0c2461;
            color: #ffffff;
            text-align: left;
            padding: 9px 11px;
            font-weight: 600;
        }}
        table.styled-table td {{
            padding: 8px 11px;
            border-bottom: 1px solid #edf2f7;
        }}
        table.styled-table tr:nth-child(even) {{
            background: #f8fafc;
        }}
        .muted-note {{
            color: var(--muted);
            font-style: italic;
        }}
        .report-footer {{
            text-align: center;
            color: var(--muted);
            font-size: 12.5px;
            margin-top: 24px;
            padding: 12px;
        }}
        @media print {{
            body {{ background: #ffffff; padding: 0; }}
            .report-section {{ page-break-inside: avoid; box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <header class="report-header">
            <h1>✈️ {html.escape(title)}</h1>
            <div class="meta">
                <b>Platform:</b> FlightML Global Airspace Network Intelligence &amp; ML &nbsp;|&nbsp;
                <b>Active Cycle:</b> Eurocontrol {html.escape(ds.period)} &nbsp;|&nbsp;
                <b>Active Flight ID:</b> #{int(flight_id)} &nbsp;|&nbsp;
                <b>Currency:</b> Euros (€ EUR) &nbsp;|&nbsp;
                <b>Generated:</b> {now_str}
            </div>
        </header>
        {body_html}
        <footer class="report-footer">
            Generated by FlightML Static HTML Report Engine — Self-contained offline document with embedded Base64 PNG figures &amp; ICAO/Eurocontrol PRU scorecards.
        </footer>
    </div>
</body>
</html>
"""
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(doc_html, encoding="utf-8")

    return doc_html
