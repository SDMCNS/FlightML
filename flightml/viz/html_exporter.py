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

from flightml.data.dataset import FlightDataset
from flightml.eval.cdo_analyzer import analyze_cco_climb, analyze_cdo_descent
from flightml.eval.contrail_climate import analyze_flight_wind_and_contrails
from flightml.eval.crco_economics import compute_flight_crco_charges
from flightml.eval.diagnostics import diagnose_flight_bottlenecks
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
