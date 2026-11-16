from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from flightml.data.dataset import FlightDataset
from flightml.data.archive_ingest import list_available_periods, list_selectable_periods, FULL_YEAR_LABEL, has_contiguous_12m_archive
from flightml.geo.trajectory import analyze_trajectory
from flightml.weather.metar_client import MetarClient
from flightml.eval.diagnostics import diagnose_flight_bottlenecks
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
from flightml.eval.aircraft_engine_perf import (
    AircraftEnginePerformance,
    CruiseAltitudeEfficiency,
    compute_cruise_altitude_efficiency,
    resolve_aircraft_engine_perf,
)
from flightml.eval.multi_level_kpis import (
    query_multi_level_kpis,
    get_dataset_date_bounds,
    get_available_firs,
    query_fir_traffic_segmentation,
    FIRTrafficSegmentSummary,
)
from flightml.eval.flight_twins import compare_twin_flights, render_twin_comparison_map
from flightml.eval.cdo_analyzer import (
    analyze_cdo_descent,
    analyze_cco_climb,
    analyze_vertical_profile_efficiency,
)
from flightml.eval.pru_kpis import (
    compute_flight_pru_kpis,
    analyze_network_level_capping,
)
from flightml.geo.fra_analyzer import (
    analyze_flight_fra_adherence,
    compare_fra_evolution,
)
from flightml.geo.fra_boundary_engine import (
    analyze_fir_boundary_inefficiencies,
    compare_fra_boundary_evolution as compare_fra_boundary_evo,
    get_boundary_detail,
    get_worst_flights_for_boundary,
    calculate_fra_long_term_forecast,
)
from flightml.features.radio_workload import estimate_airport_and_sector_radio_workload
from flightml.eval.crco_economics import (
    compute_flight_crco_charges,
    detect_route_charge_avoidance,
)
from flightml.geo.conflict_scanner import (
    scan_airspace_proximity_events,
    compute_sector_complexity_index,
    compute_hourly_grid_saturation,
)
from flightml.geo.rnp_navaid_analyzer import (
    analyze_flight_navaids_and_rnp,
    query_unified_navaids,
)
from flightml.weather.runway_safety import assess_flight_runway_safety
from flightml.eval.contrail_climate import analyze_flight_wind_and_contrails
from flightml.sim.tail_swap_optimizer import optimize_hub_tail_swaps
from flightml.sim.gdp_optimizer import optimize_ground_delay_program
from flightml.eval.recurring_series import analyze_recurring_flight_series
from flightml.eval.macro_comparison import compare_aviation_periods
from flightml.weather.network_weather_impact import analyze_network_weather_impact
from flightml.geo.airac_diff import compare_airac_cycles
from flightml.models.airline_stress import stress_test_airline
from flightml.sim.what_if import simulate_turnaround_mitigation, simulate_runway_weather_scenario
from flightml.features.sector_bottlenecks import analyze_fir_bottlenecks
from flightml.features.fir_saturation import (
    calculate_fir_hourly_saturation,
    calculate_flight_fir_dwell_inefficiency,
)
from flightml.features.aua_saturation import (
    calculate_aua_hourly_saturation,
    calculate_flight_aua_dwell_inefficiency,
)
from flightml.eval.surface_taxi import (
    analyze_airport_taxi_out_performance,
    analyze_flight_taxi_out,
)
from flightml.eval.jet_fuel_prices import get_all_weekly_fuel_prices_df
from flightml.viz.map_builder import (
    create_trajectory_map,
    create_flight_fir_charge_map,
    create_grid_saturation_map,
    create_conflict_map,
    create_world_navaids_map,
    create_fra_boundary_map,
)
from flightml.viz.html_exporter import (
    generate_static_html_report,
    REPORT_PAGE_TITLES,
)
from flightml.eval.longitudinal import LongitudinalEngine
from flightml.models.airspace_forecaster import AirspaceForecaster
from flightml.eval.airline_economics import AirlineEconomicsEngine
from flightml.models.delay_contagion_graph import DelayContagionGraphEngine
from flightml.models.flight2vec import Flight2VecEngine
from flightml.models.pinn_mass import PINNMassInverter
from flightml.models.contrail_surrogate import ContrailNeuralSurrogate
from flightml.models.st_gnn import SpatioTemporalDelayEngine
from flightml.sim.airspace_closure import AirspaceClosureSimulator
from flightml.sim.marl_atfm import ASSUMPTIONS_GLOSSARY, MultiAgentATFMSimulator
from flightml.eval.passenger_journey import PassengerJourneyEngine



st.set_page_config(
    page_title="FlightML - Airspace Network Intelligence",
    page_icon="✈️",
    layout="wide",
)


# ----------------- Sidebar: Period & Global Controls -----------------
st.sidebar.title("✈️ FlightML Control Deck")


def _get_available_periods() -> list[str]:
    try:
        periods = list_selectable_periods(include_full_year=True)
        if periods:
            return periods
    except Exception:
        pass
    try:
        periods = list_available_periods()
        if periods:
            return periods
    except Exception:
        pass
    try:
        if hasattr(FlightDataset, "list_available_periods"):
            return FlightDataset.list_available_periods()
    except Exception:
        pass
    return ["201611", "202510"]


available_periods = _get_available_periods()
default_idx = available_periods.index("202510") if "202510" in available_periods else 0
selected_period = st.sidebar.selectbox(
    "Active Eurocontrol Period",
    options=available_periods,
    index=default_idx,
    help="Select Eurocontrol archive period. Includes individual monthly archives as well as the contiguous 12-month full year baseline."
)

if "Full Year" in selected_period or "202411-202510" in selected_period:
    st.sidebar.success("🗓️ **Full Contiguous Year Active:** 12 contiguous months (Nov 2024 – Oct 2025, 10.2M trajectories federated).")


@st.cache_resource(show_spinner=False)
def get_dataset(period: str) -> FlightDataset:
    """Cached dataset accessor preventing repeated DuckDB view initialization on reruns."""
    return FlightDataset(period=period)


@st.cache_resource(show_spinner=False)
def get_metar_client() -> MetarClient:
    """Cached IEM ASOS METAR client."""
    return MetarClient()


@st.cache_data(show_spinner=False)
def _cached_grid_saturation(period: str, hour_utc: int, grid_size_deg: float = 5.0):
    return compute_hourly_grid_saturation(get_dataset(period), hour_utc=hour_utc, grid_size_deg=grid_size_deg)


@st.cache_data(show_spinner=False)
def _cached_flight_navaids_rnp(period: str, flight_id: int):
    return analyze_flight_navaids_and_rnp(get_dataset(period), int(flight_id))


@st.cache_data(show_spinner=False)
def _cached_world_navaids(
    period: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    nav_types_tuple: tuple[str, ...],
    search_query: str | None,
    limit: int,
):
    return query_unified_navaids(
        get_dataset(period),
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        nav_types=list(nav_types_tuple),
        search_query=search_query,
        limit=limit,
    )


@st.cache_data(show_spinner=False)
def _cached_tail_swaps(period: str, operator: str | None, hub_airport: str | None):
    return optimize_hub_tail_swaps(get_dataset(period), operator=operator, hub_airport=hub_airport)


@st.cache_data(show_spinner=False)
def _cached_proximity_scan(period: str, hour: int, horiz_sep_nm: float):
    return scan_airspace_proximity_events(get_dataset(period), hour=hour, horiz_sep_nm=horiz_sep_nm)


@st.cache_data(show_spinner=False)
def _cached_gdp_optimizer(
    period: str,
    target_airport: str,
    start_hour: int,
    end_hour: int,
    aar: int,
    program_date: str | None = None,
):
    return optimize_ground_delay_program(
        get_dataset(period),
        target_airport=target_airport,
        program_date=program_date,
        start_hour=start_hour,
        end_hour=end_hour,
        degraded_capacity_rate_per_hr=aar,
    )


def get_sample_flight_id(ds: FlightDataset) -> int:
    """Retrieve an active sample flight ID for the currently selected period."""
    try:
        sample_df = ds.query("SELECT ectrl_id FROM flights LIMIT 1")
        if not sample_df.empty:
            return int(sample_df.iloc[0]["ectrl_id"])
    except Exception:
        pass
    return 202007026


ds = get_dataset(selected_period)
metar_client = get_metar_client()
sample_flight_id = get_sample_flight_id(ds)

# ----------------- Sidebar: Interactive Flight Filter -----------------
with st.sidebar.expander("🔍 Filter & Select Flights", expanded=True):
    st.markdown("**Search flights across multiple operational dimensions:**")
    f_adep = st.text_input("Origin (ADEP)", value="", placeholder="e.g. EGLL, LFPG", key=f"f_adep_{selected_period}").strip().upper()
    f_ades = st.text_input("Destination (ADES)", value="", placeholder="e.g. KORD, JFK", key=f"f_ades_{selected_period}").strip().upper()
    f_op = st.text_input("Operator", value="", placeholder="e.g. BAW, RYR, DLH", key=f"f_op_{selected_period}").strip().upper()
    f_ac = st.text_input("Aircraft Type", value="", placeholder="e.g. B77W, A320, B738", key=f"f_ac_{selected_period}").strip().upper()
    f_tail = st.text_input("Tail Number (Reg)", value="", placeholder="e.g. G-STBA, EI-DCL", key=f"f_tail_{selected_period}").strip().upper()
    f_tod = st.selectbox(
        "Time of Day Bank",
        ["All", "Morning (06:00-12:00)", "Afternoon (12:00-18:00)", "Evening (18:00-24:00)", "Night (00:00-06:00)"],
        key=f"f_tod_{selected_period}",
    )
    f_dist = st.slider("Distance (nm)", 0, 7000, (0, 7000), step=250, key=f"f_dist_{selected_period}")
    f_fl = st.slider("Cruising Flight Level", 0, 450, (0, 450), step=10, key=f"f_fl_{selected_period}")

    if st.button("Apply Filters & Find Flights", key=f"btn_search_{selected_period}"):
        matched_df = ds.search_flights(
            adep=f_adep if f_adep else None,
            ades=f_ades if f_ades else None,
            ac_operator=f_op if f_op else None,
            ac_type=f_ac if f_ac else None,
            ac_reg=f_tail if f_tail else None,
            time_of_day=f_tod,
            min_distance_nm=f_dist[0],
            max_distance_nm=f_dist[1],
            min_fl=f_fl[0],
            max_fl=f_fl[1],
            limit=30,
        )
        if not matched_df.empty:
            st.session_state[f"flight_options_{selected_period}"] = dict(zip(matched_df["label"], matched_df["ectrl_id"]))
            st.success(f"Matched {len(matched_df)} flights!")
        else:
            st.warning("No flights matched the specified criteria.")

# Pre-populate options if not searched yet
if f"flight_options_{selected_period}" not in st.session_state:
    sample_df = ds.search_flights(limit=15)
    if not sample_df.empty:
        st.session_state[f"flight_options_{selected_period}"] = dict(zip(sample_df["label"], sample_df["ectrl_id"]))

current_options = st.session_state.get(f"flight_options_{selected_period}", {})

# Direct jump by ECTRL ID in sidebar
with st.sidebar.expander("⚡ Direct Jump by ECTRL ID", expanded=False):
    jump_id_str = st.text_input("Enter ECTRL ID", placeholder="e.g. 291262259", key=f"jump_input_{selected_period}").strip()
    if st.button("Load Flight", key=f"btn_jump_{selected_period}"):
        if jump_id_str.isdigit():
            target_id = int(jump_id_str)
            try:
                target_flight = ds.get_flight(target_id)
            except Exception as e:
                target_flight = None
                st.error(f"Could not load flight {target_id}: {e}")

            if target_flight:
                lbl = ds.format_flight_label(target_flight)
                current_options[lbl] = target_id
                st.session_state[f"flight_options_{selected_period}"] = current_options
                st.session_state[f"active_flight_select_{selected_period}"] = lbl
                st.session_state[f"active_flight_id_{selected_period}"] = target_id
                st.success(f"Loaded {lbl}!")
                st.rerun()
            elif not target_flight:
                st.error(f"Flight ID {target_id} not found in period {selected_period}.")
        else:
            st.error("Please enter a valid numeric ECTRL ID.")

if current_options:
    selected_flight_label = st.sidebar.selectbox(
        "Active Flight Selection",
        options=list(current_options.keys()),
        key=f"active_flight_select_{selected_period}",
    )
    active_flight_id = current_options[selected_flight_label]
else:
    active_flight_id = sample_flight_id

st.session_state[f"active_flight_id_{selected_period}"] = active_flight_id


def render_page_export_bar(page_key: str, widget_suffix: str) -> None:
    """Render an inline one-click Static HTML Report generator & download button for any page."""
    title_lbl = REPORT_PAGE_TITLES.get(page_key, "Static HTML Report")
    state_key = f"html_export_blob_{selected_period}_{page_key}_{widget_suffix}"
    c_exp1, c_exp2 = st.columns([1.8, 2.2])
    with c_exp1:
        if st.button(
            f"📥 Prepare Static HTML Report (with Images)",
            key=f"btn_prep_html_{selected_period}_{page_key}_{widget_suffix}",
            help=f"Render '{title_lbl}' with embedded high-DPI Base64 PNG charts & maps into a self-contained .html file.",
        ):
            with st.spinner(f"Rendering static HTML report & Base64 PNG figures for {title_lbl}..."):
                st.session_state[state_key] = generate_static_html_report(
                    ds,
                    page_key=page_key,
                    flight_id=int(active_flight_id),
                    hour_utc=int(st.session_state.get(f"grid_map_hour_{selected_period}", 10)),
                    show_navaids=bool(st.session_state.get(f"tog_route_navaids_{selected_period}", True)),
                    nav_type_filter=st.session_state.get(f"flt_route_navaids_{selected_period}"),
                    world_nav_types=st.session_state.get(f"wn_types_{selected_period}"),
                    world_region_name=str(st.session_state.get(f"wn_region_{selected_period}", "🌍 Entire World")),
                    world_search_query=str(st.session_state.get(f"wn_search_{selected_period}", "")) or None,
                    world_limit=int(st.session_state.get(f"wn_limit_{selected_period}", 2500)),
                    twin_adep=str(st.session_state.get(f"t_adep_{selected_period}", "RKSI")),
                    twin_ades=str(st.session_state.get(f"t_ades_{selected_period}", "EGLL")),
                    twin_ac=str(st.session_state.get(f"t_ac_{selected_period}", "B77W")),
                    rec_adep=str(st.session_state.get(f"r_adep_{selected_period}", "EGLL")),
                    rec_ades=str(st.session_state.get(f"r_ades_{selected_period}", "KORD")),
                    rec_op=str(st.session_state.get(f"r_op_{selected_period}", "BAW")),
                    conflict_hour=int(st.session_state.get(f"cs_hr_{selected_period}", 10)),
                    conflict_nm=float(st.session_state.get(f"cs_nm_{selected_period}", 5.0)),
                    gdp_airport=str(st.session_state.get(f"gdp_apt_{selected_period}", "EGLL")),
                    gdp_aar=int(st.session_state.get(f"gdp_aar_{selected_period}", 25)),
                    gdp_date=str(st.session_state.get(f"gdp_date_{selected_period}", "")) or None,
                    fra_boundary_pair=str(st.session_state.get("fra_bnd_diag_pair", "")) or None,
                    fra_min_flights=int(st.session_state.get("fra_bnd_min_flts", 500)),
                    fra_annual_growth=float(st.session_state.get("fra_fc_growth_slider", 2.0)) / 100.0,
                    fra_realization=float(st.session_state.get("fra_fc_realization_slider", 85)) / 100.0,
                    fra_fuel_price=float(st.session_state.get("fra_fc_fuel_price_input", 0.6421)),
                    fra_fuel_opt=float(st.session_state.get("fra_fc_fuel_opt_slider", 1.5)) / 100.0,
                    fra_fuel_med=float(st.session_state.get("fra_fc_fuel_med_slider", 3.5)) / 100.0,
                    fra_fuel_shock=float(st.session_state.get("fra_fc_fuel_shock_slider", 6.5)) / 100.0,
                )
    with c_exp2:
        if state_key in st.session_state:
            fname = f"flightml_{page_key}_{selected_period}_{active_flight_id}.html"
            st.download_button(
                label=f"⬇️ Download {fname} (Self-Contained HTML + PNGs)",
                data=st.session_state[state_key],
                file_name=fname,
                mime="text/html",
                key=f"dl_html_{selected_period}_{page_key}_{widget_suffix}",
            )


with st.sidebar.expander("📄 Export Static HTML Report (with Images)", expanded=True):
    st.caption("Export any page (or the full executive bundle) to a standalone `.html` file with embedded high-DPI Base64 PNG charts & maps.")
    sb_page_choice = st.selectbox(
        "Select Page to Export",
        options=list(REPORT_PAGE_TITLES.keys()),
        format_func=lambda k: REPORT_PAGE_TITLES[k],
        index=1,
        key=f"sb_export_page_{selected_period}",
    )
    sb_state_key = f"sb_html_blob_{selected_period}_{sb_page_choice}"
    if st.button("🔨 Generate Static HTML Report", key=f"btn_sb_gen_html_{selected_period}"):
        with st.spinner(f"Building {REPORT_PAGE_TITLES[sb_page_choice]}..."):
            st.session_state[sb_state_key] = generate_static_html_report(
                ds,
                page_key=sb_page_choice,
                flight_id=int(active_flight_id),
                hour_utc=int(st.session_state.get(f"grid_map_hour_{selected_period}", 10)),
                show_navaids=bool(st.session_state.get(f"tog_route_navaids_{selected_period}", True)),
                nav_type_filter=st.session_state.get(f"flt_route_navaids_{selected_period}"),
                world_nav_types=st.session_state.get(f"wn_types_{selected_period}"),
                world_region_name=str(st.session_state.get(f"wn_region_{selected_period}", "🌍 Entire World")),
                world_search_query=str(st.session_state.get(f"wn_search_{selected_period}", "")) or None,
                world_limit=int(st.session_state.get(f"wn_limit_{selected_period}", 2500)),
                twin_adep=str(st.session_state.get(f"t_adep_{selected_period}", "RKSI")),
                twin_ades=str(st.session_state.get(f"t_ades_{selected_period}", "EGLL")),
                twin_ac=str(st.session_state.get(f"t_ac_{selected_period}", "B77W")),
                rec_adep=str(st.session_state.get(f"r_adep_{selected_period}", "EGLL")),
                rec_ades=str(st.session_state.get(f"r_ades_{selected_period}", "KORD")),
                rec_op=str(st.session_state.get(f"r_op_{selected_period}", "BAW")),
                conflict_hour=int(st.session_state.get(f"cs_hr_{selected_period}", 10)),
                conflict_nm=float(st.session_state.get(f"cs_nm_{selected_period}", 5.0)),
                gdp_airport=str(st.session_state.get(f"gdp_apt_{selected_period}", "EGLL")),
                gdp_aar=int(st.session_state.get(f"gdp_aar_{selected_period}", 25)),
                gdp_date=str(st.session_state.get(f"gdp_date_{selected_period}", "")) or None,
                fra_boundary_pair=str(st.session_state.get("fra_bnd_diag_pair", "")) or None,
                fra_min_flights=int(st.session_state.get("fra_bnd_min_flts", 500)),
                fra_annual_growth=float(st.session_state.get("fra_fc_growth_slider", 2.0)) / 100.0,
                fra_realization=float(st.session_state.get("fra_fc_realization_slider", 85)) / 100.0,
                fra_fuel_price=float(st.session_state.get("fra_fc_fuel_price_input", 0.6421)),
                fra_fuel_opt=float(st.session_state.get("fra_fc_fuel_opt_slider", 1.5)) / 100.0,
                fra_fuel_med=float(st.session_state.get("fra_fc_fuel_med_slider", 3.5)) / 100.0,
                fra_fuel_shock=float(st.session_state.get("fra_fc_fuel_shock_slider", 6.5)) / 100.0,
            )
            st.success("Report ready to download!")
    if sb_state_key in st.session_state:
        sb_fname = f"flightml_{sb_page_choice}_{selected_period}_{active_flight_id}.html"
        st.download_button(
            label=f"⬇️ Download {sb_fname}",
            data=st.session_state[sb_state_key],
            file_name=sb_fname,
            mime="text/html",
            key=f"btn_sb_dl_html_{selected_period}_{sb_page_choice}",
        )

st.sidebar.markdown(f"**Period Active:** `{selected_period}`")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "### ✨ Platform Highlights\n"
    "- **5°×5° Airspace Maps:** Hourly density, complexity & 4D proximity alerts\n"
    "- **ICAO GANP & PRU KPIs:** `ATOT`, `KEP`, `KEA`, `ASMA`, `CCO`, `CDO` & level-capping\n"
    "- **FRA & Airway Snapping:** 86,280 fixes, 17,090 navaids & VHF radio load\n"
    "- **Fuel, LTO & Contrails:** OpenAP burn, ICAO EDB v32B emissions & ERA5 ISSR\n"
    "- **Eurocontrol CRCO (€):** Route charges, weekly Jet-A spot prices & cost trade-offs\n"
    "- **Operations Research:** Hub tail-swap recovery, RBS GDP slots & fleet stress tests"
)

st.title("✈️ FlightML: Global Airspace Network Intelligence & ML")
st.markdown(
    f"Unified operational analytics, 4D trajectory physics, environmental accounting (`€ EUR`), and prescriptive operations research "
    f"fusing multi-gigabyte radar trajectories, global aeronautical navigation databases, and ASOS/ERA5 meteorology. "
    f"Active dataset: **Eurocontrol Cycle {selected_period}**."
)


(
    tab_cat_single,
    tab_cat_system,
    tab_cat_longitudinal,
    tab_cat_whatif,
    tab_cat_maps,
    tab_cat_econ,
    tab_cat_ai_lab,
    tab_cat_passenger,
) = st.tabs(
    [
        "✈️ Single Aircraft / Flight Views",
        "🌐 System-Level Summary & Analysis Views",
        "📈 Longitudinal & Predictive Trends",
        "🔮 What-If Scenario Views",
        "🗺️ Information & Map Views",
        "💶 General Economic Views (€)",
        "🧪 Experimental AI & Machine Learning Lab",
        "🧳 Passenger Travel Copilot & Itinerary Risk",
    ]
)

# =============================================================================
# 1. SINGLE AIRCRAFT / FLIGHT VIEWS
# =============================================================================
with tab_cat_single:
    st.caption(
        "Focused telemetry, 4D trajectory tracking, navaid reception, vertical climb/descent profile (CCO/CDO), "
        "delay root-cause diagnostics, and environmental/LTO emissions accounting for the selected active flight."
    )
    render_page_export_bar("single_flight", "cat_single")
    sf_sub1, sf_sub2, sf_sub3, sf_sub4 = st.tabs(
        [
            "🗺️ 4D Trajectory & RNP Profile",
            "📉 Vertical Profile: CDO & CCO",
            "🩺 Delay & Runway Diagnostics",
            "🌱 Fuel, Carbon & Surface LTO",
        ]
    )

    with sf_sub1:
        st.header("4D Trajectory, Active Route Navaids, RNP/RNAV Capability & Holding Stack Visualizer")
        render_page_export_bar("trajectory_rnp", "tab2")

        flight = ds.get_flight(int(active_flight_id))
        if flight:
            st.info(f"📍 **Active Flight:** `{ds.format_flight_label(flight)}`")
            trajs = ds.get_trajectory(int(active_flight_id))
            metrics = analyze_trajectory(trajs["actual"], trajs["planned"])
            nav_rnp_rep = _cached_flight_navaids_rnp(selected_period, int(active_flight_id))

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Route", f"{flight['adep']} ➔ {flight['ades']}")
            m2.metric("Operator", f"{flight['ac_operator']} ({flight['ac_type']})")
            m3.metric("Actual Distance", f"{metrics.actual_distance_nm:.0f} nm")
            m4.metric("Inefficiency Ratio", f"{metrics.horizontal_extension_ratio:.3f}x")
            m5.metric("Holding Duration", f"{metrics.total_holding_duration_min:.1f} min")

            col_nav_tog1, col_nav_tog2 = st.columns([1, 2])
            with col_nav_tog1:
                show_route_navaids = st.toggle(
                    "📡 Show Active Route Navaids on Map",
                    value=True,
                    key=f"tog_route_navaids_{selected_period}",
                    help="Toggle the colored navaid station icons and slant-range reception sightlines on or off.",
                )
            with col_nav_tog2:
                route_nav_type_filter = st.multiselect(
                    "Filter Route Navaid Types Shown on Map",
                    options=["VOR-DME", "DME", "VOR", "ILS_LOC", "NDB"],
                    default=["VOR-DME", "DME", "VOR", "ILS_LOC", "NDB"],
                    disabled=not show_route_navaids,
                    key=f"flt_route_navaids_{selected_period}",
                )

            filtered_route_navaids = (
                [n for n in nav_rnp_rep.active_navaids if n.nav_type in route_nav_type_filter]
                if (nav_rnp_rep and nav_rnp_rep.active_navaids and show_route_navaids)
                else []
            )

            st.caption(
                "🗺️ **Interactive 4D Route & Active Navaids Map:** Colored dots show ground radio navigation aids auto-tuned along the entire corridor "
                "(🔵 **VOR-DME / VORTAC** | 🔷 **DME / TACAN** | 🟢 **VOR** | 🟣 **ILS/LOC** | 🟠 **NDB**) "
                "with dashed sightlines to the aircraft's closest 4D trajectory pass. Use the toggle above to switch navaid icons on or off."
            )
            m = create_trajectory_map(
                flight,
                trajs["actual"],
                trajs["planned"],
                active_navaids=filtered_route_navaids,
                show_navaids=show_route_navaids,
            )
            st_folium(
                m,
                width=1200,
                height=540,
                returned_objects=[],
                key=f"traj_map_{selected_period}_{active_flight_id}_{int(show_route_navaids)}_{len(filtered_route_navaids)}",
            )

            if nav_rnp_rep and nav_rnp_rep.profile_points:
                st.markdown("#### 📡 Active Route Navaids & Anticipated RNP/RNAV Capability vs. Vertical Profile")
                st.caption(f"**Navigation & PBN Summary:** {nav_rnp_rep.overall_navigation_summary}")

                nr1, nr2, nr3, nr4, nr5 = st.columns(5)
                nr1.metric(
                    "Likely Used Route Navaids",
                    f"{nav_rnp_rep.total_active_navaids} stations",
                    f"{nav_rnp_rep.dme_stations_used} DME / {nav_rnp_rep.vor_stations_used} VOR / {nav_rnp_rep.ils_stations_used} ILS",
                )
                nr2.metric("Best Terminal RNP", f"RNP {nav_rnp_rep.best_rnp_nm:.1f} nm", "Approach / TMA")
                nr3.metric(
                    "En-Route RNP / RNAV",
                    f"RNAV {nav_rnp_rep.mean_enroute_rnp_nm:.1f} nm",
                    f"Max/Worst: {nav_rnp_rep.worst_rnp_nm:.1f} nm",
                )
                nr4.metric(
                    "Multi-DME (DME/DME) Coverage",
                    f"{nav_rnp_rep.dme_dme_coverage_pct:.1f}%",
                    "≥2 DMEs in Radio Horizon",
                )
                nr5.metric(
                    "Beyond-RLOS Oceanic Share",
                    f"{nav_rnp_rep.gnss_only_oceanic_pct:.1f}%",
                    "GNSS / IRS Only" if nav_rnp_rep.gnss_only_oceanic_pct > 0 else "100% Terrestrial RLOS",
                )

                prof_df = pd.DataFrame([p.to_dict() for p in nav_rnp_rep.profile_points])
                prof_df["total_vhf_navaids_in_rlos"] = prof_df["dme_in_rlos_count"] + prof_df["vor_in_rlos_count"]

                # Dual-Axis Altair Graph: Vertical Flight Profile (FL) + Anticipated RNP/RNAV Available (nm)
                base_x = alt.X("cumulative_dist_nm:Q", title="Cumulative Flown Distance Along Route (nm)")

                fl_area = (
                    alt.Chart(prof_df)
                    .mark_area(opacity=0.22, color="#8e44ad", line={"color": "#8e44ad", "strokeWidth": 2.5})
                    .encode(
                        x=base_x,
                        y=alt.Y("flight_level:Q", title="Vertical Profile — Flight Level (FL, 100s ft)", scale=alt.Scale(zero=True)),
                        tooltip=[
                            alt.Tooltip("seq_num:Q", title="Waypoint #"),
                            alt.Tooltip("time_utc:N", title="Time (UTC)"),
                            alt.Tooltip("cumulative_dist_nm:Q", title="Distance Flown (nm)", format=".1f"),
                            alt.Tooltip("flight_level:Q", title="Flight Level (FL)", format=".0f"),
                            alt.Tooltip("flight_phase:N", title="Flight Phase"),
                            alt.Tooltip("anticipated_rnp_nm:Q", title="Anticipated RNP/RNAV (nm)", format=".1f"),
                            alt.Tooltip("pbn_specification:N", title="PBN Specification"),
                            alt.Tooltip("dme_in_rlos_count:Q", title="DMEs in Radio Horizon"),
                            alt.Tooltip("vor_in_rlos_count:Q", title="VORs in Radio Horizon"),
                            alt.Tooltip("primary_navaids_tuned:N", title="Primary Tuned Navaids"),
                            alt.Tooltip("availability_reason:N", title="Why This RNP/RNAV Is Available"),
                        ],
                    )
                )

                rnp_line = (
                    alt.Chart(prof_df)
                    .mark_line(interpolate="step-after", color="#d35400", strokeWidth=2.5, strokeDash=[5, 3])
                    .encode(
                        x=base_x,
                        y=alt.Y(
                            "anticipated_rnp_nm:Q",
                            title="Anticipated RNP / RNAV Accuracy (nm — lower is tighter)",
                            scale=alt.Scale(zero=True),
                        ),
                    )
                )

                rnp_pts = (
                    alt.Chart(prof_df)
                    .mark_circle(size=68, opacity=0.95)
                    .encode(
                        x=base_x,
                        y=alt.Y("anticipated_rnp_nm:Q", title="Anticipated RNP / RNAV Accuracy (nm — lower is tighter)"),
                        color=alt.Color(
                            "pbn_specification:N",
                            title="Anticipated PBN / RNP Specification",
                            legend=alt.Legend(orient="bottom", columns=2),
                        ),
                        tooltip=[
                            alt.Tooltip("seq_num:Q", title="Waypoint #"),
                            alt.Tooltip("time_utc:N", title="Time (UTC)"),
                            alt.Tooltip("cumulative_dist_nm:Q", title="Distance Flown (nm)", format=".1f"),
                            alt.Tooltip("flight_level:Q", title="Flight Level (FL)", format=".0f"),
                            alt.Tooltip("anticipated_rnp_nm:Q", title="Anticipated RNP/RNAV (nm)", format=".1f"),
                            alt.Tooltip("pbn_specification:N", title="PBN Specification"),
                            alt.Tooltip("dme_in_rlos_count:Q", title="DMEs in Radio Horizon"),
                            alt.Tooltip("vor_in_rlos_count:Q", title="VORs in Radio Horizon"),
                            alt.Tooltip("primary_navaids_tuned:N", title="Primary Tuned Navaids"),
                            alt.Tooltip("availability_reason:N", title="Why This RNP/RNAV Is Available"),
                        ],
                    )
                )

                combined_profile_chart = (
                    alt.layer(fl_area, rnp_line + rnp_pts)
                    .resolve_scale(y="independent")
                    .properties(
                        height=320,
                        title="Vertical Flight Profile (FL, Purple) vs. Anticipated RNP/RNAV Capability (nm, Colored Step Curve) — Hover Any Point for 'Why'",
                    )
                )
                st.altair_chart(combined_profile_chart, use_container_width=True)

                # Secondary strip: Count of Active DME & VOR stations within Radio Line-of-Sight along route
                rlos_chart = (
                    alt.Chart(prof_df)
                    .mark_bar(color="#00a8ff", opacity=0.75)
                    .encode(
                        x=base_x,
                        y=alt.Y("dme_in_rlos_count:Q", title="DMEs in RLOS"),
                        tooltip=[
                            alt.Tooltip("cumulative_dist_nm:Q", title="Distance (nm)", format=".1f"),
                            alt.Tooltip("flight_level:Q", title="Flight Level (FL)", format=".0f"),
                            alt.Tooltip("dme_in_rlos_count:Q", title="DME Stations in RLOS"),
                            alt.Tooltip("vor_in_rlos_count:Q", title="VOR Stations in RLOS"),
                            alt.Tooltip("primary_navaids_tuned:N", title="Primary Tuned Navaids"),
                            alt.Tooltip("availability_reason:N", title="Why This RNP/RNAV Is Available"),
                        ],
                    )
                    .properties(
                        height=135,
                        title="Terrestrial Multi-DME Station Redundancy in Radio Line-of-Sight (RLOS) Along Trajectory",
                    )
                )
                st.altair_chart(rlos_chart, use_container_width=True)

                col_nav_tbl, col_rnp_tbl = st.columns(2)
                with col_nav_tbl:
                    with st.expander(f"📡 Likely Used Route Navaids ({len(nav_rnp_rep.active_navaids)} Colored Dots on Map)", expanded=True):
                        if nav_rnp_rep.active_navaids:
                            nav_rows = [
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
                                for n in nav_rnp_rep.active_navaids
                            ]
                            st.dataframe(pd.DataFrame(nav_rows), use_container_width=True)
                        else:
                            st.info("No terrestrial VHF/UHF navaids within corridor range.")

                with col_rnp_tbl:
                    with st.expander("🧭 Anticipated RNP/RNAV Availability & 'Why' Along Route", expanded=True):
                        rnp_rows = [
                            {
                                "Dist (nm)": f"{p.cumulative_dist_nm:.0f} nm",
                                "FL": f"FL{p.flight_level:.0f}",
                                "RNP (nm)": f"{p.anticipated_rnp_nm:.1f} nm",
                                "PBN Specification": p.pbn_specification,
                                "DME / VOR in RLOS": f"{p.dme_in_rlos_count} DME / {p.vor_in_rlos_count} VOR",
                                "Primary Tuned": p.primary_navaids_tuned,
                                "Why Available": p.availability_reason,
                            }
                            for p in nav_rnp_rep.profile_points
                        ]
                        st.dataframe(pd.DataFrame(rnp_rows), use_container_width=True)

            pru_sc = compute_flight_pru_kpis(ds, int(active_flight_id))
            if pru_sc:
                st.markdown("#### 📏 ICAO GANP & Eurocontrol PRU Horizontal & Terminal KPIs")
                pk1, pk2, pk3, pk4, pk5 = st.columns(5)
                pk1.metric("KPI04 (KEP Filed)", f"{pru_sc.kpi04_kep_pct:.2f}%", f"{pru_sc.enroute_filed_dist_nm:.0f} nm (>40nm)")
                pk2.metric("KPI05 (KEA Actual)", f"{pru_sc.kpi05_kea_pct:.2f}%", pru_sc.enroute_routing_verdict)
                pk3.metric("Tactical KEA - KEP", f"{pru_sc.tactical_kea_minus_kep_pct:+.2f}%", f"{pru_sc.tactical_distance_delta_nm:+.1f} nm")
                pk4.metric("KPI08 ASMA (40nm)", f"{pru_sc.asma_40nm_actual_min:.1f} min", f"+{pru_sc.kpi08_asma_40nm_additional_min:.1f}m add")
                pk5.metric(
                    "KPI19 Cruise FL",
                    f"FL{pru_sc.max_actual_fl:.0f} (Req FL{pru_sc.requested_fl:.0f})",
                    "CAPPED" if pru_sc.is_level_capped else "OPTIMAL",
                )

            fra_rep = analyze_flight_fra_adherence(ds, int(active_flight_id))
            if fra_rep:
                st.markdown("#### 🛣️ Free Route Airspace (FRA) vs. Structured ATS Airway Snapping")
                st.caption(
                    "ℹ️ **Single European Sky CP1 Mandate (EU 2021/116):** Free Route Airspace (FRA) is formally operational and required "
                    "at or above **FL305** (30,500 ft). Flights below FL305 navigate structured ATS airways and terminal transitions."
                )
                fr1, fr2, fr3, fr4, fr5 = st.columns(5)
                fr1.metric("En-Route Directness", f"{fra_rep.fra_directness_index_pct:.1f}%", f"{fra_rep.fra_direct_distance_nm:.0f} nm DCT")
                if fra_rep.fl305_enroute_distance_nm > 0:
                    fr2.metric("Cruise Directness (≥FL305)", f"{fra_rep.fl305_fra_directness_index_pct:.1f}%", f"{fra_rep.fl305_fra_direct_distance_nm:.0f} nm DCT")
                else:
                    fr2.metric("Cruise Directness (≥FL305)", "N/A", "Below FL305")
                fr3.metric("ATS Airway Adherence", f"{fra_rep.ats_airway_adherence_pct:.1f}%", f"{fra_rep.ats_airway_distance_nm:.0f} nm")
                fr4.metric("Snapped ATS Airways", f"{len(fra_rep.distinct_airways_used)}", ", ".join(fra_rep.distinct_airways_used[:4]) or "DCT Only")
                fr5.metric("Snapped 5-Letter Fixes", f"{len(fra_rep.sample_snapped_fixes)}", ", ".join(fra_rep.sample_snapped_fixes[:4]) or "—")

            if ds.has_fir_traversals():
                st.markdown("#### 🧭 En-Route Sector Traversal & Dwell Efficiency")
                dwell_res = calculate_flight_fir_dwell_inefficiency(ds, int(active_flight_id))
                if dwell_res.get("has_data") and dwell_res.get("sectors"):
                    st.caption(f"**Verdict:** {dwell_res['verdict']}")
                    c_dw1, c_dw2, c_dw3 = st.columns(3)
                    c_dw1.metric("Total Actual Dwell", f"{dwell_res['total_actual_dwell_min']} min")
                    c_dw2.metric("Total Filed Dwell", f"{dwell_res['total_filed_dwell_min']} min")
                    c_dw3.metric("Net Sector Dwell Delta", f"{dwell_res['net_dwell_diff_min']:+.1f} min")
                    st.dataframe(pd.DataFrame(dwell_res["sectors"]), use_container_width=True)

            if ds.has_aua_traversals():
                st.markdown("#### 📡 ATC Unit Airspace (AUA: TMA / CTA / UTA) Traversal & Dwell Breakdown")
                aua_dwell = calculate_flight_aua_dwell_inefficiency(ds, int(active_flight_id))
                if aua_dwell.get("has_data") and aua_dwell.get("sectors"):
                    ca1, ca2, ca3, ca4 = st.columns(4)
                    ca1.metric("Actual AUA Dwell", f"{aua_dwell['total_actual_aua_min']:.1f} min")
                    ca2.metric("Filed AUA Dwell", f"{aua_dwell['total_filed_aua_min']:.1f} min")
                    ca3.metric("TMA (Terminal) Delta", f"{aua_dwell['tma_dwell_delta_min']:+.1f} min")
                    ca4.metric("CTA/UTA (En-Route) Delta", f"{aua_dwell['enroute_cta_uta_dwell_delta_min']:+.1f} min")
                    st.dataframe(pd.DataFrame(aua_dwell["sectors"]), use_container_width=True)
        else:
            st.warning(f"Flight ID {active_flight_id} not found in period {selected_period}.")

        # ----------------- TAB WORLD NAV: Global Navaids World Map Explorer -----------------

    with sf_sub2:
        st.header("Vertical Flight Efficiency: Continuous Descent (CDO), Climb (CCO) & Level-Capping")
        render_page_export_bar("vertical_cdo_cco", "sub_cdo")
        st.markdown(
            "Identifies inefficient staircase descents, intermediate climb/descent level-offs, altitude reversals, and en-route cruise level caps. "
            "Uninterrupted continuous climb and 3° idle-thrust descent profiles minimize fuel burn, noise, and emissions."
        )
        flight = ds.get_flight(int(active_flight_id))
        if not flight:
            st.warning(f"Flight ID {active_flight_id} not found in period {selected_period}.")
        else:
            st.info(f"📍 **Active Flight:** `{ds.format_flight_label(flight)}`")
            with st.spinner("Analyzing vertical descent profile and level-off segments..."):
                cdo_rep = analyze_cdo_descent(ds, int(active_flight_id))

            if not cdo_rep:
                st.warning(f"No completed descent profile detected for flight {active_flight_id}.")
            else:
                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                col_c1.metric("CDO Compliance Score", f"{cdo_rep.cdo_compliance_score:.1f}%")
                col_c2.metric("Descent Duration", f"{cdo_rep.total_descent_duration_min:.1f} min")
                col_c3.metric("Level-Off Inefficiencies", f"{cdo_rep.inefficiency_events_count} events")
                col_c4.metric("Excess Descent Fuel", f"{cdo_rep.excess_fuel_burned_kg:.1f} kg")

                st.markdown(
                    f"**Route:** `{cdo_rep.adep}` ➔ `{cdo_rep.ades}` | **Aircraft:** `{cdo_rep.ac_type}` | "
                    f"**Cruise Flight Level:** `FL{cdo_rep.cruise_flight_level:.0f}` | "
                    f"**Un-optimized Level Flight:** `{cdo_rep.total_level_or_climb_min:.1f} min` | "
                    f"**Excess CO₂:** `{cdo_rep.excess_co2_kg:.1f} kg`"
                )

                # Trajectory vertical profile chart
                trajs = ds.get_trajectory(int(active_flight_id))
                if not trajs["actual"].empty:
                    chart_df = trajs["actual"][["time_over", "flight_level"]].copy()
                    chart_df["time_over"] = pd.to_datetime(chart_df["time_over"])
                    chart_df = chart_df.set_index("time_over")
                    st.subheader("Vertical Flight Profile (Altitude / FL vs Time)")
                    st.line_chart(chart_df)

                if cdo_rep.inefficiency_events:
                    st.subheader("Detected Descent Inefficiency Events (Level-Offs & Climbing Reversals)")
                    ev_data = [
                        {
                            "Event": ev.event_type,
                            "Start (UTC)": ev.start_time,
                            "End (UTC)": ev.end_time,
                            "Duration (min)": f"{ev.duration_min:.1f}",
                            "Flight Level": f"FL{ev.flight_level:.0f}",
                            "Latitude": f"{ev.latitude:.2f}°",
                            "Longitude": f"{ev.longitude:.2f}°",
                        }
                        for ev in cdo_rep.inefficiency_events
                    ]
                    st.dataframe(pd.DataFrame(ev_data), use_container_width=True)

                st.divider()
                st.subheader("🛫 Continuous Climb Operations (KPI17 CCO) & KPI19 Cruise Level-Capping")
                cco_rep = analyze_cco_climb(ds, int(active_flight_id))
                pru_vert = compute_flight_pru_kpis(ds, int(active_flight_id))
                if cco_rep:
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("CCO Climb Compliance", f"{cco_rep.cco_compliance_score:.1f}%")
                    cc2.metric("Climb Duration to TOC", f"{cco_rep.total_climb_duration_min:.1f} min")
                    cc3.metric("Climb Level-Offs", f"{cco_rep.inefficiency_events_count} ({cco_rep.total_level_off_min:.1f}m)")
                    cc4.metric("Excess Climb Fuel", f"{cco_rep.excess_fuel_burned_kg:.1f} kg", f"{cco_rep.excess_co2_kg:.1f} kg CO₂")
                if pru_vert:
                    opt_val = getattr(pru_vert, "optimal_fl", 360.0)
                    st.caption(
                        f"📏 **KPI19 Cruise Altitude & Level-Capping:** Airframe/Engine Ideal `FL{opt_val:.0f}` | "
                        f"Requested `FL{pru_vert.requested_fl:.0f}` vs. Actual Max `FL{pru_vert.max_actual_fl:.0f}` "
                        f"(Deficit: `-{pru_vert.level_cap_delta_fl:.0f} FL`, Level-Capped: `{pru_vert.is_level_capped}`, "
                        f"Estimated Altitude Penalty: `+{pru_vert.level_cap_excess_fuel_kg:.1f} kg Jet-A`)"
                    )

        with st.expander("🌐 Network-Wide KPI19 Cruise Level-Capping Corridors (Top Capped City Pairs)", expanded=False):
            cap_res = analyze_network_level_capping(ds, top_n=12)
            if cap_res.get("top_capped_corridors"):
                st.caption(
                    f"Evaluated **{cap_res.get('analyzed_flights', 0):,}** flights "
                    f"(**{cap_res.get('level_capped_flights', 0):,}** capped / **{cap_res.get('level_capped_pct', 0.0):.1f}%**, "
                    f"mean deficit **-{cap_res.get('mean_fl_deficit', 0.0):.1f} FL**). "
                    "Identifies city pairs where ATC tactical or RAD restrictions systematically cap aircraft ≥2,000 ft below their filed optimal cruise flight level."
                )
                st.dataframe(pd.DataFrame(cap_res["top_capped_corridors"]), use_container_width=True)

        # ----------------- TAB 5: Delay & Runway Diagnostics -----------------

    with sf_sub3:
        st.header("Multi-Factor Delay Root-Cause Attribution & Runway Braking Safety")
        render_page_export_bar("delay_diagnostics", "sub_delay")
        flight = ds.get_flight(int(active_flight_id))
        if not flight:
            st.warning(f"Flight ID {active_flight_id} not found in period {selected_period}.")
        else:
            st.info(f"📍 **Active Flight:** `{ds.format_flight_label(flight)}`")
            if st.button("Run Diagnostic Lab for Active Flight", key=f"btn_diag_{selected_period}"):
                with st.spinner("Analyzing weather, tail turns, and holding stacks..."):
                    diag = diagnose_flight_bottlenecks(ds, metar_client, int(active_flight_id))

                st.subheader(f"Diagnostic Report: ECTRL {diag.ectrl_id}")
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("Arrival Delay", f"{diag.total_arr_delay_min} min")
                dc2.metric("Departure Delay", f"{diag.total_dep_delay_min} min")
                dc3.metric("Primary Factor", diag.primary_bottleneck_factor)

                st.info(f"💡 **Mitigation Recommendation:** {diag.mitigation_recommendation}")
                st.json(diag.to_dict())

            st.divider()
            st.subheader("🛬 Arrival Runway Braking, Density Altitude & Overrun Safety Margin")
            rwy_saf = assess_flight_runway_safety(ds, metar_client=None, ectrl_id=int(active_flight_id), fetch_live_metar=False)
            if rwy_saf:
                rs1, rs2, rs3, rs4 = st.columns(4)
                rs1.metric("Recommended Runway", f"RWY {rwy_saf.recommended_runway}", rwy_saf.ils_precision_category)
                rs2.metric("Density Altitude", f"{rwy_saf.density_altitude_ft:,.0f} ft", f"ISA {rwy_saf.isa_temp_deviation_c:+.1f}°C")
                rs3.metric("Stopping Safety Margin", f"+{rwy_saf.landing_distance_safety_margin_m:,.0f} m", f"{rwy_saf.lda_utilization_pct:.1f}% of LDA")
                rs4.metric("Runway Safety Risk", rwy_saf.safety_risk_level, rwy_saf.braking_action_code)

        # ----------------- TAB 6: Fuel, Carbon & Route Charges -----------------

    with sf_sub4:
        st.header("Environmental, Surface LTO, Contrail & Eurocontrol CRCO Cost Quantification (€)")
        render_page_export_bar("fuel_carbon", "sub_fuel")
        flight = ds.get_flight(int(active_flight_id))
        if not flight:
            st.warning(f"Flight ID {active_flight_id} not found in period {selected_period}.")
        else:
            st.info(f"📍 **Active Flight:** `{ds.format_flight_label(flight)}`")
            ac_type = str(flight.get("ac_type") or "A320").upper()
            base_eng_perf = resolve_aircraft_engine_perf(ds, ac_type)

            col_e1, col_e2 = st.columns([2, 1])
            with col_e1:
                selected_engine_opt = st.selectbox(
                    "🔧 Aircraft Engine Variant (OpenAP / ICAO EDB Performance Model):",
                    options=base_eng_perf.engine_options,
                    index=0,
                    key=f"fuel_engine_select_{active_flight_id}_{selected_period}",
                    help="Select installed engine variant to evaluate aerodynamic cruise specific fuel consumption and ideal cruise flight level."
                )
            with col_e2:
                cur_eng_perf = resolve_aircraft_engine_perf(ds, ac_type, engine_override=selected_engine_opt)
                st.metric(
                    "Engine BPR & Pressure Ratio",
                    f"BPR {cur_eng_perf.bypass_ratio:.1f} : 1",
                    f"PR {cur_eng_perf.pressure_ratio:.1f} | {cur_eng_perf.engine_type_class}",
                )

            try:
                impact = compute_flight_fuel_and_emissions(
                    ds,
                    int(active_flight_id),
                    engine_override=selected_engine_opt,
                )

                st.markdown("#### ✈️ 3-Way Cruise Altitude Optimization: Ideal vs. Requested vs. Actual Flown")
                fl1, fl2, fl3, fl4 = st.columns(4)
                fl1.metric(
                    "Ideal Cruise Altitude",
                    f"FL{impact.optimal_cruise_fl:.0f}",
                    f"Airframe & Engine Optimum",
                )
                fl2.metric(
                    "Filed / Requested Altitude",
                    f"FL{impact.requested_fl:.0f}",
                    f"Δ to Ideal: {impact.fl_delta_requested_vs_optimal:+.0f} FL",
                    delta_color="normal" if impact.fl_delta_requested_vs_optimal >= 0 else "inverse",
                )
                fl3.metric(
                    "Actual Radar Flown Altitude",
                    f"FL{impact.actual_cruise_fl:.0f}",
                    f"Δ to Ideal: {impact.fl_delta_actual_vs_optimal:+.0f} FL",
                    delta_color="normal" if impact.fl_delta_actual_vs_optimal >= 0 else "inverse",
                )
                fl4.metric(
                    "Tactical ATC Capping",
                    f"{impact.fl_delta_actual_vs_requested:+.0f} FL",
                    f"Actual vs. Requested",
                    delta_color="normal" if impact.fl_delta_actual_vs_requested >= 0 else "inverse",
                )

                if impact.potential_fuel_savings_kg > 0:
                    st.success(
                        f"💡 **Potential Savings at Optimal FL{impact.optimal_cruise_fl:.0f}:** "
                        f"Cruising at the aircraft & engine combination's ideal altitude would save "
                        f"**{impact.potential_fuel_savings_kg:,.1f} kg Jet-A** (**€{impact.potential_fuel_savings_eur:,.2f} EUR**) "
                        f"or **{impact.potential_fuel_savings_pct:.1f}%** of en-route cruise fuel burn!"
                    )
                else:
                    st.success(
                        f"✅ **Optimal Vertical Cruise Efficiency:** Flight cruised at or near its ideal aerodynamic flight level (FL{impact.optimal_cruise_fl:.0f}). Zero altitude fuel penalty incurred."
                    )

                st.markdown("#### ⛽ Multi-Dimensional Fuel Burn & Emissions Breakdown")
                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("Excess Distance (Detour)", f"{impact.excess_distance_nm:.1f} nm", f"{impact.excess_routing_fuel_kg:,.0f} kg burn")
                fc2.metric("Terminal Holding Burn", f"{impact.excess_holding_fuel_kg:,.1f} kg", f"{impact.holding_duration_min:.1f} min loops")
                fc3.metric(
                    "Suboptimal Altitude Penalty",
                    f"{impact.excess_altitude_fuel_kg:,.1f} kg",
                    f"+{impact.actual_burn_penalty_pct:.1f}% cruise penalty",
                    delta_color="inverse" if impact.excess_altitude_fuel_kg > 0 else "normal",
                )
                fc4.metric(
                    "Total Excess Fuel & Cost",
                    f"{impact.total_excess_fuel_kg:,.1f} kg",
                    f"€{impact.estimated_excess_fuel_cost_eur:,.2f} EUR",
                    delta_color="inverse" if impact.total_excess_fuel_kg > 0 else "normal",
                )

                with st.expander("📐 Vertical Cruise Inefficiency Attribution (Planning vs. Tactical ATC)", expanded=False):
                    va1, va2 = st.columns(2)
                    with va1:
                        st.markdown(
                            f"**Flight Planning Disparity (Requested vs. Ideal FL):**\n"
                            f"- Filed Altitude: `FL{impact.requested_fl:.0f}` (Deficit: `{impact.fl_delta_requested_vs_optimal:+.0f} FL` vs FL{impact.optimal_cruise_fl:.0f})\n"
                            f"- Inherent Planning Fuel Penalty: **{impact.planning_excess_fuel_kg:,.1f} kg** (**€{impact.planning_excess_fuel_cost_eur:,.2f} EUR**)\n"
                            f"- Reflects airline dispatch decisions prior to departure."
                        )
                    with va2:
                        atc_sign = "+" if impact.atc_tactical_excess_fuel_kg > 0 else ""
                        st.markdown(
                            f"**Tactical ATC Deviation (Actual vs. Requested FL):**\n"
                            f"- Tactical Deviation: `{impact.fl_delta_actual_vs_requested:+.0f} FL`\n"
                            f"- ATC Induced Fuel Variance: **{atc_sign}{impact.atc_tactical_excess_fuel_kg:,.1f} kg** (**€{impact.atc_tactical_excess_fuel_cost_eur:,.2f} EUR**)\n"
                            f"- Positive values indicate tactical level capping; negative values indicate ATC climbed the aircraft higher towards optimal."
                        )

                price_gal = getattr(impact, "fuel_price_eur_per_gallon", 1.265 if selected_period == "201611" else 1.992)
                price_kg = getattr(impact, "fuel_price_eur_per_kg", price_gal / 3.043)
                price_week = getattr(impact, "fuel_price_week", "Weekly Spot Price")
                price_source = getattr(impact, "fuel_price_source", "EIA Weekly Spot Price (EUR Converted)")

                st.caption(
                    f"⛽ **Weekly Jet Fuel Pricing Applied:** `€{price_gal:.3f}/gal` "
                    f"(`€{price_kg:.4f}/kg`) | **Week:** `{price_week}` "
                    f"| **Benchmark:** `{price_source}` | **Nominal Cruise Flow:** `{impact.nominal_fuel_flow_kg_hr:,.0f} kg/h`"
                )
                # Contextual comparison for historical vs modern into-plane fuel price
                modern_into_plane_rate = 0.85
                diff_pct = ((modern_into_plane_rate - price_kg) / price_kg) * 100.0 if price_kg > 0 else 0.0
                st.info(
                    f"ℹ️ **Historical Spot vs. Modern Into-Plane Reality:** The prices above reflect the **exact historical wholesale spot price** "
                    f"for the selected dataset period (`€{price_kg:.4f}/kg`). Wholesale spot excludes airport handling, into-plane fueling, and distribution fees. "
                    f"By comparison, modern (2024–2026) airline into-plane Jet-A averages **~€0.85/kg** (+{diff_pct:.0f}% vs. period spot), "
                    f"which scales total trip fuel cost accordingly."
                )
            except Exception as e:
                st.error(f"Error computing emissions for flight {active_flight_id}: {e}")

            if ds.has_fir_traversals():
                taxi_rep = analyze_flight_taxi_out(ds, int(active_flight_id))
                if taxi_rep:
                    st.divider()
                    st.subheader("🛞 Departure Surface Movement (TAXI_OUT) & ICAO EDB v32B LTO Idle Emissions")
                    tx1, tx2, tx3, tx4 = st.columns(4)
                    tx1.metric("Actual Taxi-Out", f"{taxi_rep.actual_taxi_out_min:.1f} min", f"UTOT p20: {taxi_rep.unimpeded_utot_min:.1f}m")
                    tx2.metric("ICAO KPI02 ATOT", f"+{taxi_rep.additional_atot_min:.1f} min", taxi_rep.congestion_status)
                    tx3.metric("Excess Queue Fuel & CO₂", f"{taxi_rep.excess_atot_fuel_kg:,.1f} kg", f"{taxi_rep.excess_co2_kg:,.1f} kg CO₂")
                    tx4.metric(
                        "Excess LTO NOx / CO / nvPM",
                        f"{taxi_rep.excess_nox_g:,.0f}g NOx / {taxi_rep.excess_co_g:,.0f}g CO",
                        f"{taxi_rep.excess_nvpm_mg:,.0f} mg nvPM soot",
                    )
                    st.caption(
                        f"🔧 **Engine Profile:** `{taxi_rep.ac_type}` (Wake `{taxi_rep.wake_category}`, Default Engine `{taxi_rep.default_engine}`) "
                        f"| **7% Idle + APU Rate:** `{taxi_rep.idle_fuel_rate_kg_min:.2f} kg/min` "
                        f"| **Excess Queue Cost:** `€{taxi_rep.excess_fuel_cost_eur:,.2f} EUR`"
                    )

            crco_rep = compute_flight_crco_charges(ds, int(active_flight_id))
            if crco_rep:
                st.divider()
                st.subheader("💶 Eurocontrol CRCO En-Route Navigation Charges & Direct Operating Cost (€)")
                cr1, cr2, cr3, cr4 = st.columns(4)
                cr1.metric("Aircraft Weight Factor", f"{crco_rep.crco_weight_factor:.3f}", f"MTOW: {crco_rep.mtow_tons:.1f} t")
                cr2.metric("CRCO Service Units", f"{crco_rep.total_service_units:.2f} SU", f"{crco_rep.actual_distance_km:.0f} km")
                cr3.metric("Actual CRCO Fee", f"€{crco_rep.actual_crco_charge_eur:,.2f}", f"Filed: €{crco_rep.filed_crco_charge_eur:,.2f}")
                cr4.metric(
                    "Total Direct Route Cost",
                    f"€{crco_rep.total_direct_route_cost_eur:,.2f}",
                    f"{crco_rep.crco_share_of_direct_cost_pct:.1f}% CRCO share",
                )
                mod_fuel_cost = crco_rep.estimated_total_fuel_kg * 0.85
                st.caption(
                    f"✈️ **Direct Cost Composition:** Total Trip Fuel: **{crco_rep.estimated_total_fuel_kg:,.1f} kg** "
                    f"(**€{crco_rep.estimated_total_fuel_cost_eur:,.2f} EUR** at period spot price | **€{mod_fuel_cost:,.2f} EUR** at modern €0.85/kg into-plane) "
                    f"+ Eurocontrol CRCO Navigation Fees: **€{crco_rep.actual_crco_charge_eur:,.2f} EUR**."
                )
                if crco_rep.fir_breakdown:
                    st.subheader("🗺️ Traversed FIR Airspace & CRCO Charging Zones Map")
                    st.caption(
                        "Interactive map displaying the planned and flown flight path underlaid with the certified "
                        "boundary polygon of each Flight Information Region (FIR) traversed, color-coded with Eurocontrol CRCO national charging rates (€/SU)."
                    )
                    fir_charge_map = create_flight_fir_charge_map(ds, int(active_flight_id), crco_report=crco_rep)
                    st_folium(
                        fir_charge_map,
                        width=1200,
                        height=480,
                        returned_objects=[],
                        key=f"crco_fir_map_{selected_period}_{active_flight_id}",
                    )
                    st.dataframe(pd.DataFrame([b.__dict__ for b in crco_rep.fir_breakdown]), use_container_width=True)

            contrail_rep = analyze_flight_wind_and_contrails(ds, int(active_flight_id))
            if contrail_rep:
                st.divider()
                st.subheader("🌬️ Upper-Air Winds, Equivalent Still Air Distance (ESAD) & Contrail Climate")

                # Wind Routing & Dispatch Efficiency Verdict
                v_text = getattr(
                    contrail_rep,
                    "wind_efficiency_verdict",
                    getattr(contrail_rep, "wind_routing_verdict", "NEUTRAL_WIND_PROFILE"),
                )
                if "SMART_WIND_RIDING" in v_text or "FAVORABLE" in v_text:
                    st.success(f"💨 **Wind Routing Verdict:** {v_text}")
                elif "HEADWIND" in v_text:
                    st.warning(f"💨 **Wind Routing Verdict:** {v_text}")
                elif "STRUCTURAL" in v_text:
                    st.info(f"💨 **Wind Routing Verdict:** {v_text}")
                else:
                    st.caption(f"💨 **Wind Routing Verdict:** {v_text}")

                # Metric Row 1: Upper-Air Wind & Aerodynamic Air Distance (ESAD)
                ct1, ct2, ct3, ct4 = st.columns(4)
                ct1.metric("Avg GS vs TAS", f"{contrail_rep.avg_groundspeed_kts:.0f} kts GS", f"TAS: {contrail_rep.avg_true_airspeed_kts:.0f} kts")
                ct2.metric("Net Upper-Air Wind", f"{contrail_rep.avg_tailwind_component_kts:+.1f} kts", f"{contrail_rep.wind_time_savings_min:+.1f}m wind time")
                g_dist = getattr(contrail_rep, "ground_distance_nm", contrail_rep.cruise_distance_nm)
                esad = getattr(contrail_rep, "esad_nm", contrail_rep.cruise_distance_nm)
                air_benefit = getattr(contrail_rep, "wind_distance_benefit_nm", 0.0)
                ct3.metric(
                    "Air Distance (ESAD)",
                    f"{esad:,.0f} nm",
                    f"{air_benefit:+.0f} nm air savings vs Ground: {g_dist:,.0f} nm",
                )
                f_kg = getattr(contrail_rep, "wind_fuel_impact_kg", 0.0)
                f_eur = getattr(contrail_rep, "wind_fuel_impact_eur", 0.0)
                fuel_lbl = "Saved" if f_kg >= 0 else "Penalty"
                ct4.metric(
                    f"Wind Fuel {fuel_lbl}",
                    f"{f_kg:+,.0f} kg Jet-A",
                    f"€{f_eur:+,.2f} EUR vs still air",
                )

                # Metric Row 2: Persistent Contrail Cirrus & Climate Multiplier
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("Persistent Contrail (ISSR)", f"{contrail_rep.persistent_contrail_distance_nm:.1f} nm", f"{contrail_rep.persistent_contrail_pct:.1f}% of cruise")
                cc2.metric("Direct Engine CO₂", f"{contrail_rep.direct_co2_tons:.2f} t CO₂")
                cc3.metric("Contrail Cirrus Forcing", f"+{contrail_rep.contrail_co2e_tons:.2f} t CO₂e")
                cc4.metric(
                    "Total Climate Multiplier",
                    f"{contrail_rep.total_climate_co2e_multiplier:.2f}x CO₂e",
                    f"{contrail_rep.issr_segments_count} ISSR segments",
                )

                if contrail_rep.issr_segments:
                    st.dataframe(pd.DataFrame([s.__dict__ for s in contrail_rep.issr_segments]), use_container_width=True)

                with st.expander("📖 Why Flown Distance Does Not Equal Fuel Efficiency: Air Distance (ESAD) & Jet Stream Mechanics", expanded=False):
                    st.markdown(
                        """
                        #### 1. The Paradox of Geometric Directness (KEA vs. ESAD)
                        Traditional air navigation performance indicators (such as Eurocontrol PRU **KEA** and **KEP**) measure flight efficiency
                        purely as the geometric ratio between actual flown ground distance and the Great Circle distance:
                        $$\\text{KEA (\\%)} = \\left(\\frac{D_{\\text{ground, actual}} - D_{\\text{GCD}}}{D_{\\text{GCD}}}\\right) \\times 100$$

                        However, at commercial jet cruising altitudes (FL300–FL410), high-altitude **Jet Stream winds commonly exceed 80 to 160+ knots**:
                        - **The Smart Detour**: An airline dispatch team may deliberately file a $+150\\text{ nm}$ ground detour to ride a $+110\\text{ kt}$ jetstream tailwind.
                          While its geometric KEA indicates a +5% inefficiency, the flight reaches its destination 40 minutes faster and burns **2,000 kg less fuel**
                          than if it had flown a straight Great Circle route.
                        - **The Straight-Line Penalty**: An aircraft flying a straight Great Circle path directly into a severe $-120\\text{ kt}$ headwind core
                          achieves "perfect 0.0% directness" geometrically, but experiences severe air-mass drag, prolonged air time, and massive excess fuel burn.

                        #### 2. Equivalent Still Air Distance (ESAD)
                        To measure whether a flight was truly aerodynamically and environmentally efficient, FlightML evaluates **Equivalent Still Air Distance (ESAD)** —
                        the physical distance the aircraft engines had to push the airframe through the moving air mass:
                        $$\\text{ESAD} = \\int \\text{TAS}(t) \\, dt = \\sum_{i} D_{\\text{ground},i} \\times \\left(\\frac{\\text{TAS}_i}{\\text{GS}_i}\\right)$$
                        - **Air Distance Benefit ($\\Delta D_{\\text{air}}$)**:
                          $$\\Delta D_{\\text{air}} = D_{\\text{ground}} - \\text{ESAD}$$
                          - When $\\Delta D_{\\text{air}} > 0$: Favorable tailwinds "shortened" the effective atmosphere, saving engine air miles and fuel.
                          - When $\\Delta D_{\\text{air}} < 0$: Adverse headwinds "stretched" the effective atmosphere, forcing the engines to work harder.

                        ---

                        > **📌 Data Architecture & Implementation Qualifier**:
                        > *Upper-air atmospheric winds (horizontal $U$ and $V$ wind vectors, ambient temperature, and relative humidity across cruise isobaric pressure levels 300, 250, 200, and 150 hPa)
                        > are retrieved **on-demand per flight** from the **Open-Meteo European Reanalysis & Forecast API** and cached in a local-first SQLite datastore (`flightml/data/weather_cache/upper_air_era5.sqlite`).
                        > This local-first, on-demand caching approach is implemented as a lightweight, performant operational solution in place of downloading multi-terabyte bulk ECMWF Copernicus Climate Data Store (CDS) ERA5 global reanalysis grids.*
                        """
                    )

        st.divider()
        with st.expander("💶 Corridor Route-Charge Avoidance vs. Extra Fuel Burn Trade-Off Scanner", expanded=False):
            st.markdown(
                "Analyzes city-pair corridors to detect whether longer horizontal routes are driven by **CRCO navigation charge avoidance** "
                "(flying around high-unit-rate FIRs) versus ATC tactical weather/congestion detours."
            )
            col_rc1, col_rc2, col_rc3 = st.columns(3)
            rc_adep = col_rc1.text_input("Origin (ADEP)", value="EGLL", key=f"rc_adep_{selected_period}").strip().upper()
            rc_ades = col_rc2.text_input("Destination (ADES)", value="LIRF", key=f"rc_ades_{selected_period}").strip().upper()
            rc_ac = col_rc3.text_input("Aircraft Type (Optional)", value="A320", key=f"rc_ac_{selected_period}").strip().upper()
            if st.button("Analyze Corridor CRCO Economics", key=f"btn_crco_corridor_{selected_period}"):
                crco_corr = detect_route_charge_avoidance(
                    ds, adep=rc_adep, ades=rc_ades, ac_type=rc_ac if rc_ac else None
                )
                if not crco_corr.get("flights"):
                    st.warning(f"No matching flights found on corridor `{rc_adep} ➔ {rc_ades}`.")
                else:
                    cc_m1, cc_m2, cc_m3, cc_m4 = st.columns(4)
                    cc_m1.metric("Flights Analyzed", f"{crco_corr['analyzed_flights']}")
                    cc_m2.metric("Shortest Route Dist", f"{crco_corr['shortest_route_distance_nm']:.1f} nm")
                    cc_m3.metric("Shortest Route CRCO Fee", f"€{crco_corr['shortest_route_crco_eur']:,.2f} EUR")
                    cc_m4.metric("Avoidance Detour Candidates", f"{crco_corr['avoidance_candidates_count']} flights")
                    st.dataframe(pd.DataFrame(crco_corr["flights"]), use_container_width=True)

        with st.expander(f"📊 View Weekly Average Jet Fuel Spot Prices in Euros ({selected_period})", expanded=False):
            all_fuel_df = get_all_weekly_fuel_prices_df()
            period_fuel_df = all_fuel_df[all_fuel_df["period_id"] == selected_period]
            if not period_fuel_df.empty:
                st.dataframe(
                    period_fuel_df[["week_label", "price_eur_per_gallon", "price_eur_per_kg", "price_eur_per_metric_ton", "source"]],
                    use_container_width=True,
                )
            else:
                st.dataframe(all_fuel_df, use_container_width=True)

        # ----------------- TAB 7: Twin-Flight Efficiency Benchmarking -----------------

# =============================================================================
# 2. SYSTEM-LEVEL SUMMARY & ANALYSIS VIEWS
# =============================================================================
with tab_cat_system:
    st.caption(
        "Macro air navigation metrics, en-route sector (FIR/AUA) saturation, airport surface movement (ATOT), "
        "cross-decade historical traffic growth, and network-wide hub weather correlation."
    )
    render_page_export_bar("system_summary", "cat_system")
    sys_sub_kpi, sys_sub1, sys_sub_fra, sys_sub2 = st.tabs(
        [
            "📊 Multi-Level KPI Suite (Flight / Fleet / Airport / Network)",
            "🌐 Network Status & Airspace Saturation",
            "🗺️ Free Route Airspace (FRA) & FIR Boundary Inefficiency",
            "📈 Macro Evolution & AIRAC Diff",
        ]
    )

    with sys_sub_kpi:
        st.header(f"📊 Multi-Level Operational & Environmental KPI Suite ({selected_period})")
        st.caption(
            "Hierarchical operational, environmental, and punctuality scorecard evaluated across "
            "**Flight**, **Fleet**, **Airport**, and **Network** tiers with interactive calendar date & diurnal hour scoping."
        )

        ds_min_date, ds_max_date = get_dataset_date_bounds(ds)
        try:
            d_min_obj = pd.to_datetime(ds_min_date).date()
            d_max_obj = pd.to_datetime(ds_max_date).date()
        except Exception:
            d_min_obj = pd.to_datetime("2016-11-01").date()
            d_max_obj = pd.to_datetime("2016-11-30").date()

        # FIR Filter options
        avail_firs = get_available_firs(ds)
        fir_opt_map = {f["label"]: f["fir_id"] for f in avail_firs}
        fir_options = ["All FIRs (Network-Wide)"] + list(fir_opt_map.keys())

        # Timeframe & Level Scoping Bar
        with st.container():
            col_tf1, col_tf2 = st.columns([1, 1])
            with col_tf1:
                date_range_sel = st.date_input(
                    "📅 Calendar Timeframe (Start — End):",
                    value=(d_min_obj, d_max_obj),
                    min_value=d_min_obj,
                    max_value=d_max_obj,
                    key=f"kpi_dates_{selected_period}",
                    help="Filter KPIs across specific calendar days in the dataset.",
                )
                if isinstance(date_range_sel, (tuple, list)) and len(date_range_sel) == 2:
                    filter_start_str = date_range_sel[0].strftime("%Y-%m-%d")
                    filter_end_str = date_range_sel[1].strftime("%Y-%m-%d")
                elif isinstance(date_range_sel, (tuple, list)) and len(date_range_sel) == 1:
                    filter_start_str = date_range_sel[0].strftime("%Y-%m-%d")
                    filter_end_str = filter_start_str
                else:
                    filter_start_str = ds_min_date
                    filter_end_str = ds_max_date

            with col_tf2:
                hour_range_sel = st.slider(
                    "⏰ Diurnal Departure Window (UTC Hours):",
                    min_value=0,
                    max_value=23,
                    value=(0, 23),
                    step=1,
                    key=f"kpi_hours_{selected_period}",
                    help="Filter flights departing within a specific diurnal time-of-day window (e.g. 06:00 to 09:00 morning rush).",
                )
                h_start, h_end = hour_range_sel[0], hour_range_sel[1]

            col_tf3, col_tf4 = st.columns([2, 1])
            with col_tf3:
                kpi_tier = st.radio(
                    "🎯 KPI Aggregation Tier:",
                    options=["🌐 Network Level", "📡 FIR Airspace Tier", "✈️ Fleet Level", "🛫 Airport Level", "🔍 Flight Level"],
                    index=0,
                    horizontal=True,
                    key=f"kpi_tier_radio_{selected_period}",
                )
            with col_tf4:
                selected_fir_label = st.selectbox(
                    "🗺️ Filter by FIR Airspace:",
                    options=fir_options,
                    index=0,
                    key=f"kpi_fir_filter_{selected_period}",
                    help="Scope metrics to flights that traversed or arrived/departed within a specific European Flight Information Region (FIR).",
                )
                selected_fir_id = (
                    fir_opt_map.get(selected_fir_label, None)
                    if selected_fir_label != "All FIRs (Network-Wide)"
                    else None
                )

        st.divider()

        # Tier Views
        if kpi_tier == "🌐 Network Level":
            fir_scope_desc = f" (Scoped to {selected_fir_id})" if selected_fir_id else ""
            with st.spinner("Aggregating network-wide KPIs across filtered timeframe..."):
                net_summary = query_multi_level_kpis(
                    ds,
                    level="network",
                    fir_filter=selected_fir_id,
                    start_date=filter_start_str,
                    end_date=filter_end_str,
                    hour_start=h_start,
                    hour_end=h_end,
                )

            st.subheader(f"🌐 Network-Wide Operational & Environmental KPI Scorecard{fir_scope_desc}")
            st.info(f"📅 **Timeframe:** `{net_summary.time_window}` | **Diurnal Window:** `{net_summary.hour_window}` | **Active Network Flights:** `{net_summary.total_flights:,}`")

            nk1, nk2, nk3, nk4 = st.columns(4)
            nk1.metric("Total Processed Flights", f"{net_summary.total_flights:,}")
            nk2.metric("Avg Departure Delay", f"{net_summary.avg_dep_delay_min:+.1f} min")
            nk3.metric("Avg Arrival Delay", f"{net_summary.avg_arr_delay_min:+.1f} min")
            nk4.metric("Severe Arrival Delays (>30m)", f"{net_summary.pct_severe_arr_30m:.1f}%")

            nk5, nk6, nk7, nk8 = st.columns(4)
            nk5.metric("On-Time Performance (OTP15)", f"{100.0 - net_summary.pct_arr_delayed_15m:.1f}%", f"{net_summary.pct_arr_delayed_15m:.1f}% delayed >15m")
            nk6.metric("Avg Stage Distance", f"{net_summary.avg_actual_dist_nm:,.0f} nm")
            nk7.metric("Avg Requested Altitude", f"FL{net_summary.avg_requested_fl:.0f}")
            nk8.metric("Est. Detour Fuel Cost", f"€{net_summary.total_estimated_cost_eur:,.0f} EUR", f"{net_summary.total_estimated_fuel_penalty_kg:,.0f} kg fuel")

            if selected_fir_id:
                with st.spinner(f"Extracting traffic segmentation breakdown for {selected_fir_id}..."):
                    net_fir_seg = query_fir_traffic_segmentation(
                        ds,
                        fir_id=selected_fir_id,
                        start_date=filter_start_str,
                        end_date=filter_end_str,
                        hour_start=h_start,
                        hour_end=h_end,
                    )
                st.markdown(f"#### 📡 {selected_fir_id} Airspace Traffic Breakdown ({net_fir_seg.country_name})")
                st.caption(f"Certified CRCO Unit Rate: **€{net_fir_seg.unit_rate_eur:.2f} / SU** | Total FIR Traffic: **{net_fir_seg.total_flights:,} movements**")
                bf1, bf2, bf3, bf4 = st.columns(4)
                bf1.metric("Over-flights", f"{net_fir_seg.overflights_count:,}", f"{net_fir_seg.overflights_pct:.1f}% share")
                bf2.metric("Airport Arrivals", f"{net_fir_seg.arrivals_count:,}", f"{net_fir_seg.arrivals_pct:.1f}% share")
                bf3.metric("Airport Departures", f"{net_fir_seg.departures_count:,}", f"{net_fir_seg.departures_pct:.1f}% share")
                bf4.metric("Internal / Domestic", f"{net_fir_seg.internal_count:,}", f"{net_fir_seg.internal_pct:.1f}% share")

        elif kpi_tier == "📡 FIR Airspace Tier":
            active_fir = selected_fir_id or (avail_firs[0]["fir_id"] if avail_firs else "EGTTFIR")
            with st.spinner(f"Analyzing FIR airspace traffic segmentation for {active_fir}..."):
                fir_seg = query_fir_traffic_segmentation(
                    ds,
                    fir_id=active_fir,
                    start_date=filter_start_str,
                    end_date=filter_end_str,
                    hour_start=h_start,
                    hour_end=h_end,
                )

            st.subheader(f"📡 {fir_seg.fir_id} Airspace Operational & Traffic Segmentation Scorecard")
            st.caption(
                f"**Member State / Country:** `{fir_seg.country_name}` | "
                f"**Certified CRCO Unit Rate:** `€{fir_seg.unit_rate_eur:.2f} / SU` | "
                f"**Timeframe:** `{fir_seg.time_window}` ({fir_seg.hour_window})"
            )

            sf1, sf2, sf3, sf4, sf5 = st.columns(5)
            sf1.metric("Total Movements", f"{fir_seg.total_flights:,}")
            sf2.metric("Over-flights", f"{fir_seg.overflights_count:,}", f"{fir_seg.overflights_pct:.1f}% share")
            sf3.metric("Airport Arrivals", f"{fir_seg.arrivals_count:,}", f"{fir_seg.arrivals_pct:.1f}% share")
            sf4.metric("Airport Departures", f"{fir_seg.departures_count:,}", f"{fir_seg.departures_pct:.1f}% share")
            sf5.metric("Internal / Domestic", f"{fir_seg.internal_count:,}", f"{fir_seg.internal_pct:.1f}% share")

            # Traffic Segmentation Visual & Table
            st.markdown("#### 📊 Airspace Traffic Segmentation Breakdown (Overflights vs. Arrivals vs. Departures vs. Internal)")
            if fir_seg.categories:
                cat_df = pd.DataFrame(fir_seg.categories)

                col_cg1, col_cg2 = st.columns([1, 1])
                with col_cg1:
                    chart_vol = alt.Chart(cat_df).mark_bar().encode(
                        x=alt.X("traffic_type:N", title="Traffic Category", sort="-y"),
                        y=alt.Y("flight_count:Q", title="Flight Movements"),
                        color=alt.Color(
                            "traffic_type:N",
                            scale=alt.Scale(
                                domain=["Overflight", "Arrival", "Departure", "Internal"],
                                range=["#3498db", "#2ecc71", "#e67e22", "#9b59b6"],
                            ),
                            legend=None,
                        ),
                        tooltip=["traffic_type", "flight_count", "share_pct", "avg_dwell_min", "avg_arr_delay_min"],
                    ).properties(height=260, title="Flight Volume by Airspace Role")
                    st.altair_chart(chart_vol, use_container_width=True)

                with col_cg2:
                    chart_delay = alt.Chart(cat_df).mark_bar().encode(
                        x=alt.X("traffic_type:N", title="Traffic Category", sort="-y"),
                        y=alt.Y("avg_arr_delay_min:Q", title="Avg Arrival Delay (min)"),
                        color=alt.Color(
                            "traffic_type:N",
                            scale=alt.Scale(
                                domain=["Overflight", "Arrival", "Departure", "Internal"],
                                range=["#3498db", "#2ecc71", "#e67e22", "#9b59b6"],
                            ),
                            legend=None,
                        ),
                        tooltip=["traffic_type", "avg_arr_delay_min", "avg_dep_delay_min", "pct_severe_arr_30m"],
                    ).properties(height=260, title="Arrival Delay Punctuality by Airspace Role")
                    st.altair_chart(chart_delay, use_container_width=True)

                st.dataframe(cat_df, use_container_width=True)

            # Top City Pair Corridors Breakdown
            st.markdown(f"#### 🛫 Top Route Corridors Crossing {fir_seg.fir_id}")
            t_ov, t_ar, t_dp, t_in = st.tabs([
                "✈️ Over-flight Corridors",
                "🛬 Airport Arrival Corridors",
                "🛫 Airport Departure Corridors",
                "🔄 Internal / Domestic Corridors",
            ])
            with t_ov:
                ov_corr = fir_seg.top_corridors.get("Overflight", [])
                if ov_corr:
                    st.dataframe(pd.DataFrame(ov_corr).rename(columns={"pair": "City Pair Corridor", "count": "Flights"}), use_container_width=True)
                else:
                    st.info("No recorded overflight city pairs matching the active filters.")
            with t_ar:
                ar_corr = fir_seg.top_corridors.get("Arrival", [])
                if ar_corr:
                    st.dataframe(pd.DataFrame(ar_corr).rename(columns={"pair": "City Pair Corridor", "count": "Flights"}), use_container_width=True)
                else:
                    st.info("No recorded arrival city pairs matching the active filters.")
            with t_dp:
                dp_corr = fir_seg.top_corridors.get("Departure", [])
                if dp_corr:
                    st.dataframe(pd.DataFrame(dp_corr).rename(columns={"pair": "City Pair Corridor", "count": "Flights"}), use_container_width=True)
                else:
                    st.info("No recorded departure city pairs matching the active filters.")
            with t_in:
                in_corr = fir_seg.top_corridors.get("Internal", [])
                if in_corr:
                    st.dataframe(pd.DataFrame(in_corr).rename(columns={"pair": "City Pair Corridor", "count": "Flights"}), use_container_width=True)
                else:
                    st.info("No recorded internal city pairs matching the active filters.")

            # Network-Wide FIR League Table
            st.markdown("#### 🗺️ European FIR League Table (Ranked by Total Airspace Movements)")
            all_firs_summary = query_multi_level_kpis(
                ds,
                level="fir",
                start_date=filter_start_str,
                end_date=filter_end_str,
                hour_start=h_start,
                hour_end=h_end,
                min_flights=20,
                limit=30,
            )
            if all_firs_summary.records:
                league_df = pd.DataFrame(all_firs_summary.records)
                rename_cols = {
                    "entity_id": "FIR Airspace",
                    "country_name": "Country / State",
                    "unit_rate_eur": "CRCO Unit Rate (€/SU)",
                    "total_flights": "Total Movements",
                    "overflights_count": "Over-flights",
                    "arrivals_count": "Arrivals",
                    "departures_count": "Departures",
                    "internal_count": "Internal / Domestic",
                    "overflights_pct": "Overflight Share (%)",
                    "avg_arr_delay_min": "Avg Arr Delay (min)",
                    "pct_severe_arr_30m": "Severe Delays >30m (%)",
                    "avg_requested_fl": "Avg Requested FL",
                    "total_excess_fuel_cost_eur": "Est. Detour Cost (€)",
                }
                disp_df = league_df.rename(columns={k: v for k, v in rename_cols.items() if k in league_df.columns})
                primary_cols = [
                    "FIR Airspace", "Country / State", "Total Movements", "Over-flights", "Arrivals", "Departures",
                    "Internal / Domestic", "Overflight Share (%)", "CRCO Unit Rate (€/SU)", "Avg Arr Delay (min)", "Severe Delays >30m (%)"
                ]
                final_cols = [c for c in primary_cols if c in disp_df.columns] + [c for c in disp_df.columns if c not in primary_cols]
                st.dataframe(disp_df[final_cols], use_container_width=True)

        elif kpi_tier == "✈️ Fleet Level":
            col_fl_sub1, col_fl_sub2, col_fl_sub3 = st.columns([2, 1, 1])
            with col_fl_sub1:
                fleet_mode = st.radio(
                    "Fleet Categorization:",
                    options=["Airline Operators (ac_operator)", "Aircraft Types (ac_type)"],
                    horizontal=True,
                    key=f"fleet_mode_{selected_period}",
                )
            with col_fl_sub2:
                fleet_min_flts = st.slider("Min Flights Threshold", 10, 500, 50, key=f"fl_min_{selected_period}")
            with col_fl_sub3:
                fleet_top_n = st.slider("Top Entities Count", 5, 50, 15, key=f"fl_top_{selected_period}")

            grp_key = "operator" if "operator" in fleet_mode.lower() else "aircraft"
            fir_scope_desc = f" (Scoped to {selected_fir_id})" if selected_fir_id else ""
            with st.spinner(f"Aggregating fleet KPIs by {fleet_mode}..."):
                fleet_summary = query_multi_level_kpis(
                    ds,
                    level="fleet",
                    group_by=grp_key,
                    fir_filter=selected_fir_id,
                    start_date=filter_start_str,
                    end_date=filter_end_str,
                    hour_start=h_start,
                    hour_end=h_end,
                    min_flights=fleet_min_flts,
                    limit=fleet_top_n,
                )

            st.subheader(f"✈️ Fleet Performance Scorecard — by {fleet_summary.group_by}{fir_scope_desc}")
            st.info(f"📅 **Timeframe:** `{fleet_summary.time_window}` | **Diurnal Window:** `{fleet_summary.hour_window}` | **Analyzed Fleets:** `{len(fleet_summary.records)}` (≥{fleet_min_flts} flights)")

            if fleet_summary.records:
                f_df = pd.DataFrame(fleet_summary.records)
                chart_fleet = alt.Chart(f_df).mark_bar(color="#3498db").encode(
                    x=alt.X("entity_id:N", title=fleet_summary.group_by, sort="-y"),
                    y=alt.Y("avg_arr_delay_min:Q", title="Avg Arrival Delay (min)"),
                    tooltip=["entity_id", "total_flights", "avg_arr_delay_min", "pct_arr_delayed_15m", "pct_severe_arr_30m", "total_excess_fuel_cost_eur"],
                ).properties(height=260, title=f"Top {len(f_df)} {fleet_summary.group_by} Ranked by Average Arrival Delay")
                st.altair_chart(chart_fleet, use_container_width=True)

                st.dataframe(f_df, use_container_width=True)

        elif kpi_tier == "🛫 Airport Level":
            col_ap_sub1, col_ap_sub2, col_ap_sub3 = st.columns([2, 1, 1])
            with col_ap_sub1:
                airport_mode = st.radio(
                    "Airport Categorization:",
                    options=["Departure Aerodromes (ADEP)", "Arrival Aerodromes (ADES)"],
                    horizontal=True,
                    key=f"ap_mode_{selected_period}",
                )
            with col_ap_sub2:
                ap_min_flts = st.slider("Min Flights Threshold", 10, 500, 50, key=f"ap_min_{selected_period}")
            with col_ap_sub3:
                ap_top_n = st.slider("Top Hubs Count", 5, 50, 15, key=f"ap_top_{selected_period}")

            grp_key = "departure" if "departure" in airport_mode.lower() else "arrival"
            fir_scope_desc = f" (Scoped to {selected_fir_id})" if selected_fir_id else ""
            with st.spinner(f"Aggregating airport KPIs by {airport_mode}..."):
                airport_summary = query_multi_level_kpis(
                    ds,
                    level="airport",
                    group_by=grp_key,
                    fir_filter=selected_fir_id,
                    start_date=filter_start_str,
                    end_date=filter_end_str,
                    hour_start=h_start,
                    hour_end=h_end,
                    min_flights=ap_min_flts,
                    limit=ap_top_n,
                )

            st.subheader(f"🛫 Airport Performance Scorecard — {airport_summary.group_by}{fir_scope_desc}")
            st.info(f"📅 **Timeframe:** `{airport_summary.time_window}` | **Diurnal Window:** `{airport_summary.hour_window}` | **Analyzed Hubs:** `{len(airport_summary.records)}` (≥{ap_min_flts} flights)")

            if airport_summary.records:
                ap_df = pd.DataFrame(airport_summary.records)
                delay_col = "avg_dep_delay_min" if grp_key == "departure" else "avg_arr_delay_min"
                chart_ap = alt.Chart(ap_df).mark_bar(color="#e67e22").encode(
                    x=alt.X("entity_id:N", title="Aerodrome ICAO", sort="-y"),
                    y=alt.Y(f"{delay_col}:Q", title=f"Avg {'Departure' if grp_key == 'departure' else 'Arrival'} Delay (min)"),
                    tooltip=["entity_id", "total_flights", "avg_dep_delay_min", "avg_arr_delay_min", "pct_severe_arr_30m", "total_excess_fuel_cost_eur"],
                ).properties(height=260, title=f"Top {len(ap_df)} Aerodromes Ranked by Delay ({airport_summary.group_by})")
                st.altair_chart(chart_ap, use_container_width=True)

                st.dataframe(ap_df, use_container_width=True)

        else: # 🔍 Flight Level
            fir_scope_desc = f" traversing {selected_fir_id}" if selected_fir_id else ""
            st.subheader(f"🔍 Micro Flight-Level Search & KPI Inspector{fir_scope_desc}")
            st.caption(f"Browse individual flight trajectories and efficiency metrics matching `{filter_start_str} to {filter_end_str}` ({h_start:02d}:00–{h_end:02d}:59 UTC){fir_scope_desc}.")

            with st.spinner("Querying individual flight records matching timeframe..."):
                flight_summary = query_multi_level_kpis(
                    ds,
                    level="flight",
                    fir_filter=selected_fir_id,
                    start_date=filter_start_str,
                    end_date=filter_end_str,
                    hour_start=h_start,
                    hour_end=h_end,
                    limit=100,
                )

            if flight_summary.records:
                fl_df = pd.DataFrame(flight_summary.records)
                st.dataframe(fl_df, use_container_width=True)

    with sys_sub1:
        st.header(f"Global Network Status & Airspace Saturation ({selected_period})")
        render_page_export_bar("airspace_overview", "tab1")
        stats = ds.get_summary()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Processed Flights", f"{stats['total_flights']:,}")
        c2.metric("Aircraft Operators", f"{stats['total_operators']:,}")
        c3.metric("Departure Airports", f"{stats['unique_dep_airports']:,}")
        c4.metric("Avg Arrival Delay", f"{stats['avg_arr_delay_min']} min")

        st.divider()
        st.subheader("🗺️ European Airspace Hourly Saturation & Dynamic Complexity Map (Period Average by Hour of Day)")
        st.markdown(
            "Visualizes the **average hourly flight density, peak traffic load, vertical climb/descent transition intensity, "
            "and Dynamic Complexity Score** across geographic squares for any selected hour of the day (`00:00`–`23:00 UTC`). "
            "Adjust the resolution slider to zoom in from macro $5^\\circ \\times 5^\\circ$ squares all the way down to fine $0.5^\\circ \\times 0.5^\\circ$ micro cells."
        )
        col_gm1, col_gm2 = st.columns([1, 1])
        with col_gm1:
            map_hour_utc = st.slider(
                "Select Hour of Day (UTC) — Averaged Across All Days in Period",
                min_value=0,
                max_value=23,
                value=10,
                format="%02d:00 UTC",
                key=f"grid_map_hour_{selected_period}",
            )
        with col_gm2:
            grid_size_sel = st.select_slider(
                "🗺️ Grid Square Resolution (Square Size):",
                options=[5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5],
                value=5.0,
                format_func=lambda x: f"{x:g}° × {x:g}°" + (" (Macro 5v5)" if x == 5.0 else (" (Micro 0.5v0.5)" if x == 0.5 else "")),
                help="Select cell dimensions from 5v5 macro squares down to 0.5v0.5 fine micro cells.",
                key=f"grid_size_slider_{selected_period}",
            )

        grid_sat_data = _cached_grid_saturation(selected_period, map_hour_utc, grid_size_deg=grid_size_sel)

        gm_c1, gm_c2, gm_c3, gm_c4 = st.columns(4)
        gm_c1.metric(f"Active {grid_size_sel:g}°×{grid_size_sel:g}° Squares", f"{grid_sat_data['total_active_squares']:,}")
        gm_c2.metric("Critical Saturation", f"{grid_sat_data['critical_squares_count']} squares")
        gm_c3.metric("High Density", f"{grid_sat_data['high_density_squares_count']} squares")
        gm_c4.metric("Peak Square Avg/Hr", f"{grid_sat_data['max_avg_flights_per_hour']:.1f} flts/hr")

        grid_map = create_grid_saturation_map(grid_sat_data)
        st_folium(
            grid_map,
            width=1200,
            height=540,
            returned_objects=[],
            key=f"folium_grid_sat_{selected_period}_{map_hour_utc}_{grid_size_sel:g}",
        )

        with st.expander(f"📊 View {grid_size_sel:g}° × {grid_size_sel:g}° Square Saturation Table for {map_hour_utc:02d}:00–{map_hour_utc:02d}:59 UTC", expanded=False):
            if grid_sat_data.get("squares"):
                sq_df = pd.DataFrame(grid_sat_data["squares"])[
                    [
                        "cell_label",
                        "saturation_status",
                        "avg_flights_per_hour",
                        "peak_flights_per_hour",
                        "vertical_transition_pct",
                        "avg_flight_level",
                        "dynamic_complexity_score",
                        "active_days",
                    ]
                ].rename(
                    columns={
                        "cell_label": f"{grid_size_sel:g}°×{grid_size_sel:g}° Square",
                        "saturation_status": "Status",
                        "avg_flights_per_hour": "Avg Flights / Hr",
                        "peak_flights_per_hour": "Peak Flights / Hr",
                        "vertical_transition_pct": "Vertical Climb/Desc (%)",
                        "avg_flight_level": "Mean FL",
                        "dynamic_complexity_score": "Complexity Index",
                        "active_days": "Active Days",
                    }
                )
                st.dataframe(sq_df, use_container_width=True)

        st.divider()
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Busiest Hubs by Departures")
            top_apt = ds.get_top_airports(10)
            st.dataframe(top_apt, use_container_width=True)

        with col_right:
            st.subheader("Worst Delayed Corridors")
            top_delays = ds.get_top_delayed_routes(min_flights=50, limit=10)
            st.dataframe(top_delays, use_container_width=True)

        if ds.has_fir_traversals():
            st.subheader("🌐 En-Route FIR Sector Hourly Saturation & Bottlenecks")
            col_s1, col_s2 = st.columns([1, 3])
            with col_s1:
                sat_thresh = st.slider("Saturation Threshold (flts/hr)", 20, 100, 40, key=f"fir_thresh_{selected_period}")
            sat_data = calculate_fir_hourly_saturation(ds, threshold_flights_per_hr=sat_thresh, top_n=10)
            if sat_data.get("top_saturated_firs"):
                sat_df = pd.DataFrame(sat_data["top_saturated_firs"])
                st.dataframe(sat_df, use_container_width=True)

            if ds.has_aua_traversals():
                st.subheader("📡 ATC Unit Airspaces (AUA: TMA / CTA / UTA) Saturation")
                col_a1, col_a2 = st.columns([1, 2])
                with col_a1:
                    aua_type_sel = st.selectbox(
                        "Filter AUA Type",
                        ["All (TMA/CTA/UTA)", "TMA", "CTA", "UTA"],
                        key=f"aua_type_{selected_period}",
                    )
                with col_a2:
                    aua_thresh = st.slider(
                        "AUA Hourly Saturation Threshold (flts/hr)",
                        15,
                        90,
                        30,
                        key=f"aua_thresh_{selected_period}",
                    )
                aua_filter = None if aua_type_sel.startswith("All") else aua_type_sel
                aua_sat = calculate_aua_hourly_saturation(
                    ds, aua_type=aua_filter, threshold_flights_per_hr=aua_thresh, top_n=10
                )
                if aua_sat.get("top_saturated_auas"):
                    st.dataframe(pd.DataFrame(aua_sat["top_saturated_auas"]), use_container_width=True)

            st.subheader("🛞 Airport Surface Movement & ICAO KPI02 Additional Taxi-Out Time (ATOT)")
            taxi_res = analyze_airport_taxi_out_performance(ds, min_flights=50, top_n=12)
            if taxi_res.get("airports"):
                st.caption(
                    "UTOT = 20th-percentile unimpeded taxi-out baseline per (airport, wake category) | "
                    "ATOT = Additional Taxi-Out Time (ICAO GANP KPI02) | LTO Idle Emissions via ICAO EDB v32B"
                )
                st.dataframe(pd.DataFrame(taxi_res["airports"]), use_container_width=True)
        else:
            st.subheader("Top Congested European FIR Airspace Sectors")
            fir_df = analyze_fir_bottlenecks(ds, limit=8)
            st.dataframe(fir_df, use_container_width=True)

        with st.expander("📻 ATC Controller-Pilot VHF Radio Frequency Workload & Comms Occupancy", expanded=False):
            col_rad1, col_rad2 = st.columns([1, 3])
            with col_rad1:
                rad_apt = st.text_input("Airport ICAO", value="EGLL", key=f"rad_apt_{selected_period}").strip().upper()
            rad_res = estimate_airport_and_sector_radio_workload(ds, airport_icao=rad_apt)
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Peak Movements / Hr", f"{rad_res['peak_hourly_movements']}")
            rc2.metric("Cataloged VHF Channels", f"{rad_res['total_vhf_channels_cataloged']}")
            rc3.metric("Terminal ILS/VOR/DME", f"{rad_res['associated_ils_vor_dme_count']}")
            rc4.metric("Primary Comms Bottleneck", rad_res["primary_comms_bottleneck"], f"{rad_res['peak_voice_occupancy_pct']:.1f}% occ")
            st.dataframe(pd.DataFrame(rad_res["positions"]), use_container_width=True)

        # ----------------- TAB 2: Trajectory Explorer -----------------

    # =========================================================================
    # SUB-TAB: Free Route Airspace (FRA) & FIR Boundary Inefficiency
    # =========================================================================
    with sys_sub_fra:
        st.header(f"🗺️ Free Route Airspace (FRA) & FIR Boundary Inefficiency Explorer ({selected_period})")
        st.caption(
            "Deterministic analysis of European Flight Information Region (FIR) boundary routing inefficiencies. "
            "Detects active cross-border Free Route Airspace (FRA) implementations, spotlights prime candidate "
            "boundaries for immediate FRA rollout, and ranks all European boundary interfaces in descending order of priority."
        )
        st.info(
            "ℹ️ **Regulatory Baseline (EU 2021/116 - Common Project 1 AF3):** Free Route Airspace (FRA) is formally "
            "mandated and operational across the Single European Sky **at and above FL305** (30,500 ft). "
            "Airspace below FL305 remains governed by structured ATS airways, terminal routing (SIDs/STARs), and military airspace reservations."
        )

        render_page_export_bar("fra_boundaries", "tab_fra_bnd")

        # Controls Bar
        col_fb1, col_fb2, col_fb3, col_fb4 = st.columns([1, 1, 1, 1])
        with col_fb1:
            fra_period_sel = st.selectbox(
                "📅 Analysis Period:",
                available_periods,
                index=available_periods.index(selected_period) if selected_period in available_periods else 0,
                key="fra_bnd_period_select",
                help="Select historical Eurocontrol archive period month (e.g. 202510 or 202410).",
            )
        with col_fb2:
            fra_min_flts = st.slider(
                "✈️ Min Flight Volume Threshold:",
                min_value=200,
                max_value=10000,
                value=1000,
                step=200,
                key="fra_bnd_min_flts",
                help="Filter out low-volume FIR boundaries.",
            )
        with col_fb3:
            fra_xb_only = st.checkbox(
                "🌍 Cross-Border Only",
                value=True,
                key="fra_bnd_xb_only",
                help="Filter strictly to international state boundaries (excludes domestic lower/upper sector handovers).",
            )
        with col_fb4:
            fra_top_n = st.selectbox(
                "📊 League Table Limit:",
                [15, 25, 50, 100],
                index=1,
                key="fra_bnd_top_n",
            )

        fra_ds = ds if fra_period_sel == selected_period else FlightDataset(period=fra_period_sel)

        with st.spinner(f"Computing deterministic FIR boundary inefficiencies for period {fra_period_sel}..."):
            fra_report = analyze_fir_boundary_inefficiencies(
                dataset=fra_ds,
                min_flights=fra_min_flts,
                limit=fra_top_n,
                cross_border_only=fra_xb_only,
            )

        # Macro Performance Scorecard
        st.markdown("### 🌐 European FIR Boundary Macro Performance Scorecard")
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Boundaries Analyzed", f"{fra_report.total_boundaries_analyzed:,}")
        sc2.metric("Cross-Border Interfaces", f"{fra_report.cross_border_boundaries_count:,}")
        sc3.metric("Operating Likely FRA", f"{fra_report.likely_fra_boundaries_count:,}", "🟢 Harmonized")
        sc4.metric("Prime FRA Candidates", f"{fra_report.prime_candidate_count:,}", "🚨 Bottlenecks")
        sc5.metric(
            "Detour Fuel Penalty",
            f"€{fra_report.total_network_excess_cost_eur:,.0f} EUR",
            f"{fra_report.total_network_excess_fuel_tonnes:,.0f}t Jet-A ({fra_report.total_network_excess_co2_tonnes:,.0f}t CO₂)",
        )

        st.divider()

        # Six Analytical Views
        fb_tab_prime, fb_tab_likely, fb_tab_league, fb_tab_diag, fb_tab_evo, fb_tab_forecast = st.tabs([
            "🚨 Prime Implementation Candidates",
            "🟢 Boundaries Likely Providing FRA",
            "📊 Full Ranked Priority League Table",
            "🔬 Single Boundary Diagnostic & Worst Flights Map",
            "📈 2024 vs. 2025 Longitudinal Evolution",
            "💰 Long-Term FRA Benefit Forecaster (1 / 5 / 20 Years)",
        ])

        # ---------------------------------------------------------------------
        # VIEW 1: Prime Candidates for Immediate FRA Implementation
        # ---------------------------------------------------------------------
        with fb_tab_prime:
            st.subheader("🚨 Prime Candidate FIR Boundaries for Immediate FRA Implementation")
            st.markdown(
                "These high-volume European airspace boundaries suffer from the most severe **routing detours, "
                "fixed-waypoint coordination funneling, and heading deflections**. Eliminating rigid cross-border "
                "coordination points and extending Free Route Airspace across these specific interfaces yields "
                "the largest network-wide fuel, emissions, and airspace capacity gains."
            )

            prime_candidates = [b for b in fra_report.boundaries if b.fra_classification == "🚨 PRIME_CANDIDATE_IMMEDIATE_FRA"]
            if not prime_candidates:
                st.info("No boundaries meeting the 'Prime Candidate' criteria at the selected volume threshold.")
            else:
                for idx, c in enumerate(prime_candidates, start=1):
                    with st.container():
                        col_card1, col_card2, col_card3, col_card4 = st.columns([2, 1, 1, 1])
                        with col_card1:
                            st.markdown(f"#### #{idx}. `{c.boundary_pair}` ({c.country_a} ↔ {c.country_b})")
                            st.caption(
                                f"Directional Flow: **{c.fir_a} ➔ {c.fir_b}:** `{c.flow_a_to_b:,}` flts | "
                                f"**{c.fir_b} ➔ {c.fir_a}:** `{c.flow_b_to_a:,}` flts | "
                                f"Waypoint Funneling Index: **{c.waypoint_funneling_score:.1f}/100**"
                            )
                        with col_card2:
                            st.metric("Total Crossings", f"{c.total_crossings:,}")
                        with col_card3:
                            st.metric("Mean Detour", f"+{c.mean_detour_nm:.1f} NM", f"Directness: {c.directness_index_pct:.1f}%")
                        with col_card4:
                            st.metric("Excess Fuel Cost", f"€{c.excess_cost_eur:,.0f} EUR", f"{c.excess_fuel_tonnes:,.0f}t fuel ({c.excess_co2_tonnes:,.0f}t CO₂)")

                        st.progress(min(1.0, c.fra_priority_score / 100.0), text=f"FRA Boundary Priority Score: {c.fra_priority_score:.1f} / 100.0")
                        st.divider()

        # ---------------------------------------------------------------------
        # VIEW 2: Boundaries Likely Already Providing FRA
        # ---------------------------------------------------------------------
        with fb_tab_likely:
            st.subheader("🟢 European FIR Boundaries Likely Already Providing Free Route Airspace (FRA)")
            st.markdown(
                r"These interfaces exhibit **high directness indices ($\ge 96.5\%$), minimal boundary heading deflections ($\le 8^\circ$), "
                r"and widely dispersed lateral crossing points**, confirming successful cross-border Free Route integration. "
                "Prominent examples include **NEFRA** (North European FRA), **SECSI FRA** (South-East Europe Common Sky Initiative), and **SEE FRA**."
            )

            likely_fra = [b for b in fra_report.boundaries if b.fra_classification == "🟢 LIKELY_CROSS_BORDER_FRA"]
            if likely_fra:
                st.info(f"Identified **{len(likely_fra)} boundaries** deterministically operating in near-direct Free Route alignment.")
                fra_table_data = [
                    {
                        "Boundary Pair": b.boundary_pair,
                        "Countries / States": f"{b.country_a} ↔ {b.country_b}",
                        "Crossings": b.total_crossings,
                        "Directness Index": f"{b.directness_index_pct:.1f}%",
                        "Mean Detour": f"+{b.mean_detour_nm:.1f} NM",
                        "Deflection Angle": f"{b.deflection_angle_deg:.1f}°",
                        "Funneling Score": f"{b.waypoint_funneling_score:.1f}/100",
                        "Status": "🟢 Cross-Border Free Route Operating",
                    }
                    for b in likely_fra
                ]
                st.dataframe(pd.DataFrame(fra_table_data), use_container_width=True)
            else:
                st.info("No boundaries meeting the Likely FRA criteria in the active slice.")

        # ---------------------------------------------------------------------
        # VIEW 3: Full Ranked Priority League Table
        # ---------------------------------------------------------------------
        with fb_tab_league:
            st.subheader("📊 European FIR Boundary Priority League Table (Ranked by Need for FRA)")
            st.markdown(
                "Comprehensive ranking of European airspace boundaries sorted in descending order of the **Free Route Airspace Boundary "
                "Priority Index (FRA-BPI, 0–100)**. Scores combine traffic exposure, detour nautical miles, waypoint funneling, and excess fuel penalty."
            )

            if fra_report.boundaries:
                scatter_df = pd.DataFrame([b.to_dict() for b in fra_report.boundaries])
                chart = (
                    alt.Chart(scatter_df)
                    .mark_circle(size=140)
                    .encode(
                        x=alt.X("total_crossings:Q", title="Total Boundary Crossings"),
                        y=alt.Y("mean_detour_nm:Q", title="Mean Boundary Detour (NM)"),
                        color=alt.Color(
                            "fra_classification:N",
                            title="Classification",
                            scale=alt.Scale(
                                domain=[
                                    "🚨 PRIME_CANDIDATE_IMMEDIATE_FRA",
                                    "⚠️ MEDIUM_PRIORITY_FRA",
                                    "🟢 LIKELY_CROSS_BORDER_FRA",
                                    "ℹ️ DOMESTIC_OR_LOW_IMPACT",
                                ],
                                range=["#e74c3c", "#f39c12", "#2ecc71", "#95a5a6"],
                            ),
                        ),
                        size=alt.Size("excess_cost_eur:Q", title="Excess Fuel Cost (€ EUR)"),
                        tooltip=[
                            alt.Tooltip("boundary_pair:N", title="Boundary"),
                            alt.Tooltip("country_a:N", title="Country A"),
                            alt.Tooltip("country_b:N", title="Country B"),
                            alt.Tooltip("total_crossings:Q", title="Crossings", format=","),
                            alt.Tooltip("mean_detour_nm:Q", title="Detour (NM)", format=".1f"),
                            alt.Tooltip("directness_index_pct:Q", title="Directness", format=".1f"),
                            alt.Tooltip("excess_cost_eur:Q", title="Excess Cost (€)", format=",.0f"),
                            alt.Tooltip("fra_priority_score:Q", title="Priority Score", format=".1f"),
                        ],
                    )
                    .properties(height=380)
                    .interactive()
                )
                st.altair_chart(chart, use_container_width=True)

                search_q = st.text_input("🔍 Search Boundary Pair or Country:", key="fra_bnd_search").strip().upper()
                display_df = scatter_df.copy()
                if search_q:
                    display_df = display_df[
                        display_df["boundary_pair"].str.contains(search_q, case=False)
                        | display_df["country_a"].str.contains(search_q, case=False)
                        | display_df["country_b"].str.contains(search_q, case=False)
                    ]

                rename_map = {
                    "priority_rank": "Rank",
                    "boundary_pair": "Boundary Pair",
                    "country_a": "Country A",
                    "country_b": "Country B",
                    "fra_classification": "Classification",
                    "total_crossings": "Crossings",
                    "directness_index_pct": "Directness (%)",
                    "mean_detour_nm": "Detour (NM)",
                    "waypoint_funneling_score": "Funneling (0-100)",
                    "excess_fuel_tonnes": "Excess Fuel (t)",
                    "excess_co2_tonnes": "Excess CO2 (t)",
                    "excess_cost_eur": "Excess Cost (€ EUR)",
                    "fra_priority_score": "Priority Score (0-100)",
                }
                st.dataframe(
                    display_df[[c for c in rename_map.keys() if c in display_df.columns]].rename(columns=rename_map),
                    use_container_width=True,
                )

        # ---------------------------------------------------------------------
        # VIEW 4: Single Boundary Diagnostic
        # ---------------------------------------------------------------------
        with fb_tab_diag:
            st.subheader("🔬 Single Boundary Deep-Dive Diagnostic")
            st.markdown("Inspect directional flow splits, diurnal congestion peaks, and top airlines navigating any specific boundary interface.")

            avail_pairs = [b.boundary_pair for b in fra_report.boundaries]
            if not avail_pairs:
                st.info("No boundaries available for diagnostic.")
            else:
                default_pair = "EGTTUIR <-> LFFFUIR" if "EGTTUIR <-> LFFFUIR" in avail_pairs else avail_pairs[0]
                sel_pair = st.selectbox(
                    "Select Boundary Interface to Inspect:",
                    avail_pairs,
                    index=avail_pairs.index(default_pair) if default_pair in avail_pairs else 0,
                    key="diag_sel_boundary",
                )

                diag_info = get_boundary_detail(fra_ds, sel_pair)
                if "error" in diag_info:
                    st.error(diag_info["error"])
                else:
                    d_c1, d_c2 = st.columns([1, 1])
                    with d_c1:
                        st.markdown(f"#### ✈️ Top Airlines Navigating `{sel_pair}`")
                        st.caption(f"{diag_info['country_a']} ↔ {diag_info['country_b']}")
                        if diag_info.get("top_operators"):
                            st.dataframe(pd.DataFrame(diag_info["top_operators"]).rename(columns={"ac_operator": "Airline Operator", "flights_count": "Crossings"}), use_container_width=True)
                        else:
                            st.info("No operator telemetry recorded.")

                    with d_c2:
                        st.markdown(f"#### ⏰ Diurnal Crossing Volume by Hour (UTC)")
                        if diag_info.get("hourly_distribution"):
                            h_df = pd.DataFrame(diag_info["hourly_distribution"])
                            h_chart = (
                                alt.Chart(h_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X("hour_utc:N", title="Hour of Day (UTC)"),
                                    y=alt.Y("flights_count:Q", title="Flight Movements"),
                                    color=alt.Color("origin_fir:N", title="Origin FIR"),
                                    tooltip=["hour_utc", "origin_fir", "flights_count"],
                                )
                                .properties(height=280)
                            )
                            st.altair_chart(h_chart, use_container_width=True)
                        else:
                            st.info("No hourly profile recorded.")

                    st.divider()
                    st.markdown("### 🗺️ Interactive Map: Worst 10 Flights vs. Direct FRA Benchmark")
                    st.markdown(
                        f"Visualizes the 10 flights with the highest lateral routing detours across `{sel_pair}` during {fra_period_sel}. "
                        "Compare each flight's actual radar trajectory (solid color-coded line) "
                        "against its direct geodesic Free Route Airspace benchmark path (dashed emerald green line)."
                    )

                    col_m1, col_m2 = st.columns([1, 1])
                    with col_m1:
                        geo_scope = st.radio(
                            "Flight Distance Scope:",
                            ["All Flights (including Intercontinental)", "Intra-European / Continental (≤ 2,500 NM)"],
                            index=0,
                            key=f"fra_geo_scope_{sel_pair}",
                        )
                    max_gc = 2500.0 if "Intra-European" in geo_scope else None

                    with col_m2:
                        fl_scope = st.radio(
                            "Altitude Stratum:",
                            ["Cruising at or above FL305 (Mandatory FRA Stratum)", "All Flight Levels"],
                            index=0,
                            key=f"fra_fl_scope_{sel_pair}",
                        )
                    min_fl_val = 305.0 if "FL305" in fl_scope else None

                    with st.spinner(f"Querying radar trajectories for top 10 detoured flights across {sel_pair}..."):
                        worst_flts = get_worst_flights_for_boundary(
                            fra_ds,
                            sel_pair,
                            limit=10,
                            max_gc_dist_nm=max_gc,
                            include_trajectories=True,
                            min_fl=min_fl_val,
                        )

                    if not worst_flts:
                        st.info(f"No flight trajectories found crossing {sel_pair} matching the active distance and altitude filters.")
                    else:
                        parts = [p.strip() for p in sel_pair.split("<->")]
                        fir_polys = fra_ds.get_fir_boundaries(parts)
                        fra_map = create_fra_boundary_map(
                            boundary_pair=sel_pair,
                            worst_flights=worst_flts,
                            fir_polygons=fir_polys,
                        )
                        st_folium(fra_map, width=1200, height=520, returned_objects=[])

                        st.markdown("#### 📋 Top 10 Detoured Flights Metric Summary (ECAC Airspace Envelope)")
                        st.caption(
                            "🇪🇺 **ECAC Airspace Envelope:** Free Route Airspace (FRA) is evaluated strictly within European Civil Aviation Conference airspace. "
                            "Flights originating or terminating outside ECAC are assessed from their ECAC Entry Point to their destination or ECAC Exit Point."
                        )
                        summary_rows = []
                        for f in worst_flts:
                            summary_rows.append({
                                "Rank": f"#{f['rank']}",
                                "Flight ID": f["ectrl_id"],
                                "Callsign / Operator": f"{f['ac_operator']} ({f['ac_type']})",
                                "Max FL": f"FL{int(f.get('max_cruise_fl', 0))}" if f.get("max_cruise_fl") else "—",
                                "Route (ECAC Envelope)": f"{f.get('adep_label', f['adep'])} ➔ {f.get('ades_label', f['ades'])}",
                                "Off-Block": f["actual_off_block_time"][:16] if f.get("actual_off_block_time") else "N/A",
                                "ECAC Actual (NM)": f"{f['actual_distance_nm']:,.0f}",
                                "ECAC Direct (NM)": f"{f['great_circle_nm']:,.0f}",
                                "Detour (NM)": f"+{f['detour_nm']:,.1f}",
                                "Detour (%)": f"+{f['detour_pct']:.1f}%",
                                "Excess Fuel": f"{f['excess_fuel_kg']:,.0f} kg ({f['excess_fuel_tonnes']:.2f} t)",
                                "Excess Cost (€ EUR)": f"€{f['excess_cost_eur']:,.2f}",
                                "Excess CO₂": f"{f['excess_co2_tonnes']:.2f} t",
                            })
                        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # ---------------------------------------------------------------------
        # VIEW 5: 2024 vs. 2025 Longitudinal Evolution
        # ---------------------------------------------------------------------
        with fb_tab_evo:
            st.subheader("📈 2024 vs. 2025 Free Route Airspace Evolution Tracker")
            st.markdown(
                "Evaluates multi-period trajectory shifts between October 2024 (`202410`) and October 2025 (`202510`). "
                "Detects which boundaries expanded Free Route Direct (`DCT`) routing versus which interfaces experienced growing congestion."
            )

            with st.spinner("Comparing 2024 vs 2025 boundary routing shifts..."):
                evo_res = compare_fra_boundary_evo(period1="202410", period2="202510", min_flights=fra_min_flts, top_n=25)

            if evo_res.get("comparison_table"):
                evo_df = pd.DataFrame(evo_res["comparison_table"])
                evo_rename = {
                    "boundary_pair": "Boundary Pair",
                    "country_a": "Country A",
                    "country_b": "Country B",
                    "volume_2024": "2024 Flights",
                    "volume_2025": "2025 Flights",
                    "volume_growth_pct": "Growth (%)",
                    "directness_2024_pct": "Directness 2024 (%)",
                    "directness_2025_pct": "Directness 2025 (%)",
                    "directness_delta_pp": "Directness Shift (pp)",
                    "detour_delta_nm": "Detour Shift (NM)",
                    "trend": "Operational Trend",
                }
                st.dataframe(
                    evo_df[[c for c in evo_rename.keys() if c in evo_df.columns]].rename(columns=evo_rename),
                    use_container_width=True,
                )
            else:
                st.info("Insufficient longitudinal data for the selected filter threshold.")

        # ---------------------------------------------------------------------
        # VIEW 6: Long-Term Multi-Horizon Benefit Forecaster (1, 5, 20 Years)
        # ---------------------------------------------------------------------
        with fb_tab_forecast:
            st.subheader("💰 Long-Term Free Route Airspace Benefit Forecaster (1, 5, and 20 Years)")
            st.markdown(
                "Deterministically models and compounds the network-wide benefits of Free Route Airspace (FRA) "
                "harmonization across European airspace boundaries over **1-year, 5-year, and 20-year** investment horizons. "
                "Incorporates empirical multi-season synthesis across **Spring (02), Summer Peak (06), and Autumn (10)**, "
                "and evaluates forward-looking **energy pathways (Optimistic, Median Expected, and Geopolitical Shock pricing)**."
            )

            col_fc0, col_fc1, col_fc2 = st.columns([1, 1.4, 1.6])
            with col_fc0:
                fc_year = st.selectbox(
                    "Baseline Year:",
                    ["2025", "2024"],
                    index=0,
                    key="fra_fc_year_select",
                    help="Eurocontrol multi-season archive baseline year.",
                )
            with col_fc1:
                fc_baseline_mode = st.selectbox(
                    "Baseline Extrapolation Source:",
                    [
                        "🗓️ 12-Month Full Year (202411-202510) [Empirical 12M]",
                        "🌸 Multi-Season Synthesis (Spring, Summer, Autumn)",
                        "⚡ Active Sidebar Period Annualization",
                    ],
                    index=0,
                    key="fra_fc_baseline_mode_select",
                    help="Choose whether to extrapolate from all 12 contiguous empirical monthly archives, 3 seasonal anchor months, or the single active period.",
                )
            with col_fc2:
                fc_scope_label = st.selectbox(
                    "Boundary Implementation Scope:",
                    [
                        "🚨 Prime Candidates Only (Top Choke Points)",
                        "⚠️ Prime + Medium Priority Boundaries",
                        "🌐 All Cross-Border European Boundaries",
                        "🔝 Top 10 Ranked Priority Boundaries",
                        "🔝 Top 25 Ranked Priority Boundaries",
                    ],
                    index=0,
                    key="fra_fc_scope_select",
                )

            col_fc3, col_fc4, col_fc5 = st.columns([1, 1, 1])
            with col_fc3:
                fc_growth = st.slider(
                    "Annual Traffic Growth (%/yr):",
                    min_value=0.0,
                    max_value=6.0,
                    value=2.0,
                    step=0.1,
                    key="fra_fc_growth_slider",
                    help="Eurocontrol STATFOR 20-year baseline projection is ~2.0% annual growth for European civil aviation.",
                )
            with col_fc4:
                fc_realization = st.slider(
                    "Directness Realization (%):",
                    min_value=50,
                    max_value=100,
                    value=85,
                    step=5,
                    key="fra_fc_realization_slider",
                    help="Percentage of theoretical geodesic detour eliminated in practice, factoring in adverse weather avoidance and military reserved airspace (TRA/TSA).",
                )
            with col_fc5:
                fc_fuel_price = st.number_input(
                    "Baseline Jet-A Price (€/kg):",
                    min_value=0.30,
                    max_value=2.00,
                    value=0.6421,
                    step=0.05,
                    format="%.4f",
                    key="fra_fc_fuel_price_input",
                    help="Eurocontrol standard benchmark reference Jet-A fuel price is €0.6421/kg.",
                )

            with st.expander("⛽ 20-Year Jet-A & Sustainable Aviation Fuel (SAF) Price Growth Pathways", expanded=True):
                st.markdown(
                    "Aviation fuel costs over 20 years will escalate due to crude markets, **ReFuelEU SAF blending mandates** "
                    "(2% in 2025, 6% in 2030, 20% in 2035, 70% in 2050), and **EU ETS carbon allowance costs**. "
                    "Configure annual fuel escalation rates across the three economic pathways:"
                )
                col_fp1, col_fp2, col_fp3 = st.columns(3)
                with col_fp1:
                    fc_fuel_opt = st.slider(
                        "🟢 Optimistic Escalation (%/yr):",
                        min_value=0.0,
                        max_value=4.0,
                        value=1.5,
                        step=0.1,
                        key="fra_fc_fuel_opt_slider",
                        help="Clean energy scale-up, declining SAF parity costs, low crude oil volatility.",
                    )
                with col_fp2:
                    fc_fuel_med = st.slider(
                        "🟡 Median Expected Escalation (%/yr):",
                        min_value=1.0,
                        max_value=6.0,
                        value=3.5,
                        step=0.1,
                        key="fra_fc_fuel_med_slider",
                        help="Baseline SESAR / IATA transition pathway: ReFuelEU SAF mandates + rising EU ETS carbon pricing.",
                    )
                with col_fp3:
                    fc_fuel_shock = st.slider(
                        "🔴 Shock Pricing Escalation (%/yr):",
                        min_value=3.0,
                        max_value=12.0,
                        value=6.5,
                        step=0.1,
                        key="fra_fc_fuel_shock_slider",
                        help="Geopolitical disruptions, crude oil embargoes, SAF feedstock shortages, peak carbon penalty surcharges.",
                    )

            scope_map = {
                "🚨 Prime Candidates Only (Top Choke Points)": "prime_only",
                "⚠️ Prime + Medium Priority Boundaries": "prime_and_medium",
                "🌐 All Cross-Border European Boundaries": "all_cross_border",
                "🔝 Top 10 Ranked Priority Boundaries": "top_10",
                "🔝 Top 25 Ranked Priority Boundaries": "top_25",
            }
            sel_scope_key = scope_map.get(fc_scope_label, "prime_only")
            b_mode = "full_year" if "12-Month" in fc_baseline_mode else ("seasonal_synthesis" if "Multi-Season" in fc_baseline_mode else "single_period")

            with st.spinner("Synthesizing baseline profiles & fuel price pathways..."):
                fc_res = calculate_fra_long_term_forecast(
                    boundary_metrics=fra_report.boundaries,
                    dataset=ds,
                    year_baseline=fc_year,
                    baseline_mode=b_mode,
                    use_seasonal_synthesis=(b_mode != "single_period"),
                    annual_growth_rate=fc_growth / 100.0,
                    realization_rate=fc_realization / 100.0,
                    horizons=[1, 5, 20],
                    jet_a_eur_per_kg=fc_fuel_price,
                    fuel_growth_optimistic=fc_fuel_opt / 100.0,
                    fuel_growth_median=fc_fuel_med / 100.0,
                    fuel_growth_shock=fc_fuel_shock / 100.0,
                    scope=sel_scope_key,
                )

            # Official Eurocontrol Network Traffic Benchmark Card
            net_flts = fc_res.get("network_total_flights", 11124247)
            official_yoy = fc_res.get("official_yoy_growth_pct", 4.0)
            arch_cov = fc_res.get("archive_coverage_pct", 92.4)
            ann = fc_res.get("seasonal_breakdown", {}).get("annual_blended", {})
            ann_scoped_crossings = ann.get("annual_crossings", 0)
            crossings_per_flt = (ann_scoped_crossings / max(1, net_flts))

            st.info(
                f"🇪🇺 **Eurocontrol Network Baseline ({fc_year}):** **{net_flts:,} Total Flights** "
                f"(`+{official_yoy:.1f}%` YoY) ➔ Generated **{ann_scoped_crossings:,} Scoped Boundary Crossings** across the {fc_res['total_boundaries_evaluated']} evaluated interfaces "
                f"(averaging **{crossings_per_flt:.2f} crossings per flight** | Baseline Mode: **{fc_res.get('baseline_mode', 'full_year').replace('_', ' ').title()}** | Coverage: **{arch_cov:.1f}%**)."
            )

            if fc_res.get("is_12m_full_year") and fc_res.get("monthly_breakdown"):
                st.success(
                    "🗓️ **Full Contiguous 12-Month Empirical Baseline Active (Nov 2024 – Oct 2025):** "
                    f"Evaluates all 12 consecutive Eurocontrol monthly archives (10,235,149 ingested flights) "
                    f"calibrated against official Eurocontrol traffic (11,061,273 flights) across {fc_res['total_boundaries_evaluated']} scoped boundary interfaces."
                )
                with st.expander("📊 Contiguous 12-Month Empirical Monthly Breakdown (Nov 2024 – Oct 2025)", expanded=True):
                    m_breakdown = fc_res["monthly_breakdown"]
                    m_df = pd.DataFrame(m_breakdown)

                    st.markdown("#### 📊 Monthly Scoped Crossings & Jet-A Conserved Progression (Nov 2024 – Oct 2025)")
                    st.caption(
                        f"Displays verified empirical progression across the 12 contiguous months for the **{fc_res['total_boundaries_evaluated']} scoped boundary interfaces** "
                        f"(Scope: `{fc_scope_label}`). Crossings follow natural European airline seasonality from the winter trough (~402k/mo in Jan–Feb) "
                        f"to the summer peak (~638k/mo in Jul–Aug), calibrated against official Eurocontrol traffic with a verified ~92.5% archive capture rate."
                    )
                    chronological_months = list(m_df["month_name"])
                    c_chart1, c_chart2 = st.columns(2)
                    with c_chart1:
                        st.caption("Monthly Scoped Crossings Across Bottlenecks (Chronological)")
                        chart_crossings = (
                            alt.Chart(m_df)
                            .mark_bar(color="#3498db")
                            .encode(
                                x=alt.X(
                                    "month_name:N",
                                    title="Chronological Month (Nov 2024 – Oct 2025)",
                                    sort=chronological_months,
                                    axis=alt.Axis(labelAngle=-45),
                                ),
                                y=alt.Y(
                                    "crossings_scaled:Q",
                                    title="Calibrated Crossings",
                                    axis=alt.Axis(format="~s"),
                                ),
                                tooltip=[
                                    alt.Tooltip("month_name:N", title="Month"),
                                    alt.Tooltip("season:N", title="Season"),
                                    alt.Tooltip("flights_eurocontrol:Q", title="Eurocontrol Flights", format=","),
                                    alt.Tooltip("flights_archive:Q", title="Archive Flights", format=","),
                                    alt.Tooltip("archive_coverage_pct:Q", title="Coverage", format=".1f"),
                                    alt.Tooltip("crossings_raw:Q", title="Raw Crossings", format=","),
                                    alt.Tooltip("crossings_scaled:Q", title="Calibrated Crossings", format=","),
                                    alt.Tooltip("fuel_saved_tonnes:Q", title="Jet-A Saved (t)", format=",.1f"),
                                    alt.Tooltip("cost_saved_eur:Q", title="Value (€ EUR)", format=",.2f"),
                                ],
                            )
                            .properties(height=280)
                        )
                        st.altair_chart(chart_crossings, use_container_width=True)

                    with c_chart2:
                        st.caption("Monthly Jet-A Fuel Saved in Tonnes (Chronological)")
                        chart_fuel = (
                            alt.Chart(m_df)
                            .mark_line(color="#2ecc71", strokeWidth=2.5, point=alt.OverlayMarkDef(color="#27ae60", size=60))
                            .encode(
                                x=alt.X(
                                    "month_name:N",
                                    title="Chronological Month (Nov 2024 – Oct 2025)",
                                    sort=chronological_months,
                                    axis=alt.Axis(labelAngle=-45),
                                ),
                                y=alt.Y(
                                    "fuel_saved_tonnes:Q",
                                    title="Jet-A Saved (Tonnes)",
                                    axis=alt.Axis(format="~s"),
                                ),
                                tooltip=[
                                    alt.Tooltip("month_name:N", title="Month"),
                                    alt.Tooltip("season:N", title="Season"),
                                    alt.Tooltip("fuel_saved_tonnes:Q", title="Jet-A Saved (t)", format=",.1f"),
                                    alt.Tooltip("co2_mitigated_tonnes:Q", title="CO₂ Mitigated (t)", format=",.1f"),
                                    alt.Tooltip("cost_saved_eur:Q", title="Operating Value (€)", format=",.2f"),
                                    alt.Tooltip("crossings_scaled:Q", title="Calibrated Crossings", format=","),
                                ],
                            )
                            .properties(height=280)
                        )
                        st.altair_chart(chart_fuel, use_container_width=True)

                    rename_cols = {
                        "month_name": "Month",
                        "season": "Season",
                        "flights_eurocontrol": "Eurocontrol Flights",
                        "flights_archive": "Archive Flights",
                        "archive_coverage_pct": "Coverage (%)",
                        "coverage_scale_factor": "Scale Factor",
                        "crossings_raw": "Raw Crossings",
                        "crossings_scaled": "Calibrated Crossings",
                        "fuel_saved_tonnes": "Fuel Saved (t)",
                        "co2_mitigated_tonnes": "CO₂ Mitigated (t)",
                        "cost_saved_eur": "Operating Value (€ EUR)",
                        "detour_deviation_pct": "Mean Detour (%)",
                    }
                    display_m_df = m_df[[c for c in rename_cols.keys() if c in m_df.columns]].rename(columns=rename_cols)
                    st.dataframe(
                        display_m_df.style.format({
                            "Eurocontrol Flights": "{:,}",
                            "Archive Flights": "{:,}",
                            "Coverage (%)": "{:.1f}%",
                            "Scale Factor": "{:.4f}x",
                            "Raw Crossings": "{:,}",
                            "Calibrated Crossings": "{:,}",
                            "Fuel Saved (t)": "{:,.1f}",
                            "CO₂ Mitigated (t)": "{:,.1f}",
                            "Operating Value (€ EUR)": "€{:,.2f}",
                            "Mean Detour (%)": "+{:.2f}%",
                        }),
                        use_container_width=True,
                    )

            if not fc_res.get("is_12m_full_year"):
                with st.expander("📅 3-Anchor Seasonal Synthesis Mapping (How 3 Archives Extrapolate 12 Months)", expanded=False):
                    mb_df = pd.DataFrame(fc_res.get("eurocontrol_monthly_breakdown", []))
                    if not mb_df.empty:
                        mb_rename = {
                            "month_num": "Month #",
                            "month_name": "Month",
                            "official_flights": "Official Eurocontrol Flights",
                            "seasonal_cluster": "Season Category",
                            "anchor_represented": "Representing Archive Anchor",
                        }
                        st.dataframe(
                            mb_df.rename(columns=mb_rename).style.format({"Official Eurocontrol Flights": "{:,}"}),
                            use_container_width=True,
                        )

            # Seasonal Realization Profiles Card
            sb = fc_res.get("seasonal_breakdown", {})
            if sb and "annual_blended" in sb:
                st.markdown(f"### 🌸 Seasonal Boundary Crossing Profiles across {fc_res['total_boundaries_evaluated']} Scoped Interfaces")
                st.caption(
                    f"Quantifies empirical FIR boundary crossings across seasons: "
                    f"Summer peaks with holiday surges and convective detours, Autumn carries high-density business trunk flows, and Spring reflects shoulder operations. "
                    f"Generated by **{net_flts:,} European flights** in {fc_year}."
                )
                sc_sp, sc_su, sc_au, sc_ann = st.columns(4)
                with sc_sp:
                    st.info("#### 🌸 Spring (Shoulder)")
                    st.metric("4-Month Boundary Crossings", f"{sb['spring']['seasonal_crossings']:,}")
                    st.metric("Jet-A Saved", f"{sb['spring']['seasonal_fuel_saved_tonnes']:,.1f} t")
                    st.metric("Seasonal Value", f"€{sb['spring']['seasonal_cost_saved_eur']:,.2f} EUR")
                    st.caption(f"Detour Dev: `+{sb['spring'].get('detour_deviation_pct', 0.0):.2f}%` | CO₂ Mit: `-{sb['spring'].get('co2_mitigation_deviation_pct', 0.0):.2f}%`")

                with sc_su:
                    st.warning("#### ☀️ Summer Peak (Surge)")
                    st.metric("4-Month Boundary Crossings", f"{sb['summer']['seasonal_crossings']:,}")
                    st.metric("Jet-A Saved", f"{sb['summer']['seasonal_fuel_saved_tonnes']:,.1f} t")
                    st.metric("Seasonal Value", f"€{sb['summer']['seasonal_cost_saved_eur']:,.2f} EUR")
                    st.caption(f"Detour Dev: `+{sb['summer'].get('detour_deviation_pct', 0.0):.2f}%` | CO₂ Mit: `-{sb['summer'].get('co2_mitigation_deviation_pct', 0.0):.2f}%`")

                with sc_au:
                    st.info("#### 🍂 Autumn (Trunk)")
                    st.metric("4-Month Boundary Crossings", f"{sb['autumn']['seasonal_crossings']:,}")
                    st.metric("Jet-A Saved", f"{sb['autumn']['seasonal_fuel_saved_tonnes']:,.1f} t")
                    st.metric("Seasonal Value", f"€{sb['autumn']['seasonal_cost_saved_eur']:,.2f} EUR")
                    st.caption(f"Detour Dev: `+{sb['autumn'].get('detour_deviation_pct', 0.0):.2f}%` | CO₂ Mit: `-{sb['autumn'].get('co2_mitigation_deviation_pct', 0.0):.2f}%`")

                with sc_ann:
                    ann = sb.get("annual_blended", {})
                    st.success("#### 🌐 Synthesized Annual")
                    st.metric("Annual Boundary Crossings", f"{ann.get('annual_crossings', 0):,}")
                    st.metric("Annual Jet-A Saved", f"{ann.get('annual_fuel_saved_tonnes', 0):,.1f} t")
                    st.metric("Annual Base Value", f"€{ann.get('annual_cost_saved_eur', 0):,.2f} EUR")
                    st.caption(f"{ann_scoped_crossings/1e6:.2f}M crossings generated across {net_flts/1e6:.2f}M European flights ({fc_year} baseline)")

            st.divider()

            h_data = fc_res["horizons"]
            h1 = h_data.get(1, {})
            h5 = h_data.get(5, {})
            h20 = h_data.get(20, {})

            st.markdown(
                f"##### 🎯 Multi-Horizon Investment Scorecards | Traffic Growth: **{fc_growth:.1f}%/yr** | "
                f"Scope: **{fc_scope_label}** ({fc_res['total_boundaries_evaluated']} Interfaces)"
            )

            # Three Executive Horizon Cards with Fuel Price Scenarios
            c_h1, c_h5, c_h20 = st.columns(3)
            with c_h1:
                st.info("### 📅 1-Year Horizon (Immediate)")
                st.metric("Boundary Crossings", f"{h1.get('cumulative_crossings', 0):,}")
                st.caption(f"Generated across ~{h1.get('cumulative_network_flights', 0)/1e6:.2f}M European flights in Year 1")
                st.metric("Excess Detour Avoided", f"{h1.get('cumulative_detour_nm_avoided', 0):,.0f} NM")
                st.metric("Jet-A Fuel Saved", f"{h1.get('cumulative_fuel_saved_tonnes', 0):,.1f} tonnes")
                st.metric(
                    "Gross CO₂ Mitigated",
                    f"{h1.get('cumulative_co2_mitigated_tonnes', 0):,.1f} tonnes",
                    delta=f"-{h1.get('co2_mitigation_deviation_pct', 0.0):.2f}% crossing deviation avoided",
                    delta_color="inverse",
                )
                st.caption(
                    f"🌱 **ReFuelEU SAF:** **{h1.get('saf_mandate_pct', 0.0):.1f}%** blend "
                    f"(`-{h1.get('lifecycle_carbon_intensity_reduction_pct', 0.0):.1f}%` lifecycle intensity) ➔ "
                    f"**{h1.get('cumulative_net_lifecycle_co2_tonnes', 0):,.1f} t** net lifecycle CO₂."
                )
                st.markdown("**Cumulative Financial Savings Across Fuel Pathways:**")
                st.markdown(f"- 🟢 **Optimistic (+{fc_fuel_opt:.1f}%):** `€{h1.get('savings_optimistic_eur', 0):,.2f} EUR` *(€{h1.get('fuel_price_optimistic_eur', 0):.4f}/kg)*")
                st.markdown(f"- 🟡 **Median (+{fc_fuel_med:.1f}%):** `€{h1.get('savings_median_eur', 0):,.2f} EUR` *(€{h1.get('fuel_price_median_eur', 0):.4f}/kg)*")
                st.markdown(f"- 🔴 **Shock (+{fc_fuel_shock:.1f}%):** `€{h1.get('savings_shock_eur', 0):,.2f} EUR` *(€{h1.get('fuel_price_shock_eur', 0):.4f}/kg)*")

            with c_h5:
                st.warning("### 📅 5-Year Horizon (Medium-Term)")
                st.metric("Boundary Crossings", f"{h5.get('cumulative_crossings', 0):,}")
                st.caption(f"Cumulative crossings generated across ~{h5.get('cumulative_network_flights', 0)/1e6:.1f}M European flights over 5 years")
                st.metric("Excess Detour Avoided", f"{h5.get('cumulative_detour_nm_avoided', 0):,.0f} NM")
                st.metric("Jet-A Fuel Saved", f"{h5.get('cumulative_fuel_saved_tonnes', 0):,.1f} tonnes")
                st.metric(
                    "Gross CO₂ Mitigated",
                    f"{h5.get('cumulative_co2_mitigated_tonnes', 0):,.1f} tonnes",
                    delta=f"-{h5.get('co2_mitigation_deviation_pct', 0.0):.2f}% crossing deviation avoided",
                    delta_color="inverse",
                )
                st.caption(
                    f"🌱 **ReFuelEU SAF:** **{h5.get('saf_mandate_pct', 0.0):.1f}%** blend "
                    f"(`-{h5.get('lifecycle_carbon_intensity_reduction_pct', 0.0):.1f}%` lifecycle intensity) ➔ "
                    f"**{h5.get('cumulative_net_lifecycle_co2_tonnes', 0):,.1f} t** net lifecycle CO₂."
                )
                st.markdown("**Cumulative Financial Savings Across Fuel Pathways:**")
                st.markdown(f"- 🟢 **Optimistic (+{fc_fuel_opt:.1f}%):** `€{h5.get('savings_optimistic_eur', 0):,.2f} EUR` *(€{h5.get('fuel_price_optimistic_eur', 0):.4f}/kg)*")
                st.markdown(f"- 🟡 **Median (+{fc_fuel_med:.1f}%):** `€{h5.get('savings_median_eur', 0):,.2f} EUR` *(€{h5.get('fuel_price_median_eur', 0):.4f}/kg)*")
                st.markdown(f"- 🔴 **Shock (+{fc_fuel_shock:.1f}%):** `€{h5.get('savings_shock_eur', 0):,.2f} EUR` *(€{h5.get('fuel_price_shock_eur', 0):.4f}/kg)*")

            with c_h20:
                st.success("### 📅 20-Year Horizon (Long-Term SESAR)")
                st.metric("Boundary Crossings", f"{h20.get('cumulative_crossings', 0):,}")
                st.caption(f"Cumulative crossings generated across ~{h20.get('cumulative_network_flights', 0)/1e6:.1f}M European flights over 20 years")
                st.metric("Excess Detour Avoided", f"{h20.get('cumulative_detour_nm_avoided', 0):,.0f} NM")
                st.metric("Jet-A Fuel Saved", f"{h20.get('cumulative_fuel_saved_tonnes', 0):,.1f} tonnes")
                st.metric(
                    "Gross CO₂ Mitigated",
                    f"{h20.get('cumulative_co2_mitigated_tonnes', 0):,.1f} tonnes",
                    delta=f"-{h20.get('co2_mitigation_deviation_pct', 0.0):.2f}% crossing deviation avoided",
                    delta_color="inverse",
                )
                st.caption(
                    f"🌱 **ReFuelEU SAF:** **{h20.get('saf_mandate_pct', 0.0):.1f}%** blend "
                    f"(`-{h20.get('lifecycle_carbon_intensity_reduction_pct', 0.0):.1f}%` lifecycle intensity) ➔ "
                    f"**{h20.get('cumulative_net_lifecycle_co2_tonnes', 0):,.1f} t** net lifecycle CO₂."
                )
                st.markdown("**Cumulative Financial Savings Across Fuel Pathways:**")
                st.markdown(f"- 🟢 **Optimistic (+{fc_fuel_opt:.1f}%):** `€{h20.get('savings_optimistic_eur', 0):,.2f} EUR` *(€{h20.get('fuel_price_optimistic_eur', 0):.4f}/kg)*")
                st.markdown(f"- 🟡 **Median (+{fc_fuel_med:.1f}%):** `€{h20.get('savings_median_eur', 0):,.2f} EUR` *(€{h20.get('fuel_price_median_eur', 0):.4f}/kg)*")
                st.markdown(f"- 🔴 **Shock (+{fc_fuel_shock:.1f}%):** `€{h20.get('savings_shock_eur', 0):,.2f} EUR` *(€{h20.get('fuel_price_shock_eur', 0):.4f}/kg)*")

            st.divider()

            # Multi-Scenario Compounding Curves Chart
            st.markdown("#### 📈 Compounding 20-Year Fuel Savings: Energy Price Pathways Comparison")
            prog_df = pd.DataFrame(fc_res["yearly_progression"])
            if not prog_df.empty:
                prog_df["Optimistic (€M)"] = prog_df["cum_savings_optimistic_eur"] / 1_000_000.0
                prog_df["Median Expected (€M)"] = prog_df["cum_savings_median_eur"] / 1_000_000.0
                prog_df["Shock Pricing (€M)"] = prog_df["cum_savings_shock_eur"] / 1_000_000.0
                prog_df["Constant Fuel (€M)"] = prog_df["cum_savings_constant_eur"] / 1_000_000.0
                prog_df["Cumulative CO₂ Mitigated (k tonnes)"] = prog_df["cum_co2_mitigated_tonnes"] / 1_000.0

                ch_c1, ch_c2 = st.columns(2)
                with ch_c1:
                    chart_fuel_scenarios = (
                        alt.Chart(
                            prog_df.melt(
                                id_vars=["year"],
                                value_vars=["Optimistic (€M)", "Median Expected (€M)", "Shock Pricing (€M)", "Constant Fuel (€M)"],
                                var_name="Pricing Pathway",
                                value_name="Cumulative Savings (€M)",
                            )
                        )
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("year:O", title="Timeline (Year)"),
                            y=alt.Y("Cumulative Savings (€M):Q", title="Cumulative Savings (€ Millions EUR)"),
                            color=alt.Color(
                                "Pricing Pathway:N",
                                scale=alt.Scale(
                                    domain=["Optimistic (€M)", "Median Expected (€M)", "Shock Pricing (€M)", "Constant Fuel (€M)"],
                                    range=["#27ae60", "#f39c12", "#c0392b", "#7f8c8d"],
                                ),
                            ),
                            tooltip=["year", "Pricing Pathway", alt.Tooltip("Cumulative Savings (€M):Q", format=",.2f")],
                        )
                        .properties(height=340, title="Cumulative Value Across 20-Year Fuel Price Pathways (€M EUR)")
                    )
                    st.altair_chart(chart_fuel_scenarios, use_container_width=True)

                with ch_c2:
                    chart_co2 = (
                        alt.Chart(prog_df)
                        .mark_area(
                            line={"color": "#2980b9", "width": 2.5},
                            color=alt.Gradient(
                                gradient="linear",
                                stops=[alt.GradientStop(color="#2980b9", offset=0), alt.GradientStop(color="rgba(41, 128, 185, 0.08)", offset=1)],
                                x1=1, x2=1, y1=1, y2=0,
                            ),
                        )
                        .encode(
                            x=alt.X("year:O", title="Timeline (Year)"),
                            y=alt.Y("Cumulative CO₂ Mitigated (k tonnes):Q", title="Cumulative CO₂ Mitigated (Thousand Tonnes)"),
                            tooltip=["year", alt.Tooltip("Cumulative CO₂ Mitigated (k tonnes):Q", format=",.1f"), alt.Tooltip("cum_fuel_saved_tonnes:Q", title="Cumulative Fuel Saved (t)", format=",.1f")],
                        )
                        .properties(height=340, title="Cumulative Environmental CO₂ Mitigation (Thousand Tonnes)")
                    )
                    st.altair_chart(chart_co2, use_container_width=True)

            st.divider()

            # Boundary-by-Boundary Benefit Contribution Table
            st.markdown("#### 📊 Boundary-by-Boundary Long-Term Contribution Breakdown")
            st.caption("Displays empirical seasonal crossing counts (Spring, Summer, Autumn) alongside 1-year, 5-year, and 20-year savings across the three fuel price regimes.")
            contribs = fc_res.get("boundary_contributions", [])
            if contribs:
                contrib_df = pd.DataFrame(contribs)
                c_rename = {
                    "boundary_pair": "Boundary Pair",
                    "countries": "States / Sovereignties",
                    "classification": "Classification",
                    "spring_crossings": "Spring (Feb)",
                    "summer_crossings": "Summer (Jun)",
                    "autumn_crossings": "Autumn (Oct)",
                    "annual_baseline_crossings": "Annual Baseline Flights",
                    "mean_detour_nm": "Mean Detour (NM)",
                    "detour_deviation_pct": "Detour % Dev",
                    "co2_mitigation_deviation_pct": "CO₂ Mit % Dev",
                    "savings_1yr_eur": "1-Year Median (€)",
                    "savings_5yr_eur": "5-Year Median (€)",
                    "savings_20yr_optimistic_eur": "20-Yr Optimistic (€)",
                    "savings_20yr_median_eur": "20-Yr Median (€)",
                    "savings_20yr_shock_eur": "20-Yr Shock (€)",
                    "co2_mitigated_20yr_tonnes": "20-Yr CO₂ (t)",
                }
                disp_df = contrib_df[[c for c in c_rename.keys() if c in contrib_df.columns]].rename(columns=c_rename)
                st.dataframe(
                    disp_df.style.format({
                        "Spring (Feb)": "{:,}",
                        "Summer (Jun)": "{:,}",
                        "Autumn (Oct)": "{:,}",
                        "Annual Baseline Flights": "{:,}",
                        "Mean Detour (NM)": "{:.1f}",
                        "Detour % Dev": "+{:.2f}%",
                        "CO₂ Mit % Dev": "-{:.2f}%",
                        "1-Year Median (€)": "€{:,.2f}",
                        "5-Year Median (€)": "€{:,.2f}",
                        "20-Yr Optimistic (€)": "€{:,.2f}",
                        "20-Yr Median (€)": "€{:,.2f}",
                        "20-Yr Shock (€)": "€{:,.2f}",
                        "20-Yr CO₂ (t)": "{:,.1f}",
                    }),
                    use_container_width=True,
                )

        # ----------------- TAB 2: Trajectory Explorer -----------------

    with sys_sub2:
        st.header("Macro Evolution, FRA Structural Shift, Weather Correlation & AIRAC Diffing")
        render_page_export_bar("macro_reference", "sys_macro")
        st.markdown(
            "Compare two Eurocontrol monthly archives to uncover macro shifts in global air traffic, "
            "airline market expansion, delay patterns, Free Route Airspace (`FRA`) adoption, and fleet modernization."
        )

        col_p1, col_p2 = st.columns(2)
        p1_val = col_p1.selectbox("Baseline Period", available_periods, index=0, key="macro_p1")
        p2_val = col_p2.selectbox("Comparison Period", available_periods, index=min(1, len(available_periods) - 1), key="macro_p2")

        if st.button("Compare Monthly Archives", key="btn_compare_macro"):
            with st.spinner(f"Performing macro comparative analytics {p1_val} vs {p2_val}..."):
                comp = compare_aviation_periods(period1=p1_val, period2=p2_val)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Baseline Flights", f"{comp.total_flights_p1:,}")
            mc2.metric("Comparison Flights", f"{comp.total_flights_p2:,}")
            mc3.metric("Traffic Growth", f"{comp.traffic_growth_pct:+.1f}%")
            mc4.metric("Delay Shift", f"{comp.delay_change_min:+.2f} min")

            st.subheader("Airline Market Growth & Contraction")
            if comp.top_growing_airlines:
                st.dataframe(pd.DataFrame(comp.top_growing_airlines), use_container_width=True)

            st.subheader("Fleet Modernization & Aircraft Family Shifts")
            if comp.fleet_modernization_shifts:
                st.dataframe(pd.DataFrame(comp.fleet_modernization_shifts), use_container_width=True)

            st.subheader("Top Hub Movements Comparison")
            if comp.busiest_airports_comparison:
                st.dataframe(pd.DataFrame(comp.busiest_airports_comparison), use_container_width=True)

        st.divider()
        st.subheader("🛣️ Free Route Airspace (FRA) vs. Fixed ATS Airway Structural Evolution")
        if st.button("Compare FRA Directness Evolution Across Archives", key="btn_fra_evo"):
            with st.spinner(f"Sampling trajectories in {p1_val} and {p2_val} against 86,280 global waypoints & airways..."):
                fra_evo = compare_fra_evolution(period1=p1_val, period2=p2_val, sample_flights=150)
            fe1, fe2, fe3 = st.columns(3)
            fe1.metric(f"{p1_val} FRA Direct Share", f"{fra_evo['period1']['fra_direct_pct']:.1f}%", f"{fra_evo['period1']['fra_dct_waypoints']:,} DCT pts")
            fe2.metric(f"{p2_val} FRA Direct Share", f"{fra_evo['period2']['fra_direct_pct']:.1f}%", f"{fra_evo['period2']['fra_dct_waypoints']:,} DCT pts")
            fe3.metric("FRA Structural Shift", f"{fra_evo['fra_shift_percentage_points']:+.1f} pp")

        st.divider()
        st.subheader("🌦️ European Hub Weather Disruption Correlation (Daily IFR/Wind vs. Arrival Delays)")
        st.caption(f"Correlates historical European aerodrome METAR weather disruptions across the **entire {selected_period} timeframe** with global arrival delays.")

        wx_state_key = f"wx_rep_data_{selected_period}"
        if st.button(f"Analyze Hub Weather vs. Delay Correlation ({selected_period})", key=f"btn_wx_corr_{selected_period}"):
            with st.spinner(f"Correlating IEM ASOS METAR hub weather severity against daily arrival delays for entire {selected_period}..."):
                st.session_state[wx_state_key] = analyze_network_weather_impact(ds, metar_client)

        wx_rep = st.session_state.get(wx_state_key)
        if wx_rep:
            st.info(
                f"📅 **Analyzed Timeframe:** `{wx_rep.start_date}` to `{wx_rep.end_date}` "
                f"({len(wx_rep.daily_time_series)} days analyzed across {len(wx_rep.analyzed_hubs)} hubs: {', '.join(wx_rep.analyzed_hubs)})"
            )
            wx1, wx2, wx3, wx4 = st.columns(4)
            wx1.metric("Clear-Sky Baseline Delay", f"{wx_rep.clear_sky_baseline_delay_min:+.2f} min")
            wx2.metric("Delay / Severity Sensitivity Slope", f"{wx_rep.delay_sensitivity_minutes_per_severity_point:+.2f} min/pt")
            wx3.metric("Delay vs IFR Correlation (r)", f"{wx_rep.pearson_corr_delay_vs_ifr:+.3f}")
            worst_d = wx_rep.worst_weather_day.get("date", "N/A")
            worst_del = wx_rep.worst_weather_day.get("avg_delay_min", 0.0)
            wx4.metric("Worst Weather Day", f"{worst_d}", f"{worst_del:+.1f}m delay")

            if wx_rep.daily_time_series:
                wx_df = pd.DataFrame([r.__dict__ for r in wx_rep.daily_time_series])
                # Interactive daily trend chart: Arrival Delay vs. Weather Severity Index
                if "date_str" in wx_df.columns and "avg_arr_delay_min" in wx_df.columns:
                    base_chart = alt.Chart(wx_df).encode(x=alt.X("date_str:T", title="Date"))
                    line_delay = base_chart.mark_line(color="#e74c3c", strokeWidth=2.5, point=True).encode(
                        y=alt.Y("avg_arr_delay_min:Q", title="Avg Arrival Delay (min)", scale=alt.Scale(zero=False)),
                        tooltip=["date_str:T", "total_flights:Q", "avg_arr_delay_min:Q", "composite_weather_severity_index:Q", "ifr_hours_pct:Q", "high_wind_hours_pct:Q"],
                    )
                    line_sev = base_chart.mark_line(color="#3498db", strokeWidth=2.0, strokeDash=[4, 4], point=True).encode(
                        y=alt.Y("composite_weather_severity_index:Q", title="Weather Severity Index (0-10)", scale=alt.Scale(domain=[0, 10])),
                        tooltip=["date_str:T", "composite_weather_severity_index:Q", "ifr_hours_pct:Q", "high_wind_hours_pct:Q"],
                    )
                    combo_chart = alt.layer(line_delay, line_sev).resolve_scale(y="independent").properties(
                        title=f"Daily Network Arrival Delay (Red Solid) vs. European Hub Weather Severity (Blue Dashed) — {selected_period}",
                        height=280,
                    )
                    st.altair_chart(combo_chart, use_container_width=True)

                st.dataframe(wx_df, use_container_width=True)

        st.divider()

        st.subheader("Aeronautical Configuration Diffing (AIRAC Cycles)")
        st.markdown("Track airway commissionings, decommissionings, and FIR boundary revisions across cycles.")
        col_a1, col_a2 = st.columns(2)
        ac_c1 = col_a1.text_input("Cycle 1", "2509", key="airac_c1")
        ac_c2 = col_a2.text_input("Cycle 2", "2510", key="airac_c2")

        if st.button("Diff AIRAC Configurations", key="btn_airac_diff"):
            with st.spinner(f"Diffing AIRAC {ac_c1} ➔ {ac_c2}..."):
                diff_rep = compare_airac_cycles(cycle1=ac_c1, cycle2=ac_c2)

            rc = diff_rep.route_changes
            fc = diff_rep.fir_changes

            ad1, ad2, ad3, ad4 = st.columns(4)
            ad1.metric("Added Airways", f"+{rc.added_routes_count}")
            ad2.metric("Deleted Airways", f"-{rc.deleted_routes_count}")
            ad3.metric("Modified Airways", f"~{rc.modified_routes_count}")
            ad4.metric("Re-profiled FIRs", f"~{fc.modified_boundaries_count}")

            st.json(diff_rep.to_dict())

        # ----------------- TAB 10: Methodology & KPI Reference -----------------

# =============================================================================
# 3. LONGITUDINAL & PREDICTIVE TRENDS
# =============================================================================
with tab_cat_longitudinal:
    st.caption(
        "Multi-period airspace network intelligence, longitudinal historical evolution, Year-over-Year (YoY) benchmarking, "
        "annual 3-season surge decomposition, predictive sector saturation forecasting, and airline economics."
    )
    render_page_export_bar("longitudinal_trends", "cat_long")
    long_sub1, long_sub2, long_sub3, long_sub4 = st.tabs(
        [
            "📊 Multi-Period Timeline & Fleet Evolution",
            "🔄 Year-over-Year (YoY) & Seasonality",
            "🔮 Predictive Airspace Saturation Forecaster",
            "💶 Airline Economics & Delay Contagion",
        ]
    )

    # ----------------- SUB-TAB 1: Multi-Period Timeline & Fleet Evolution -----------------
    with long_sub1:
        st.header("📊 Multi-Period Longitudinal Airspace Timeline & Fleet Evolution")
        st.markdown(
            "Synthesizes cross-sectional operational metrics, daily traffic patterns, mean arrival delays, "
            "environmental CO₂ footprints, and the market penetration of new-generation eco-efficient aircraft "
            "(A320neo, A321neo, B737 MAX, A220/BCS3, A350, B787) across historical archives."
        )

        @st.cache_data(show_spinner=False)
        def _get_cached_longitudinal_df(cohort: str = "all"):
            eng = LongitudinalEngine()
            return eng.get_timeline_dataframe(cohort=cohort)

        col_scope_sel, col_scope_info = st.columns([2, 3])
        with col_scope_sel:
            cohort_choice = st.radio(
                "Cohort Scope",
                ["✈️ Contemporary YoY Cohort (2024–2025)", "🏛️ All Archives (Include 2016 Baseline)"],
                horizontal=True,
                key="tl_cohort_choice",
            )
        is_yoy_only = "Contemporary" in cohort_choice
        with col_scope_info:
            if is_yoy_only:
                st.info("💡 **YoY Continuous Processing**: Showing the balanced 2024–2025 multi-season archives. The 2016 archive is an isolated 9-year historical benchmark.")
            else:
                st.caption("Showing all indexed historical periods including the 2016 historical reference.")

        with st.spinner("Compiling longitudinal timeline..."):
            raw_tl_df = _get_cached_longitudinal_df(cohort="yoy" if is_yoy_only else "all")

        if raw_tl_df.empty:
            st.warning("No longitudinal archive periods currently detected in database.")
        else:
            t_df = raw_tl_df.copy()
            col_tl1, col_tl2, col_tl3, col_tl4 = st.columns(4)
            col_tl1.metric("Indexed Periods", f"{len(t_df)} archives", "2024 – 2025" if is_yoy_only else "2016 – 2025")
            latest_row = t_df.iloc[-1]
            earliest_row = t_df.iloc[0]
            col_tl2.metric(
                "Fleet Modernization",
                f"{latest_row['newgen_share_pct']:.1f}% Neo/MAX",
                f"+{latest_row['newgen_share_pct'] - earliest_row['newgen_share_pct']:.1f} pp vs {earliest_row['period_id']}",
            )
            peak_row = t_df.loc[t_df['daily_flights'].idxmax()]
            col_tl3.metric(
                "Peak Observed Daily Traffic",
                f"{peak_row['daily_flights']:,.0f} /day",
                f"{peak_row['period_id']} ({peak_row['season']})",
            )
            col_tl4.metric("Total Indexed Flights", f"{t_df['total_flights'].sum():,}")

            st.subheader("Historical Timeline Summary Table")
            display_tl_df = t_df.rename(
                columns={
                    "period_id": "Period",
                    "year": "Year",
                    "month": "Month",
                    "season": "Season",
                    "days": "Days",
                    "total_flights": "Total Flights",
                    "daily_flights": "Daily Flights",
                    "avg_arr_delay_min": "Avg Arr Delay (min)",
                    "avg_dep_delay_min": "Avg Dep Delay (min)",
                    "otp_15_pct": "OTP 15 (%)",
                    "avg_distance_nm": "Avg Distance (nm)",
                    "avg_air_time_min": "Avg Air Time (min)",
                    "newgen_share_pct": "Neo/MAX Share (%)",
                    "cohort_type": "Cohort",
                }
            )
            st.dataframe(display_tl_df, use_container_width=True)

            st.subheader("Key Longitudinal Trajectories")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Daily Flight Volume Progression**")
                vol_chart = t_df.set_index("period_id")[["daily_flights"]].rename(columns={"daily_flights": "Daily Flights"})
                st.line_chart(vol_chart)
            with c2:
                st.markdown("**New-Gen Eco Fleet Penetration (% A320neo, B737 MAX, A350, B787)**")
                fleet_chart = t_df.set_index("period_id")[["newgen_share_pct"]].rename(columns={"newgen_share_pct": "Neo/MAX Share (%)"})
                st.line_chart(fleet_chart)

            c3, c4 = st.columns(2)
            with c3:
                st.markdown("**Average Arrival Delay Trend (min)**")
                delay_chart = t_df.set_index("period_id")[["avg_arr_delay_min"]].rename(columns={"avg_arr_delay_min": "Avg Arr Delay (min)"})
                st.line_chart(delay_chart)
            with c4:
                st.markdown("**Mean Flown Distance per Flight (NM)**")
                dist_chart = t_df.set_index("period_id")[["avg_distance_nm"]].rename(columns={"avg_distance_nm": "Avg Distance (nm)"})
                st.line_chart(dist_chart)

    # ----------------- SUB-TAB 2: Year-over-Year (YoY) & Seasonality -----------------
    with long_sub2:
        st.header("🔄 Year-over-Year (YoY) Benchmarking & Seasonal Demand Surge Dynamics")
        st.markdown(
            "Performs rigorous pairwise delta benchmarking across identical historical months (e.g. October 2024 vs October 2025) "
            "and decomposes annual seasonal elasticity between winter troughs, summer peaks, and autumn shoulders."
        )

        st.subheader("1. Pairwise Year-over-Year (YoY) Analyzer")
        eng_inst = LongitudinalEngine()
        yoy_pairs = eng_inst.get_yoy_matching_pairs()
        yoy_periods_only = eng_inst.get_yoy_capable_periods()

        mode_col, pair_col = st.columns([1, 2])
        pair_mode = mode_col.radio("Selection Mode", ["Strict Seasonal Matching Pairs (Recommended)", "Custom Periods"], key="yoy_pair_mode")

        if pair_mode == "Strict Seasonal Matching Pairs (Recommended)" and yoy_pairs:
            pair_options = [p["label"] for p in yoy_pairs]
            chosen_pair_idx = pair_col.selectbox("Select Seasonally Matched YoY Pair", range(len(yoy_pairs)), format_func=lambda i: pair_options[i], index=len(yoy_pairs)-1 if len(yoy_pairs)>0 else 0, key="yoy_matched_pair_sel")
            base_p = yoy_pairs[chosen_pair_idx]["baseline"]
            targ_p = yoy_pairs[chosen_pair_idx]["target"]
            st.caption(f"Analyzing identical month across consecutive years: `{base_p}` ➔ `{targ_p}` ({yoy_pairs[chosen_pair_idx]['season']})")
        else:
            include_hist = pair_col.checkbox("Include 2016 Historical Baseline in selector", value=False, key="yoy_inc_hist")
            all_selectable = (yoy_periods_only + ["201611"]) if include_hist else yoy_periods_only
            y_col1, y_col2 = st.columns(2)
            base_p = y_col1.selectbox("Baseline Period (Prior Year)", all_selectable, index=all_selectable.index("202410") if "202410" in all_selectable else 0, key="yoy_base")
            targ_p = y_col2.selectbox("Target Period (Recent Year)", all_selectable, index=all_selectable.index("202510") if "202510" in all_selectable else len(all_selectable)-1, key="yoy_targ")

        if st.button("Compute Year-over-Year Delta Analysis", key="btn_run_yoy"):
            with st.spinner(f"Analyzing delta between {base_p} and {targ_p}..."):
                eng = LongitudinalEngine()
                yoy = eng.compare_yoy(base_p, targ_p)

            if not yoy.is_matched_season:
                st.warning(f"⚠️ Note: {base_p} (Month {base_p[4:6]}) and {targ_p} (Month {targ_p[4:6]}) represent different seasonal months. Results reflect seasonal cross-effects in addition to annual shifts.")
            elif yoy.year_gap != 1:
                st.info(f"🏛️ Long-Term Historical Era Comparison: Spanning a {yoy.year_gap}-year delta ({base_p[:4]} ➔ {targ_p[:4]}).")

            ym1, ym2, ym3, ym4 = st.columns(4)
            ym1.metric("Daily Flights Delta", f"{yoy.daily_flights_target:,.0f}/day", f"{yoy.daily_flights_growth_pct:+.2f}% ({yoy.daily_flights_delta:+.0f}/day)")
            ym2.metric("Mean Arrival Delay Delta", f"{yoy.avg_arr_delay_target_min:+.1f} min", f"{yoy.arr_delay_delta_min:+.1f} min")
            ym3.metric("Neo/MAX Fleet Penetration", f"{yoy.newgen_share_target_pct:.2f}%", f"{yoy.newgen_share_delta_pct:+.2f} percentage points")
            ym4.metric("Total Monthly Movements", f"{yoy.total_flights_target:,}", f"{yoy.total_flights_growth_pct:+.2f}% ({yoy.total_flights_delta:+,})")

            y_t1, y_t2 = st.columns(2)
            with y_t1:
                st.markdown("#### Top Expanding Airline Operators")
                st.dataframe(pd.DataFrame(yoy.top_growing_airlines), use_container_width=True)
            with y_t2:
                st.markdown("#### Aircraft Type Modernization Shifts")
                st.dataframe(pd.DataFrame(yoy.top_aircraft_type_shifts), use_container_width=True)

            if yoy.top_city_pairs_evolution:
                st.markdown("#### Top City Pair Corridors Evolution")
                st.dataframe(pd.DataFrame(yoy.top_city_pairs_evolution), use_container_width=True)

        st.markdown("---")
        st.subheader("2. Annual Seasonal Decomposition (Winter ➔ Summer ➔ Autumn)")
        s_col1, s_col2 = st.columns([1, 2])
        target_year = s_col1.selectbox("Select Seasonality Year", [2024, 2025], index=0, key="seas_year")
        if s_col1.button("Decompose Annual Seasonality", key="btn_seas"):
            with st.spinner(f"Decomposing {target_year} seasonal elasticity..."):
                eng = LongitudinalEngine()
                decomp = eng.decompose_seasonality(int(target_year))

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Winter Trough (Feb)", f"{decomp.winter_daily_flights:,.0f} /day")
            sm2.metric("Summer Peak (Jun)", f"{decomp.summer_daily_flights:,.0f} /day", f"{decomp.summer_surge_pct:+.1f}% vs Feb")
            sm3.metric("Autumn Shoulder (Oct)", f"{decomp.autumn_daily_flights:,.0f} /day", f"{decomp.autumn_taper_pct:+.1f}% vs Jun")
            sm4.metric("Delay Surge Elasticity", f"{decomp.delay_surge_ratio:.2f}x", f"Delay jumped {decomp.summer_delay_surge_pct:+.1f}%")

            st.info(
                f"**Seasonal Elasticity Finding ({target_year}):** In {target_year}, traffic surged by "
                f"**+{decomp.summer_surge_pct:.1f}%** from winter trough to summer peak, but network arrival delays surged by "
                f"**+{decomp.summer_delay_surge_pct:.1f}%** (a **{decomp.delay_surge_ratio:.2f}x elasticity**), "
                f"demonstrating non-linear airspace congestion escalation when capacity buffers saturate."
            )

            col_s_dest, col_s_fir = st.columns(2)
            with col_s_dest:
                st.markdown("#### Top Summer Leisure Surge Destination Airports")
                st.dataframe(pd.DataFrame(decomp.top_summer_surge_airports), use_container_width=True)
            with col_s_fir:
                st.markdown("#### Top Summer Surge Airspace Sectors / FIRs")
                st.dataframe(pd.DataFrame(decomp.top_summer_surge_firs), use_container_width=True)

    # ----------------- SUB-TAB 3: Predictive Airspace Saturation Forecaster -----------------
    with long_sub3:
        st.header("🔮 Predictive Airspace Saturation & Bottleneck Forecaster")
        st.markdown(
            "Extrapolates longitudinal traffic growth and fleet mix trajectories to project future European airspace saturation, "
            "identifying sectors and spatial cells projected to breach safe controller workload and hourly capacity thresholds."
        )

        fc_c1, fc_c2, fc_c3, fc_c4 = st.columns(4)
        fc_year = fc_c1.selectbox("Forecast Horizon Year", [2026, 2027, 2028], index=0, key="fc_year")
        fc_season = fc_c2.selectbox("Target Season", ["summer", "autumn", "winter"], index=0, key="fc_season")
        fc_growth = fc_c3.slider("Annual Compound Traffic Growth Rate (%)", min_value=1.0, max_value=8.0, value=3.5, step=0.5, key="fc_growth")
        fc_grid = fc_c4.selectbox("Spatial Grid Size Resolution", [0.5, 1.0, 2.0, 5.0], index=1, format_func=lambda x: f"{x:.1f}° × {x:.1f}°", key="fc_grid")

        if st.button("Generate Airspace Bottleneck Forecast", key="btn_run_forecast"):
            with st.spinner(f"Simulating forward airspace demand for {fc_season.capitalize()} {fc_year} at +{fc_growth}% annual growth..."):
                forecaster = AirspaceForecaster()
                report = forecaster.generate_comprehensive_forecast(
                    forecast_year=int(fc_year),
                    season=str(fc_season),
                    growth_scenario_pct=float(fc_growth),
                    grid_size_deg=float(fc_grid),
                )

            fm1, fm2, fm3, fm4 = st.columns(4)
            fm1.metric("Forecast Scope", report.forecast_label)
            fm2.metric("Sectors Analyzed", f"{report.total_sectors_analyzed} FIR sectors", f"Base: {report.baseline_period}")
            fm3.metric("Capacity Breaches", f"{report.breached_sectors_count} Critical / {report.severe_sectors_count} Severe", f"{report.elevated_sectors_count} Elevated")
            fm4.metric("Network Risk Index", f"{report.network_atfm_risk_index:.1f} / 10.0", f"Res: {fc_grid}°")

            st.subheader("Projected Airspace Sector Capacity Breaches")
            if not report.top_capacity_breaches:
                st.success("No airspace sectors projected to breach hourly capacity threshold under specified growth parameters.")
            else:
                b_df = pd.DataFrame([
                    {
                        "Sector / FIR": b.sector_id,
                        "Nominal Hourly Capacity": b.nominal_capacity_hourly,
                        "Baseline Peak (flt/hr)": b.historical_peak_hourly_baseline,
                        "Target Peak (flt/hr)": b.historical_peak_hourly_target,
                        "Historical Growth (%)": f"{b.historical_growth_pct:+.1f}%",
                        "Projected Peak (flt/hr)": round(b.projected_peak_hourly, 1),
                        "Projected Saturation (%)": f"{b.projected_saturation_pct:.1f}%",
                        "Breach Level": b.breach_level,
                        "Estimated Daily Delay (min)": b.estimated_daily_atfm_delay_min,
                        "Mitigation Strategy": b.mitigation_recommendation,
                    }
                    for b in report.top_capacity_breaches
                ])
                st.dataframe(b_df, use_container_width=True)

            st.subheader(f"Top Congested Spatial Cells ({fc_grid}° × {fc_grid}° Resolution)")
            grid_df = pd.DataFrame([
                {
                    "Cell Coordinates (Lat/Lon)": f"{c.cell_lat:.1f}°, {c.cell_lon:.1f}°",
                    "Projected Flights": round(c.projected_flights, 1),
                    "Peak Hourly Flights": c.peak_hourly_flights,
                    "Baseline Flights": c.flights_baseline,
                    "Target Flights": c.flights_target,
                    "Growth (%)": f"{c.historical_growth_pct:+.1f}%",
                    "Complexity Score": c.complexity_score,
                }
                for c in report.top_grid_hotspots
            ])
            st.dataframe(grid_df, use_container_width=True)

    # ----------------- SUB-TAB 4: Airline Economics & Delay Contagion -----------------
    with long_sub4:
        st.header("💶 Airline Economics, Route Charge Avoidance & Airframe Delay Contagion")
        st.markdown(
            "Evaluates multi-objective airline flight operating cost trade-offs, detects detour trajectories flown to circumvent "
            "high CRCO route charge unit rates, and tracks sequential airframe delay contagion across daily aircraft rotations."
        )

        col_ec_left, col_ec_right = st.columns(2)

        with col_ec_left:
            st.subheader("1. Direct Operating Cost (DOC) Flight Evaluator")
            st.markdown(
                "Decomposes Total DOC into Fuel Burn (€), Crew & Airframe Time Cost (€), and Eurocontrol CRCO Navigation Fees (€)."
            )
            doc_flt_id = st.text_input("Eurocontrol Flight ID", value=str(active_flight_id), key="doc_flt_id").strip()
            if st.button("Evaluate Flight Operating Cost (€)", key="btn_doc_eval"):
                with st.spinner(f"Evaluating DOC for flight {doc_flt_id}..."):
                    econ_eng = AirlineEconomicsEngine(ds, period=selected_period)
                    try:
                        doc = econ_eng.evaluate_flight_doc(int(doc_flt_id))
                        if doc:
                            dm1, dm2, dm3, dm4 = st.columns(4)
                            dm1.metric("Total DOC", f"€{doc.total_doc_eur:,.2f}")
                            dm2.metric("Fuel Cost", f"€{doc.fuel_cost_eur:,.2f}", f"{doc.fuel_cost_share_pct:.1f}%")
                            dm3.metric("Crew/Time Cost", f"€{doc.time_crew_cost_eur:,.2f}", f"{doc.time_cost_share_pct:.1f}%")
                            dm4.metric("CRCO Route Charges", f"€{doc.crco_navigation_charges_eur:,.2f}", f"{doc.crco_cost_share_pct:.1f}%")

                            st.write(f"**Route:** `{doc.route}` | **Operator:** `{doc.ac_operator}` | **Type:** `{doc.ac_type}`")
                            st.write(f"**Flown Distance:** `{doc.distance_nm:.1f} nm` | **Air Time:** `{doc.flight_duration_min:.1f} min` | **Fuel Burn:** `{doc.fuel_burn_kg:,.1f} kg`")
                        else:
                            st.warning(f"Could not calculate DOC for flight {doc_flt_id}.")
                    except Exception as e:
                        st.error(f"Error evaluating DOC for flight {doc_flt_id}: {e}")

            st.markdown("---")
            st.subheader("2. Network Route Charge Avoidance Scanner")
            st.markdown("Identifies high-frequency city-pair corridors where flights fly detours to avoid high navigation rate FIRs.")
            min_c_flts = st.slider("Minimum Corridor Flights", min_value=20, max_value=200, value=50, step=10, key="avoid_min_flts")
            if st.button("Scan Avoidance Corridors", key="btn_scan_avoid"):
                with st.spinner("Scanning network corridors for route charge avoidance detours..."):
                    econ_eng = AirlineEconomicsEngine(ds, period=selected_period)
                    avoid_corridors = econ_eng.scan_network_route_charge_avoidance(min_corridor_flights=int(min_c_flts), top_n=5)

                if not avoid_corridors:
                    st.info("No significant route avoidance corridors identified with current criteria.")
                else:
                    av_data = [
                        {
                            "Corridor": c.route,
                            "Flights Analyzed": c.total_flights_analyzed,
                            "Avoidance Flights": c.avoidance_flights_count,
                            "Avoidance Share (%)": f"{c.avoidance_share_pct:.1f}%",
                            "Mean Detour (nm)": f"+{c.mean_extra_distance_nm:.1f} nm",
                            "Mean CRCO Saved (€)": f"€{c.mean_crco_savings_eur:,.2f}",
                            "Extra Fuel (kg)": f"+{c.mean_extra_fuel_kg:.1f} kg",
                            "Net Cost Benefit (€)": f"€{c.net_economic_benefit_eur:+,.2f}",
                            "Avoided FIR": c.avoided_fir or "None",
                            "Preferred Cheaper FIR": c.preferred_lower_fee_fir or "None",
                        }
                        for c in avoid_corridors
                    ]
                    st.dataframe(pd.DataFrame(av_data), use_container_width=True)

        with col_ec_right:
            st.subheader("3. Single European Sky (SES) Rate Harmonization Simulator")
            st.markdown(
                "Simulates counterfactual policy impact if Eurocontrol instituted a single flat unit rate across all European airspace."
            )
            harm_rate = st.slider("Harmonized European Unit Rate (€ / Service Unit)", min_value=50.0, max_value=100.0, value=72.50, step=2.5, key="harm_rate")
            if st.button("Simulate SES Policy Harmonization", key="btn_sim_ses"):
                with st.spinner("Simulating European rate harmonization impact..."):
                    econ_eng = AirlineEconomicsEngine(ds, period=selected_period)
                    ses_res = econ_eng.simulate_harmonized_eurocontrol_rate(harmonized_rate_eur=float(harm_rate), sample_size=100)

                sm_c1, sm_c2 = st.columns(2)
                sm_c1.metric("Eliminated Detour Flights", f"{ses_res.estimated_annual_avoidance_flights_eliminated:,} flights/yr")
                sm_c2.metric("Annual Jet-A Fuel Saved", f"{ses_res.estimated_annual_fuel_saved_tonnes:,.0f} tonnes")
                sm_c3, sm_c4 = st.columns(2)
                sm_c3.metric("Annual CO₂ Emissions Mitigated", f"{ses_res.estimated_annual_co2_saved_tonnes:,.0f} tonnes")
                sm_c4.metric("Airlines Net Annual Saving", f"€{ses_res.estimated_annual_airline_cost_saved_eur:,.0f} EUR")

                st.success(
                    f"**SES Policy Implication:** Flat rate of €{harm_rate:.2f}/SU eliminates detour flight incentives, "
                    f"returning direct traffic to central European airspace while mitigating ~{ses_res.estimated_annual_co2_saved_tonnes:,.0f} tonnes of CO₂ annually."
                )

            st.markdown("---")
            st.subheader("4. Airframe Delay Contagion & Repeat Offender Analyzer")
            p_start_date, p_end_date = get_dataset_date_bounds(ds)
            st.markdown(
                f"Detects morning departures whose delays trigger downstream cascading infections along aircraft rotations, "
                f"and identifies chronic repeat offender routes and operators causing systemic network delay propagation.\n\n"
                f"🗓️ **Analyzed Reporting Period:** `{selected_period}` | **Coverage Window:** **{p_start_date}** to **{p_end_date}**"
            )

            contagion_scope = st.radio(
                "Contagion Analysis Scope",
                ["🔁 Period-Wide Repeat Offenders (Routes & Operators)", "📅 Single-Date Super-Spreaders"],
                horizontal=True,
                key="contagion_scope_radio",
            )

            if contagion_scope == "🔁 Period-Wide Repeat Offenders (Routes & Operators)":
                offender_sub_choice = st.radio(
                    "Target Dimension",
                    ["🛣️ Repeat Offender Routes (City Pairs)", "✈️ Repeat Offender Airline Operators"],
                    horizontal=True,
                    key="offender_dim_radio",
                )
                st.info(f"📊 **Period-Wide Scan Scope:** Analyzing all sequential airframe rotations between **{p_start_date}** and **{p_end_date}** (Period `{selected_period}`)")
                col_of1, col_of2 = st.columns(2)
                min_pri_del = col_of1.slider("Min Primary Departure Delay (min)", min_value=15, max_value=60, value=25, key="of_min_pri_del")
                min_prop_del = col_of2.slider("Min Propagated Reactionary Delay (min)", min_value=5, max_value=30, value=10, key="of_min_prop_del")

                if st.button("Scan Reporting Period for Repeat Offenders", key="btn_scan_offenders"):
                    with st.spinner(f"Scanning archive {selected_period} ({p_start_date} to {p_end_date}) for repeat contagion sources..."):
                        delay_eng = DelayContagionGraphEngine(ds, period=selected_period)
                        if "Routes" in offender_sub_choice:
                            off_routes = delay_eng.scan_repeat_offender_routes(
                                min_primary_delay_min=float(min_pri_del),
                                min_propagated_delay_min=float(min_prop_del),
                                min_incidents=3,
                                top_n=20,
                            )
                            if not off_routes:
                                st.info(f"No repeat offender routes found between {p_start_date} and {p_end_date} with current criteria.")
                            else:
                                st.success(f"**Period Scope:** Analyzed all rotations from **{p_start_date}** to **{p_end_date}** (Period `{selected_period}`).")
                                om0, om1, om2, om3 = st.columns(4)
                                om0.metric("Analysis Period", f"{selected_period}", f"{p_start_date} to {p_end_date}")
                                om1.metric("Worst Offender Route", off_routes[0].route, f"{off_routes[0].incident_count} delay infection days")
                                om2.metric("Max Reactionary Delay", f"{off_routes[0].total_reactionary_delay_hours:.1f} hrs", f"Avg Leg 1 Delay: {off_routes[0].avg_primary_dep_delay_min:.0f}m")
                                om3.metric("Max Financial Damage", f"€{off_routes[0].total_reactionary_cost_eur:,.0f} EUR", f"{off_routes[0].total_downstream_legs_infected} legs infected")

                                r_df = pd.DataFrame([
                                    {
                                        "Route": r.route,
                                        "Origin": r.adep,
                                        "Destination": r.ades,
                                        "Contagion Days": r.incident_count,
                                        "Avg Initial Delay": f"{r.avg_primary_dep_delay_min:.1f} min",
                                        "Downstream Legs Infected": r.total_downstream_legs_infected,
                                        "Reactionary Delay (hrs)": f"{r.total_reactionary_delay_hours:.1f} hrs",
                                        "Reactionary Cost (€)": f"€{r.total_reactionary_cost_eur:,.0f}",
                                        "Period Window": f"{p_start_date} to {p_end_date}",
                                    }
                                    for r in off_routes
                                ])
                                st.dataframe(r_df, use_container_width=True)
                        else:
                            off_ops = delay_eng.scan_repeat_offender_operators(
                                min_primary_delay_min=float(min_pri_del),
                                min_propagated_delay_min=float(min_prop_del),
                                min_spreader_flights=5,
                                top_n=20,
                            )
                            if not off_ops:
                                st.info(f"No repeat offender operators found between {p_start_date} and {p_end_date} with current criteria.")
                            else:
                                st.success(f"**Period Scope:** Analyzed all carrier rotations from **{p_start_date}** to **{p_end_date}** (Period `{selected_period}`).")
                                om0, om1, om2, om3 = st.columns(4)
                                om0.metric("Analysis Period", f"{selected_period}", f"{p_start_date} to {p_end_date}")
                                om1.metric("Highest Contagion Carrier", off_ops[0].operator, f"{off_ops[0].total_root_spreader_flights} spreader flights")
                                om2.metric("Total Downstream Delay", f"{off_ops[0].total_reactionary_delay_hours:,.1f} hrs", f"{off_ops[0].total_downstream_legs_infected} legs infected")
                                om3.metric("Total Reactionary Cost", f"€{off_ops[0].total_reactionary_cost_eur:,.0f} EUR", f"Multiplier: {off_ops[0].downstream_infection_multiplier:.2f}x")

                                op_df = pd.DataFrame([
                                    {
                                        "Airline Operator": o.operator,
                                        "Root Spreader Flights": o.total_root_spreader_flights,
                                        "Avg Primary Delay": f"{o.avg_primary_dep_delay_min:.1f} min",
                                        "Downstream Legs Infected": o.total_downstream_legs_infected,
                                        "Infection Multiplier": f"{o.downstream_infection_multiplier:.2f}x",
                                        "Reactionary Delay (hrs)": f"{o.total_reactionary_delay_hours:.1f} hrs",
                                        "Reactionary Cost (€)": f"€{o.total_reactionary_cost_eur:,.0f}",
                                        "Period Window": f"{p_start_date} to {p_end_date}",
                                    }
                                    for o in off_ops
                                ])
                                st.dataframe(op_df, use_container_width=True)

            else:
                col_sp_dt, col_sp_del = st.columns(2)
                default_date = p_start_date if p_start_date else "2025-10-15"
                sp_date = col_sp_dt.text_input(f"Operating Date (within {p_start_date} to {p_end_date})", value=default_date, key="sp_date").strip()
                sp_min_del = col_sp_del.slider("Min Primary Departure Delay (min)", min_value=10, max_value=60, value=25, key="sp_min_del")

                if st.button("Detect Delay Super-Spreaders", key="btn_find_spreaders"):
                    with st.spinner(f"Tracing airframe delay contagion for {sp_date}..."):
                        delay_eng = DelayContagionGraphEngine(ds, period=selected_period)
                        spreaders = delay_eng.identify_super_spreaders(date_str=sp_date, min_primary_delay_min=float(sp_min_del), top_n=15)

                    if not spreaders:
                        st.info(f"No multi-leg super-spreaders found for date {sp_date} with >={sp_min_del}m departure delay.")
                    else:
                        st.success(f"**Date Analyzed:** {sp_date} (within period `{selected_period}` coverage: **{p_start_date}** to **{p_end_date}**)")
                        sp_df = pd.DataFrame([
                            {
                                "Root Flight ID": s.ectrl_id,
                                "Tail": s.ac_reg,
                                "Carrier": s.operator,
                                "Origin Hub": s.origin_hub,
                                "Destination": s.destination,
                                "Initial Departure Delay": f"{s.primary_delay_min:+.0f} min",
                                "Infected Downstream Legs": s.infected_downstream_legs,
                                "Downstream Reactionary Delay": f"{s.total_downstream_delay_min:.0f} min",
                                "Reactionary Delay Cost (€)": f"€{s.reactionary_delay_cost_eur:,.0f}",
                                "Period Analyzed": f"{p_start_date} to {p_end_date}",
                            }
                            for s in spreaders[:15]
                        ])
                        st.dataframe(sp_df, use_container_width=True)
                        st.dataframe(sp_df, use_container_width=True)


# =============================================================================
# 4. WHAT-IF SCENARIO VIEWS
# =============================================================================
with tab_cat_whatif:
    st.caption(
        "Prescriptive operations research and counterfactual airspace stress testing: turnaround buffers, "
        "crosswind runway collapse, fleet infection fragility, hub tail swaps, 4D co-altitude proximity conflicts, and GDP arrival slots."
    )
    render_page_export_bar("what_if_scenarios", "cat_whatif")
    st.header("Counterfactual Simulations, Airspace Conflict Scanner & Operations Research")
    render_page_export_bar("what_if_stress_tests", "tab8")

    st.subheader("Scenario A: Turnaround Buffer Scheduling Optimization")
    st.markdown(
        "Simulate how scheduling extra ground buffer prevents downstream reactionary delays across multiple airline fleets. "
        "Compare operator delay reduction curves across buffer sizes (5m to 25m) side-by-side on a single graph."
    )
    col_a_ops, col_a_buf = st.columns([2, 1])
    with col_a_ops:
        selected_turn_ops = st.multiselect(
            "Select Operators to Compare",
            options=["RYR", "DLH", "BAW", "EZY", "AFR", "KLM", "SAS", "THY"],
            default=["RYR", "DLH", "BAW", "EZY"],
            key=f"turn_ops_{selected_period}",
        )
    with col_a_buf:
        extra_buffer = st.slider(
            "Target Extra Buffer (Minutes)", 5, 25, 10, step=5, key=f"buf_sim_{selected_period}"
        )

    if st.button("Run Turnaround Buffer Simulation", key=f"btn_sim_turn_{selected_period}"):
        if not selected_turn_ops:
            st.warning("Please select at least one airline operator to evaluate.")
        else:
            with st.spinner(f"Simulating turnaround buffer impacts across {len(selected_turn_ops)} operators..."):
                buf_steps = [5, 10, 15, 20, 25]
                curve_data: dict[str, list[float]] = {op: [] for op in selected_turn_ops}
                target_summary = []

                for op in selected_turn_ops:
                    target_res = simulate_turnaround_mitigation(ds, operator=op, extra_buffer_min=float(extra_buffer))
                    target_summary.append({
                        "Operator": op,
                        "Buffer Added": f"+{extra_buffer} min",
                        "Incidents Prevented": target_res.get("incidents_prevented", 0),
                        "Delay Min Saved": target_res.get("delay_minutes_saved", 0),
                        "Delay Reduction %": f"{target_res.get('delay_reduction_percentage', 0.0):.1f}%",
                    })

                    for b in buf_steps:
                        b_res = simulate_turnaround_mitigation(ds, operator=op, extra_buffer_min=float(b))
                        curve_data[op].append(b_res.get("delay_reduction_percentage", 0.0))

            st.subheader(f"📊 Comparative Reactionary Delay Reduction at +{extra_buffer} min Buffer")
            st.dataframe(pd.DataFrame(target_summary), use_container_width=True)

            st.subheader("📈 Multi-Operator Buffer Sensitivity Curve (Delay Reduction % vs Extra Buffer)")
            st.caption("X-axis: Extra Scheduled Ground Buffer (minutes) | Y-axis: Reactionary Delay Reduction (%)")
            chart_df = pd.DataFrame(curve_data, index=buf_steps)
            chart_df.index.name = "Extra Buffer (min)"
            st.line_chart(chart_df)

    st.divider()

    st.subheader("Scenario B: Adverse Weather & Runway Crosswind Collapse")
    st.markdown("Evaluate aerodrome capacity retention under shifting wind vectors.")
    sim_apt = st.text_input("Airport ICAO", "KORD", key=f"sim_apt_{selected_period}")
    w_dir = st.slider("Wind Direction (Degrees)", 0, 360, 310, step=10, key=f"wdir_{selected_period}")
    w_spd = st.slider("Wind Speed (Knots)", 5, 50, 28, step=1, key=f"wspd_{selected_period}")
    w_gst = st.slider("Gusts (Knots)", 0, 60, 38, step=1, key=f"wgst_{selected_period}")

    if st.button("Simulate Runway Impact", key=f"btn_sim_rw_{selected_period}"):
        rw_res = simulate_runway_weather_scenario(
            ds,
            sim_apt,
            float(w_dir),
            float(w_spd),
            float(w_gst) if w_gst > 0 else None,
        )
        if "error" in rw_res:
            st.error(rw_res["error"])
        else:
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric(
                "Usable Runways",
                f"{rw_res['usable_runway_ends']} / {rw_res['total_runway_ends']}",
            )
            rc2.metric(
                "Capacity Retention", f"{rw_res['estimated_capacity_retention_pct']}%"
            )
            if rw_res["optimal_runway"]:
                rc3.metric(
                    "Optimal Runway",
                    f"{rw_res['optimal_runway']['runway_ident']} (Heading {rw_res['optimal_runway']['runway_heading_deg']}°)",
                )

            if rw_res.get("runways_status"):
                st.dataframe(pd.DataFrame(rw_res["runways_status"]), use_container_width=True)

    st.divider()

    st.subheader("Scenario C: Airline Fleet Operational Shock Stress Testing")
    st.markdown(
        "Inject operational delay shocks (+15m, +30m, +45m, +60m, +90m) into multiple airlines' rotation networks "
        "to construct comparative fragility curves and benchmark fleet resilience thresholds on a single graph."
    )
    selected_stress_ops = st.multiselect(
        "Select Operators to Stress Test",
        options=["RYR", "BAW", "DLH", "AFR", "KLM", "EZY", "SAS", "THY"],
        default=["RYR", "BAW", "DLH", "AFR"],
        key=f"stress_ops_{selected_period}",
    )

    if st.button("Run Multi-Operator Shock Stress Test", key=f"btn_stress_{selected_period}"):
        if not selected_stress_ops:
            st.warning("Please select at least one airline operator to stress test.")
        else:
            with st.spinner(f"Simulating cascading delay wave contagion across {len(selected_stress_ops)} operators..."):
                shock_steps = [15, 30, 45, 60, 90]
                frag_curve_data: dict[str, list[float]] = {op: [] for op in selected_stress_ops}
                frag_summary = []

                for op in selected_stress_ops:
                    s_res = stress_test_airline(ds, operator=op)
                    frag_summary.append({
                        "Operator": op,
                        "Rotations Analyzed": f"{s_res.total_rotations:,}",
                        "Mean Turnaround": f"{s_res.mean_scheduled_turnaround_min:.1f} min",
                        "Resilience Grade": s_res.resilience_grade,
                        "50% Shock Threshold": f"+{s_res.resilience_threshold_shock_min:.0f}m" if s_res.resilience_threshold_shock_min else ">90m",
                    })

                    c_map = {c["injected_shock_min"]: c["infection_rate_pct"] for c in s_res.fragility_curve}
                    for s in shock_steps:
                        frag_curve_data[op].append(c_map.get(float(s), 0.0))

            st.subheader("📊 Airline Resilience & Fragility Benchmarking")
            st.dataframe(pd.DataFrame(frag_summary), use_container_width=True)

            st.subheader("📈 Multi-Operator Fleet Infection Fragility Curves")
            st.caption("X-axis: Injected Operational Delay Shock (minutes) | Y-axis: Fleet Rotation Infection Rate (%)")
            frag_chart_df = pd.DataFrame(frag_curve_data, index=shock_steps)
            frag_chart_df.index.name = "Injected Delay Shock (min)"
            st.line_chart(frag_chart_df)

    st.divider()
    st.subheader("Scenario D: Prescriptive Hub Tail-Swap Recovery Optimizer")
    st.markdown(
        "Automatically scans aircraft turnarounds across the network to discover **concrete same-operator, same-aircraft-type tail-swap pairs (`Tail A ⇄ Tail B`)** "
        "on the ground at the same hub that absorb inbound arrival delays and prevent downstream reactionary departure delays."
    )
    col_ts1, col_ts2 = st.columns(2)
    ts_op_choice = col_ts1.selectbox(
        "Filter Operator (or Scan All Operators Network-Wide)",
        ["All Operators (Network-Wide)", "RYR", "BAW", "EZY", "DLH", "AFR", "KLM", "SAS", "THY"],
        index=0,
        key=f"ts_op_{selected_period}",
    )
    ts_hub = col_ts2.text_input(
        "Filter Hub Airport ICAO (Optional — leave blank to scan all hubs)",
        value="",
        placeholder="e.g. EGSS, EGLL, EDDF, EHAM",
        key=f"ts_hub_{selected_period}",
    ).strip().upper()

    ts_op_param = None if ts_op_choice.startswith("All") else ts_op_choice
    ts_res = _cached_tail_swaps(
        selected_period,
        operator=ts_op_param,
        hub_airport=ts_hub if ts_hub else None,
    )
    tm1, tm2, tm3, tm4 = st.columns(4)
    tm1.metric("Candidate Rotations Scanned", f"{ts_res.candidate_rotations_analyzed:,}")
    tm2.metric("Viable Tail-Swap Pairs Found", f"{ts_res.recommended_swaps_count}")
    tm3.metric("Total Delay Minutes Saved", f"{ts_res.total_delay_minutes_saved:,.1f} min")
    tm4.metric("Est. Delay Cost Saved (€)", f"€{ts_res.estimated_cost_saved_eur:,.0f} EUR")

    if ts_res.swaps:
        swap_rows = [
            {
                "Date": s.swap_date,
                "Hub": s.hub_airport,
                "Operator": s.operator,
                "Aircraft": s.ac_type,
                "Tail Swap Pair": f"{s.delayed_tail_reg} ⇄ {s.donor_tail_reg}",
                "Delayed Outbound Leg": f"#{s.delayed_outbound_flight_id} (➔ {s.delayed_outbound_ades} @ {s.delayed_outbound_std[-8:-3] if len(s.delayed_outbound_std) >= 8 else s.delayed_outbound_std})",
                "Donor Outbound Leg": f"#{s.donor_outbound_flight_id} (➔ {s.donor_outbound_ades} @ {s.donor_outbound_std[-8:-3] if len(s.donor_outbound_std) >= 8 else s.donor_outbound_std})",
                "Baseline Delay": f"{s.baseline_reactionary_delay_min:.0f} min",
                "Post-Swap Delay": f"{s.post_swap_reactionary_delay_min:.0f} min",
                "Net Delay Saved": f"-{s.net_delay_saved_min:.0f} min",
                "Cost Saved (€)": f"€{s.estimated_cost_saved_eur:,.0f}",
            }
            for s in ts_res.swaps
        ]
        st.dataframe(pd.DataFrame(swap_rows), use_container_width=True)
    else:
        st.info("No positive-saving tail swaps found for this specific filter combination. Try clearing the Hub filter or selecting 'All Operators (Network-Wide)'.")

    st.divider()
    st.subheader("Scenario E: 4D Co-Altitude Airspace Proximity & Complexity Scanner")
    st.markdown(
        "Interpolates 4D trajectories at 1-minute epochs (`:00`, `:15`, `:30`, `:45`) to detect **co-altitude aircraft pairs** "
        "within horizontal separation thresholds (`< 5 nm` / `1,000 ft` vertical) and overlays them on the $5^\\circ \\times 5^\\circ$ airspace saturation map."
    )
    col_cs1, col_cs2 = st.columns(2)
    cs_hr = col_cs1.slider("Scan Hour (UTC)", 0, 23, 10, format="%02d:00 UTC", key=f"cs_hr_{selected_period}")
    cs_nm = col_cs2.slider("Horizontal Proximity Threshold (nm)", 2.0, 10.0, 5.0, step=0.5, key=f"cs_nm_{selected_period}")

    prox_rep = _cached_proximity_scan(selected_period, hour=cs_hr, horiz_sep_nm=cs_nm)
    pe1, pe2, pe3, pe4 = st.columns(4)
    pe1.metric("Core Traffic Date Scanned", prox_rep.date_str)
    pe2.metric("Scan Window (UTC)", f"{prox_rep.hour_utc:02d}:00–{prox_rep.hour_utc:02d}:59")
    pe3.metric("Co-Altitude Proximity Pairs", f"{prox_rep.events_detected} pairs")
    pe4.metric("Loss-of-Sep Alerts (<3nm)", f"{prox_rep.loss_of_sep_alerts} alerts")

    if prox_rep.events:
        prox_rows = [
            {
                "Timestamp (UTC)": e.timestamp_utc,
                "Severity": e.severity,
                "Aircraft Pair": f"{e.operator_1} #{e.flight_1_id} ({e.ac_type_1}) ⇄ {e.operator_2} #{e.flight_2_id} ({e.ac_type_2})",
                "Routes": f"{e.route_1} ⇄ {e.route_2}",
                "Horizontal Sep": f"{e.horizontal_sep_nm:.2f} nm",
                "Vertical Sep": f"{e.vertical_sep_ft:.0f} ft",
                "Flight Level": f"FL{e.fl_1:.0f} / FL{e.fl_2:.0f}",
                "Coordinates": f"({e.latitude:.2f}°, {e.longitude:.2f}°)",
            }
            for e in prox_rep.events
        ]
        st.dataframe(pd.DataFrame(prox_rows), use_container_width=True)

        map_state_key = f"show_cs_map_{selected_period}"
        if map_state_key not in st.session_state:
            st.session_state[map_state_key] = False

        col_btn_map1, col_btn_map2 = st.columns([1, 3])
        with col_btn_map1:
            if not st.session_state[map_state_key]:
                if st.button("🗺️ View Conflicts on Airspace Map", key=f"btn_open_cs_map_{selected_period}"):
                    st.session_state[map_state_key] = True
                    st.rerun()
            else:
                if st.button("✖️ Hide Conflict Map", key=f"btn_close_cs_map_{selected_period}"):
                    st.session_state[map_state_key] = False
                    st.rerun()

        if st.session_state[map_state_key]:
            st.caption(
                "🔴 **Red Pins:** Loss-of-Separation Alerts (`< 3.0 nm` horiz & `≤ 1,000 ft` vert) | "
                "🟠 **Orange Pins:** High-Proximity Co-Altitude Pairs (`3.0–5.0 nm` horiz & `≤ 1,000 ft` vert). "
                "Click any pin to inspect the two aircraft, routes, flight levels, and separation distances."
            )
            cs_map = create_conflict_map(
                proximity_events=prox_rep.events,
                hour_utc=cs_hr,
                date_str=prox_rep.date_str,
            )
            st_folium(
                cs_map,
                width=1200,
                height=500,
                returned_objects=[],
                key=f"cs_prox_map_{selected_period}_{cs_hr}_{cs_nm}",
            )
    else:
        st.info("No co-altitude pairs detected within the selected threshold at this hour. Try increasing the horizontal proximity slider or selecting peak daytime hours (08:00–17:00 UTC).")

    st.divider()
    st.subheader("Scenario F: Ration-by-Schedule Ground Delay Program (GDP / ATFM Slot) Optimizer")
    st.markdown(
        "Simulates how enforcing a degraded Airport Arrival Rate (`AAR`) cap via ATFM Ground Delay slots (`CTOT` / `CTA`) shifts expensive airborne holding stacks "
        "onto the origin departure gate with engines off, quantifying net fuel (`kg`), CO₂ (`kg`), and Euro (`€`) savings."
    )
    col_gdp_dt, col_gdp1, col_gdp2, col_gdp3 = st.columns([1.4, 1.2, 1.4, 1.6])

    ds_min_date, ds_max_date = get_dataset_date_bounds(ds)
    try:
        d_min_obj = pd.to_datetime(ds_min_date).date()
        d_max_obj = pd.to_datetime(ds_max_date).date()
        d_def_obj = d_min_obj + pd.Timedelta(days=14)
        if d_def_obj > d_max_obj or d_def_obj < d_min_obj:
            d_def_obj = d_min_obj
    except Exception:
        d_min_obj = pd.to_datetime(f"{selected_period[:4]}-{selected_period[4:6]}-01").date()
        d_max_obj = pd.to_datetime(f"{selected_period[:4]}-{selected_period[4:6]}-28").date()
        d_def_obj = d_min_obj

    gdp_date_obj = col_gdp_dt.date_input(
        "Regulated Arrival Date",
        value=d_def_obj,
        min_value=d_min_obj,
        max_value=d_max_obj,
        key=f"gdp_date_{selected_period}",
    )
    gdp_date_str = gdp_date_obj.strftime("%Y-%m-%d")

    gdp_apt = col_gdp1.text_input("Arrival Hub ICAO", value="EGLL", key=f"gdp_apt_{selected_period}").strip().upper()
    gdp_aar = col_gdp2.slider(
        "Degraded AAR Cap (arrivals/hr)",
        min_value=10,
        max_value=55,
        value=25,
        key=f"gdp_aar_{selected_period}",
    )
    gdp_window = col_gdp3.slider(
        "GDP Window (UTC Hours)",
        min_value=0,
        max_value=24,
        value=(6, 14),
        key=f"gdp_win_{selected_period}",
    )

    gdp_res = _cached_gdp_optimizer(
        selected_period,
        target_airport=gdp_apt if gdp_apt else "EGLL",
        program_date=gdp_date_str,
        start_hour=int(gdp_window[0]),
        end_hour=int(gdp_window[1]),
        aar=int(gdp_aar),
    )
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(
        "Regulated Arrival Window",
        f"{gdp_res.program_start[:10]} ({gdp_window[0]:02d}:00–{gdp_window[1]:02d}:00 UTC)",
        f"Declared AAR Cap: {gdp_res.degraded_capacity_rate_per_hr}/hr",
    )
    g2.metric(
        "Flights Regulated (RBS Slots)",
        f"{gdp_res.total_flights_regulated:,} arrivals",
        f"Avg Slot Delay: {(gdp_res.total_ground_delay_assigned_min / gdp_res.total_flights_regulated if gdp_res.total_flights_regulated else 0.0):.1f} min",
    )
    g3.metric(
        "Airborne Holding Prevented",
        f"{gdp_res.total_airborne_delay_prevented_min:,.1f} min",
        f"-{gdp_res.total_fuel_saved_kg:,.0f} kg Jet-A saved",
    )
    g4.metric(
        "Net Fuel Cost Saved (€)",
        f"€{gdp_res.total_fuel_cost_saved_eur:,.2f} EUR",
        f"-{gdp_res.total_co2_saved_kg:,.0f} kg CO₂ prevented",
    )

    if gdp_res.slot_assignments:
        slot_rows = [
            {
                "Flight ID": f"#{s.ectrl_id}",
                "Operator": s.ac_operator,
                "Origin (ADEP)": s.adep,
                "Hub (ADES)": s.ades,
                "Aircraft": s.ac_type,
                "Filed ETA (UTC)": s.original_arrival_time[-8:] if len(s.original_arrival_time) >= 8 else s.original_arrival_time,
                "Assigned CTA Slot": s.assigned_slot_time[-8:] if len(s.assigned_slot_time) >= 8 else s.assigned_slot_time,
                "Ground Delay Assigned": f"{s.ground_delay_assigned_min:.1f} min",
                "Holding Fuel Saved": f"{s.fuel_saved_kg:,.1f} kg",
                "CO₂ Prevented": f"{s.co2_saved_kg:,.1f} kg",
                "Fuel Cost Saved (€)": f"€{s.fuel_cost_saved_eur:,.2f}",
            }
            for s in gdp_res.slot_assignments[:60]
        ]
        st.dataframe(pd.DataFrame(slot_rows), use_container_width=True)
    else:
        st.info(f"No arrivals found for `{gdp_apt}` on `{gdp_date_str}` in the selected UTC window `{gdp_window[0]:02d}:00–{gdp_window[1]:02d}:00 UTC`.")

    # ----------------- TAB 9: Macro Evolution & AIRAC Diff -----------------

# =============================================================================
# 5. INFORMATION & MAP VIEWS
# =============================================================================
with tab_cat_maps:
    st.caption(
        "Dedicated geographic and aeronautical information visualization: 5°×5° European airspace saturation heatmaps, "
        "4D co-altitude aircraft conflict radar, global radio navaids world directory, and regulatory KPI references."
    )
    render_page_export_bar("maps_navigation", "cat_maps")
    map_sub1, map_sub2 = st.tabs(
        [
            "📡 Global Radio Navaids World Map",
            "📖 Methodology & KPI Reference",
        ]
    )

    with map_sub1:
        st.header("🌍 Global Radio Navaids World Map & Station Directory")
        render_page_export_bar("global_navaids", "tab_world_nav")
        st.markdown(
            "Explore **all global terrestrial radio navigation aids** (`VOR-DME / VORTAC`, `DME / TACAN`, `VOR`, `ILS / LOC`, and `NDB`) "
            "fused across the X-Plane 1100 global navigation database and OurAirports navaid registry. "
            "Filter by navaid type, geographic region/viewport, or station identifier."
        )

        WORLD_REGIONS: dict[str, tuple[float, float, float, float, float, float, int]] = {
            "🌍 Entire World": (-85.0, 85.0, -180.0, 180.0, 25.0, 10.0, 2),
            "🇪🇺 Europe & North Atlantic": (34.0, 72.0, -25.0, 45.0, 50.0, 10.0, 4),
            "🌎 North America": (15.0, 72.0, -170.0, -50.0, 40.0, -98.0, 3),
            "🌏 Asia-Pacific": (-48.0, 60.0, 65.0, 180.0, 22.0, 118.0, 3),
            "🌍 Middle East & Africa": (-38.0, 40.0, -20.0, 65.0, 12.0, 25.0, 3),
            "🌎 South & Central America": (-58.0, 15.0, -95.0, -30.0, -18.0, -60.0, 3),
        }

        wn_c1, wn_c2, wn_c3, wn_c4 = st.columns([2, 2, 1.5, 1.5])
        with wn_c1:
            wn_types = st.multiselect(
                "Filter Navaid Types",
                options=["VOR-DME", "DME", "VOR", "ILS_LOC", "NDB"],
                default=["VOR-DME", "DME", "VOR"],
                key=f"wn_types_{selected_period}",
                help="Select which navaid categories to render on the global map (VOR-DME/VORTAC, standalone DME/TACAN, VOR, ILS/LOC, NDB).",
            )
        with wn_c2:
            wn_region = st.selectbox(
                "Geographic Region / Viewport",
                options=list(WORLD_REGIONS.keys()),
                index=0,
                key=f"wn_region_{selected_period}",
            )
        with wn_c3:
            wn_search = st.text_input(
                "Search Ident or Name",
                value="",
                placeholder="e.g. BPK, LAM, FRANKFURT",
                key=f"wn_search_{selected_period}",
            ).strip()
        with wn_c4:
            wn_limit = st.slider(
                "Max Map Markers",
                min_value=500,
                max_value=6000,
                value=2500,
                step=500,
                key=f"wn_limit_{selected_period}",
                help="Maximum number of navaid markers rendered simultaneously on the HTML5 Canvas world map.",
            )

        min_lat, max_lat, min_lon, max_lon, c_lat, c_lon, z_lvl = WORLD_REGIONS[wn_region]
        world_nav_df = _cached_world_navaids(
            selected_period,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            nav_types_tuple=tuple(wn_types if wn_types else ["NONE"]),
            search_query=wn_search if wn_search else None,
            limit=wn_limit,
        )

        # Compute breakdown metrics
        vordme_cnt = int((world_nav_df["nav_type"] == "VOR-DME").sum()) if not world_nav_df.empty else 0
        dme_cnt = int((world_nav_df["nav_type"] == "DME").sum()) if not world_nav_df.empty else 0
        vor_cnt = int((world_nav_df["nav_type"] == "VOR").sum()) if not world_nav_df.empty else 0
        ils_cnt = int((world_nav_df["nav_type"] == "ILS_LOC").sum()) if not world_nav_df.empty else 0
        ndb_cnt = int((world_nav_df["nav_type"] == "NDB").sum()) if not world_nav_df.empty else 0

        wm1, wm2, wm3, wm4, wm5, wm6 = st.columns(6)
        wm1.metric("Stations Rendered", f"{len(world_nav_df):,}")
        wm2.metric("🔵 VOR-DME / VORTAC", f"{vordme_cnt:,}")
        wm3.metric("🔷 DME / TACAN", f"{dme_cnt:,}")
        wm4.metric("🟢 VOR", f"{vor_cnt:,}")
        wm5.metric("🟣 ILS / LOC", f"{ils_cnt:,}")
        wm6.metric("🟠 NDB", f"{ndb_cnt:,}")

        world_map = create_world_navaids_map(
            world_nav_df,
            center_lat=c_lat,
            center_lon=c_lon,
            zoom_start=z_lvl,
        )
        st_folium(
            world_map,
            width=1200,
            height=560,
            returned_objects=[],
            key=f"world_nav_map_{selected_period}_{wn_region}_{'_'.join(wn_types)}_{wn_search}_{wn_limit}",
        )

        with st.expander(f"📋 View Filtered Global Navaids Station Directory ({len(world_nav_df):,} stations)", expanded=False):
            if not world_nav_df.empty:
                disp_nav_df = world_nav_df[
                    ["ident", "name", "nav_type", "frequency_mhz", "range_nm", "elevation_ft", "latitude", "longitude"]
                ].rename(
                    columns={
                        "ident": "Ident",
                        "name": "Station Name",
                        "nav_type": "Navaid Type",
                        "frequency_mhz": "Freq (MHz)",
                        "range_nm": "Range (nm)",
                        "elevation_ft": "Elevation (ft)",
                        "latitude": "Latitude",
                        "longitude": "Longitude",
                    }
                )
                st.dataframe(disp_nav_df, use_container_width=True)
            else:
                st.info("No navaids match the current filter selection.")

        # ----------------- TAB 3: Recurring Series & Diurnal Trends -----------------

    with map_sub2:
        st.header("📖 Aviation Intelligence Methodology, Physics Models & KPI Reference")
        st.markdown(
            "Use this reference guide to look up the mathematical definitions, international standards "
            "(**ICAO GANP**, **Eurocontrol PRU**, **Eurocontrol CRCO**, **ICAO Aircraft Engine Emissions Databank v32B**, and **OpenAP**), "
            "and operations research algorithms powering every metric in the FlightML Workbench."
        )

        ref_sub1, ref_sub_pbn, ref_sub2, ref_sub3, ref_sub4, ref_sub5 = st.tabs(
            [
                "📏 ICAO GANP & PRU KPIs",
                "📡 RNAV & RNP Requirements (PBN)",
                "🗺️ Airspace Saturation & 4D Conflicts",
                "🌱 Fuel, LTO Emissions & Contrails",
                "💶 Eurocontrol CRCO & Cost Economics",
                "🔮 Operations Research & Safety Models",
            ]
        )

        with ref_sub1:
            st.subheader("ICAO Global Air Navigation Plan (GANP) & Eurocontrol PRU Indicators")
            st.markdown(
                """
        | KPI Code | Official Indicator Name | Mathematical Formulation & Operational Meaning |
        | :--- | :--- | :--- |
        | **KPI02 (ATOT)** | **Additional Taxi-Out Time** | $\\text{ATOT} = \\max(0,\\; \\text{Actual Taxi-Out} - \\text{UTOT}_{p20})$, where $\\text{UTOT}_{p20}$ is the **20th percentile unimpeded taxi-out time** for that specific `(airport, wake turbulence category)` pair. Measures departure runway queue congestion and surface hold time. |
        | **KPI04 (KEP)** | **Horizontal En-Route Flight Efficiency (Filed)** | $\\text{KEP (\\%)} = \\left(\\frac{L_{\\text{filed, >40nm}} - D_{\\text{GCD, >40nm}}}{D_{\\text{GCD, >40nm}}}\\right) \\times 100$. Evaluates how much extra distance the airline's filed flight plan adds outside the 40 nm origin and destination terminal cylinders relative to the achieved Great Circle Distance ($D_{\\text{GCD}}$). Indicates structural route availability, RAD restrictions, or route-charge avoidance. |
        | **KPI05 (KEA)** | **Horizontal En-Route Flight Efficiency (Actual)** | $\\text{KEA (\\%)} = \\left(\\frac{L_{\\text{actual, >40nm}} - D_{\\text{GCD, >40nm}}}{D_{\\text{GCD, >40nm}}}\\right) \\times 100$. Measures the actual flown en-route distance extension outside the 40 nm cylinders. **Tactical ATC Attribution ($\\text{KEA} - \\text{KEP}$)**: When $\\text{KEA} < \\text{KEP}$, ATC cleared tactical directs (`DCT`); when $\\text{KEA} > \\text{KEP}$, tactical vectoring or weather deviations lengthened the flight. |
        | **KPI08 (ASMA)** | **Additional Time in Terminal Airspace (40 nm)** | $\\text{ASMA}_{\\text{add}} = \\max(0,\\; T_{\\text{40nm\\to ALDT}} - T_{\\text{unimpeded, p20}})$. Tracks arrival sequencing, trombone vectoring, and airborne holding inside the 40 nm Arrival Sequencing and Metering Area cylinder. |
        | **KPI17 (CCO)** | **Continuous Climb Operations (Climb Profile)** | Identifies intermediate level-offs ($\\lvert\\text{ROCD}\\rvert < 200\\text{ ft/min}$ for $\\ge 60\\text{ s}$) below Top of Climb (`TOC`). Level-offs at low/medium altitudes force high drag and higher specific fuel consumption compared to an uninterrupted climb to cruise altitude. |
        | **KPI19 (CDO & Level-Cap)** | **Continuous Descent Operations & Cruise Level-Capping** | **CDO** measures uninterrupted idle-thrust descent from Top of Descent (`TOD`) to landing without throttle-up level segments. **Cruise Level-Capping** compares the filed requested Flight Level ($\\text{FL}_{\\text{req}}$) against the maximum actual achieved cruise level ($\\text{FL}_{\\text{max,act}}$); a deficit $\\ge 20\\text{ FL}$ ($2,000\\text{ ft}$) increases cruise fuel burn by ~1.25% per $1,000\\text{ ft}$. |
        | **FRA vs. ATS** | **Free Route Airspace Directness Index** | Matches actual trajectory legs against **86,280 global 5-letter waypoints (`fix.dat`)**, **17,090 VHF navaids (`nav.dat`)**, and published ATS airways. Segments where consecutive waypoints lie on the same published airway are classified as **Structured ATS Airway**; off-airway point-to-point segments are classified as **Free Route Airspace (`FRA / DCT`)**. |
        | **RNP / RNAV & RLOS** | **Anticipated PBN Navigation Capability (ICAO Doc 9613)** | Evaluates terrestrial `DME`, `VOR`, `ILS/LOC`, and `NDB` stations within the VHF/UHF **Radio Line-of-Sight (`RLOS`)** horizon $D_{\\text{RLOS}} (\\text{nm}) = 1.23 \\times (\\sqrt{h_{\\text{ac, ft}}} + \\sqrt{h_{\\text{nav, ft}}})$ at each 4D waypoint: **`0.1 nm`** (`xLS / RNP AR APCH 0.1` with ILS/LOC lock), **`0.3 nm`** (`RNP APCH 0.3`), **`1.0 nm`** (`RNAV 1 / RNP 1` with $\\ge 2$–$3$ DMEs in RLOS), **`2.0 nm`** (`RNAV 2` dual-DME), **`5.0 nm`** (`RNAV 5 / B-RNAV` single VOR/DME), and **`10.0 nm`** (`RNP 4 / RNP 10` Oceanic/Remote beyond-RLOS relying on GNSS + IRS). |
        """
            )

        with ref_sub_pbn:
            st.subheader("Performance-Based Navigation (PBN): RNAV vs. RNP Meanings, Physics & Regulatory Requirements")
            st.markdown(
                """
        #### 1. Fundamental Distinction: RNAV (Area Navigation) vs. RNP (Required Navigation Performance)
        Under **ICAO Doc 9613 (Performance-Based Navigation Manual)**, global air navigation transitioned from **sensor-specific flying** (hopping between ground radio beacons like VORs and NDBs) to **performance-based navigation specifications** defined by accuracy, integrity, continuity, and functionality.

        The core difference between **RNAV** and **RNP** comes down to **On-Board Performance Monitoring and Alerting (OBPMA)**:

        - **RNAV (Area Navigation)**:
          - A method of navigation that permits aircraft operation on any desired flight path within the coverage of station-referenced navigation aids, self-contained aids, or a combination.
          - **Critical Limitation**: RNAV systems **do NOT require on-board performance monitoring and alerting**. If navigation accuracy degrades beyond the specified limit (e.g. due to sensor loss or inertial drift), the system does **not** guarantee an automated warning to the flight crew.
          - Because of this lack of autonomous containment alerting, air traffic controllers must maintain wider radar separation buffers around RNAV routes.

        - **RNP (Required Navigation Performance)**:
          - An area navigation specification that **mandates on-board performance monitoring and alerting (OBPMA)**.
          - The onboard avionics (FMS, GNSS receivers, and inertial reference units) continuously calculate the estimated position uncertainty. If the probability that the **Total System Error (TSE)** exceeds the containment boundary is greater than $10^{-5}$, an immediate visual and aural alert is raised to the pilots (e.g., `"UNABLE RNP"`).
          - Because containment is guaranteed with high statistical integrity, air traffic authorities can safely reduce route separation (e.g., parallel routes spaced 7 nm apart instead of 50 nm) and design curved, obstacle-hugging **Radius-to-Fix (`RF`)** legs through mountainous terrain.

        ---

        #### 2. Total System Error (TSE) Physics & Containment Boundaries
        The lateral deviation between the true aircraft position and the desired path is governed by the **Total System Error (TSE)**:

        $$\\text{TSE} = \\sqrt{\\text{NSE}^2 + \\text{FTE}^2 + \\text{PDE}^2}$$

        Where:
        - **Navigation System Error (NSE)**: The difference between the true aircraft position and the position calculated by the onboard sensors (GNSS satellite clock/ephemeris errors, atmospheric delays, DME multi-lateration geometry, or IRS drift).
        - **Flight Technical Error (FTE)**: The accuracy with which the autopilot or pilot follows the flight director commands to keep the aircraft on the intended centerline (typically $\\le 0.2\\text{ nm}$ with autopilot engaged).
        - **Path Definition Error (PDE)**: The difference between the intended flight track and the path defined by the Flight Management Computer (FMC) navigation database (waypoint latitude/longitude rounding, leg transitions).

        **95% Accuracy vs. Containment Area**:
        - For an $\\text{RNP } X$ specification (e.g. $\\text{RNP 1}$ or $\\text{RNP 0.3}$), the aircraft must remain within $\\pm X\\text{ nautical miles}$ of the cleared track for **at least 95% of total flight time**.
        - The **RNP Containment Area** extends laterally to **$2 \\times X\\text{ nm}$** (twice the nominal accuracy). The avionics must guarantee with an integrity probability of **$99.999\\%$ ($1 - 10^{-5}$)** that the aircraft does not cross this containment limit without raising an alert.

        ---

        #### 3. Complete ICAO PBN Specification Hierarchy & Infrastructure Requirements
        The table below details every operational PBN specification, its required 95% lateral accuracy, mandatory sensor suites, and operational airspace applications:

        | PBN Specification | Flight Phase | 95% Lateral Accuracy | On-Board Alerting (OBPMA) | Eligible Sensors | Operational Purpose & Mandatory Equipment |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | **`RNAV 10` (RNP 10)** | Oceanic / Remote | $\\pm 10.0\\text{ nm}$ | No | Dual INS/IRS or GNSS | Oceanic crossings (e.g. North Atlantic Tracks). Permits 50 nm lateral separation without ground radar. Requires dual long-range navigation systems with time-limited inertial drift certification. |
        | **`RNAV 5` (B-RNAV)** | En-Route Continental | $\\pm 5.0\\text{ nm}$ | No | DME/DME, VOR/DME, GNSS, or IRS | **Mandatory in European airspace (ECAC)** above FL95 since 1998. Allows direct routing outside fixed ATS airways. Primary sensor in Europe is multi-DME triangulation or single VOR/DME. |
        | **`RNAV 2`** | En-Route Continental | $\\pm 2.0\\text{ nm}$ | No | GNSS or DME/DME/IRU | Modern high-altitude en-route domestic corridors (e.g., US FAA Q-routes). Requires high-density DME coverage or GNSS. |
        | **`RNAV 1` (P-RNAV)** | Terminal (SID / STAR) | $\\pm 1.0\\text{ nm}$ | No | GNSS or DME/DME/IRU | European Precision RNAV (`P-RNAV`) for departure SIDs and arrival STARs below FL150. Eliminates vectoring noise; requires dual DME coverage within line of sight. |
        | **`RNP 4`** | Oceanic / Remote | $\\pm 4.0\\text{ nm}$ | **Yes** | GNSS | Oceanic airspace with 30 nm lateral / 30 nm longitudinal separation. Requires ADS-C and CPDLC datalink communications. |
        | **`RNP 2`** | Continental En-Route / Remote | $\\pm 2.0\\text{ nm}$ | **Yes** | GNSS | En-route airspace with sparse terrestrial navigation aids (e.g., Australian outback, Canadian north). |
        | **`RNP 1`** | Terminal (SID / STAR) | $\\pm 1.0\\text{ nm}$ | **Yes** | GNSS or DME/DME/IRU | Advanced terminal arrivals connecting to initial approach fixes. Supports RF (Radius-to-Fix) curved legs to deconflict closely spaced airport arrival flows. |
        | **`RNP APCH` (RNP 0.3)** | Initial, Intermediate & Final Approach | $\\pm 0.3\\text{ nm}$ (Final) / $\\pm 1.0\\text{ nm}$ (Initial) | **Yes** | GNSS with RAIM / SBAS (EGNOS, WAAS) | Standard instrument approach down to LNAV, LNAV/VNAV, LP, or LPV minima. Replaces ground ILS glideslope with Barometric VNAV or Satellite-Based Augmentation. |
        | **`RNP AR APCH`** | Confined Approach & Missed Approach | $\\pm 0.1\\text{ nm}$ to $\\pm 0.3\\text{ nm}$ | **Yes (High Integrity)** | Dual GNSS + Dual FMC + Inertial Reference | **Authorization Required (AR)** for terrain-critical airports (e.g. Innsbruck LOWI, Queenstown NZQN). Mandatory Radius-to-Fix (RF) curved legs, dual autopilot coupling, and specialized flight crew simulator training. |

        ---

        #### 4. Terrestrial Navaid Infrastructure & Radio Line-of-Sight (RLOS) Physics
        When aircraft navigate without GNSS or experience satellite signal interference, onboard Flight Management Systems rely on terrestrial VHF/UHF stations (**DME**, **VOR**, **ILS/LOC**, and **NDB**):

        1. **Radio Line-of-Sight Horizon Formula**:
           VHF and UHF radio signals propagate in direct straight lines and are blocked by the curvature of the Earth and terrain. The maximum theoretical line-of-sight distance $D_{\\text{RLOS}}$ between an aircraft at altitude $h_{\\text{ac}}$ (feet MSL) and a ground station at elevation $h_{\\text{nav}}$ (feet MSL) is:

           $$D_{\\text{RLOS}} (\\text{nm}) = 1.23 \\times \\left(\\sqrt{h_{\\text{aircraft, ft}}} + \\sqrt{h_{\\text{navaid, ft}}}\\right)$$

           - At cruising altitude (**FL360**), an aircraft enjoys an RLOS horizon of **$\\sim 235\\text{ nm}$**, bringing dozens of European DME stations into view simultaneously.
           - During terminal approach (**3,000 ft AGL**), the RLOS horizon shrinks to **$\\sim 70\\text{ nm}$**, significantly restricting available ground stations.

        2. **DME/DME Multi-Lateration Geometry**:
           - To calculate an RNAV position fix from DME stations without GNSS, the FMC requires **at least two independent DME stations** in line of sight.
           - **Subtended Angle Constraint**: The intersecting radials from the two stations must cross at an angle between **$30^\\circ$ and $150^\\circ$**. Angles sharper than $30^\\circ$ or flatter than $150^\\circ$ produce severe Geometric Dilution of Precision (GDOP), causing position calculation errors.
           - Optimal positioning uses **DME/DME/DME (triple-DME)** multi-lateration with continuous Kalman filtering.

        3. **Why Terrestrial Navaids Cannot Support RNP Approaches**:
           - Terrestrial DME and VOR stations cannot support RNP approaches ($0.3\\text{ nm}$ or $0.1\\text{ nm}$) because ground stations lack real-time autonomous integrity broadcast to the aircraft, and multipath reflections in terminal valleys corrupt DME ranging accuracy. Therefore, **RNP APCH and RNP AR APCH strictly require GNSS with RAIM (Receiver Autonomous Integrity Monitoring) or SBAS (EGNOS/WAAS)**.

        ---

        #### 5. How FlightML Calculates Anticipated Navigation in the Front End
        In the **Single Aircraft / Flight View** (Category 1 $\\rightarrow$ *Terrestrial Navaids & PBN Navigation*), FlightML models 4D trajectory points against **17,090 global VHF navaids (`nav.dat`)**:
        - Computes active terrestrial stations within the radio line-of-sight horizon at each trajectory altitude.
        - Calculates the geometry and density of available DME/VOR stations.
        - Color-codes the vertical flight profile beneath the route map into:
          - 🟣 **`0.1 nm (xLS / RNP AR APCH)`**: Terminal runway alignment with ILS/LOC beam lock.
          - 🔵 **`0.3 nm (RNP APCH)`**: Final approach segment with SBAS/GNSS integrity.
          - 🟢 **`1.0 nm (RNAV 1 / RNP 1)`**: Terminal arrival/departure with $\\ge 2$–$3$ DMEs in line of sight and healthy geometry.
          - 🟡 **`2.0 nm (RNAV 2)`**: En-route domestic with dual-DME coverage.
          - 🟠 **`5.0 nm (RNAV 5 / B-RNAV)`**: En-route coverage with single VOR/DME.
          - ⚪ **`10.0 nm (RNP 10 / Oceanic)`**: Remote or oceanic airspace beyond terrestrial line of sight.
        """
            )

        with ref_sub2:
            st.subheader("5° × 5° Airspace Saturation, Dynamic Complexity & 4D Proximity Detection")
            st.markdown(
                """
        #### 1. 5° × 5° Hourly Grid Saturation (Period Average by Hour of Day)
        To give air navigation analysts a clear macro view of European traffic waves, trajectories across all days in the selected archive are bucketed into **$5^\\circ \\text{ Latitude} \\times 5^\\circ \\text{ Longitude}$ geographic squares** for each hour of the day (`00:00` to `23:00 UTC`):
        - **Average Flights / Hour ($\\bar{N}_{h}$)**: Total distinct aircraft traversing the $5^\\circ \\times 5^\\circ$ square during hour $h$ divided by the number of active days in the archive.
        - **Peak Flights / Hour ($N_{\\max,h}$)**: Highest single-day aircraft count in that square during hour $h$.
        - **Vertical Transition Share ($V_{\\%}$)**: Percentage of 4D trajectory points inside the square that are actively climbing or descending ($|\\Delta \\text{FL}| \\ge 5$) rather than in level cruise.
        - **Dynamic Complexity Index**:
          $$\\text{Complexity Index} = \\bar{N}_{h} \\times \\left(1 + \\frac{V_{\\%}}{100}\\right)$$
          Squares are classified into four operational tiers:
          - 🔴 **`CRITICAL_SATURATION`**: $\\bar{N}_{h} \\ge 80\\text{ flts/hr}$ or $\\text{Complexity Index} \\ge 100$
          - 🟠 **`HIGH_DENSITY`**: $\\bar{N}_{h} \\ge 40\\text{ flts/hr}$ or $\\text{Complexity Index} \\ge 55$
          - 🟡 **`MODERATE_FLOW`**: $\\bar{N}_{h} \\ge 15\\text{ flts/hr}$
          - 🟢 **`LOW_DENSITY`**: $< 15\\text{ flts/hr}$

        #### 2. FIR & AUA (TMA / CTA / UTA) Sector Dwell Inefficiency
        - **FIR (`Fir_Airspace`)**: National/regional Flight Information Regions (`EGTTFIR`, `LFFFFIR`, `EDGGFIR`).
        - **AUA (`AUA_Airspace`)**: Granular Air Traffic Control Unit Airspaces separating **Terminal Manoeuvring Areas (`TMA`)**, **Control Areas (`CTA`)**, and **Upper Control Areas (`UTA`)**. Comparing `entry_time`/`exit_time` between ` Actual ` and ` Filed ` traversals pinpoints whether excess dwell occurred in terminal approach airspace or en-route upper sectors.

        #### 3. 4D Co-Altitude Airspace Proximity Scanner (Scenario E)
        - Trajectories are linearly interpolated in 4D `(latitude, longitude, flight_level, groundspeed, track)` at synchronized 1-minute UTC epochs (`:00`, `:15`, `:30`, `:45`).
        - Spatial bounding-box pruning followed by exact Haversine distance and altitude filtering flags every aircraft pair operating simultaneously within **$< 1,000\\text{ ft}$ vertical ($10\\text{ FL}$)** and **$\\le 5.0\\text{ nm}$ horizontal separation** (with `< 3.0 nm` flagged as a **Loss-of-Separation Alert**).

        #### 4. ATC VHF Radio Frequency Occupancy
        - Cross-references airport/TMA VHF voice channels (`apt_frequencies.csv`) and navaids (`nav.dat`) against peak hourly movements. Assuming 3.5 controller-pilot voice exchanges per flight in a sector at 9 seconds per exchange (including readback), occupancy above **60%** indicates controller voice saturation where an additional frequency split is required.
        """
            )

        with ref_sub3:
            st.subheader("Fuel Burn, ICAO EDB v32B LTO Surface Emissions & Contrail Physics")
            st.markdown(
                """
        #### 1. OpenAP Aircraft-Specific Fuel Burn & Carbon Quantification
        - Each aircraft type (`B738`, `A320`, `A20N`, `B77W`, `A359`, etc.) is mapped to its **OpenAP** aerodynamic drag polar and engine fuel-flow model across climb, cruise, descent, and terminal holding regimes.
        - **Carbon Dioxide ($\\text{CO}_2$)**: Computed via the standard ICAO stoichometric Jet-A emission factor of **$3.16\\text{ kg CO}_2$ per $1.0\\text{ kg}$ of Jet-A fuel**.

        #### 2. Surface Taxi-Out (`TAXI_OUT`) & ICAO Engine Emissions Databank (EDB v32B)
        - Ground movement between Actual Off-Block Time (`AOBT`) and Actual Take-Off Time (`ATOT`) operates main turbofan engines at **7% ICAO Landing/Take-Off (LTO) Idle thrust** plus an Auxiliary Power Unit (`APU`) fuel flow.
        - Using empirical emission indices from `data/external/icao_edb_emissions.parquet` (**ICAO EDB v32B**), FlightML quantifies exact ground-level airport air quality pollutants generated during excess departure queue delays:
          - **Nitrogen Oxides ($\\text{NO}_x$)** in grams per kg fuel at 7% idle
          - **Carbon Monoxide ($\\text{CO}$)** and **Unburned Hydrocarbons ($\\text{HC}$)** (which peak at low idle combustion temperatures)
          - **Non-Volatile Particulate Matter ($\\text{nvPM}$ soot mass)** in milligrams

        #### 3. Upper-Air Winds, Equivalent Still Air Distance (ESAD) & Contrail Climate Forcing
        - **Ground Distance vs. Air Distance (ESAD)**:
          Traditional flight efficiency metrics (like Eurocontrol PRU **KEA** and **KEP**) evaluate flights purely geometrically against the Great Circle distance.
          However, high-altitude **Jet Stream winds (80–160+ kts)** mean that ground distance does not equal fuel burn:
          - A flight taking a $+150\\text{ nm}$ ground detour to ride a $+110\\text{ kt}$ jetstream tailwind may show $+5\\%$ geometric detour inefficiency,
            yet arrive 40 minutes faster and burn **2,000 kg less fuel** than a direct flight.
          - Conversely, a flight flying straight into a $-120\\text{ kt}$ headwind core achieves "perfect directness" geometrically, but experiences severe fuel waste.
        - **Equivalent Still Air Distance (ESAD) Formula**:
          $$\\text{ESAD} = \\int \\text{TAS}(t) \\, dt = \\sum_{i} D_{\\text{ground},i} \\times \\left(\\frac{\\text{TAS}_i}{\\text{GS}_i}\\right)$$
          - **Air Distance Benefit ($\\Delta D_{\\text{air}}$)**: $\\Delta D_{\\text{air}} = D_{\\text{ground}} - \\text{ESAD}$.
            Positive values denote air miles saved by tailwinds; negative values denote air miles added by headwind resistance.
        - **Schmidt-Appleman Criterion & Ice-Supersaturated Regions (`ISSR`)**:
          When a flight cruises between `FL280` and `FL430` in ambient temperatures $T < -40^\\circ\\text{C}$ ($233\\text{ K}$) and high relative humidity over ice ($\\text{RH}_{\\text{ice}} \\ge 100\\%$),
          exhaust water vapor nucleates into persistent contrail-cirrus clouds that trap outgoing longwave radiation, multiplying total radiative climate forcing (`CO₂e`).

        > **📌 Implementation Note & Data Architecture Qualifier**:
        > *Upper-air atmospheric winds (horizontal $U$ and $V$ wind vectors, ambient temperature, and relative humidity across cruise pressure levels 300, 250, 200, and 150 hPa)
        > are sampled on-demand per flight using the **Open-Meteo European Reanalysis & Forecast API** and cached in a local SQLite datastore (`data/weather_cache/upper_air_era5.sqlite`).
        > This local-first, on-demand caching approach is implemented as an operational, low-overhead solution in place of downloading multi-terabyte bulk ECMWF Copernicus Climate Data Store (CDS) ERA5 global reanalysis grids.*
        """
            )

        with ref_sub4:
            st.subheader("Eurocontrol CRCO Route Charges & Direct Operating Cost Economics (€)")
            st.markdown(
                """
        #### 1. Eurocontrol Central Route Charges Office (CRCO) Formula
        Every commercial flight traversing European Flight Information Regions pays an en-route navigation charge billed in **Euros (`€`)** according to the official Eurocontrol CRCO formula:
        $$\\text{CRCO Charge (€)} = d_i \\times p \\times u_i$$
        where:
        - **Distance Factor ($d_i$)**: Entry-to-exit great-circle distance inside charging zone $i$ (in kilometers, minus $20\\text{ km}$ for each take-off or landing within the zone) divided by $100$:
          $$d_i = \\frac{D_{i,\\text{km}}}{100}$$
        - **Aircraft Weight Factor ($p$)**: Derived from the aircraft type's certified Maximum Take-Off Weight ($\\text{MTOW}$ in metric tons) via OpenAP:
          $$p = \\sqrt{\\frac{\\text{MTOW (tons)}}{50}}$$
        - **National Unit Rate ($u_i$)**: Official Eurocontrol CRCO unit rate in **Euros (`€`) per Service Unit (`SU`)** for FIR state $i$ (e.g., UK, Switzerland, Germany, France, Spain).
        - **Total Service Units (`SU`)**: $\\text{SU} = \\sum_i (d_i \\times p)$.

        #### 2. Weekly Jet-A Spot Fuel Pricing in Euros (`€`)
        - Historical U.S. Energy Information Administration (EIA) / Platts weekly Jet-A spot prices are matched to each flight's exact departure week and converted into **Euros (`€`)** using the historical ECB reference exchange rate for the archive period (`0.91 €/$` for November 2016; `0.92 €/$` for October 2025).

        #### 3. Route-Charge Avoidance vs. Fuel Burn Trade-Off
        - Airlines occasionally file longer horizontal routes (`KPI04 KEP`) to skirt FIRs with high unit rates ($u_i > €95/\\text{SU}$). The Corridor CRCO Economics scanner compares the shortest-path and longest-path flights on any city pair to test whether CRCO fee savings exceed or are outweighed by the extra Jet-A fuel cost (`€`) and $\\text{CO}_2$ emissions.
        """
            )

        with ref_sub5:
            st.subheader("Prescriptive Operations Research & Runway Safety Physics")
            st.markdown(
                """
        #### 1. Turnaround Buffer & Reactionary Delay Contagion (Scenarios A & C)
        - Chains aircraft tail numbers (`AC_Registration`) chronologically across their daily multi-leg rotations.
        - **Reactionary Delay Propagation**: If inbound leg $k$ arrives with delay $D_{\\text{arr},k}$ and scheduled ground turnaround slack above the minimum aircraft turn time ($T_{\\min}$) is $S_k$, then downstream departure delay on leg $k+1$ is:
          $$D_{\\text{react},k+1} = \\max(0,\\; D_{\\text{arr},k} - S_k)$$
        - **Cost of Delay**: Quantified at the Eurocontrol Standard Inputs benchmark of **€88.00 EUR per minute** of tactical delay (crew, passenger EU261 compensation, maintenance, and missed connections).

        #### 2. Prescriptive Hub Tail-Swap Recovery Optimizer (Scenario D)
        - Scans all turnarounds across hubs and dates to find pairs of same-operator, same-aircraft-type rotations (`Delayed Tail A ⇄ Donor Tail B`) on the ground simultaneously at the same hub.
        - Evaluates the counterfactual swap of outbound legs `(Leg A_out, Leg B_out)` and recommends every conflict-free swap that yields a strictly positive net reduction in total reactionary departure delay minutes and Euro (`€`) delay cost.

        #### 3. Ration-by-Schedule Ground Delay Program (`GDP` / ATFM Slot) Optimizer (Scenario F)
        - Models an arrival capacity constraint where hourly arrival demand exceeds the declared Airport Arrival Rate (`AAR`).
        - Assigns ground delay slots in **Ration-by-Schedule (`RBS`)** order so that excess arrival queuing is absorbed at the departure gate/ground (`7% idle/gate fuel rate`) rather than in airborne terminal holding stacks inside the 40nm ASMA cylinder (`100% holding flight fuel rate`), reporting net fuel (`kg`), $\\text{CO}_2$ (`kg`), and `€ EUR` savings.

        #### 4. Runway Crosswind Vector Decomposition & Braking Safety Margin (Scenario B & Tab 5)
        - Decomposes surface wind $(V_{\\text{wind}}, \\theta_{\\text{wind}})$ against each physical runway threshold heading $\\psi_{\\text{rwy}}$ from `runways.csv`:
          $$V_{\\text{crosswind}} = |V_{\\text{wind}} \\cdot \\sin(\\theta_{\\text{wind}} - \\psi_{\\text{rwy}})|, \\qquad V_{\\text{headwind}} = V_{\\text{wind}} \\cdot \\cos(\\theta_{\\text{wind}} - \\psi_{\\text{rwy}})$$
        - Computes **Density Altitude** from airport elevation, QNH pressure altitude, and ISA temperature deviation, and scales dry vs. wet/contaminated Stopping Distance against the published Landing Distance Available (`LDA`).
        """
            )


# =============================================================================
# 6. GENERAL ECONOMIC VIEWS (€)
# =============================================================================
with tab_cat_econ:
    st.caption(
        "Airspace economics, direct operating cost quantification, Eurocontrol CRCO navigation fee avoidance trade-offs, "
        "twin-flight fuel disparity benchmarking, and longitudinal recurring series diurnal delay trends in Euros (€)."
    )
    render_page_export_bar("general_economics", "cat_econ")
    econ_sub1, econ_sub2 = st.tabs(
        [
            "👯 Twin-Flight Efficiency Benchmark",
            "📅 Recurring Series & Diurnal Trends",
        ]
    )

    with econ_sub1:
        st.header("Twin-Flight Efficiency Benchmarking & Dual-Trajectory Map")
        st.markdown(
            "Compare and contrast the most and least efficient flights between the same city pair operating the same aircraft type. "
            "Reveals route detours, airspace bypasses, divergence coordinates, and excess fuel consumption in Euros (`€`)."
        )
        col_t1, col_t2, col_t3 = st.columns(3)
        t_adep = col_t1.text_input("Origin (ADEP)", "RKSI", key=f"t_adep_{selected_period}").strip().upper()
        t_ades = col_t2.text_input("Destination (ADES)", "EGLL", key=f"t_ades_{selected_period}").strip().upper()
        t_ac = col_t3.text_input("Aircraft Type", "B77W", key=f"t_ac_{selected_period}").strip().upper()

        twin_comp = compare_twin_flights(ds, adep=t_adep, ades=t_ades, ac_type=t_ac)
        if not twin_comp:
            st.warning(f"Not enough matching flights for `{t_adep} ➔ {t_ades}` (`{t_ac}`) in period `{selected_period}`. Try another city pair or aircraft type.")
        else:
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            col_b1.metric("Distance Disparity", f"+{twin_comp.distance_penalty_nm:.1f} nm")
            col_b2.metric("Air Time Disparity", f"+{twin_comp.air_time_penalty_min:.1f} min")
            col_b3.metric("Excess Fuel Burned", f"+{twin_comp.excess_fuel_burned_kg:,.1f} kg")
            col_b4.metric("Excess Fuel Cost", f"€{twin_comp.excess_fuel_cost_eur:,.2f}")

            st.subheader("🗺️ Dual-Trajectory Comparison Map (Green = Most Efficient Twin | Red = Least Efficient Twin)")
            twin_map = render_twin_comparison_map(ds, twin_comp)
            st_folium(
                twin_map,
                width=1200,
                height=500,
                returned_objects=[],
                key=f"twin_map_{selected_period}_{twin_comp.fast_id}_{twin_comp.slow_id}",
            )

            if twin_comp.divergence_lat and twin_comp.divergence_lon:
                st.warning(
                    f"⚠️ **Lateral Route Divergence Point:** `{twin_comp.divergence_lat:.3f}°, {twin_comp.divergence_lon:.3f}°` "
                    f"(marked in orange on the map where the two trajectories separate by >30 nm)."
                )

            st.subheader("Operational Disparity Breakdown")
            twin_table = [
                {
                    "Metric": "ECTRL Flight ID",
                    "High-Efficiency Flight (Green)": str(twin_comp.fast_id),
                    "Low-Efficiency Flight (Red)": str(twin_comp.slow_id),
                    "Delta": "-",
                },
                {
                    "Metric": "Airline Operator",
                    "High-Efficiency Flight (Green)": twin_comp.fast_operator,
                    "Low-Efficiency Flight (Red)": twin_comp.slow_operator,
                    "Delta": "-",
                },
                {
                    "Metric": "Flown Distance",
                    "High-Efficiency Flight (Green)": f"{twin_comp.fast_actual_distance_nm:.1f} nm",
                    "Low-Efficiency Flight (Red)": f"{twin_comp.slow_actual_distance_nm:.1f} nm",
                    "Delta": f"+{twin_comp.distance_penalty_nm:.1f} nm",
                },
                {
                    "Metric": "Air Time",
                    "High-Efficiency Flight (Green)": f"{twin_comp.fast_air_time_min:.1f} min",
                    "Low-Efficiency Flight (Red)": f"{twin_comp.slow_air_time_min:.1f} min",
                    "Delta": f"+{twin_comp.air_time_penalty_min:.1f} min",
                },
                {
                    "Metric": "Extension Ratio (KEA)",
                    "High-Efficiency Flight (Green)": f"{twin_comp.fast_extension_ratio:.3f}x",
                    "Low-Efficiency Flight (Red)": f"{twin_comp.slow_extension_ratio:.3f}x",
                    "Delta": f"+{twin_comp.slow_extension_ratio - twin_comp.fast_extension_ratio:.3f}x",
                },
                {
                    "Metric": "Fuel Burned",
                    "High-Efficiency Flight (Green)": f"{twin_comp.fast_fuel_kg:,.1f} kg",
                    "Low-Efficiency Flight (Red)": f"{twin_comp.slow_fuel_kg:,.1f} kg",
                    "Delta": f"+{twin_comp.excess_fuel_burned_kg:,.1f} kg",
                },
                {
                    "Metric": "Estimated Fuel Cost (€)",
                    "High-Efficiency Flight (Green)": "-",
                    "Low-Efficiency Flight (Red)": "-",
                    "Delta": f"+€{twin_comp.excess_fuel_cost_eur:,.2f} EUR",
                },
            ]
            st.dataframe(pd.DataFrame(twin_table), use_container_width=True)

        # ----------------- TAB 8: What-If & Stress Tests -----------------

    with econ_sub2:
        st.header("📅 Recurring Flight Series & Diurnal Time-of-Day Assessment")
        st.markdown(
            "Track and compare the same flight (or very similar flights on the same city pair) as they occurred each day. "
            "Evaluates day-over-day operational variability, flown distance dispersion, and diurnal congestion escalation across morning, afternoon, and evening departure banks."
        )
        col_r1, col_r2, col_r3 = st.columns(3)
        r_adep = col_r1.text_input("Origin (ADEP)", value="EGLL", key=f"r_adep_{selected_period}").strip().upper()
        r_ades = col_r2.text_input("Destination (ADES)", value="KORD", key=f"r_ades_{selected_period}").strip().upper()
        r_op = col_r3.text_input("Airline Operator (Optional)", value="BAW", key=f"r_op_{selected_period}").strip().upper()

        col_r4, col_r5 = st.columns(2)
        r_ac = col_r4.text_input("Aircraft Type (Optional)", value="", placeholder="e.g. B77W, B744, A320", key=f"r_ac_{selected_period}").strip().upper()
        r_tod = col_r5.selectbox(
            "Time-of-Day Bank Filter",
            ["All", "Morning (06:00-12:00)", "Afternoon (12:00-18:00)", "Evening (18:00-24:00)", "Night (00:00-06:00)"],
            key=f"r_tod_{selected_period}",
        )

        if st.button("Analyze Recurring Series Across Period", key=f"btn_rec_series_{selected_period}"):
            with st.spinner(f"Extracting longitudinal recurring series for {r_adep} ➔ {r_ades}..."):
                rep = analyze_recurring_flight_series(
                    ds,
                    adep=r_adep,
                    ades=r_ades,
                    operator=r_op if r_op else None,
                    ac_type=r_ac if r_ac else None,
                    time_of_day=r_tod,
                )

            if not rep:
                st.warning(f"Not enough recurring flights found for corridor {r_adep} ➔ {r_ades} in period {selected_period}.")
            else:
                rm1, rm2, rm3, rm4 = st.columns(4)
                rm1.metric("Occurrences Analyzed", f"{rep.total_flights} flights")
                rm2.metric("Mean Arrival Delay", f"{rep.mean_arr_delay_min:+.1f} min", f"±{rep.std_arr_delay_min:.1f}m (std)")
                rm3.metric("Flown Distance Span", f"{rep.mean_distance_nm:.0f} nm", f"Δ{rep.distance_span_nm:.0f} nm span")
                rm4.metric("Air Time Dispersion", f"{rep.mean_air_time_min:.0f} min", f"Δ{rep.air_time_span_min:.0f} min span")

                col_best, col_worst = st.columns(2)
                col_best.success(
                    f"🏆 **Best Performing Day:** `{rep.best_performing_day['date']}` "
                    f"({rep.best_performing_day['flights']} flights, avg delay **{rep.best_performing_day['avg_arr_delay_min']:+.1f} min**)"
                )
                col_worst.error(
                    f"⚠️ **Worst Performing Day:** `{rep.worst_performing_day['date']}` "
                    f"({rep.worst_performing_day['flights']} flights, avg delay **{rep.worst_performing_day['avg_arr_delay_min']:+.1f} min**)"
                )

                st.subheader("Diurnal Time-of-Day Departure Bank Congestion Profile")
                diurnal_data = [
                    {
                        "Departure Bank": b.bank_name,
                        "Flights": b.flight_count,
                        "Avg Departure Delay": f"{b.avg_dep_delay_min:+.1f} min",
                        "Avg Arrival Delay": f"{b.avg_arr_delay_min:+.1f} min",
                        "Avg Air Time": f"{b.avg_air_time_min:.1f} min",
                    }
                    for b in rep.diurnal_banks
                ]
                st.dataframe(pd.DataFrame(diurnal_data), use_container_width=True)

                if rep.hourly_delay_curve:
                    st.subheader("Hourly Departure Delay Escalation Curve (UTC Departure Hour)")
                    h_df = pd.DataFrame(rep.hourly_delay_curve).set_index("hour")
                    st.line_chart(h_df[["avg_arr_delay_min", "avg_dep_delay_min"]])

                st.subheader("Day-by-Day Recurring Occurrences")
                occ_df = pd.DataFrame([
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
                    for o in rep.occurrences
                ])
                st.dataframe(occ_df, use_container_width=True)

# =============================================================================
# 7. EXPERIMENTAL AI & MACHINE LEARNING LAB
# =============================================================================
with tab_cat_ai_lab:
    st.caption(
        "Experimental machine learning, deep neural surrogate architectures, and multi-agent operations research "
        "advancing ATM research beyond conventional heuristics: Flight2Vec embeddings, Physics-Informed Neural Networks (PINNs), "
        "Spatio-Temporal Graph Neural Networks (ST-GNNs), 4D counterfactual airspace closure simulations, and Multi-Agent RL ATFM gaming."
    )

    ai_sub1, ai_sub2, ai_sub3, ai_sub4, ai_sub5 = st.tabs(
        [
            "🧬 Flight2Vec Latent Embeddings",
            "⚖️ PINN Mass & Contrail Surrogate",
            "🌐 ST-GNN Delay Contagion Network",
            "🌪️ 4D Trajectory & FIR Closure Simulator",
            "🎮 Multi-Agent RL ATFM Slot Gaming",
        ]
    )

    # ----------------- SUB-TAB 1: Flight2Vec Latent Embeddings -----------------
    with ai_sub1:
        st.header("🧬 Flight2Vec: Self-Supervised Trajectory Foundation Embeddings")
        st.markdown(
            "Self-supervised contrastive learning foundation model mapping multi-dimensional 4D radar telemetry, "
            "climb/descent energy states, and operational context into a dense 128-dimensional unit hypersphere $\\mathbb{S}^{127}$. "
            "Enables zero-shot operational twin search, semantic flight classification, and unsupervised anomaly detection."
        )

        f2v = Flight2VecEngine(ds, period=selected_period)
        cached_count = f2v.get_cached_count()

        col_c_info, col_c_btn1, col_c_btn2 = st.columns([2.0, 1.0, 1.0])
        with col_c_info:
            if cached_count > 0:
                st.success(
                    f"⚡ **Pre-Vectorized Corpus Active**: **{cached_count:,} flights** cached in `data/cache/flight2vec/`. "
                    f"Twin queries search across the entire pre-vectorized corpus via matrix BLAS dot product in **<5 ms**."
                )
            else:
                st.info(
                    "💡 **Pre-Vectorization Available**: Pre-vectorizing flights in batch persists 128D embeddings to disk, "
                    "enabling instantaneous full-corpus operational twin retrieval."
                )
        with col_c_btn1:
            if st.button("⚡ Vectorize 25k Batch", key=f"btn_pre_vec_25k_{selected_period}"):
                with st.spinner("Pre-vectorizing 25,000 flights via PyTorch batch matrix tensor operations..."):
                    n_vec = f2v.build_vector_cache(limit=25000)
                st.success(f"Successfully vectorized and cached {n_vec:,} flights!")
                st.rerun()
        with col_c_btn2:
            if st.button("🚀 Vectorize All in DB", key=f"btn_pre_vec_all_{selected_period}"):
                with st.spinner("Pre-vectorizing entire flight database via high-speed neural matrix tensor operations..."):
                    n_vec = f2v.build_vector_cache(limit=None)
                st.success(f"Successfully vectorized and cached all {n_vec:,} flights in the database!")
                st.rerun()

        # ----------------- FLIGHT SEARCH & FILTER ENGINE -----------------
        with st.expander("🔎 Search & Filter Flights (ADEP, ADES, Operator, Season & Delay)", expanded=False):
            st.markdown(
                "Filter and inspect candidate flights across corridors, airline operators, seasons, and operational delay bands "
                "to select a target flight for Flight2Vec embedding and operational twin analysis."
            )
            col_s1, col_s2, col_s3 = st.columns(3)
            s_adep = col_s1.text_input("Origin Airport (ADEP)", value="", placeholder="e.g. EGLL, LFPG, EHAM", key=f"f2v_s_adep_{selected_period}").strip().upper()
            s_ades = col_s2.text_input("Destination Airport (ADES)", value="", placeholder="e.g. EDDF, LEMD, LIRF", key=f"f2v_s_ades_{selected_period}").strip().upper()
            s_op = col_s3.text_input("Airline Operator", value="", placeholder="e.g. BAW, RYR, DLH, AFR", key=f"f2v_s_op_{selected_period}").strip().upper()

            col_s4, col_s5, col_s6 = st.columns([1.5, 1.5, 1])
            avail_periods = list_available_periods()
            s_season = col_s4.selectbox(
                "Season / Traffic Cycle",
                ["Current Cycle", "🌱 Early Spring / Winter (Feb)", "☀️ Summer Peak (Jun)", "🍂 Autumn Shoulder (Oct)"],
                key=f"f2v_s_season_{selected_period}",
            )
            s_delay = col_s5.selectbox(
                "Punctuality / Delay Status",
                ["All Flights", "On-Time (≤ 0 min)", "Minor Delay (0 to 15 min)", "Moderate Delay (15 to 45 min)", "Severe Delay (> 45 min)"],
                key=f"f2v_s_delay_{selected_period}",
            )
            with col_s6:
                st.write("")
                st.write("")
                btn_do_search = st.button("🔎 Search Flights", key=f"btn_f2v_search_{selected_period}")

            if btn_do_search:
                target_period = selected_period
                if "Feb" in s_season:
                    matches = [p for p in avail_periods if p.endswith("02")]
                    if matches:
                        target_period = matches[-1]
                elif "Jun" in s_season:
                    matches = [p for p in avail_periods if p.endswith("06")]
                    if matches:
                        target_period = matches[-1]
                elif "Oct" in s_season:
                    matches = [p for p in avail_periods if p.endswith("10")]
                    if matches:
                        target_period = matches[-1]

                search_ds = ds if target_period == selected_period else FlightDataset(period=target_period)

                where_clauses = ["1=1"]
                if s_adep:
                    where_clauses.append(f"adep = '{s_adep}'")
                if s_ades:
                    where_clauses.append(f"ades = '{s_ades}'")
                if s_op:
                    where_clauses.append(f"ac_operator = '{s_op}'")
                if s_delay == "On-Time (≤ 0 min)":
                    where_clauses.append("arr_delay_min <= 0")
                elif s_delay == "Minor Delay (0 to 15 min)":
                    where_clauses.append("arr_delay_min BETWEEN 0 AND 15")
                elif s_delay == "Moderate Delay (15 to 45 min)":
                    where_clauses.append("arr_delay_min BETWEEN 15 AND 45")
                elif s_delay == "Severe Delay (> 45 min)":
                    where_clauses.append("arr_delay_min > 45")

                where_sql = " AND ".join(where_clauses)
                search_results = search_ds.query(f"""
                    SELECT
                        ectrl_id, adep, ades, ac_operator, ac_type, ac_reg,
                        actual_distance_nm, actual_air_time_min, dep_delay_min, arr_delay_min,
                        requested_fl, actual_off_block_time, actual_arrival_time
                    FROM flights
                    WHERE {where_sql}
                    ORDER BY actual_off_block_time DESC
                    LIMIT 40;
                """)
                st.session_state[f"f2v_search_results_{selected_period}"] = search_results

            stored_results = st.session_state.get(f"f2v_search_results_{selected_period}")
            if stored_results is not None:
                if stored_results.empty:
                    st.warning("No flights matched the specified search criteria.")
                else:
                    st.success(f"Found {len(stored_results)} matching candidate flights.")
                    res_options = {
                        f"#{r['ectrl_id']}: {r['ac_operator']} ({r['adep']} ➔ {r['ades']}, {r['ac_type']}, Dist: {float(r['actual_distance_nm'] or 0):.0f}nm, Delay: {float(r['arr_delay_min'] or 0):+.0f}m, Dep: {str(r['actual_off_block_time'])[:16]})": int(r['ectrl_id'])
                        for _, r in stored_results.iterrows()
                    }
                    picked_label = st.selectbox(
                        "Select Target Flight from Results",
                        options=list(res_options.keys()),
                        key=f"f2v_picked_select_{selected_period}",
                    )
                    chosen_id = res_options[picked_label]
                    st.session_state[f"f2v_selected_id_{selected_period}"] = chosen_id

        # Determine effective target flight ID
        effective_target_id = st.session_state.get(f"f2v_selected_id_{selected_period}", int(active_flight_id))
        target_flt = ds.get_flight(effective_target_id)
        if not target_flt:
            flt_df = ds.query(f"SELECT * FROM flights WHERE ectrl_id = {effective_target_id} LIMIT 1;")
            target_flt = flt_df.iloc[0].to_dict() if not flt_df.empty else None

        # ----------------- SELECTED FLIGHT DETAILS CARD -----------------
        if target_flt:
            st.markdown(
                f"#### ✈️ Selected Target Flight: `{target_flt.get('ac_operator', 'UNK')} #{effective_target_id}` "
                f"({target_flt.get('adep')} ➔ {target_flt.get('ades')})"
            )
            det_c1, det_c2, det_c3, det_c4 = st.columns(4)
            det_c1.metric("Route & Operator", f"{target_flt.get('adep')} ➔ {target_flt.get('ades')}", str(target_flt.get("ac_operator")))
            det_c2.metric("Aircraft & Tail", f"{target_flt.get('ac_type')}", f"Tail: {target_flt.get('ac_reg', 'N/A')}")
            det_c3.metric(
                "Flown Distance / Air Time",
                f"{float(target_flt.get('actual_distance_nm') or 0):.0f} nm",
                f"{float(target_flt.get('actual_air_time_min') or 0):.1f} min air time",
            )
            det_c4.metric(
                "Punctuality Profile",
                f"{float(target_flt.get('arr_delay_min') or 0):+.1f} min arr",
                f"Dep: {float(target_flt.get('dep_delay_min') or 0):+.1f} min",
            )
            st.caption(
                f"Off-block: `{target_flt.get('actual_off_block_time')}` UTC | Arrival: `{target_flt.get('actual_arrival_time')}` UTC | "
                f"Requested Level: `FL{target_flt.get('requested_fl')}` | Flight Type: `{target_flt.get('flight_type', 'IFR')}` | "
                f"Segment: `{target_flt.get('market_segment', 'Commercial Air Transport')}`"
            )

        col_f1, col_f2, col_f3 = st.columns([1.5, 1, 1])
        with col_f1:
            f2v_target_input = st.text_input(
                "Target Flight ID (ECTRL ID)",
                value=str(effective_target_id),
                key=f"f2v_flt_input_{selected_period}",
            )
        with col_f2:
            f2v_top_k = st.slider("Top Twins to Retrieve", 3, 10, 5, key=f"f2v_top_k_{selected_period}")
        with col_f3:
            f2v_pool_size = st.slider("Search Corpus Sample Size (Fallback)", 50, 1000, 250, key=f"f2v_pool_{selected_period}")

        if st.button("Generate 128D Embedding & Retrieve Latent Twins", key=f"btn_f2v_{selected_period}"):
            try:
                target_f_id = int(f2v_target_input.strip())
            except ValueError:
                target_f_id = int(effective_target_id)

            with st.spinner(f"Computing 128D Flight2Vec projection for flight #{target_f_id}..."):
                emb = f2v.embed_flight(target_f_id)

            if emb is None:
                st.warning(f"Flight ID #{target_f_id} not found in period {selected_period}.")
            else:
                st.subheader(f"📊 Latent Vector Signature (ECTRL #{target_f_id})")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Latent Dimensions", "128D Space")
                m2.metric("Hypersphere Norm ||z||₂", f"{np.linalg.norm(emb):.4f}")
                m3.metric("Latent Variance", f"{np.var(emb):.4f}")
                m4.metric("Mean Activation", f"{np.mean(emb):+.4f}")

                # Activation chart across 128 dimensions
                act_df = pd.DataFrame({
                    "Dimension": [f"D{i+1}" for i in range(len(emb))],
                    "Activation": emb,
                })
                st.bar_chart(act_df.set_index("Dimension"), height=180)

                # Twin retrieval
                with st.spinner(f"Searching latent space across candidate flights..."):
                    twins = f2v.find_operational_twins(target_f_id, search_sample_size=f2v_pool_size, top_k=f2v_top_k)

                if twins:
                    st.subheader(f"👯 Latent Operational Twins (Cosine Similarity on $\\mathbb{{S}}^{{127}}$)")
                    st.caption("Identifies flights exhibiting nearest-neighbor 4D profile, energy dissipation, and delay dynamics.")
                    twin_rows = [
                        {
                            "Flight ID": t.ectrl_id,
                            "Route": t.route,
                            "Operator": t.operator,
                            "Aircraft": t.ac_type,
                            "Distance": f"{t.flown_distance_nm:.0f} nm",
                            "Cruise FL": f"FL{t.cruise_fl:.0f}",
                            "Arr Delay": f"{t.arr_delay_min:+.1f} min",
                            "Cosine Sim (cos θ)": f"{t.cosine_similarity:.4f}",
                            "Match Score": f"{t.similarity_pct:.1f}%",
                        }
                        for t in twins
                    ]
                    st.dataframe(pd.DataFrame(twin_rows), use_container_width=True)

                    # ----------------- TOP 5 TWIN SIMILARITY EXPLANATIONS -----------------
                    st.subheader("💡 Why They Are Similar (Operational Twin Similarity Explanations)")
                    st.caption("Multi-dimensional operational alignment explaining why the Flight2Vec foundation model paired each twin in the 128D latent manifold.")
                    for idx, t in enumerate(twins, 1):
                        with st.expander(
                            f"#{idx} Twin: {t.operator} #{t.ectrl_id} ({t.route}) — Match Score: {t.similarity_pct:.1f}% (cos θ = {t.cosine_similarity:.4f})",
                            expanded=(idx <= 2),
                        ):
                            st.markdown(f"**Operational Summary:** {t.explanation}")
                            if t.similarity_reasons:
                                st.markdown("**Key Dimensional Alignments:**")
                                for r_item in t.similarity_reasons:
                                    st.markdown(f"- {r_item}")

                # 2D PCA projection of sample flights
                with st.spinner("Projecting latent manifold via Principal Component Analysis (PCA)..."):
                    pca_df = f2v.project_latent_space_2d(sample_size=40)

                if not pca_df.empty:
                    st.subheader("🗺️ 2D Latent Manifold Projection (PCA)")
                    st.caption("Visualizing high-dimensional trajectory clusters: flights with similar routing, cruise altitude, and operational pacing group together.")
                    scatter_chart = (
                        alt.Chart(pca_df)
                        .mark_circle(size=80)
                        .encode(
                            x=alt.X("PCA_Dim_1:Q", title="Principal Component 1 (Kinematic Energy / Distance)"),
                            y=alt.Y("PCA_Dim_2:Q", title="Principal Component 2 (Delay / Pacing Dynamics)"),
                            color=alt.Color("Operator:N", title="Operator"),
                            tooltip=["Flight_Label", "Route", "Operator", "Arr_Delay_Min"],
                        )
                        .properties(height=340)
                        .interactive()
                    )
                    st.altair_chart(scatter_chart, use_container_width=True)
                    st.altair_chart(scatter_chart, use_container_width=True)

    # ----------------- SUB-TAB 2: PINN Mass & Contrail Surrogate -----------------
    with ai_sub2:
        st.header("⚖️ Physics-Informed Neural Networks (PINN) & Fast Contrail Surrogate")
        st.markdown(
            "Combines physical conservation laws with deep neural architectures: **(1)** inverting confidential aircraft takeoff mass and payload "
            "directly from kinematic climb radar data, and **(2)** evaluating thermodynamic contrail radiative forcing at microsecond latency ($<15\\,\\mu\\text{s}$)."
        )

        col_p1, col_p2 = st.columns(2)

        # Module A: PINN Mass Inversion
        with col_p1:
            st.subheader("1. In-Flight Aircraft Mass & Payload Inversion (PINN)")
            st.markdown(
                "Aircraft Takeoff Gross Weight (TGW) and passenger load factors are confidential airline commercial data never broadcast over ADS-B. "
                "The PINN inverts unknown mass $m$ directly from climb kinematics by enforcing Newton's second law: "
                "$$T(M, h) - D(C_L, C_D, m) - mg\\sin\\gamma = m\\frac{dV}{dt}$$"
            )

            pinn_target_input = st.text_input(
                "Flight to Invert Mass",
                value=str(active_flight_id),
                key=f"pinn_flt_input_{selected_period}",
            )

            if st.button("Invert Aircraft Mass via Physics Constraints", key=f"btn_pinn_mass_{selected_period}"):
                try:
                    pinn_f_id = int(pinn_target_input.strip())
                except ValueError:
                    pinn_f_id = int(active_flight_id)

                with st.spinner(f"Inverting aerodynamic mass balance for flight #{pinn_f_id}..."):
                    inverter = PINNMassInverter(ds, period=selected_period)
                    mass_res = inverter.invert_flight_mass(pinn_f_id)

                if not mass_res:
                    st.warning(f"Could not invert mass for flight #{pinn_f_id}.")
                else:
                    pm1, pm2 = st.columns(2)
                    pm1.metric("Aircraft Type", mass_res.ac_type)
                    pm1.metric("Operating Empty Weight (OEW)", f"{mass_res.oew_kg:,.0f} kg")
                    pm1.metric("Maximum Takeoff Weight (MTOW)", f"{mass_res.mtow_kg:,.0f} kg")

                    pm2.metric("Inverted Takeoff Weight (TGW)", f"{mass_res.estimated_takeoff_weight_kg:,.0f} kg")
                    pm2.metric("Estimated Commercial Payload", f"{mass_res.estimated_payload_kg:,.0f} kg")
                    pm2.metric("Inferred Passenger Load Factor", f"{mass_res.estimated_load_factor_pct:.1f}%")

                    st.info(
                        f"🔬 **Physics Residual Error:** `{mass_res.physics_residual_error:.4f}` | "
                        f"**Climb ROCD:** `{mass_res.climb_rate_fpm:,.0f} fpm` | "
                        f"**Acceleration:** `{mass_res.acceleration_ms2:+.3f} m/s²` | "
                        f"**Inversion Confidence:** `{mass_res.confidence_score * 100:.0f}%`"
                    )

        # Module B: Contrail Neural Surrogate
        with col_p2:
            st.subheader("2. Microsecond Contrail Neural Surrogate Engine")
            st.markdown(
                "Thermodynamic Schmidt-Appleman and Ice Super-Saturated Region (ISSR) radiative forcing models "
                "require microsecond-latency evaluation ($<15\\,\\mu\\text{s}$) to enable real-time en-route tactical ATC re-clearing."
            )

            cs_fl = st.slider("Waypoint Flight Level (FL)", 280, 420, 360, step=10, key=f"cs_fl_{selected_period}")
            cs_temp = st.slider("Ambient Temperature (°C)", -65.0, -35.0, -52.0, step=0.5, key=f"cs_temp_{selected_period}")
            cs_rhi = st.slider("Relative Humidity over Ice (RHi %)", 60.0, 150.0, 115.0, step=1.0, key=f"cs_rhi_{selected_period}")
            cs_bpr = st.slider("Engine Bypass Ratio (BPR)", 5.0, 15.0, 11.0, step=0.5, key=f"cs_bpr_{selected_period}")

            if st.button("Evaluate Contrail Radiative Forcing (<15 μs)", key=f"btn_cs_{selected_period}"):
                surrogate = ContrailNeuralSurrogate()
                c_out = surrogate.evaluate_atmospheric_point(
                    temperature_c=float(cs_temp),
                    rh_ice_pct=float(cs_rhi),
                    flight_level=float(cs_fl),
                    bypass_ratio=float(cs_bpr),
                )

                if c_out.climate_forcing_severity == "SEVERE_CLIMATE_FORCING":
                    st.error(
                        f"⚠️ **Severe Persistent Warming Contrail!** Instantaneous RF: **{c_out.instantaneous_rf_mw_m2:+.2f} mW/m²** "
                        f"(Optical Depth τ = {c_out.optical_depth:.3f}). Tactical vertical offset advised: descend 2,000 ft to exit ISSR layer."
                    )
                elif c_out.climate_forcing_severity == "MODERATE_WARMING":
                    st.warning(
                        f"⚠️ **Moderate Warming Contrail.** Instantaneous RF: **{c_out.instantaneous_rf_mw_m2:+.2f} mW/m²** "
                        f"(Optical Depth τ = {c_out.optical_depth:.3f})."
                    )
                elif c_out.climate_forcing_severity == "LOW_IMPACT":
                    st.info(
                        f"ℹ️ **Short-Lived / Low Impact Contrail.** Instantaneous RF: **{c_out.instantaneous_rf_mw_m2:+.2f} mW/m²**."
                    )
                else:
                    st.success(
                        f"✅ **Clear Air - No Persistent Contrail Formation.** Ambient air is ice-subsaturated (RHi < 100%)."
                    )

                cm1, cm2, cm3 = st.columns(3)
                cm1.metric("Schmidt-Appleman Criterion", "Satisfied" if c_out.schmidt_appleman_satisfied else "Not Met")
                cm2.metric("ISSR Ice Super-Saturation", "Persistent (RHi ≥ 100%)" if c_out.is_persistent_issr else "Evaporative")
                cm3.metric("Neural Inference Latency", f"{c_out.inference_latency_us:.1f} μs", "Sub-millisecond")

    # ----------------- SUB-TAB 3: ST-GNN Delay Contagion Network -----------------
    with ai_sub3:
        st.header("🌐 Spatio-Temporal Graph Neural Network (ST-GNN) Delay Contagion")
        st.markdown(
            "Simulates multi-hop network delay propagation and knock-on ripple shockwaves across the European airport "
            "and airspace sector graph $\\mathcal{G} = (\\mathcal{V}, \\mathcal{E}, W)$ using Spatio-Temporal Graph WaveNet architectures. "
            "Combines spatial diffusion message passing with temporal dilated causal convolutions across multi-hour forward lookaheads."
        )

        col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns([1.3, 1.1, 0.7, 0.7, 0.8])
        with col_g1:
            gnn_shock_node = st.selectbox(
                "Shock Origin Node (Airport Hub or En-Route Sector)",
                options=["EGLL", "LFPG", "EHAM", "EDDF", "LEMD", "LIRF", "EDDM", "LEBL", "EBBR", "EIDW", "EDUUUIR", "LFFFFIR", "EGTTFIR", "LSASFIR", "EDMMFIR"],
                key=f"gnn_shock_{selected_period}",
            )
        with col_g2:
            start_date_bound, end_date_bound = get_dataset_date_bounds(ds)
            default_gnn_date = pd.to_datetime(start_date_bound).date()
            try:
                row_op = ds.query(
                    "SELECT CAST(COALESCE(actual_off_block_time, filed_off_block_time) AS DATE) AS d "
                    "FROM flights GROUP BY 1 HAVING COUNT(*) > 500 ORDER BY 1 LIMIT 1"
                )
                if not row_op.empty:
                    default_gnn_date = pd.to_datetime(str(row_op.iloc[0]["d"])).date()
            except Exception:
                pass
            gnn_shock_date = st.date_input(
                "Shock Introduction Date",
                value=default_gnn_date,
                min_value=pd.to_datetime(start_date_bound).date(),
                max_value=pd.to_datetime(end_date_bound).date(),
                key=f"gnn_date_{selected_period}",
            )
        with col_g3:
            gnn_shock_hour = st.slider(
                "Shock Start (UTC)",
                0, 23, 10,
                key=f"gnn_hour_{selected_period}",
            )
        with col_g4:
            gnn_shock_dur = st.slider(
                "Duration (Hours)",
                1.0, 6.0, 2.0, step=0.5,
                key=f"gnn_dur_{selected_period}",
            )
        with col_g5:
            gnn_injected_delay = st.slider(
                "Injected Delay (Min)",
                15, 180, 60, step=15,
                key=f"gnn_delay_{selected_period}",
            )

        if st.button("Simulate Spatio-Temporal Delay Contagion", key=f"btn_st_gnn_{selected_period}"):
            with st.spinner(f"Propagating {gnn_injected_delay} min delay shock from {gnn_shock_node} on {gnn_shock_date} from {gnn_shock_hour:02d}:00 UTC (+{gnn_shock_dur}h) across European flight schedule..."):
                st_engine = SpatioTemporalDelayEngine(ds, period=selected_period)
                prop_res = st_engine.simulate_delay_propagation(
                    shock_origin=gnn_shock_node,
                    injected_delay_min=float(gnn_injected_delay),
                    shock_date=str(gnn_shock_date),
                    shock_time_hour=int(gnn_shock_hour),
                    shock_duration_hours=float(gnn_shock_dur),
                )

            gm1, gm2, gm3, gm4, gm5 = st.columns(5)
            gm1.metric("Shock Origin & Window", f"{prop_res.shock_origin_node} ({prop_res.shock_duration_hours:.1f}h)", f"{gnn_shock_hour:02d}:00 - {int(gnn_shock_hour + gnn_shock_dur):02d}:00 UTC")
            gm2.metric("Injected Shock", f"{prop_res.injected_shock_delay_min:.0f} min")
            gm3.metric("Total Network Reactionary Delay", f"{prop_res.total_network_reactionary_delay_hours:,.1f} delay hours")
            gm4.metric("Total Flights Impacted", f"{prop_res.total_flights_impacted:,} flights", f"{len(prop_res.sample_turnaround_rotations)} turnaround rotations")
            peak_downstream = [n for n in prop_res.top_vulnerable_nodes if n.node_id != prop_res.shock_origin_node]
            peak_node_name = peak_downstream[0].node_id if peak_downstream else "N/A"
            peak_react = peak_downstream[0].reactionary_delay_hours if peak_downstream else 0.0
            gm5.metric("Peak Downstream Hub Impact", peak_node_name, f"+{peak_react:.1f} reactionary hrs")

            if prop_res.origin_saturation_pct >= 85.0:
                st.warning(
                    f"⚠️ **High Runway Saturation Active ({prop_res.origin_saturation_pct:.1f}% at {prop_res.shock_origin_node})**: "
                    f"Scheduled movements at {prop_res.shock_origin_node} operate near physical runway throughput capacity "
                    f"with minimal spare recovery buffer ({prop_res.origin_spare_capacity:.1f} mvts/hr). "
                    f"Because departure and arrival movements cannot outstrip runway capacity, the point-queue bottleneck persists across multi-hour horizons (t=4h and t=6h), "
                    f"preventing rapid local dissipation and transmitting substantial rotational delay downstream."
                )

            st.subheader("📈 Multi-Horizon Delay Escalation Curve (1h, 2h, 4h, 6h Lookaheads)")
            st.caption("Tracks how initial disruption at the shock origin hub dissipates locally while propagating downstream to connected European hubs.")

            # Build line chart for top 6 affected nodes
            curve_records = []
            time_labels = [f"t={h}h" for h in prop_res.lookahead_hours]
            for node_state in prop_res.top_vulnerable_nodes[:6]:
                delays = prop_res.node_delay_matrix.get(node_state.node_id, [0.0] * len(time_labels))
                for h_idx, h_lbl in enumerate(time_labels):
                    val = delays[h_idx] if h_idx < len(delays) else 0.0
                    curve_records.append({
                        "Time Horizon": h_lbl,
                        "Node": node_state.node_id,
                        "Predicted Delay (min)": val,
                    })

            if curve_records:
                st.line_chart(
                    pd.DataFrame(curve_records).pivot(index="Time Horizon", columns="Node", values="Predicted Delay (min)")
                )

            st.subheader("🎯 Ranked Vulnerable Network Nodes & Contagion Risk")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Node ID": n.node_id,
                        "Node Type": n.node_type,
                        "Nominal Capacity": f"{n.nominal_hourly_capacity} mvts/hr",
                        "Scheduled Movements": f"{n.scheduled_movements} flights",
                        "Runway Saturation": f"{n.saturation_ratio_pct:.1f}%",
                        "Spare Capacity": f"{n.spare_hourly_capacity:.1f} mvts/hr",
                        "Baseline Delay": f"{n.baseline_delay_min:+.1f} min",
                        "t=0h Initial": f"{n.initial_delay_min:.1f} min",
                        "t=1h Delay": f"{n.predicted_delay_1h_min:.1f} min",
                        "t=2h Delay": f"{n.predicted_delay_2h_min:.1f} min",
                        "t=4h Delay": f"{n.predicted_delay_4h_min:.1f} min",
                        "t=6h Delay": f"{n.predicted_delay_6h_min:.1f} min",
                        "Reactionary Added": f"{n.reactionary_delay_hours:,.1f} delay hrs",
                        "Vulnerability Index": f"{n.vulnerability_index:.1f}",
                        "Contagion Risk Level": n.risk_level,
                    }
                    for n in prop_res.top_vulnerable_nodes
                ]),
                use_container_width=True,
            )

            if prop_res.sample_impacted_flights:
                st.subheader(f"✈️ Direct Downstream Flights Departing {prop_res.shock_origin_node} Disrupted by Shock")
                st.caption(f"Representative operational sample of {len(prop_res.sample_impacted_flights)} flights out of {prop_res.total_flights_impacted:,} total network movements impacted by the {prop_res.shock_origin_node} disruption.")
                impacted_df = pd.DataFrame([
                    {
                        "Flight ID": f["ectrl_id"],
                        "Operator": f["operator"],
                        "Route": f["route"],
                        "Aircraft": f["ac_type"],
                        "Airframe Tail": f["ac_reg"],
                        "Scheduled Off-Block": f["off_block_time"],
                        "Baseline Delay": f"{f['baseline_delay_min']:+.0f} min",
                        "Induced Shock Delay": f"+{f['induced_delay_min']:.0f} min",
                        "Predicted Total Delay": f"{f['predicted_total_delay_min']:.0f} min",
                        "Downstream Arrival Hub": f["downstream_node"],
                    }
                    for f in prop_res.sample_impacted_flights
                ])
                st.dataframe(impacted_df, use_container_width=True)

            if prop_res.sample_turnaround_rotations:
                st.subheader("🔄 Aircraft Tail Tracking: Downstream Rotations & Liberated Slot Capacity")
                st.caption(
                    "Follows physical aircraft tails (`ac_reg`) across scheduled daily rotations: when an aircraft never takes off or arrives with severe delay, "
                    "downstream departures relying on that tail cannot operate, spreading contagion to other hubs. "
                    "Crucially, because those unserved flights cannot fly, they vacate arrival and departure slots, liberating runway capacity at congested hubs to accelerate backlog clearance."
                )

                tm1, tm2, tm3, tm4 = st.columns(4)
                tm1.metric("Unserved Rotations", f"{getattr(prop_res, 'unserved_rotations_count', 0)} flights", "Missing Aircraft Tails")
                tm2.metric("Freed Runway Slots", f"{getattr(prop_res, 'total_freed_slots_count', 0)} slots", "Liberated Runway Capacity")
                tm3.metric("Downstream Hubs Impacted", f"{len(getattr(prop_res, 'downstream_airports_impacted', []))} airports", "Contagion Footprint")
                tm4.metric("Queue Drainage Acceleration", f"+{getattr(prop_res, 'backlog_acceleration_pct', 0.0):.1f}%", "Accelerated Clearance")

                turns_df = pd.DataFrame([
                    {
                        "Aircraft Tail": tr.get("ac_reg", "—"),
                        "Leg 1 Initial Flight": f"#{tr['leg1_id']} ({tr['leg1_route']})",
                        "Downstream Rotation": f"#{tr['leg2_id']} ({tr['leg2_route']})",
                        "Downstream Hub": tr.get("downstream_outstation", tr["leg2_route"].split("➔")[-1].strip()),
                        "Downstream Status": tr.get("downstream_status", "REACTIONARY_ROTATIONAL_DELAY"),
                        "Reactionary Delay": f"+{tr['reactionary_delay_min']:.0f} min" if tr['reactionary_delay_min'] > 0 else "0 min",
                        "Slots Liberated": f"+{tr.get('slots_liberated', 0)} slots" if tr.get("slots_liberated", 0) > 0 else "0 slots",
                        "Operational Consequence": tr.get("operational_consequence", "Rotational delay transmitted along tail rotation."),
                    }
                    for tr in prop_res.sample_turnaround_rotations[:25]
                ])
                st.dataframe(turns_df, use_container_width=True)

            with st.expander("ℹ️ How the ST-GNN Delay Contagion Physics & Spatio-Temporal Graph WaveNet Work", expanded=False):
                st.markdown(
                    """
                    ### Spatio-Temporal Network Delay Propagation Mechanics
                    In dense European airspace, delays do not stay confined to their point of origin:

                    1. **Dynamic Schedule-Aware Adjacency $\\mathcal{W}(t)$**:
                       - Instead of a static geographic topology, the model builds an empirical flight connectivity matrix from the **real historical schedule** operating on the selected date and hour.
                       - Hubs with high scheduled traffic from the shock origin (e.g. Heathrow $\\to$ Amsterdam Schiphol, Frankfurt, Paris) receive high dynamic message-passing weights.
                    2. **Multi-Hop Spatio-Temporal Graph WaveNet Architecture**:
                       - Combines **spatial diffusion graph convolutions** ($K$-hop neighborhood information diffusion) with **dilated causal temporal 1D convolutions**.
                       - Allows the neural network to capture both immediate spatial neighbors ($t=1\\text{h}$) and multi-hop network echo waves ($t=2\\text{h}, 4\\text{h}, 6\\text{h}$).
                    3. **Aircraft Tail Tracking & Downstream Contagion**:
                       - Narrowbody commercial airliners fly 4 to 8 flight legs per day across Europe with tight 35–45 minute turnaround buffers.
                       - When an aircraft never takes off or arrives with catastrophic delay, the subsequent flight relying on that tail number cannot depart on schedule.
                       - This spreads contagion outward to other European airports (outstations where return flights are cancelled and passengers stranded).
                    4. **Countervailing Slot Relief (Backlog Clearance Acceleration)**:
                       - Because those tail numbers cannot operate their subsequent return flights, the scheduled departure and arrival slots for those flights are **vacated**.
                       - This thins competing movement demand during the disruption window, liberating runway capacity for remaining flights with aircraft and crew on the ramp.
                       - Consequently, the delay backlog clears significantly faster than naive linear queue models predict.
                    5. **Vulnerability Index & Shock Absorber Classification**:
                       - Hubs are ranked by their vulnerability index $\\mathcal{V}_i$, distinguishing **Critical Shock Absorbers** (hubs absorbing high reactionary delay while buffering downstream neighbors) from **Nominal Absorbers**.
                    """
                )

    # ----------------- SUB-TAB 4: 4D Trajectory & FIR Closure Simulator -----------------
    with ai_sub4:
        st.header("🌪️ 4D Trajectory Counterfactual & FIR Closure Simulator")
        st.markdown(
            "Simulates the geopolitical, meteorological, or ATC disruption impact of a temporary Flight Information Region (FIR) closure. "
            "Predicts traffic diversions, reroute detour nautical miles, extra fuel burn and $\\text{CO}_2$, departure ground delays, "
            "cancellations, and post-reopening non-linear recovery queue dynamics."
        )

        col_c1, col_c2, col_c3, col_c4 = st.columns([1.4, 1.1, 0.8, 0.8])
        with col_c1:
            fir_target_choice = st.selectbox(
                "Closed Flight Information Region (FIR)",
                options=[
                    "LSASFIR (Switzerland - Zurich/Geneva)",
                    "EDUUUIR (Germany - Rhein Upper Airspace)",
                    "LFFFFIR (France - Paris Upper Area)",
                    "EISNFIR (Ireland - Shannon Upper)",
                    "EGTTFIR (United Kingdom - London)",
                    "EDMMFIR (Germany - Munich Upper)",
                    "LOVVFIR (Austria - Vienna)",
                ],
                key=f"fir_target_{selected_period}",
            )
            fir_code = fir_target_choice.split()[0]
        with col_c2:
            start_date_bound, end_date_bound = get_dataset_date_bounds(ds)
            # Find default operational day (with > 500 flights if start_date is sparse)
            default_sim_date = pd.to_datetime(start_date_bound).date()
            try:
                row_op = ds.query(
                    "SELECT CAST(COALESCE(actual_off_block_time, filed_off_block_time) AS DATE) AS d "
                    "FROM flights GROUP BY 1 HAVING COUNT(*) > 500 ORDER BY 1 LIMIT 1"
                )
                if not row_op.empty:
                    default_sim_date = pd.to_datetime(str(row_op.iloc[0]["d"])).date()
            except Exception:
                pass
            fir_date_choice = st.date_input(
                "Closure Date",
                value=default_sim_date,
                min_value=pd.to_datetime(start_date_bound).date(),
                max_value=pd.to_datetime(end_date_bound).date(),
                key=f"fir_date_{selected_period}",
            )
        with col_c3:
            fir_start_hr = st.slider("Closure Start Time (UTC Hour)", 0, 23, 10, key=f"fir_start_hr_{selected_period}")
        with col_c4:
            fir_duration = st.slider("Closure Duration (Hours)", 1.0, 8.0, 3.0, step=0.5, key=f"fir_dur_{selected_period}")

        if st.button("Run FIR Closure Counterfactual Simulation", key=f"btn_fir_sim_{selected_period}"):
            with st.spinner(f"Simulating counterfactual closure of {fir_code} on {fir_date_choice} from {fir_start_hr:02d}:00 UTC (+{fir_duration}h)..."):
                fir_sim = AirspaceClosureSimulator(ds, period=selected_period)
                c_res = fir_sim.simulate_fir_closure(
                    target_fir=fir_code,
                    closure_start_hour=fir_start_hr,
                    closure_duration_hours=fir_duration,
                    date_str=str(fir_date_choice),
                )

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Impacted Flights", f"{c_res.total_flights_impacted} flights")
            cm2.metric("Rerouted Airborne Flights", f"{c_res.rerouted_flights_count} flights")
            cm3.metric("ATFM Ground Held Flights", f"{c_res.ground_held_flights_count} flights")
            cm4.metric("Cancellation Risk Flights", f"{c_res.cancellation_risk_flights_count} flights")

            em1, em2, em3, em4 = st.columns(4)
            em1.metric("Total Detour Flown", f"{c_res.total_extra_distance_flown_nm:,.0f} nm")
            em2.metric("Excess Fuel Burned", f"{c_res.total_extra_fuel_burned_tonnes:,.1f} tonnes")
            em3.metric("Airline Direct Loss (€)", f"€{c_res.total_airline_economic_loss_eur:,.2f} EUR")
            em4.metric("Post-Reopening Recovery Duration", f"{c_res.post_reopening_recovery_hours:.1f} hours", f"Rebound @ {c_res.network_stability_rebound_time}")

            tm1, tm2, tm3, tm4 = st.columns(4)
            tm1.metric("Unserved Return Flights", f"{getattr(c_res, 'unserved_return_flights_count', 0)} flights", "Missing Aircraft Tails")
            tm2.metric("Freed Runway Slots", f"{getattr(c_res, 'total_freed_slots_count', 0)} slots", "Liberated Runway Capacity")
            tm3.metric("Downstream Outstations Impacted", f"{len(getattr(c_res, 'downstream_airports_impacted', []))} airports", "Network Contagion")
            tm4.metric("Backlog Clearance Boost", f"+{getattr(c_res, 'backlog_acceleration_pct', 0.0):.1f}%", "Accelerated Throughput")

            if c_res.sample_rerouted_flights:
                st.subheader(f"✈️ Sample 4D Traffic Detours Bypassing {fir_code}")
                detour_df = pd.DataFrame([
                    {
                        "Flight ID": f.ectrl_id,
                        "Operator": f.ac_operator,
                        "Route": f.route,
                        "Aircraft": f.ac_type,
                        "Original Dist": f"{f.original_distance_nm:.0f} nm",
                        "Detour Dist": f"{f.detour_distance_nm:.0f} nm",
                        "Extra Dist": f"+{f.extra_distance_nm:.0f} nm",
                        "Extra Time": f"+{f.extra_air_time_min:.1f} min",
                        "Extra Fuel": f"+{f.extra_fuel_kg:,.0f} kg",
                        "Added Cost (€)": f"€{f.extra_cost_eur:,.2f} EUR",
                        "Bypass Corridor": f.avoidance_fir_diverted_to,
                    }
                    for f in c_res.sample_rerouted_flights
                ])
                st.dataframe(detour_df, use_container_width=True)

            tail_rots = getattr(c_res, "sample_tail_rotations", [])
            if tail_rots:
                st.subheader("🔄 Aircraft Tail Tracking: Downstream Rotations & Liberated Slot Capacity")
                st.caption(
                    "Follows physical aircraft tails (`ac_reg`): if a flight never took off or suffered extreme ground hold, "
                    "the downstream flight relying on that tail cannot depart on schedule, spreading contagion to outstations. "
                    "However, because those return rotations cannot fly, they vacate arrival and departure slots, liberating runway throughput to clear the backlog."
                )
                rot_df = pd.DataFrame([
                    {
                        "Aircraft Tail": tr["tail_number"],
                        "Operator": tr["operator"],
                        "Leg 1 Initial Flight": f"#{tr['leg1_id']} ({tr['leg1_route']})",
                        "Leg 1 Status": tr["leg1_status"],
                        "Downstream Leg": f"#{tr['leg2_id']} ({tr['leg2_route']})",
                        "Outstation Impacted": tr["downstream_outstation"],
                        "Downstream Status": tr["downstream_status"],
                        "Slots Liberated": f"+{tr['slots_liberated']} slots" if tr['slots_liberated'] > 0 else "0 slots",
                        "Operational Consequence": tr["operational_consequence"],
                    }
                    for tr in tail_rots[:25]
                ])
                st.dataframe(rot_df, use_container_width=True)

            if c_res.recovery_curve:
                st.subheader("⏱️ Post-Reopening Backlog Depletion & Recovery Queue Dynamics")
                st.caption("Models non-linear queue drainage once the FIR reopens: comparing actual accelerated drainage (benefiting from liberated return slots) against naive unmitigated queue decay.")
                rec_df = pd.DataFrame(c_res.recovery_curve).set_index("time_utc")
                chart_cols = [c for c in ["backlog_flights", "naive_backlog_flights", "sector_surge_capacity"] if c in rec_df.columns]
                st.line_chart(rec_df[chart_cols])

            with st.expander("ℹ️ Operational Flow Management Logic & Physics Behind FIR Closure Disruption", expanded=False):
                st.markdown(
                    """
                    ### How ATFM and Airline Dispatch Respond to a Closed FIR
                    When an entire Flight Information Region (such as Switzerland `LSASFIR`) shuts down unexpectedly (e.g. radar outage, power glitch, industrial action, or weather), traffic is managed under strict Eurocontrol Network Manager procedures:

                    1. **ATFM Ground Delay Programs (Internal Traffic `LS*`)**:
                       - Flights scheduled to land at or depart from airports within the closed FIR (e.g. Zurich `LSZH`, Geneva `LSGG`, Basel `LFSB`) **cannot** be rerouted around the FIR because their flight plan terminates or originates inside it.
                       - To prevent severe airborne holding stacks, fuel starvation, and safety hazards, Eurocontrol issues mandatory **Ground Delay Program (GDP)** slots. Aircraft are held at their departure gates across Europe (or at Swiss gates) absorbing delay on the ground with engines off.
                    2. **Cancellation Risk (Duty Period & Curfew Breaches)**:
                       - Flights receiving extreme ground delays (>135 minutes) risk breaching flight crew maximum duty limits under **EASA Subpart-FTL** (Flight Time Limitations) or missing destination airport night noise curfews. Airlines voluntarily cancel or consolidate these flights.
                    3. **Aircraft Tail Tracking & Outstation Contagion**:
                       - Physical airframes (`ac_reg`) operate multi-leg rotations throughout the day. When an aircraft never arrives at its destination (or arrives too late for turnaround), the subsequent flight using that airframe **cannot take off**.
                       - The contagion spreads to outstation airports across Europe where passengers on return legs are stranded.
                    4. **Countervailing Slot Relief & Accelerated Backlog Drainage**:
                       - Because those missing aircraft cannot serve their return flights, those scheduled flights drop out of the departure and arrival queues, **liberating runway and gate slots**.
                       - This liberated capacity relieves the bottleneck at congested airports, allowing the remaining backlog of flights with aircraft and crew on the ramp to clear significantly faster.
                    5. **Lateral En-Route Rerouting (Overflights)**:
                       - Transiting flights that neither take off nor land in Switzerland (e.g., London to Rome, Frankfurt to Madrid, Amsterdam to Nice) cannot enter `LSASFIR`.
                       - Eurocontrol Network Manager and airline flight dispatch re-clear them onto adjacent bypass corridors (`LFFFFIR`, `EDUUUIR`, `LIRRFIR`).
                    """
                )

    # ----------------- SUB-TAB 5: Multi-Agent RL ATFM Slot Gaming -----------------
    with ai_sub5:
        st.header("🎮 Multi-Agent RL ATFM Slot Gaming & Nash Equilibrium")
        st.markdown(
            "Decentralized game-theoretic simulation of heterogeneous airline agents competing under ATFM arrival slot restrictions. "
            "Evaluates how independent airline agents—**Network Legacy**, **Low-Cost Carrier (LCC)**, **Hybrid Carrier**, and **Express Cargo**—"
            "strategically execute slot swaps, en-route speedups, and voluntary flight cancellations to minimize their individual cost functions, "
            "reaching a mixed-strategy Nash equilibrium."
        )

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns([1.3, 1.1, 0.7, 0.7, 0.8])
        with col_m1:
            marl_airport = st.selectbox(
                "Regulated Bottleneck Airport",
                options=["EGLL", "EHAM", "LFPG", "EDDF", "LEMD"],
                key=f"marl_apt_{selected_period}",
            )
        with col_m2:
            start_date_bound, end_date_bound = get_dataset_date_bounds(ds)
            default_marl_date = pd.to_datetime(start_date_bound).date()
            try:
                row_op = ds.query(
                    "SELECT CAST(COALESCE(actual_off_block_time, filed_off_block_time) AS DATE) AS d "
                    "FROM flights GROUP BY 1 HAVING COUNT(*) > 500 ORDER BY 1 LIMIT 1"
                )
                if not row_op.empty:
                    default_marl_date = pd.to_datetime(str(row_op.iloc[0]["d"])).date()
            except Exception:
                pass
            marl_date = st.date_input(
                "Regulation Date",
                value=default_marl_date,
                min_value=pd.to_datetime(start_date_bound).date(),
                max_value=pd.to_datetime(end_date_bound).date(),
                key=f"marl_date_{selected_period}",
            )
        with col_m3:
            marl_hour = st.slider(
                "Start (UTC)",
                0, 23, 10,
                key=f"marl_hour_{selected_period}",
            )
        with col_m4:
            marl_dur = st.slider(
                "Duration (h)",
                1.0, 6.0, 3.0, step=0.5,
                key=f"marl_dur_{selected_period}",
            )
        with col_m5:
            marl_cut = st.slider(
                "Capacity Cut (%)",
                15, 65, 40, step=5,
                key=f"marl_cut_{selected_period}",
            )

        if st.button("Simulate Competitive Multi-Agent ATFM Equilibrium", key=f"btn_marl_{selected_period}"):
            with st.spinner(f"Running multi-agent reinforcement learning simulation for {marl_airport} (-{marl_cut}% capacity) on {marl_date} from {marl_hour:02d}:00 UTC (+{marl_dur}h)..."):
                marl_sim = MultiAgentATFMSimulator(ds, period=selected_period)
                marl_res = marl_sim.run_marl_simulation(
                    regulated_airport=marl_airport,
                    capacity_cut_pct=float(marl_cut),
                    date_str=str(marl_date),
                    start_time_hour=int(marl_hour),
                    duration_hours=float(marl_dur),
                )

            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Regulated Airport & Window", f"{marl_res.regulated_airport} ({marl_res.regulation_duration_hours:.1f}h)", f"{marl_hour:02d}:00 - {int(marl_hour + marl_dur):02d}:00 UTC")
            mm2.metric("Regulated Arrival Flights", f"{marl_res.total_flights_competing} flights")
            mm3.metric("System Delay Saved", f"{marl_res.total_network_delay_saved_min:,.0f} min")
            mm4.metric("Net Economic Benefit (€)", f"€{marl_res.total_net_economic_benefit_eur:,.2f} EUR")

            st.info(
                f"⚖️ **Cooperative vs. Competitive Efficiency Gap:** `{marl_res.cooperative_vs_competitive_efficiency_gap_pct:.1f}%`. "
                f"Decentralized airline slot swapping achieves **{100.0 - marl_res.cooperative_vs_competitive_efficiency_gap_pct:.1f}%** "
                f"of theoretical central planner Pareto efficiency while preserving proprietary airline passenger connection priorities."
            )

            # Infilled Domain Assumptions Glossary
            glossary = getattr(marl_res, "assumptions_glossary", None) or ASSUMPTIONS_GLOSSARY
            if glossary:
                with st.expander("📖 Infilled Domain Assumptions Glossary (Operational Context Infilled for Raw DDR2 Trajectories)", expanded=False):
                    st.markdown(
                        "Raw Eurocontrol DDR2 radar feeds record aircraft trajectory positions and timings, but do not contain passenger ticketing manifests, "
                        "crew duty rosters, or commercial airline contracts. Autonomous airline agents infill the following domain parameters to compute the true marginal cost of delay:"
                    )
                    glossary_df = pd.DataFrame([
                        {"Operational Dimension": k, "Infilled Assumption & Modeling Logic": v}
                        for k, v in glossary.items()
                    ])
                    st.dataframe(glossary_df, use_container_width=True)

            # Blow-by-Blow Autonomous Decision Log
            dec_log = getattr(marl_res, "decision_log", None) or []
            if dec_log:
                st.subheader("📜 Blow-by-Blow Autonomous Decision Log")
                st.caption(
                    "Comprehensive chronological stream of autonomous decisions across all participating airline agents: "
                    "Eurocontrol GDP initiation, assumption infilling, tactical regional cancellations, intra-fleet swaps, bilateral slot trades, en-route speedups, and Nash equilibrium convergence."
                )

                # Aggregate tactical decision metric cards
                tot_swaps = sum(getattr(s, "slots_swapped_count", 0) for s in marl_res.airline_strategies)
                tot_speedups = sum(getattr(s, "tactical_speedups_count", 0) for s in marl_res.airline_strategies)
                tot_cancellations = sum(getattr(s, "tactical_cancellations_count", 0) for s in marl_res.airline_strategies)
                tot_decisions = len(dec_log)

                dl_c1, dl_c2, dl_c3, dl_c4 = st.columns(4)
                dl_c1.metric("Total Decisions Logged", f"{tot_decisions} steps")
                dl_c2.metric("Tactical Slot Swaps", f"{tot_swaps} swaps")
                dl_c3.metric("Tactical Speedups", f"{tot_speedups} speedups")
                dl_c4.metric("Tactical Cancellations", f"{tot_cancellations} cancelled")

                # Interactive Filtering Controls
                fcol1, fcol2, fcol3 = st.columns([1.2, 1.2, 2.0])
                all_agents = sorted({entry.agent_name for entry in dec_log if entry.agent_name})
                all_actions = sorted({entry.action_type for entry in dec_log if entry.action_type})

                sel_agent = fcol1.selectbox("Filter by Airline Agent", ["All Agents"] + all_agents, key="marl_filter_agent")
                sel_action = fcol2.selectbox("Filter by Action Type", ["All Actions"] + all_actions, key="marl_filter_action")
                search_kw = fcol3.text_input(
                    "Search Flight / Route / Rationale",
                    placeholder="e.g. BAW, A320, cancellation, swap, connection...",
                    key="marl_filter_search",
                )

                filtered_log = dec_log
                if sel_agent != "All Agents":
                    filtered_log = [e for e in filtered_log if e.agent_name == sel_agent]
                if sel_action != "All Actions":
                    filtered_log = [e for e in filtered_log if e.action_type == sel_action]
                if search_kw.strip():
                    kw = search_kw.strip().lower()
                    filtered_log = [
                        e for e in filtered_log
                        if kw in e.flight_id.lower()
                        or kw in e.route.lower()
                        or kw in e.operational_reasoning.lower()
                        or kw in e.infilled_assumptions.lower()
                        or kw in e.counterparty.lower()
                    ]

                st.caption(f"Showing **{len(filtered_log)}** of **{len(dec_log)}** decisions matching active filters.")

                log_df = pd.DataFrame([
                    {
                        "Step": f"#{entry.step_number}",
                        "Time (Rel)": entry.timestamp_rel,
                        "Agent": entry.agent_name,
                        "Action Type": entry.action_type,
                        "Flight & Route": f"{entry.flight_id} ({entry.route})",
                        "Infilled Assumptions": entry.infilled_assumptions,
                        "Operational Decision Rationale": entry.operational_reasoning,
                        "Counterparty": entry.counterparty,
                        "Delay Impact": f"{entry.delay_impact_min:+.0f} min" if entry.delay_impact_min != 0 else "0 min",
                        "Net Value Created (€)": f"€{entry.economic_benefit_eur:,.2f} EUR" if entry.economic_benefit_eur != 0 else "€0.00 EUR",
                    }
                    for entry in filtered_log
                ])
                st.dataframe(log_df, use_container_width=True, height=500)

            # Per-Airline Agent Net Gains & Losses Balance Sheet
            st.subheader("💰 Per-Airline Agent Net Gains & Losses Balance Sheet")
            st.caption(
                "Direct financial balance sheet comparing baseline unmitigated costs (blind FCFS ground delay) against post-gaming optimized costs. "
                "Highlights delay saved, passenger connections protected, EU261 claims avoided, fuel costs, cancellation costs, and net economic payoffs in Euros (€ EUR)."
            )
            balance_df = pd.DataFrame([
                {
                    "Airline Agent": s.airline_name,
                    "Flights Regulated": s.flights_regulated,
                    "Baseline Cost (Blind Delay)": f"€{getattr(s, 'baseline_cost_no_gaming_eur', 0.0):,.2f} EUR",
                    "Post-Gaming Cost": f"€{getattr(s, 'post_gaming_cost_eur', 0.0):,.2f} EUR",
                    "Net Economic Payoff (Gain)": f"€{s.net_economic_payoff_eur:,.2f} EUR",
                    "Gross Delay Saved": f"{s.gross_delay_saved_min:,.0f} min",
                    "Connections Saved": f"€{getattr(s, 'connection_costs_saved_eur', 0.0):,.2f} EUR",
                    "EU261 Fines Prevented": f"€{s.eu261_penalties_prevented_eur:,.2f} EUR",
                    "Speedup Fuel Cost": f"€{getattr(s, 'fuel_cost_incurred_eur', 0.0):,.2f} EUR",
                    "Cancellation Costs": f"€{getattr(s, 'cancellation_penalties_eur', 0.0):,.2f} EUR",
                    "Dominant Strategy": s.dominant_strategy,
                }
                for s in marl_res.airline_strategies
            ])
            st.dataframe(balance_df, use_container_width=True)

            if marl_res.sample_swapped_flights:
                st.subheader("🔄 Autonomous Slot Swaps & High-Value Connection Protection")
                st.caption("Real flights swapped by autonomous airline agents: prioritizing high-yield connecting flights while trading regional slots.")
                swaps_df = pd.DataFrame([
                    {
                        "Carrier": sw["carrier"],
                        "Prioritized High-Yield Flight": sw["high_priority_flight"],
                        "Deprioritized Regional Flight": sw["deprioritized_flight"],
                        "Autonomous Action": sw["action"],
                        "Pax Connections Saved": sw["passenger_connections_protected"],
                        "Net Value Created (€)": sw["delay_mitigated_eur"],
                    }
                    for sw in marl_res.sample_swapped_flights
                ])
                st.dataframe(swaps_df, use_container_width=True)

            st.subheader("👥 Airline Agent Strategic Responses & Operational Summary")
            strat_df = pd.DataFrame([
                {
                    "Airline Archetype": s.airline_name,
                    "Flights Regulated": s.flights_regulated,
                    "Slot Swaps Executed": s.slots_swapped_count,
                    "En-Route Speedups": s.tactical_speedups_count,
                    "Tactical Cancellations": s.tactical_cancellations_count,
                    "Delay Saved": f"{s.gross_delay_saved_min:,.0f} min",
                    "Extra Fuel Burned": f"{s.additional_fuel_burned_kg:,.0f} kg",
                    "EU261 Fines Prevented": f"€{s.eu261_penalties_prevented_eur:,.2f} EUR",
                    "Net Economic Payoff": f"€{s.net_economic_payoff_eur:,.2f} EUR",
                    "Dominant Strategy": s.dominant_strategy,
                }
                for s in marl_res.airline_strategies
            ])
            st.dataframe(strat_df, use_container_width=True)

            if marl_res.sample_agent_decisions:
                st.subheader("📋 Sample Tactical Multi-Agent Decision Log")
                st.dataframe(pd.DataFrame(marl_res.sample_agent_decisions), use_container_width=True)

            with st.expander("ℹ️ How Multi-Agent Reinforcement Learning (MARL) Models Airline Strategic ATFM Gaming", expanded=False):
                st.markdown(
                    """
                    ### Game-Theoretic Slot Gaming & Nash Equilibrium Under Capacity Restrictions
                    When an airport declares an ATFM ground delay regulation (e.g. 40% arrival rate cut due to weather or staffing):

                    1. **Heterogeneous Airline Objective Functions**:
                       - **Network Legacy Carriers (Hub & Spoke)**: Heavy connecting bank penalties. A 20-minute delay on an inbound flight can cause dozens of missed long-haul transatlantic connections. They are highly willing to burn extra Jet-A on tactical speedups or swap arrival slots between high-value flights and low-yield regional flights.
                       - **Point-to-Point Low-Cost Carriers (LCC)**: Strict direct operating cost minimizers with low connecting passenger obligations. They refuse expensive en-route speedups and prefer gate holds or tactical cancellations of late-night rotations.
                       - **Express Cargo Operators**: Night-centric schedules with high crew and cargo curfew penalties, deferring arrival slots to off-peak periods.
                    2. **Decentralized Slot Swaps vs. Centralized Mandates**:
                       - Centralized First-Come-First-Served (FCFS) slot allocation destroys economic value by treating a 300-passenger Boeing 777 with 150 tight connections identically to an empty regional turboprop.
                       - Decentralized slot trading allows airlines to autonomously exchange slots within their portfolio, achieving **~82% of central planner Pareto efficiency** while keeping passenger itineraries proprietary.
                    3. **Nash Equilibrium Convergence**:
                       - Each airline optimizes its mixed strategy $\\pi_i$ against competitors' responses. The simulation computes the stable Nash equilibrium where no airline can unilaterally improve its payoff.
                    """
                )

# =============================================================================
# 8. PASSENGER TRAVEL COPILOT & ITINERARY RISK
# =============================================================================
with tab_cat_passenger:
    st.header("🧳 Passenger Travel Copilot & Multi-Leg Journey Predictor")
    st.caption(
        "Empowering the general travelling public with empirical airline delay distributions, live 7-day airport "
        "meteorological forecasts, transfer hub missed connection probabilities (Minimum Connection Time MCT), "
        "and EU261 / UK261 statutory compensation rights."
    )

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        p_origin = st.text_input("🛫 Origin Airport (ICAO)", value="EINN", help="e.g. EINN (Shannon), EHAM (Amsterdam)").strip().upper()
    with p_col2:
        p_hub = st.text_input("🔄 Transfer Connection Hub (ICAO)", value="", help="e.g. EGLL (London Heathrow), EHAM, EDDF. Leave blank for direct flights.").strip().upper()
    with p_col3:
        p_dest = st.text_input("🛬 Final Destination (ICAO)", value="EBBR", help="e.g. EBBR (Brussels), LFPG (Paris)").strip().upper()

    p_r2_col1, p_r2_col2, p_r2_col3, p_r2_col4, p_r2_col5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.3])
    with p_r2_col1:
        p_dow = st.selectbox("📅 Day of the Week", ["Tuesday", "Monday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=0)
    with p_r2_col2:
        p_wave = st.selectbox("🕒 Departure Window", ["Morning (06:00 - 11:00 UTC)", "Midday (11:00 - 16:00 UTC)", "Evening (17:00 - 22:00 UTC)"], index=0)
    with p_r2_col3:
        p_layover = st.slider("⏳ Scheduled Layover at Hub (min)", min_value=30, max_value=240, value=75, step=5)
    with p_r2_col4:
        p_target_date = st.text_input("📆 Target Flight Date", value="", help="YYYY-MM-DD or leave empty for upcoming 7-day live weather forecast").strip()
    with p_r2_col5:
        p_period_mode = st.selectbox(
            "🗄️ Baseline Data Period",
            ["⚡ Auto (Best seasonal fit)", f"Sidebar Period ({selected_period})"],
            index=0,
            help="Auto selects the best fitting Eurocontrol archive based on season & modernity for the flight date."
        )

    p_debug = st.checkbox("🛠️ Detailed Accounting & Data Provenance Audit (--debug)", value=False, help="Show exact historical flight records, DuckDB SQL queries, NWP weather endpoints, and buffer calculations.")
    p_run = st.button("🔍 Assess My Flight Journey & Delay Risk", type="primary", use_container_width=True)

    if p_run or True:
        with st.spinner("Analyzing empirical flight distributions, live 7-day meteorological models & connection buffers..."):
            chosen_p = "auto" if "Auto" in p_period_mode else selected_period
            p_engine = PassengerJourneyEngine(period=chosen_p)
            clean_hub_val = p_hub if p_hub and p_hub not in (p_origin, p_dest, "NONE", "") else None
            p_res = p_engine.assess_itinerary(
                origin=p_origin,
                destination=p_dest,
                hub=clean_hub_val,
                day_of_week=p_dow,
                time_window=p_wave,
                scheduled_layover_min=p_layover,
                target_date=p_target_date if p_target_date else None,
                period=chosen_p,
                debug=p_debug,
            )

        st.markdown("---")
        st.info(f"📅 **Baseline Data Archive Selected**: `{p_res.period}` — {p_res.period_selection_reason}")

        # Top KPI Scoreboard
        k1, k2, k3, k4 = st.columns(4)
        risk_color = "🔴" if p_res.overall_delay_risk_score > 60 else "🟡" if p_res.overall_delay_risk_score > 30 else "🟢"
        k1.metric(
            "Overall Journey Risk",
            f"{p_res.overall_delay_risk_score:.1f} / 100",
            f"{risk_color} {p_res.overall_delay_risk_level}",
        )
        if p_res.connection:
            mc_color = "🔴" if p_res.connection.misconnect_probability_pct > 40 else "🟡" if p_res.connection.misconnect_probability_pct > 15 else "🟢"
            k2.metric(
                "Missed Connection Risk",
                f"{p_res.connection.misconnect_probability_pct:.1f}%",
                f"{mc_color} {p_res.connection.misconnect_risk_level}",
            )
            lug_color = "🔴" if "CRITICAL" in p_res.connection.luggage_status else "🟡" if "MODERATE" in p_res.connection.luggage_status else "🟢"
            k3.metric(
                "Luggage Transfer Risk",
                f"{p_res.connection.luggage_transfer_risk_pct:.0f}%",
                f"{lug_color} {p_res.connection.luggage_status.replace('_', ' ')}",
            )
        else:
            k2.metric("Flight Type", "Direct Non-Stop", "No Hub Layover")
            k3.metric("Baggage Routing", "Direct Transfer", "🟢 LOW RISK")

        k4.metric(
            "EU261 Statutory Rights",
            f"€{p_res.passenger_rights.statutory_compensation_amount_eur:,.0f} EUR",
            f"If delayed ≥ {p_res.passenger_rights.delay_threshold_hours:.0f}h",
        )

        # Actionable Passenger Advisories
        st.subheader("💡 Actionable Travel Recommendations & Alerts")
        for rec in p_res.actionable_recommendations:
            if "🚨" in rec:
                st.error(rec)
            elif "🧳" in rec:
                st.warning(rec)
            elif "💨" in rec:
                st.info(rec)
            else:
                st.success(rec)

        st.markdown("---")

        # Flight Legs Breakdown
        st.subheader("✈️ Flight Legs & Meteorological Risk Analysis")
        for idx, leg in enumerate(p_res.legs):
            with st.container():
                st.markdown(
                    f"#### Leg {idx + 1}: **{leg.origin}** ({leg.origin_name}) ➔ **{leg.destination}** ({leg.destination_name})"
                )
                l_col1, l_col2, l_col3, l_col4 = st.columns(4)
                l_col1.metric("Operating Airline", leg.operator_name, f"Aircraft: {leg.typical_aircraft}")
                l_col2.metric("Average / Median Delay", f"{leg.mean_arr_delay_min:+.1f} min", f"Median: {leg.median_arr_delay_min:+.1f} min")
                l_col3.metric("P90 Delay (Worst 10%)", f"{leg.p90_arr_delay_min:+.1f} min", f"Delay > 15m: {leg.prob_delay_gt_15m_pct:.1f}%")
                l_col4.metric(
                    "Weather-Adjusted Risk",
                    f"{leg.weather_adjusted_prob_delay_pct:.1f}%",
                    f"Multiplier: {leg.weather_risk_multiplier:.2f}x ({leg.expected_delay_bracket})",
                )

                # Weather Forecast Sub-cards
                wx_c1, wx_c2 = st.columns(2)
                with wx_c1:
                    ow = leg.origin_weather
                    live_tag = "🟢 Live 7-Day Forecast" if ow.is_live_api else "ℹ️ Climatological Model"
                    st.info(
                        f"**{leg.origin} Weather ({ow.date_str}) [{live_tag}]:** {ow.weather_description} | "
                        f"Temp: {ow.temperature_c}°C | Wind: {ow.wind_speed_kts} kts (Gusts: **{ow.wind_gusts_kts} kts**) | "
                        f"Rain Chance: {ow.precipitation_probability_pct:.0f}% | Runway Crosswind: {ow.crosswind_kts} kts"
                    )
                with wx_c2:
                    dw = leg.destination_weather
                    live_tag = "🟢 Live 7-Day Forecast" if dw.is_live_api else "ℹ️ Climatological Model"
                    st.info(
                        f"**{leg.destination} Weather ({dw.date_str}) [{live_tag}]:** {dw.weather_description} | "
                        f"Temp: {dw.temperature_c}°C | Wind: {dw.wind_speed_kts} kts (Gusts: **{dw.wind_gusts_kts} kts**) | "
                        f"Rain Chance: {dw.precipitation_probability_pct:.0f}% | Runway Crosswind: {dw.crosswind_kts} kts"
                    )
                st.markdown("<br>", unsafe_allow_html=True)

        # Connection Hub Deep-Dive
        if p_res.connection:
            st.markdown("---")
            st.subheader(f"🔄 Connection Hub Analysis: {p_res.connection.hub_name} ({p_res.connection.hub})")
            c = p_res.connection

            ch_c1, ch_c2, ch_c3, ch_c4 = st.columns(4)
            ch_c1.metric("Scheduled Layover", f"{c.scheduled_layover_min} min", "Passenger itinerary")
            ch_c2.metric("Effective MCT (Weather)", f"{c.effective_mct_min:.0f} min", f"Nominal: {c.minimum_connection_time_min}m (+{c.inter_terminal_transit_weather_penalty_min:.0f}m transit)")
            slack_delta_color = "normal" if c.effective_slack_buffer_min >= 15 else "inverse"
            ch_c3.metric("Effective Slack Buffer", f"{c.effective_slack_buffer_min:+.0f} min", f"{'Adequate' if c.effective_slack_buffer_min >= 15 else 'DEFICIT / CRITICAL'}", delta_color=slack_delta_color)
            ch_c4.metric("Missed Connection Probability", f"{c.misconnect_probability_pct:.1f}%", c.misconnect_risk_level)

            # Weather flow impact breakdown
            wh_c1, wh_c2 = st.columns(2)
            with wh_c1:
                st.info(
                    f"🛬 **Leg 1 Inbound Arrival Flow Impact:** Weather at {c.hub} restricts runway throughput, adding "
                    f"**+{c.hub_arrival_flow_penalty_min:.0f} min** expected airborne holding at outer stacks before touchdown."
                )
            with wh_c2:
                st.warning(
                    f"🛫 **Leg 2 Outbound Departure vs. Gate Conformance:** Outbound flight faces **+{c.hub_departure_taxi_penalty_min:.0f} min** "
                    f"taxiway queues, but **gate conformance closes strictly {c.boarding_conformance_cutoff_min} min prior to scheduled departure**. "
                    f"Taxi delays will NOT hold the gate open."
                )

            st.error(f"**Airport Transit Note:** {c.terminal_transfer_note}")
            if c.weather_compound_guidance:
                st.caption(f"ℹ️ **Detailed Hub Weather Guidance:** {c.weather_compound_guidance}")

        # When Should I Fly? Departure Wave Comparison
        st.markdown("---")
        st.subheader(f"🕒 'When Should I Fly?' Reliability Comparison across Departure Waves ({p_res.origin} ➔ {p_res.legs[0].destination})")
        st.caption(
            "Empirical evidence demonstrates the **First Wave Advantage**: Flights in the early morning start with fresh aircraft "
            "and rested flight crews after overnight maintenance, exhibiting minimal reactionary delay. Evening flights absorb cascading delays "
            "across 4–6 preceding rotations."
        )

        wave_rows = []
        for w in p_res.departure_waves:
            wave_rows.append({
                "Departure Wave": w.wave_name,
                "Time Window": w.hours_range,
                "Average Arrival Delay (min)": w.avg_arr_delay_min,
                "P90 Delay (min)": w.p90_delay_min,
                "On-Time Performance (≤ 15m)": f"{w.on_time_pct:.1f}%",
                "Reactionary Ripple Risk": w.reactionary_risk_level,
                "Recommendation": w.recommendation_badge,
            })
        st.dataframe(pd.DataFrame(wave_rows), use_container_width=True)

        # EU261 / UK261 Passenger Rights Explainer
        st.markdown("---")
        st.subheader("💶 Passenger Rights & Statutory Compensation Charter (EU261 & UK261)")
        pr = p_res.passenger_rights

        pr_c1, pr_c2, pr_c3 = st.columns(3)
        pr_c1.metric("Total Route Distance", f"{pr.total_journey_distance_km:,.0f} km", pr.compensation_bracket)
        pr_c2.metric("Cash Compensation Due", f"€{pr.statutory_compensation_amount_eur:,.0f} EUR", "If delayed ≥ 3 hours at arrival")
        pr_c3.metric("Right to Care Trigger", f"After {pr.right_to_care_hours:.0f} hours delay", "Meals, drinks & communication")

        st.markdown("##### 🛡️ Mandatory Duty-of-Care Entitlements:")
        for ent in pr.duty_of_care_entitlements:
            st.markdown(f"- {ent}")

        with st.expander("ℹ️ Legal Explainer: Extraordinary Circumstances vs. Airline Operational Fault", expanded=False):
            st.markdown(
                f"""
                ### What Qualifies Under European & UK Case Law?
                {pr.claimability_guidance}

                - **Technical Aircraft Defects**: The European Court of Justice (*Wallentin-Hermann v Alitalia*, C-549/07) ruled that unexpected technical defects are inherent in the normal operation of air carriers and **do not** excuse airlines from paying compensation.
                - **Missed Connections**: If an initial delay causes you to miss a connecting flight booked under a single ticket and you arrive at your final destination 3+ hours late (*Folkerts v Air France*, C-11/11), compensation is calculated based on the **final destination arrival time** across the entire journey.
                - **Right to Care is Unconditional**: Even if severe gales or volcanic ash ground your flight (*McDonagh v Ryanair*, C-12/11), the airline's obligation to provide free hotel accommodation, food, and transport remains **unlimited in cost and duration**.
                """
            )

        # Detailed Debug & Provenance Accounting (when debug is enabled)
        if p_res.debug_audit:
            dbg = p_res.debug_audit
            st.markdown("---")
            st.subheader("🛠️ Detailed FlightML Diagnostic & Data Provenance Audit (--debug)")

            # 1. Period Selection
            with st.expander("🗄️ Eurocontrol Period Selection Scoring Audit", expanded=True):
                p_sec = dbg.get("period_selection", {})
                st.markdown(f"**Selected Winning Archive:** `{p_sec.get('winning_period')}`")
                st.caption(f"Rationale: {p_sec.get('period_reason')}")
                if p_sec.get("candidates"):
                    st.dataframe(pd.DataFrame(p_sec["candidates"]), use_container_width=True)

            # 2. Historical Flight Records
            with st.expander("✈️ Historical Flight Records & Sampling Audit", expanded=True):
                for l_dbg in dbg.get("legs_audit", []):
                    st.markdown(f"#### Leg: {l_dbg.get('origin')} ➔ {l_dbg.get('destination')} ({l_dbg.get('origin_name')} to {l_dbg.get('destination_name')})")
                    st.caption(f"**Sampling Stage:** {l_dbg.get('sampling_stage')} | **Matched Historical Records:** {l_dbg.get('sample_size_flights')}")
                    st.code(l_dbg.get("query_executed"), language="sql")

                    if l_dbg.get("sample_records"):
                        st.markdown("**Sample Matched Flight Executions:**")
                        st.dataframe(pd.DataFrame(l_dbg["sample_records"]), use_container_width=True)

                    c_pct, c_ops = st.columns([1.2, 1.0])
                    with c_pct:
                        st.markdown("**Arrival Delay Distribution Percentiles (min):**")
                        st.dataframe(pd.DataFrame([l_dbg.get("delay_percentiles", {})]), use_container_width=True)
                    with c_ops:
                        if l_dbg.get("operator_breakdown"):
                            st.markdown("**Airline Market Shares:**")
                            st.dataframe(pd.DataFrame(l_dbg["operator_breakdown"]), use_container_width=True)

            # 3. Weather Observations
            with st.expander("🌦️ Live NWP Weather Observations & API Provenance Audit", expanded=True):
                if dbg.get("weather_audit"):
                    st.dataframe(pd.DataFrame(dbg["weather_audit"]), use_container_width=True)

            # 4. Connection MCT & Buffer Accounting
            if dbg.get("connection_audit"):
                with st.expander("⏳ Connection MCT & Buffer Dynamics Accounting", expanded=True):
                    st.json(dbg["connection_audit"])

            # 5. Statutory Rights Geodesic
            if dbg.get("passenger_rights_audit"):
                with st.expander("⚖️ Statutory Passenger Rights & Geodesic Calculation Audit", expanded=True):
                    st.json(dbg["passenger_rights_audit"])


