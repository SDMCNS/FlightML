from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from flightml.data.dataset import FlightDataset
from flightml.data.archive_ingest import list_available_periods
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
    create_grid_saturation_map,
    create_conflict_map,
    create_world_navaids_map,
)
from flightml.viz.html_exporter import (
    generate_static_html_report,
    REPORT_PAGE_TITLES,
)


st.set_page_config(
    page_title="FlightML - Airspace Network Intelligence",
    page_icon="✈️",
    layout="wide",
)


# ----------------- Sidebar: Period & Global Controls -----------------
st.sidebar.title("✈️ FlightML Control Deck")


def _get_available_periods() -> list[str]:
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
selected_period = st.sidebar.selectbox(
    "Active Eurocontrol Period",
    options=available_periods,
    index=0,
    help="Switch between historical baseline (201611) and modern archives (202510)."
)


@st.cache_resource(show_spinner=False)
def get_dataset(period: str) -> FlightDataset:
    """Cached dataset accessor preventing repeated DuckDB view initialization on reruns."""
    return FlightDataset(period=period)


@st.cache_resource(show_spinner=False)
def get_metar_client() -> MetarClient:
    """Cached IEM ASOS METAR client."""
    return MetarClient()


@st.cache_data(show_spinner=False)
def _cached_grid_saturation(period: str, hour_utc: int):
    return compute_hourly_grid_saturation(get_dataset(period), hour_utc=hour_utc)


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
def _cached_gdp_optimizer(period: str, target_airport: str, start_hour: int, end_hour: int, aar: int):
    return optimize_ground_delay_program(
        get_dataset(period),
        target_airport=target_airport,
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
    tab_cat_whatif,
    tab_cat_maps,
    tab_cat_econ,
) = st.tabs(
    [
        "✈️ Single Aircraft / Flight Views",
        "🌐 System-Level Summary & Analysis Views",
        "🔮 What-If Scenario Views",
        "🗺️ Information & Map Views",
        "💶 General Economic Views (€)",
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
                fr1, fr2, fr3, fr4 = st.columns(4)
                fr1.metric("FRA Directness Index", f"{fra_rep.fra_directness_index_pct:.1f}%", f"{fra_rep.fra_direct_distance_nm:.0f} nm DCT")
                fr2.metric("ATS Airway Adherence", f"{fra_rep.ats_airway_adherence_pct:.1f}%", f"{fra_rep.ats_airway_distance_nm:.0f} nm")
                fr3.metric("Snapped ATS Airways", f"{len(fra_rep.distinct_airways_used)}", ", ".join(fra_rep.distinct_airways_used[:4]) or "DCT Only")
                fr4.metric("Snapped 5-Letter Fixes", f"{len(fra_rep.sample_snapped_fixes)}", ", ".join(fra_rep.sample_snapped_fixes[:4]) or "—")

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
                if crco_rep.fir_breakdown:
                    st.dataframe(pd.DataFrame([b.__dict__ for b in crco_rep.fir_breakdown]), use_container_width=True)

            contrail_rep = analyze_flight_wind_and_contrails(ds, int(active_flight_id))
            if contrail_rep:
                st.divider()
                st.subheader("🌬️ Upper-Air Winds, Equivalent Still Air Distance (ESAD) & Contrail Climate")

                # Wind Routing & Dispatch Efficiency Verdict
                v_text = contrail_rep.wind_efficiency_verdict
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
                dist_note = f"Ground: {contrail_rep.ground_distance_nm:,.0f} nm"
                ct3.metric(
                    "Air Distance (ESAD)",
                    f"{contrail_rep.esad_nm:,.0f} nm",
                    f"{contrail_rep.wind_distance_benefit_nm:+.0f} nm air savings vs {dist_note}",
                )
                fuel_lbl = "Saved" if contrail_rep.wind_fuel_impact_kg >= 0 else "Penalty"
                ct4.metric(
                    f"Wind Fuel {fuel_lbl}",
                    f"{contrail_rep.wind_fuel_impact_kg:+,.0f} kg Jet-A",
                    f"€{contrail_rep.wind_fuel_impact_eur:+,.2f} EUR vs still air",
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
    sys_sub_kpi, sys_sub1, sys_sub2 = st.tabs(
        [
            "📊 Multi-Level KPI Suite (Flight / Fleet / Airport / Network)",
            "🌐 Network Status & Airspace Saturation",
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

        # Timeframe & Level Scoping Bar
        with st.container():
            col_tf1, col_tf2, col_tf3 = st.columns([2, 2, 2])
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

            with col_tf3:
                kpi_tier = st.radio(
                    "🎯 KPI Aggregation Tier:",
                    options=["🌐 Network Level", "✈️ Fleet Level", "🛫 Airport Level", "🔍 Flight Level"],
                    index=0,
                    horizontal=True,
                    key=f"kpi_tier_radio_{selected_period}",
                )

        st.divider()

        # Tier Views
        if kpi_tier == "🌐 Network Level":
            with st.spinner("Aggregating network-wide KPIs across filtered timeframe..."):
                net_summary = query_multi_level_kpis(
                    ds,
                    level="network",
                    start_date=filter_start_str,
                    end_date=filter_end_str,
                    hour_start=h_start,
                    hour_end=h_end,
                )

            st.subheader(f"🌐 Network-Wide Operational & Environmental KPI Scorecard")
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
            with st.spinner(f"Aggregating fleet KPIs by {fleet_mode}..."):
                fleet_summary = query_multi_level_kpis(
                    ds,
                    level="fleet",
                    group_by=grp_key,
                    start_date=filter_start_str,
                    end_date=filter_end_str,
                    hour_start=h_start,
                    hour_end=h_end,
                    min_flights=fleet_min_flts,
                    limit=fleet_top_n,
                )

            st.subheader(f"✈️ Fleet Performance Scorecard — by {fleet_summary.group_by}")
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
            with st.spinner(f"Aggregating airport KPIs by {airport_mode}..."):
                airport_summary = query_multi_level_kpis(
                    ds,
                    level="airport",
                    group_by=grp_key,
                    start_date=filter_start_str,
                    end_date=filter_end_str,
                    hour_start=h_start,
                    hour_end=h_end,
                    min_flights=ap_min_flts,
                    limit=ap_top_n,
                )

            st.subheader(f"🛫 Airport Performance Scorecard — {airport_summary.group_by}")
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
            st.subheader("🔍 Micro Flight-Level Search & KPI Inspector")
            st.caption(f"Browse individual flight trajectories and efficiency metrics matching `{filter_start_str} to {filter_end_str}` ({h_start:02d}:00–{h_end:02d}:59 UTC).")

            with st.spinner("Querying individual flight records matching timeframe..."):
                flight_summary = query_multi_level_kpis(
                    ds,
                    level="flight",
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
        st.subheader("🗺️ 5° × 5° European Airspace Hourly Saturation & Complexity Map (Period Average by Hour of Day)")
        st.markdown(
            "Visualizes the **average hourly flight density, peak traffic load, vertical climb/descent transition intensity, "
            "and Dynamic Complexity Score** across every $5^\\circ \\times 5^\\circ$ geographic square for any selected hour of the day (`00:00`–`23:00 UTC`)."
        )
        col_gm1, col_gm2 = st.columns([2, 3])
        with col_gm1:
            map_hour_utc = st.slider(
                "Select Hour of Day (UTC) — Averaged Across All Days in Period",
                min_value=0,
                max_value=23,
                value=10,
                format="%02d:00 UTC",
                key=f"grid_map_hour_{selected_period}",
            )
        grid_sat_data = _cached_grid_saturation(selected_period, map_hour_utc)
        with col_gm2:
            gm_c1, gm_c2, gm_c3, gm_c4 = st.columns(4)
            gm_c1.metric("Active 5°×5° Squares", f"{grid_sat_data['total_active_squares']}")
            gm_c2.metric("Critical Saturation", f"{grid_sat_data['critical_squares_count']} squares")
            gm_c3.metric("High Density", f"{grid_sat_data['high_density_squares_count']} squares")
            gm_c4.metric("Peak Square Avg/Hr", f"{grid_sat_data['max_avg_flights_per_hour']:.1f} flts/hr")

        grid_map = create_grid_saturation_map(grid_sat_data)
        st_folium(
            grid_map,
            width=1200,
            height=540,
            returned_objects=[],
            key=f"folium_grid_sat_{selected_period}_{map_hour_utc}",
        )

        with st.expander(f"📊 View 5° × 5° Square Saturation Table for {map_hour_utc:02d}:00–{map_hour_utc:02d}:59 UTC", expanded=False):
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
                        "cell_label": "5°×5° Square",
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

    with sys_sub2:
        st.header("Macro Evolution, FRA Structural Shift, Weather Correlation & AIRAC Diffing")
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
# 3. WHAT-IF SCENARIO VIEWS
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
    col_gdp1, col_gdp2, col_gdp3 = st.columns([1.5, 2, 1.5])
    gdp_apt = col_gdp1.text_input("Arrival Hub ICAO", value="EGLL", key=f"gdp_apt_{selected_period}").strip().upper()
    gdp_aar = col_gdp2.slider(
        "Degraded Airport Arrival Rate Cap (AAR arrivals/hr)",
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
        start_hour=int(gdp_window[0]),
        end_hour=int(gdp_window[1]),
        aar=int(gdp_aar),
    )
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(
        "Regulated Arrival Window",
        f"{gdp_res.program_start[:10]} ({gdp_window[0]:02d}:00–{gdp_window[1]:02d}:00)",
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
        st.info(f"No arrivals found for `{gdp_apt}` in the selected UTC window `{gdp_window[0]:02d}:00–{gdp_window[1]:02d}:00`.")

    # ----------------- TAB 9: Macro Evolution & AIRAC Diff -----------------

# =============================================================================
# 4. INFORMATION & MAP VIEWS
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
# 5. GENERAL ECONOMIC VIEWS (€)
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

        # ----------------- TAB 4: Vertical Profile: CDO & CCO -----------------
