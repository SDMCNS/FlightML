from __future__ import annotations

import sys
import importlib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Hot-reload FlightML modules in running Streamlit dev sessions
for _mod in ["flightml.eval.fuel_carbon", "flightml.eval.jet_fuel_prices", "flightml.data.dataset"]:
    if _mod in sys.modules:
        try:
            importlib.reload(sys.modules[_mod])
        except Exception:
            pass

from flightml.data.dataset import FlightDataset
from flightml.data.archive_ingest import list_available_periods
from flightml.geo.trajectory import analyze_trajectory
from flightml.weather.metar_client import MetarClient
from flightml.eval.diagnostics import diagnose_flight_bottlenecks
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
from flightml.eval.flight_twins import compare_twin_flights, render_twin_comparison_map
from flightml.eval.cdo_analyzer import analyze_cdo_descent
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
from flightml.eval.jet_fuel_prices import get_all_weekly_fuel_prices_df
from flightml.viz.map_builder import create_trajectory_map


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


def get_dataset(period: str) -> FlightDataset:
    """Session-isolated dataset accessor preventing cross-thread DuckDB lock contention."""
    cache_key = f"dataset_{period}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = FlightDataset(period=period)
    return st.session_state[cache_key]


def get_metar_client() -> MetarClient:
    """Session-isolated IEM ASOS METAR client."""
    if "metar_client" not in st.session_state:
        st.session_state["metar_client"] = MetarClient()
    return st.session_state["metar_client"]


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

st.sidebar.markdown(f"**Period Active:** `{selected_period}`")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "### Platform Highlights\n"
    "- **DuckDB + Parquet** high-speed engine\n"
    "- **4D Trajectory & Holding** detection\n"
    "- **Recurring Flight Series** day-over-day\n"
    "- **Diurnal Time-of-Day** congestion\n"
    "- **Continuous Descent (CDO)** analytics\n"
    "- **Twin-Flight** efficiency benchmarking\n"
    "- **IEM METAR** European weather fusion\n"
    "- **AIRAC Cycle Diffing** (airways & FIRs)"
)

st.title("✈️ FlightML: Global Airspace Network Intelligence & ML")
st.markdown(
    f"Unified operational analytics and ML intelligence platform fusing multi-gigabyte 4D trajectories, "
    f"aeronautical infrastructure, and NOAA/IEM ASOS weather observations. "
    f"Active dataset: **Eurocontrol Cycle {selected_period}**."
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "🌐 Network Overview",
        "🗺️ 4D Trajectory & Holdings",
        "📅 Recurring Series & Diurnal Trends",
        "📉 CDO Continuous Descent",
        "🩺 Delay Diagnostics",
        "🌱 Fuel & Carbon Penalty",
        "👯 Twin-Flight Benchmark",
        "🔮 What-If & Stress Tests",
        "📈 Macro Evolution & AIRAC Diff",
    ]
)

# ----------------- TAB 1: Network Overview -----------------
with tab1:
    st.header(f"Global Network Status ({selected_period})")
    stats = ds.get_summary()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Processed Flights", f"{stats['total_flights']:,}")
    c2.metric("Aircraft Operators", f"{stats['total_operators']:,}")
    c3.metric("Departure Airports", f"{stats['unique_dep_airports']:,}")
    c4.metric("Avg Arrival Delay", f"{stats['avg_arr_delay_min']} min")

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
    else:
        st.subheader("Top Congested European FIR Airspace Sectors")
        fir_df = analyze_fir_bottlenecks(ds, limit=8)
        st.dataframe(fir_df, use_container_width=True)

# ----------------- TAB 2: Trajectory Explorer -----------------
with tab2:
    st.header("4D Trajectory, Deviation & Holding Stack Visualizer")

    flight = ds.get_flight(int(active_flight_id))
    if flight:
        st.info(f"📍 **Active Flight:** `{ds.format_flight_label(flight)}`")
        trajs = ds.get_trajectory(int(active_flight_id))
        metrics = analyze_trajectory(trajs["actual"], trajs["planned"])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Route", f"{flight['adep']} ➔ {flight['ades']}")
        m2.metric("Operator", f"{flight['ac_operator']} ({flight['ac_type']})")
        m3.metric("Actual Distance", f"{metrics.actual_distance_nm:.0f} nm")
        m4.metric("Inefficiency Ratio", f"{metrics.horizontal_extension_ratio:.3f}x")
        m5.metric("Holding Duration", f"{metrics.total_holding_duration_min:.1f} min")

        m = create_trajectory_map(flight, trajs["actual"], trajs["planned"])
        st_folium(m, width=1200, height=520)

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
    else:
        st.warning(f"Flight ID {active_flight_id} not found in period {selected_period}.")

# ----------------- TAB 3: Recurring Series & Diurnal Trends -----------------
with tab3:
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

# ----------------- TAB 4: CDO Continuous Descent -----------------
with tab4:
    st.header("Continuous Descent Operations (CDO) Profile & Fuel Waste")
    st.markdown(
        "Identifies inefficient staircase descents, intermediate level-offs, and altitude reversals. "
        "A continuous 3° descent without throttle-up level segments drastically reduces terminal fuel burn, noise, and emissions."
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
            col_c4.metric("Excess Fuel Burned", f"{cdo_rep.excess_fuel_burned_kg:.1f} kg")

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
                st.subheader("Detected Inefficiency Events (Level-Offs & Climbing Reversals)")
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

# ----------------- TAB 5: Delay Diagnostics -----------------
with tab5:
    st.header("Multi-Factor Delay Root-Cause Attribution")
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

# ----------------- TAB 6: Fuel & Carbon -----------------
with tab6:
    st.header("Environmental & Fuel Burn Cost Quantification")
    flight = ds.get_flight(int(active_flight_id))
    if not flight:
        st.warning(f"Flight ID {active_flight_id} not found in period {selected_period}.")
    else:
        st.info(f"📍 **Active Flight:** `{ds.format_flight_label(flight)}`")
        try:
            impact = compute_flight_fuel_and_emissions(ds, int(active_flight_id))
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Excess Distance", f"{impact.excess_distance_nm:.1f} nm")
            fc2.metric("Excess Fuel Burned", f"{impact.total_excess_fuel_kg:,.1f} kg")
            fc3.metric("Excess CO₂ Emitted", f"{impact.excess_co2_kg:,.1f} kg")
            fc4.metric("Excess Fuel Cost", f"${impact.estimated_excess_fuel_cost_usd:,.2f}")

            price_gal = getattr(impact, "fuel_price_usd_per_gallon", 1.368 if selected_period == "201611" else 2.154)
            price_kg = getattr(impact, "fuel_price_usd_per_kg", price_gal / 3.043)
            price_week = getattr(impact, "fuel_price_week", "Weekly Spot Price")
            price_source = getattr(impact, "fuel_price_source", "EIA Weekly Spot Price (USGC/IATA)")

            st.caption(
                f"⛽ **Weekly Jet Fuel Pricing Applied:** `${price_gal:.3f}/gal` "
                f"(`${price_kg:.4f}/kg`) | **Week:** `{price_week}` "
                f"| **Benchmark:** `{price_source}`"
            )
        except Exception as e:
            st.error(f"Error computing emissions for flight {active_flight_id}: {e}")

    with st.expander(f"📊 View Weekly Average Jet Fuel Spot Prices ({selected_period})", expanded=False):
        all_fuel_df = get_all_weekly_fuel_prices_df()
        period_fuel_df = all_fuel_df[all_fuel_df["period_id"] == selected_period]
        if not period_fuel_df.empty:
            st.dataframe(
                period_fuel_df[["week_label", "price_usd_per_gallon", "price_usd_per_kg", "price_usd_per_metric_ton", "source"]],
                use_container_width=True,
            )
        else:
            st.dataframe(all_fuel_df, use_container_width=True)

    st.markdown(
        """
    - **Holding Stack Penalty:** Accounts for fuel burned while circling at low flight levels in terminal holding areas.
    - **Horizontal Inefficiency ($KEA$):** Quantifies extra nautical miles flown compared to the filed flight plan.
    - **Carbon Conversion:** Standard ICAO Jet-A emission index of **3.16 kg CO₂ per kg fuel**.
    """
    )

# ----------------- TAB 7: Twin-Flight Efficiency Benchmarking -----------------
with tab7:
    st.header("Twin-Flight Efficiency Benchmarking")
    st.markdown(
        "Compare and contrast the most and least efficient flights between the same city pair operating the same aircraft type. "
        "Reveals route detours, airspace bypasses, and excess fuel consumption."
    )
    col_t1, col_t2, col_t3 = st.columns(3)
    t_adep = col_t1.text_input("Origin (ADEP)", "RKSI", key=f"t_adep_{selected_period}")
    t_ades = col_t2.text_input("Destination (ADES)", "EGLL", key=f"t_ades_{selected_period}")
    t_ac = col_t3.text_input("Aircraft Type", "B77W", key=f"t_ac_{selected_period}")

    if st.button("Run Twin Comparison", key=f"btn_twin_{selected_period}"):
        with st.spinner("Extracting paired flights and comparing 4D geometries..."):
            twin_comp = compare_twin_flights(ds, adep=t_adep, ades=t_ades, ac_type=t_ac)

        if not twin_comp:
            st.error(f"Not enough matching flights for {t_adep} -> {t_ades} ({t_ac}) in period {selected_period}.")
        else:
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            col_b1.metric("Distance Disparity", f"+{twin_comp.distance_penalty_nm:.1f} nm")
            col_b2.metric("Air Time Disparity", f"+{twin_comp.air_time_penalty_min:.1f} min")
            col_b3.metric("Excess Fuel Burned", f"+{twin_comp.excess_fuel_burned_kg:,.1f} kg")
            col_b4.metric("Excess Fuel Cost", f"${twin_comp.excess_fuel_cost_usd:,.2f}")

            st.subheader("Operational Disparity Breakdown")
            twin_table = [
                {
                    "Metric": "ECTRL Flight ID",
                    "High-Efficiency Flight": str(twin_comp.fast_id),
                    "Low-Efficiency Flight": str(twin_comp.slow_id),
                    "Delta": "-",
                },
                {
                    "Metric": "Airline Operator",
                    "High-Efficiency Flight": twin_comp.fast_operator,
                    "Low-Efficiency Flight": twin_comp.slow_operator,
                    "Delta": "-",
                },
                {
                    "Metric": "Flown Distance",
                    "High-Efficiency Flight": f"{twin_comp.fast_actual_distance_nm:.1f} nm",
                    "Low-Efficiency Flight": f"{twin_comp.slow_actual_distance_nm:.1f} nm",
                    "Delta": f"+{twin_comp.distance_penalty_nm:.1f} nm",
                },
                {
                    "Metric": "Air Time",
                    "High-Efficiency Flight": f"{twin_comp.fast_air_time_min:.1f} min",
                    "Low-Efficiency Flight": f"{twin_comp.slow_air_time_min:.1f} min",
                    "Delta": f"+{twin_comp.air_time_penalty_min:.1f} min",
                },
                {
                    "Metric": "Extension Ratio (KEA)",
                    "High-Efficiency Flight": f"{twin_comp.fast_extension_ratio:.3f}x",
                    "Low-Efficiency Flight": f"{twin_comp.slow_extension_ratio:.3f}x",
                    "Delta": f"+{twin_comp.slow_extension_ratio - twin_comp.fast_extension_ratio:.3f}x",
                },
                {
                    "Metric": "Fuel Burned",
                    "High-Efficiency Flight": f"{twin_comp.fast_fuel_kg:,.1f} kg",
                    "Low-Efficiency Flight": f"{twin_comp.slow_fuel_kg:,.1f} kg",
                    "Delta": f"+{twin_comp.excess_fuel_burned_kg:,.1f} kg",
                },
            ]
            st.dataframe(pd.DataFrame(twin_table), use_container_width=True)

            if twin_comp.divergence_lat and twin_comp.divergence_lon:
                st.warning(
                    f"⚠️ Major route divergence detected at coordinates: "
                    f"({twin_comp.divergence_lat:.3f}°, {twin_comp.divergence_lon:.3f}°)"
                )

# ----------------- TAB 8: What-If & Stress Tests -----------------
with tab8:
    st.header("Counterfactual Simulations & Airline Fleet Stress Testing")

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

# ----------------- TAB 9: Macro Evolution & AIRAC Diff -----------------
with tab9:
    st.header("Macro Evolution, Weather Impact & AIRAC Configuration Diffing")
    st.markdown(
        "Compare two Eurocontrol monthly archives to uncover macro shifts in global air traffic, "
        "airline market expansion, delay patterns, and fleet modernization (e.g. Next-Gen Neo/MAX adoption)."
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
