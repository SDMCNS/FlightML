"""
FlightML Streamlit FIR Configuration Sandbox View.

Provides an interactive what-if experimentation lab for DME and Radar networks:
1. Load the ideal configuration for any given FIR (from precomputed pipeline runs or dynamic simulation).
2. Interactively deactivate active stations (simulating unexpected outages/maintenance) or
   reactivate standby stations (simulating bringing backup equipment online).
3. Real-time sub-millisecond calculation of:
   - En-route coverage percentage (FL50 / FL200)
   - Terminal airport integrity down to FL10
   - List of newly degraded airports and blind terminals
   - OPEX financial savings balance
   - Daily operational delay penalties (€/day and €/h)
   - Single Point of Failure (SPOF) vulnerabilities
4. Interactive Leaflet map visualizing station postures and airport integrity pins.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from flightml.data.dataset import FlightDataset
from flightml.eval.ecac_station_rationalization import (
    ECAC_FIR_REGISTRY,
    DME_ANNUAL_COST_SAVINGS_EUR,
    RADAR_ANNUAL_COST_SAVINGS_EUR,
)
from flightml.eval.interactive_sandbox import (
    get_available_fir_list,
    load_fir_ideal_configuration,
    evaluate_custom_fir_posture,
    create_sandbox_map,
)


def render_fir_configuration_sandbox(selected_period: str) -> None:
    st.header("🎛️ Interactive FIR Configuration Sandbox (DME & Radar What-If Lab)")
    st.caption(
        "Load the ideal rationalized network configuration for any sovereign ECAC FIR, then interactively toggle "
        "(deactivate or reactivate) any individual ground beacon or radar head to inspect real-time impacts on coverage, "
        "terminal airport integrity down to FL10, financial OPEX savings, operational delay penalties, and single point of failure (SPOF) risks."
    )

    # 1. Top Controls Row
    col_c0, col_c1, col_c2, col_c3, col_c4 = st.columns([1.3, 1.1, 1.2, 1.8, 1.0])

    with col_c0:
        sys_choice = st.radio(
            "System Type",
            options=["🛰️ Radar Surveillance", "📻 DME Navigation"],
            horizontal=True,
            key=f"sb_sys_choice_{selected_period}",
            help="Select whether to experiment on Radar surveillance infrastructure or DME terrestrial navigation beacons.",
        )
        sys_type = "RADAR" if "Radar" in sys_choice else "DME"

    eval_periods = ["202507", "202510", "202402", "201611"]
    with col_c1:
        sim_period = st.selectbox(
            "Traffic Cycle",
            options=eval_periods,
            index=0,
            format_func=lambda p: f"☀️ {p} (July Peak)" if p == "202507" else f"{p} (Eurocontrol)",
            key=f"sb_cycle_{selected_period}",
            help="Traffic cycle period used for stress testing and flight density scaling.",
        )

    with col_c2:
        if sys_type == "RADAR":
            std_options = ["VIEW_3", "VIEW_1", "VIEW_2"]
            std_format = {
                "VIEW_3": "View 3: FL50 Infill & FL10 Hubs",
                "VIEW_1": "View 1: FL200 Single Baseline",
                "VIEW_2": "View 2: FL200 Dual Redundant",
            }
        else:
            std_options = ["RNAV_5", "RNAV_2", "RNAV_1", "RNP_1"]
            std_format = {
                "RNAV_5": "RNAV 5: En-Route B-RNAV",
                "RNAV_2": "RNAV 2: Continental ATS",
                "RNAV_1": "RNAV 1: Terminal P-RNAV",
                "RNP_1": "RNP 1: Reversionary Hub",
            }

        sel_std = st.selectbox(
            "Target Specification",
            options=std_options,
            index=0,
            format_func=lambda x: std_format.get(x, x),
            key=f"sb_std_{selected_period}",
        )

    ds = FlightDataset(period=sim_period)
    available_firs = get_available_fir_list(sys_type, dataset=ds)
    fir_ids = [f[0] for f in available_firs]
    fir_labels = {f[0]: f[1] for f in available_firs}
    default_fir = "EDUUUIR" if "EDUUUIR" in fir_ids else (fir_ids[0] if fir_ids else "LOVVFIR")
    default_idx = fir_ids.index(default_fir) if default_fir in fir_ids else 0

    with col_c3:
        sel_fir = st.selectbox(
            "Select Sovereign FIR",
            options=fir_ids,
            index=default_idx,
            format_func=lambda fid: fir_labels.get(fid, fid),
            key=f"sb_fir_select_{selected_period}",
            help="Select which sovereign FIR network to load and experiment upon.",
        )

    with col_c4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        reset_btn = st.button("🔄 Reset Ideal", key=f"sb_btn_reset_{selected_period}", help="Reset all toggles to the optimal candidate configuration.")

    # 2. Session State Management
    state_key_active = f"sb_active_{sys_type}_{sel_fir}_{sel_std}_{sim_period}"
    state_key_ideal = f"sb_ideal_{sys_type}_{sel_fir}_{sel_std}_{sim_period}"
    state_key_cfg = f"sb_cfg_{sys_type}_{sel_fir}_{sel_std}_{sim_period}"

    if state_key_cfg not in st.session_state or reset_btn:
        with st.spinner(f"Loading ideal configuration for {sel_fir} ({sys_type})..."):
            cfg = load_fir_ideal_configuration(
                system_type=sys_type,
                fir_id=sel_fir,
                target_standard=sel_std,
                period=sim_period,
                dataset=ds,
            )
            st.session_state[state_key_cfg] = cfg
            st.session_state[state_key_ideal] = set(cfg.get("optimal_active_station_ids", []))
            st.session_state[state_key_active] = set(cfg.get("optimal_active_station_ids", []))

    cfg = st.session_state[state_key_cfg]
    ideal_active_set = set(st.session_state[state_key_ideal])
    current_active_set = set(st.session_state[state_key_active])
    all_stations = cfg.get("all_stations", [])
    all_sta_ids = [s["id"] for s in all_stations]

    st.markdown("---")

    # 3. Quick Action Preset Buttons
    p_c1, p_c2, p_c3, p_c4 = st.columns([1.5, 1.5, 1.5, 2.5])
    with p_c1:
        if st.button("★ Load Ideal Configuration", key=f"sb_act_ideal_{selected_period}"):
            st.session_state[state_key_active] = set(ideal_active_set)
            st.rerun()
    with p_c2:
        if st.button("🏢 Full Baseline (100% Active)", key=f"sb_act_base_{selected_period}"):
            st.session_state[state_key_active] = set(all_sta_ids)
            st.rerun()
    with p_c3:
        if st.button("⚠️ Simulate N-1 Critical Outage", key=f"sb_act_n1_{selected_period}"):
            # Disable top critical SPOF
            spofs = cfg.get("critical_spofs", [])
            top_spof_id = spofs[0]["id"] if spofs else (list(ideal_active_set)[0] if ideal_active_set else None)
            if top_spof_id and top_spof_id in current_active_set:
                st.session_state[state_key_active] = current_active_set - {top_spof_id}
                st.rerun()
    with p_c4:
        delta_vs_ideal = len(current_active_set) - len(ideal_active_set)
        if delta_vs_ideal == 0:
            st.caption("Active configuration is currently identical to the **Ideal Configuration**.")
        elif delta_vs_ideal < 0:
            st.caption(f"Active configuration has **{abs(delta_vs_ideal)} station(s) deactivated** below the ideal core.")
        else:
            st.caption(f"Active configuration has **{delta_vs_ideal} standby station(s) reactivated** above the ideal core.")

    # 4. Instant Real-Time Evaluation
    eval_res = evaluate_custom_fir_posture(
        system_type=sys_type,
        fir_id=sel_fir,
        active_station_ids=current_active_set,
        target_standard=sel_std,
        ideal_active_station_ids=ideal_active_set,
        period=sim_period,
        dataset=ds,
    )

    # 5. KPI Metrics Grid
    km1, km2, km3, km4, km5 = st.columns(5)
    unit_cost = RADAR_ANNUAL_COST_SAVINGS_EUR if sys_type == "RADAR" else DME_ANNUAL_COST_SAVINGS_EUR

    km1.metric(
        "Active Equipment Posture",
        f"{eval_res['active_station_count']} / {eval_res['baseline_station_count']} {sys_type}s",
        f"{eval_res['delta_active_vs_ideal']:+d} vs Ideal Core",
    )
    km2.metric(
        "Annual ANSP OPEX Savings",
        f"€{eval_res['annual_opex_savings_eur']:,.0f} / yr",
        f"{eval_res['disabled_station_count']} Deactivated ({round(eval_res['disabled_station_count']/max(1, eval_res['baseline_station_count'])*100, 1)}%)",
    )
    km3.metric(
        f"{sel_std} Airspace Coverage",
        f"{eval_res['coverage_coverable_pct']:.1f}%",
        f"{eval_res['cells_met']} / {int(eval_res['coverage_coverable_pct']/100.0 * eval_res['total_cells'] if eval_res['total_cells'] else 1)} cells met",
    )
    deg_cnt = eval_res["degraded_airports_count"]
    km4.metric(
        "Terminal / Hub Integrity (FL10)",
        f"{deg_cnt} Degraded" if deg_cnt > 0 else "100% Intact",
        "⚠️ Deficit" if deg_cnt > 0 else "✅ Dual Redundant",
    )
    pen = eval_res["daily_operational_penalty_eur"]
    km5.metric(
        "Daily Operational Delay Penalty",
        f"+€{pen:,.0f} / day" if pen > 0 else "€0 / day",
        f"+€{pen/24:,.0f} / h" if pen > 0 else "Zero Operational Deficit",
    )

    # 6. Operational Status Banners
    deg_apts = eval_res["degraded_airports_idents"]
    if deg_cnt > 0:
        deg_str = ", ".join(deg_apts[:6])
        st.error(
            f"🚨 **CRITICAL TERMINAL SURVEILLANCE / NAVAID DEFICIT:** {deg_cnt} airport(s) lose required low-level line-of-sight ({deg_str}) down to FL10! "
            f"Causes severe procedural separation expansions, stack holding, and a net operational penalty of **+€{pen:,.0f}/day (+€{pen/24:,.0f}/h)**."
        )
    elif eval_res["coverage_coverable_pct"] < 90.0:
        st.warning(
            f"⚠️ **EN-ROUTE COVERAGE DEFICIT:** Airspace coverage met dropped to {eval_res['coverage_coverable_pct']:.1f}% (below the 90% threshold). "
            f"Requires lateral airway rerouting and radar miles-in-trail flow restrictions."
        )
    elif eval_res["delta_active_vs_ideal"] < 0:
        st.warning(
            f"⚠️ **EXPERIMENTAL OUTAGE POSTURE:** {abs(eval_res['delta_active_vs_ideal'])} station(s) deactivated below the ideal core. "
            f"All airports currently maintain required coverage, but network safety margin is reduced."
        )
    elif eval_res["delta_active_vs_ideal"] > 0:
        st.info(
            f"ℹ️ **REDUNDANT AUGMENTATION:** {eval_res['delta_active_vs_ideal']} standby station(s) brought back online above the ideal core. "
            f"Increases overlap redundancy at an added OPEX cost of **+€{eval_res['delta_active_vs_ideal'] * unit_cost:,.0f}/yr**."
        )
    else:
        st.success(
            "✅ **IDEAL RATIONALIZED CORE ACTIVE:** Maximum financial OPEX savings achieved with 100% airspace coverage compliance and 0 degraded airports."
        )

    # 7. Interactive Map & Station Controls Columns
    map_col, ctrl_col = st.columns([3, 2])

    with map_col:
        st.markdown("#### 🗺️ Airspace Posture & Terminal Integrity Map")
        st.caption(
            "🟢 Green = Active Core | 🔴 Red = Standby / Decommissioned | 🟠 Orange = Toggled Outage | 🔵 Blue = Toggled Online | "
            "Pins: 🟢 Green Pin = Hub FL10 Met | 🔴 Red Pin = Hub Lost Coverage!"
        )
        sandbox_map = create_sandbox_map(
            eval_result=eval_res,
            ideal_active_station_ids=ideal_active_set,
            dataset=ds,
        )
        st_folium(sandbox_map, width=760, height=520, returned_objects=[], key=f"sb_folium_{sys_type}_{sel_fir}_{len(current_active_set)}")

    with ctrl_col:
        st.markdown(f"#### 🎛️ Station Toggles ({len(all_stations)} Stations)")
        st.caption("Select stations to keep active in the sandbox. Toggling a station triggers an instant recalculation.")

        selected_active_ids = st.multiselect(
            f"Active {sys_type} Stations in Sandbox:",
            options=all_sta_ids,
            default=sorted(list(current_active_set)),
            format_func=lambda sid: next((f"{s['name']} [{s['id']}]" for s in all_stations if s["id"] == sid), sid),
            key=f"sb_multiselect_{sys_type}_{sel_fir}_{len(current_active_set)}",
            help="Add stations to activate them; remove stations to simulate an outage or deactivation.",
        )

        if set(selected_active_ids) != current_active_set:
            st.session_state[state_key_active] = set(selected_active_ids)
            st.rerun()

        # Station Inventory Overview Table
        roster_rows = []
        for s in all_stations:
            sid = s["id"]
            is_active = sid in current_active_set
            was_ideal = sid in ideal_active_set
            if is_active and was_ideal:
                status_badge = "🟢 ACTIVE CORE"
            elif not is_active and not was_ideal:
                status_badge = "🔴 STANDBY"
            elif not is_active and was_ideal:
                status_badge = "🟠 OUTAGE (Off)"
            else:
                status_badge = "🔵 ONLINE (Added)"

            roster_rows.append({
                "Station": f"{s['name']} [{s['id']}]",
                "Role": s.get("role", "En-Route"),
                "Ideal": "Active" if was_ideal else "Standby",
                "Sandbox": status_badge,
            })
        st.dataframe(pd.DataFrame(roster_rows), use_container_width=True, height=360)

    # 8. Single Points of Failure (SPOF) Analysis for Current Sandbox State
    spofs = eval_res.get("active_spofs", [])
    st.markdown("---")
    st.markdown(f"#### 🚨 Single Point of Failure (SPOF) Vulnerability Analysis ({eval_res['active_station_count']} Active {sys_type}s)")
    if spofs:
        st.caption(
            f"Under the currently active configuration, **{len(spofs)} station(s)** are critical single points of failure. "
            f"If any of these stations experience an unplanned hardware fault, terminal coverage at the associated airports will be immediately lost."
        )
        spof_rows = [
            {
                "Critical Station ID": s["station_id"],
                "Station Name": s["station_name"],
                "Vulnerable Airports Protected": ", ".join(s["airports_vulnerable"]),
                "Vulnerability Consequence": f"Terminal radar/dual-DME blackout at {len(s['airports_vulnerable'])} airport(s)",
            }
            for s in spofs
        ]
        st.dataframe(pd.DataFrame(spof_rows), use_container_width=True)
    else:
        st.success(f"Zero Single Points of Failure detected in the current active station configuration! All terminal airports have redundant multi-station line-of-sight.")

    # 9. Download Configuration JSON
    st.markdown("---")
    cfg_export_data = {
        "fir_id": sel_fir,
        "fir_name": ECAC_FIR_REGISTRY.get(sel_fir, sel_fir),
        "system_type": sys_type,
        "target_standard": sel_std,
        "period": sim_period,
        "active_station_ids": sorted(list(current_active_set)),
        "disabled_station_ids": sorted(list(set(all_sta_ids) - current_active_set)),
        "metrics": {
            "active_count": eval_res["active_station_count"],
            "baseline_count": eval_res["baseline_station_count"],
            "coverage_compliance_pct": eval_res["coverage_coverable_pct"],
            "degraded_airports_count": eval_res["degraded_airports_count"],
            "degraded_airports_idents": eval_res["degraded_airports_idents"],
            "annual_opex_savings_eur": eval_res["annual_opex_savings_eur"],
            "daily_operational_penalty_eur": eval_res["daily_operational_penalty_eur"],
        },
        "exported_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    json_bytes = json.dumps(cfg_export_data, indent=2).encode("utf-8")
    st.download_button(
        label=f"⬇️ Export Current Custom {sel_fir} Configuration (JSON)",
        data=json_bytes,
        file_name=f"custom_config_{sys_type}_{sel_fir}_{sel_std}_{sim_period}.json",
        mime="application/json",
        key=f"sb_dl_config_{selected_period}",
    )
