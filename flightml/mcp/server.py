"""
FlightML Model Context Protocol (MCP) Server.

Exposes FlightML analytics, 4D trajectory twins, spatio-temporal delay contagion,
counterfactual FIR airspace closures, and multi-agent ATFM slot gaming simulations
to local LLMs (LM Studio, Claude Desktop, Cursor, Ollama) via standard MCP protocols.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from flightml.data.dataset import FlightDataset
from flightml.eval.multi_level_kpis import get_dataset_date_bounds
from flightml.models.contrail_surrogate import ContrailNeuralSurrogate
from flightml.models.flight2vec import Flight2VecEngine
from flightml.models.st_gnn import SpatioTemporalDelayEngine
from flightml.sim.airspace_closure import AirspaceClosureSimulator
from flightml.sim.marl_atfm import ASSUMPTIONS_GLOSSARY, MultiAgentATFMSimulator
from flightml.eval.passenger_journey import PassengerJourneyEngine
from flightml.geo.fra_boundary_engine import (
    analyze_fir_boundary_inefficiencies,
    get_boundary_detail,
    compare_fra_boundary_evolution as compare_fra_boundary_evo,
    get_worst_flights_for_boundary,
    calculate_fra_long_term_forecast,
)

# Global singleton cache for datasets by period
_DATASET_CACHE: dict[str, FlightDataset] = {}


# Map node capacities from ST-GNN topology
NOMINAL_CAPACITIES: dict[str, int] = {
    node_id: cap for node_id, _, cap in SpatioTemporalDelayEngine.MAJOR_NODES
}


def get_dataset(period: str = "202510") -> FlightDataset:
    """Retrieve or lazily initialize a FlightDataset instance for a given period."""
    p = str(period).strip()
    if p not in _DATASET_CACHE:
        _DATASET_CACHE[p] = FlightDataset(period=p)
    return _DATASET_CACHE[p]


def _safe_json(obj: Any) -> str:
    """Helper to cleanly serialize dataclasses, dicts, DataFrames, and numbers to formatted JSON."""
    import pandas as pd
    from dataclasses import asdict, is_dataclass
    if isinstance(obj, pd.DataFrame):
        data = obj.to_dict(orient="records")
    elif is_dataclass(obj) and not isinstance(obj, type):
        data = asdict(obj)
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        data = obj.to_dict()
    else:
        data = obj
    return json.dumps(data, indent=2, default=str)


def create_mcp_server() -> FastMCP:
    """Factory creating and configuring the FlightML FastMCP server."""
    server = FastMCP(
        name="FlightML Aviation Intelligence",
        instructions=(
            "FlightML provides high-fidelity European aviation analytics, 4D trajectory modeling, "
            "Eurocontrol DDR2 radar track querying, physics-informed neural contrail calculations, "
            "Spatio-Temporal Graph Neural Network (ST-GNN) delay propagation simulations, "
            "counterfactual airspace (FIR) closure rerouting, and Multi-Agent RL ATFM slot gaming."
        ),
    )

    # -------------------------------------------------------------------------
    # TOOL: get_flight_details
    # -------------------------------------------------------------------------
    @server.tool()
    def get_flight_details(flight_id: str | int, period: str = "202510") -> str:
        """Retrieve verified operational radar telemetry for a specific flight by ectrl_id.

        Args:
            flight_id: Eurocontrol unique flight identifier (e.g. '291262156' or 291262156).
            period: Eurocontrol archive month (e.g. '202510').
        """
        ds = get_dataset(period)
        fid_clean = str(flight_id).strip()
        query_sql = f"""
            SELECT
                ectrl_id, adep, ades, ac_type, ac_operator, ac_reg,
                market_segment, flight_type, actual_distance_nm,
                actual_air_time_min, filed_air_time_min,
                dep_delay_min, arr_delay_min, requested_fl,
                actual_off_block_time, actual_arrival_time,
                filed_off_block_time, filed_arrival_time
            FROM flights
            WHERE ectrl_id = '{fid_clean}'
            LIMIT 1;
        """
        try:
            df = ds.query(query_sql)
            if df.empty:
                return json.dumps({
                    "error": f"Flight ID '{fid_clean}' not found in period '{period}'."
                }, indent=2)
            row = df.iloc[0].to_dict()
            clean_row = {k: str(v) if v is not None and not isinstance(v, (int, float, bool)) else v for k, v in row.items()}
            return json.dumps(clean_row, indent=2)
        except Exception as exc:
            return json.dumps({"error": f"Query failed: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: search_flights
    # -------------------------------------------------------------------------
    @server.tool()
    def search_flights(
        ades: str = "EGLL",
        adep: str = "",
        operator: str = "",
        date_str: str = "2025-10-15",
        limit: int = 10,
        period: str = "202510",
    ) -> str:
        """Search scheduled and operated flights by destination, origin, operator, and date.

        Args:
            ades: ICAO arrival airport (e.g. 'EGLL', 'LFPG', 'EHAM').
            adep: Optional ICAO departure airport (e.g. 'KJFK', 'EDDF').
            operator: Optional ICAO airline operator code (e.g. 'BAW', 'DLH', 'AFR').
            date_str: Flight date (YYYY-MM-DD).
            limit: Maximum number of candidate flights to return (1-50).
            period: Eurocontrol archive month (e.g. '202510').
        """
        ds = get_dataset(period)
        conditions = []
        if ades and ades.strip():
            conditions.append(f"ades = '{ades.strip().upper()}'")
        if adep and adep.strip():
            conditions.append(f"adep = '{adep.strip().upper()}'")
        if operator and operator.strip():
            conditions.append(f"ac_operator = '{operator.strip().upper()}'")
        if date_str and date_str.strip():
            conditions.append(f"CAST(COALESCE(actual_arrival_time, filed_arrival_time, actual_off_block_time) AS DATE) = '{date_str.strip()}'")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        lim = max(1, min(int(limit), 50))
        query_sql = f"""
            SELECT
                ectrl_id, adep, ades, ac_type, ac_operator, ac_reg,
                actual_distance_nm, arr_delay_min, requested_fl,
                actual_off_block_time, actual_arrival_time
            FROM flights
            WHERE {where_clause}
            ORDER BY COALESCE(actual_arrival_time, filed_arrival_time) ASC
            LIMIT {lim};
        """
        try:
            df = ds.query(query_sql)
            records = []
            for _, r in df.iterrows():
                row_dict = r.to_dict()
                clean_row = {k: str(v) if v is not None and not isinstance(v, (int, float, bool)) else v for k, v in row_dict.items()}
                records.append(clean_row)
            return json.dumps({
                "query_filter": {"ades": ades, "adep": adep, "operator": operator, "date": date_str},
                "total_matches": len(records),
                "flights": records,
            }, indent=2)
        except Exception as exc:
            return json.dumps({"error": f"Search failed: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: find_operational_twins
    # -------------------------------------------------------------------------
    @server.tool()
    def find_operational_twins(flight_id: str | int, top_k: int = 5, period: str = "202510") -> str:
        """Find the top-K operational twin flights in 128D Flight2Vec hypersphere space.

        Matches flights with similar stage lengths, cruise altitudes, speed profiles,
        and identical commercial aircraft taxonomy (e.g., A320 commercial jets).

        Args:
            flight_id: Eurocontrol unique flight identifier (e.g. '291262156' or 291262156).
            top_k: Number of twin matches to return (1-20).
            period: Eurocontrol archive month (e.g. '202510').
        """
        ds = get_dataset(period)
        f2v = Flight2VecEngine(ds)
        try:
            matches = f2v.find_operational_twins(target_ectrl_id=int(flight_id), top_k=int(top_k))
            twins_list = [asdict(m) if hasattr(m, "__dataclass_fields__") else m.__dict__ for m in matches]
            return json.dumps({
                "target_flight_id": str(flight_id),
                "total_twins": len(twins_list),
                "twins": twins_list,
            }, indent=2)
        except Exception as exc:
            return json.dumps({"error": f"Flight2Vec twin search failed: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: simulate_delay_contagion
    # -------------------------------------------------------------------------
    @server.tool()
    def simulate_delay_contagion(
        shock_origin: str = "EGLL",
        injected_delay_min: float = 60.0,
        shock_date: str = "2025-10-15",
        shock_time_hour: int = 10,
        shock_duration_hours: float = 2.0,
        period: str = "202510",
    ) -> str:
        """Simulate spatio-temporal delay propagation from an airport disruption across European hubs.

        Uses Spatio-Temporal Graph WaveNet modeling runway saturation and turnaround airframe rotations.

        Args:
            shock_origin: ICAO airport code where delay shock is injected (e.g. 'EGLL', 'LFPG', 'EHAM').
            injected_delay_min: Shock magnitude in minutes (15 to 180 min).
            shock_date: Date of the shock event (YYYY-MM-DD).
            shock_time_hour: Start time hour in UTC (0 to 23).
            shock_duration_hours: Duration of disruption in hours (0.5 to 8.0).
            period: Eurocontrol archive month.
        """
        ds = get_dataset(period)
        engine = SpatioTemporalDelayEngine(ds, period=period)
        try:
            res = engine.simulate_delay_propagation(
                shock_origin=shock_origin.strip().upper(),
                injected_delay_min=float(injected_delay_min),
                shock_date=shock_date.strip() if shock_date else None,
                shock_time_hour=int(shock_time_hour),
                shock_duration_hours=float(shock_duration_hours),
            )
            return json.dumps(res.to_dict(), indent=2)
        except Exception as exc:
            return json.dumps({"error": f"Delay contagion simulation failed: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: simulate_fir_closure
    # -------------------------------------------------------------------------
    @server.tool()
    def simulate_fir_closure(
        target_fir: str = "LSASFIR",
        closure_date: str = "2025-10-15",
        start_hour: int = 10,
        duration_hours: float = 3.0,
        period: str = "202510",
    ) -> str:
        """Simulate tactical counterfactual rerouting for a temporary Flight Information Region (FIR) closure.

        Calculates diverted flights, detour nautical miles, extra fuel burn, airline economic losses (€ EUR),
        and post-reopening bottleneck recovery backlog.

        Args:
            target_fir: FIR airspace code (e.g. 'LSASFIR', 'EDUUUIR', 'LFFFFIR', 'EGTTFIR').
            closure_date: Date of the closure event (YYYY-MM-DD).
            start_hour: Start time hour in UTC (0 to 23).
            duration_hours: Duration of closure in hours (1.0 to 8.0).
            period: Eurocontrol archive month.
        """
        ds = get_dataset(period)
        sim = AirspaceClosureSimulator(ds, period=period)
        try:
            res = sim.simulate_fir_closure(
                target_fir=target_fir.strip().upper(),
                closure_start_hour=int(start_hour),
                closure_duration_hours=float(duration_hours),
                date_str=closure_date.strip() if closure_date else None,
            )
            return json.dumps(res.to_dict(), indent=2)
        except Exception as exc:
            return json.dumps({"error": f"FIR closure simulation failed: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: simulate_marl_slot_gaming
    # -------------------------------------------------------------------------
    @server.tool()
    def simulate_marl_slot_gaming(
        regulated_airport: str = "EGLL",
        capacity_cut_pct: float = 40.0,
        regulation_date: str = "2025-10-15",
        start_hour: int = 10,
        duration_hours: float = 3.0,
        iterations: int = 50,
        period: str = "202510",
    ) -> str:
        """Run Multi-Agent Reinforcement Learning (MARL) ATFM slot gaming simulation.

        Returns chronological blow-by-blow decision log, infilled domain assumptions
        (connecting passengers, EU261 exposure, crew FDP duty limits), and per-airline gains/losses balance sheets.

        Args:
            regulated_airport: Bottleneck airport ICAO code (e.g. 'EGLL', 'EHAM', 'LFPG', 'EDDF').
            capacity_cut_pct: Percentage declared arrival capacity reduction (10% to 80%).
            regulation_date: Regulation date (YYYY-MM-DD).
            start_hour: Start time hour in UTC (0 to 23).
            duration_hours: Duration of regulation in hours (0.5 to 8.0).
            iterations: Multi-agent game iterations to reach Nash equilibrium.
            period: Eurocontrol archive month.
        """
        ds = get_dataset(period)
        sim = MultiAgentATFMSimulator(ds, period=period)
        try:
            res = sim.run_marl_simulation(
                regulated_airport=regulated_airport.strip().upper(),
                capacity_cut_pct=float(capacity_cut_pct),
                date_str=regulation_date.strip() if regulation_date else None,
                start_time_hour=int(start_hour),
                duration_hours=float(duration_hours),
                iterations=int(iterations),
            )
            return json.dumps(res.to_dict(), indent=2)
        except Exception as exc:
            return json.dumps({"error": f"MARL slot gaming simulation failed: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: evaluate_contrail_climate_forcing
    # -------------------------------------------------------------------------
    @server.tool()
    def evaluate_contrail_climate_forcing(
        temperature_c: float = -54.0,
        rh_ice_pct: float = 120.0,
        flight_level: float = 360.0,
        rh_water_pct: float = 75.0,
        bypass_ratio: float = 8.5,
    ) -> str:
        """Evaluate contrail climate forcing using a Physics-Informed Neural Network surrogate.

        Computes the Schmidt-Appleman criterion, persistent Ice Super-Saturated Region (ISSR) formation,
        optical depth, and instantaneous radiative forcing (mW/m^2).

        Args:
            temperature_c: Ambient static air temperature in Celsius (e.g. -54.0 °C).
            rh_ice_pct: Relative humidity with respect to ice (e.g. 120.0%).
            flight_level: Cruising flight level (e.g. FL360).
            rh_water_pct: Relative humidity with respect to liquid water (e.g. 75.0%).
            bypass_ratio: Engine bypass ratio (e.g. 8.5 for modern turbofans).
        """
        surrogate = ContrailNeuralSurrogate()
        try:
            res = surrogate.evaluate_atmospheric_point(
                temperature_c=float(temperature_c),
                rh_water_pct=float(rh_water_pct),
                rh_ice_pct=float(rh_ice_pct),
                flight_level=float(flight_level),
                bypass_ratio=float(bypass_ratio),
            )
            return json.dumps(res.to_dict(), indent=2)
        except Exception as exc:
            return json.dumps({"error": f"Contrail evaluation failed: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_airport_capacity
    # -------------------------------------------------------------------------
    @server.tool()
    def get_airport_capacity(airport_icao: str = "EGLL") -> str:
        """Get the nominal hourly runway capacity and saturation baseline for a European airport.

        Args:
            airport_icao: 4-letter ICAO airport identifier (e.g. 'EGLL', 'LFPG', 'EHAM', 'EDDF').
        """
        apt = airport_icao.strip().upper()
        cap = NOMINAL_CAPACITIES.get(apt, 40)
        return json.dumps({
            "airport_icao": apt,
            "nominal_hourly_capacity_movements": cap,
            "profile": "Dual parallel runway saturated hub" if cap >= 85 else "Coordinated European airport / FIR node",
        }, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: assess_passenger_itinerary
    # -------------------------------------------------------------------------
    @server.tool()
    def assess_passenger_itinerary(
        origin: str = "EINN",
        destination: str = "EBBR",
        hub: str = "",
        day_of_week: str = "tuesday",
        time_window: str = "morning",
        scheduled_layover_min: int = 75,
        target_date: str = "",
        period: str = "auto",
        debug: bool = False,
    ) -> str:
        """Assess flight journey delay risk, missed connection probability, weather impact, and EU261 compensation for the general travelling public.

        Args:
            origin: Departure airport 4-letter ICAO code (e.g. 'EINN' Shannon, 'EHAM' Amsterdam).
            destination: Arrival airport 4-letter ICAO code (e.g. 'EBBR' Brussels, 'EGLL' Heathrow).
            hub: Optional transfer connection hub (e.g. 'EGLL'). Leave empty for direct flights.
            day_of_week: Day of week for departure (e.g. 'tuesday', 'monday', 'friday').
            time_window: Departure window ('morning', 'midday', 'afternoon', 'evening').
            scheduled_layover_min: Scheduled layover duration at transfer hub in minutes (e.g. 75).
            target_date: Target flight date in YYYY-MM-DD format (or leave empty for upcoming forecast).
            period: Eurocontrol baseline data period (default: 'auto' to auto-select the best fitting archive for the target date).
            debug: Set True to include full diagnostic audit accounting of historical flights, NWP weather endpoints, and buffers.
        """
        try:
            from flightml.data.dataset import FlightDataset
            parsed_date, _ = FlightDataset.parse_flight_date(target_date, day_of_week)
            resolved_date_str = parsed_date.isoformat()

            if str(period).lower().strip() in ("auto", "none", ""):
                resolved_period, _ = FlightDataset.find_best_period(target_date=resolved_date_str, day_of_week=day_of_week)
            else:
                resolved_period = str(period).strip()

            ds = get_dataset(resolved_period)
            engine = PassengerJourneyEngine(ds)
            clean_hub = hub.strip() if hub and hub.strip().upper() not in (origin.strip().upper(), destination.strip().upper(), "NONE", "") else None
            res = engine.assess_itinerary(
                origin=origin.strip().upper(),
                destination=destination.strip().upper(),
                hub=clean_hub,
                day_of_week=day_of_week,
                time_window=time_window,
                scheduled_layover_min=scheduled_layover_min,
                target_date=resolved_date_str,
                period=resolved_period,
                debug=debug,
            )
            return json.dumps(res.to_dict(), indent=2)
        except Exception as exc:
            return json.dumps({"error": f"Passenger itinerary assessment failed: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_pru_scorecard
    # -------------------------------------------------------------------------
    @server.tool()
    def get_pru_scorecard(flight_id: str | int, period: str = "202510") -> str:
        """Compute ICAO GANP & Eurocontrol PRU flight efficiency scorecard (KPI04 KEP, KPI05 KEA, KPI08 ASMA, level-capping).

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.eval.pru_kpis import compute_flight_pru_kpis
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            sc = compute_flight_pru_kpis(ds, ectrl_id=fid)
            if not sc:
                return json.dumps({"error": f"Flight {flight_id} not found in period {period}."}, indent=2)
            return _safe_json(sc)
        except Exception as exc:
            return json.dumps({"error": f"Failed to compute PRU scorecard: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_flight_doc
    # -------------------------------------------------------------------------
    @server.tool()
    def get_flight_doc(flight_id: str | int, period: str = "202510", time_cost_eur_per_hr: float = 850.0) -> str:
        """Calculate Total Direct Operating Cost (DOC = Fuel EUR + Crew/Time EUR + CRCO Charges EUR) and route charge avoidance incentive ratio.

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
            time_cost_eur_per_hr: Airline crew and aircraft time cost benchmark in EUR/hour (default: 850.0).
        """
        try:
            from flightml.eval.airline_economics import AirlineEconomicsEngine
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            econ = AirlineEconomicsEngine(ds)
            doc = econ.evaluate_flight_doc(fid, cost_of_time_eur_per_hour=time_cost_eur_per_hr)
            if not doc:
                return json.dumps({"error": f"Flight {flight_id} not found in period {period}."}, indent=2)
            return _safe_json(doc)
        except Exception as exc:
            return json.dumps({"error": f"Failed to compute Direct Operating Cost: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_crco_charges
    # -------------------------------------------------------------------------
    @server.tool()
    def get_crco_charges(flight_id: str | int, period: str = "202510") -> str:
        """Calculate Eurocontrol CRCO En-Route Navigation Charges in Euros (EUR), Service Units, and FIR charging zones.

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.eval.crco_economics import compute_flight_crco_charges
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            crco = compute_flight_crco_charges(ds, ectrl_id=fid)
            if not crco:
                return json.dumps({"error": f"Flight {flight_id} not found in period {period}."}, indent=2)
            return _safe_json(crco)
        except Exception as exc:
            return json.dumps({"error": f"Failed to compute CRCO route charges: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: scan_route_charge_avoidance
    # -------------------------------------------------------------------------
    @server.tool()
    def scan_route_charge_avoidance(
        period: str = "202510",
        min_flights: int = 50,
        detour_nm_threshold: float = 15.0,
        fee_saved_threshold_eur: float = 100.0,
        limit: int = 10,
    ) -> str:
        """Scan European network city pairs for systematic FIR route charge avoidance detours.

        Args:
            period: Eurocontrol archive month (e.g. '202510').
            min_flights: Minimum flights on city pair to assess (default: 50).
            detour_nm_threshold: Minimum nautical miles detour above Great Circle (default: 15.0).
            fee_saved_threshold_eur: Minimum CRCO fees saved in EUR (default: 100.0).
            limit: Maximum city pairs to return (default: 10).
        """
        try:
            from flightml.eval.crco_economics import detect_route_charge_avoidance
            ds = get_dataset(period)
            df = detect_route_charge_avoidance(
                ds,
                min_flights=min_flights,
                detour_nm_threshold=detour_nm_threshold,
                fee_saved_threshold_eur=fee_saved_threshold_eur,
                top_n=limit,
            )
            return _safe_json(df)
        except Exception as exc:
            return json.dumps({"error": f"Failed to scan route charge avoidance: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_fuel_and_emissions
    # -------------------------------------------------------------------------
    @server.tool()
    def get_fuel_and_emissions(
        flight_id: str | int,
        period: str = "202510",
        jet_a_price_eur_per_kg: float = 0.6421,
    ) -> str:
        """Compute OpenAP kinematic fuel burn, ICAO engine emissions (CO2, NOx), and Jet-A expenditure in Euros (EUR).

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
            jet_a_price_eur_per_kg: Jet-A price in EUR per kg (default: 0.6421).
        """
        try:
            from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            impact = compute_flight_fuel_and_emissions(
                ds, ectrl_id=fid, fuel_price_per_kg=jet_a_price_eur_per_kg
            )
            if not impact:
                return json.dumps({"error": f"Flight {flight_id} not found or has insufficient trajectory data."}, indent=2)
            return _safe_json(impact)
        except Exception as exc:
            return json.dumps({"error": f"Failed to compute fuel and emissions: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_cdo_descent_profile
    # -------------------------------------------------------------------------
    @server.tool()
    def get_cdo_descent_profile(flight_id: str | int, period: str = "202510") -> str:
        """Analyze Continuous Descent Operations (CDO), Top of Descent (TOD), low-altitude level segments (<10,000 ft), and noise/fuel penalties.

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.eval.cdo_analyzer import analyze_cdo_descent
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            res = analyze_cdo_descent(ds, ectrl_id=fid)
            if not res:
                return json.dumps({"error": f"Flight {flight_id} not found in period {period}."}, indent=2)
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to analyze CDO descent: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_cco_climb_profile
    # -------------------------------------------------------------------------
    @server.tool()
    def get_cco_climb_profile(flight_id: str | int, period: str = "202510") -> str:
        """Analyze Continuous Climb Operations (CCO), intermediate climb level-offs, and initial climb profile efficiency.

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.eval.cdo_analyzer import analyze_cco_climb
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            res = analyze_cco_climb(ds, ectrl_id=fid)
            if not res:
                return json.dumps({"error": f"Flight {flight_id} not found in period {period}."}, indent=2)
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to analyze CCO climb: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_fra_adherence
    # -------------------------------------------------------------------------
    @server.tool()
    def get_fra_adherence(
        flight_id: str | int,
        period: str = "202510",
        direct_dct_tolerance_nm: float = 5.0,
        min_flight_level: float = 305.0,
    ) -> str:
        """Evaluate Free Route Airspace (FRA) adherence, DCT direct waypoint shortcuts, and mileage savings vs structured airways.
        Specifically evaluates both overall en-route directness and the mandated upper FRA stratum (>= FL305 per Single European Sky CP1).

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
            direct_dct_tolerance_nm: Cross-track tolerance in NM for direct segment detection (default: 5.0).
            min_flight_level: Minimum flight level for mandatory FRA stratum evaluation (default: 305.0).
        """
        try:
            from flightml.geo.fra_analyzer import analyze_flight_fra_adherence
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            res = analyze_flight_fra_adherence(
                ds,
                ectrl_id=fid,
                snap_tolerance_nm=direct_dct_tolerance_nm,
                min_fl=min_flight_level,
            )
            if not res:
                return json.dumps({"error": f"Flight {flight_id} not found in period {period}."}, indent=2)
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to analyze FRA adherence: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: diagnose_flight_bottlenecks
    # -------------------------------------------------------------------------
    @server.tool()
    def diagnose_flight_bottlenecks(flight_id: str | int, period: str = "202510") -> str:
        """Run 360-degree bottleneck diagnostic: departure surface delay, en-route ATFM re-routing, arrival terminal holding, and weather attribution.

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.eval.diagnostics import diagnose_flight_bottlenecks
            from flightml.weather.metar_client import MetarClient
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            diag = diagnose_flight_bottlenecks(ds, metar_client=MetarClient(), ectrl_id=fid)
            if not diag:
                return json.dumps({"error": f"Flight {flight_id} not found in period {period}."}, indent=2)
            return _safe_json(diag)
        except Exception as exc:
            return json.dumps({"error": f"Failed to diagnose flight bottlenecks: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_fir_saturation
    # -------------------------------------------------------------------------
    @server.tool()
    def get_fir_saturation(
        fir_id: str = "EDUUUIR",
        date: str = "",
        period: str = "202510",
        threshold_pct: float = 85.0,
        limit: int = 10,
    ) -> str:
        """Calculate hourly FIR airspace saturation, peak traffic volume, nominal capacity breaches, and ATFM delay severity.

        Args:
            fir_id: European FIR identifier (e.g. 'EDUUUIR', 'LFFFFIR', 'EGTTFIR').
            date: Target date in YYYY-MM-DD format (leave blank for first date in period).
            period: Eurocontrol archive month (e.g. '202510').
            threshold_pct: Saturation capacity threshold percentage (default: 85.0).
            limit: Maximum hourly records to return (default: 10).
        """
        try:
            from flightml.features.fir_saturation import calculate_fir_hourly_saturation
            ds = get_dataset(period)
            res = calculate_fir_hourly_saturation(
                ds,
                fir_id=fir_id,
                date_str=date or None,
                threshold_flights_per_hr=int(threshold_pct),
                top_n=limit,
            )
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to calculate FIR saturation: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_flight_fir_dwell
    # -------------------------------------------------------------------------
    @server.tool()
    def get_flight_fir_dwell(flight_id: str | int, period: str = "202510") -> str:
        """Calculate flight-level sector-by-sector entry/exit timestamps, dwell durations, and dwell inefficiency delays.

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.features.fir_saturation import calculate_flight_fir_dwell_inefficiency
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            dwells = calculate_flight_fir_dwell_inefficiency(ds, ectrl_id=fid)
            return _safe_json(dwells)
        except Exception as exc:
            return json.dumps({"error": f"Failed to calculate FIR dwell inefficiency: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_airport_taxi_out
    # -------------------------------------------------------------------------
    @server.tool()
    def get_airport_taxi_out(
        airport_icao: str = "EGLL",
        date: str = "",
        period: str = "202510",
        min_flights: int = 5,
        limit: int = 10,
    ) -> str:
        """Analyze airport surface taxi-out performance: unimpeded baseline duration vs actual off-block to takeoff time and excess queue delays.

        Args:
            airport_icao: Origin aerodrome ICAO (e.g. 'EGLL', 'LFPG', 'EHAM').
            date: Target date in YYYY-MM-DD format (leave blank for first date).
            period: Eurocontrol archive month (e.g. '202510').
            min_flights: Minimum flights per hour to consider (default: 5).
            limit: Maximum hourly records to return (default: 10).
        """
        try:
            from flightml.eval.surface_taxi import analyze_airport_taxi_out_performance
            ds = get_dataset(period)
            res = analyze_airport_taxi_out_performance(
                ds,
                airport_icao=airport_icao.strip().upper(),
                date_str=date or None,
                min_flights=min_flights,
                top_n=limit,
            )
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to analyze airport taxi-out: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_flight_taxi_out
    # -------------------------------------------------------------------------
    @server.tool()
    def get_flight_taxi_out(flight_id: str | int, period: str = "202510") -> str:
        """Analyze taxi-out surface duration for a specific flight: off-block to airborne time, unimpeded reference, and excess minutes.

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.eval.surface_taxi import analyze_flight_taxi_out
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            res = analyze_flight_taxi_out(ds, ectrl_id=fid)
            if not res:
                return json.dumps({"error": f"Flight {flight_id} not found in period {period}."}, indent=2)
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to analyze flight taxi-out: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: assess_runway_safety
    # -------------------------------------------------------------------------
    @server.tool()
    def assess_runway_safety(flight_id: str | int, period: str = "202510") -> str:
        """Assess runway safety & operational wind limits: crosswind component, tailwind component, exceedance flags, and wet runway braking margins.

        Args:
            flight_id: Eurocontrol unique flight identifier (ectrl_id).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.weather.runway_safety import assess_flight_runway_safety
            ds = get_dataset(period)
            fid = int(str(flight_id).strip())
            res = assess_flight_runway_safety(ds, metar_client=None, ectrl_id=fid)
            if not res:
                return json.dumps({"error": f"Flight {flight_id} not found in period {period}."}, indent=2)
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to assess runway safety: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: analyze_reactionary_delay
    # -------------------------------------------------------------------------
    @server.tool()
    def analyze_reactionary_delay(
        operator: str = "BAW",
        start_date: str = "",
        end_date: str = "",
        period: str = "202510",
    ) -> str:
        """Analyze airline rotational airframe delay propagation, turnaround slack absorption, and downstream reactionary delays.

        Args:
            operator: Airline operator ICAO (e.g. 'BAW', 'DLH', 'AFR', 'RYR').
            start_date: Start date YYYY-MM-DD (optional).
            end_date: End date YYYY-MM-DD (optional).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.models.reactionary_delay import ReactionaryDelayAnalyzer
            from flightml.eval.multi_level_kpis import get_dataset_date_bounds
            ds = get_dataset(period)
            min_d, max_d = get_dataset_date_bounds(ds)
            d_start = start_date.strip() if start_date else min_d
            d_end = end_date.strip() if end_date else max_d
            analyzer = ReactionaryDelayAnalyzer(ds)
            res = analyzer.compute_network_reactionary_multiplier(
                operator=operator.strip().upper() if operator else None,
                date_start=d_start,
                date_end=d_end,
            )
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to analyze reactionary delay: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: optimize_tail_swaps
    # -------------------------------------------------------------------------
    @server.tool()
    def optimize_tail_swaps(
        operator: str = "BAW",
        airport_icao: str = "EGLL",
        date: str = "",
        period: str = "202510",
    ) -> str:
        """Optimize hub aircraft tail swaps to break reactionary delay chains and preserve downstream passenger connections.

        Args:
            operator: Airline operator ICAO (e.g. 'BAW', 'DLH', 'KLM').
            airport_icao: Hub aerodrome ICAO (e.g. 'EGLL', 'EDDF', 'EHAM').
            date: Target date YYYY-MM-DD (optional).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.sim.tail_swap_optimizer import optimize_hub_tail_swaps
            ds = get_dataset(period)
            res = optimize_hub_tail_swaps(
                ds,
                operator=operator.strip().upper() if operator else None,
                hub_airport=airport_icao.strip().upper() if airport_icao else None,
                date_str=date or None,
            )
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to optimize tail swaps: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: estimate_radio_workload
    # -------------------------------------------------------------------------
    @server.tool()
    def estimate_radio_workload(
        airport_icao: str = "EGLL",
        date: str = "",
        period: str = "202510",
    ) -> str:
        """Estimate ATC radio frequency transmission workload, peak communications load, and channel occupancy.

        Args:
            airport_icao: Aerodrome ICAO (e.g. 'EGLL', 'EDDF', 'LFPG').
            date: Target date YYYY-MM-DD (optional).
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.features.radio_workload import estimate_airport_and_sector_radio_workload
            ds = get_dataset(period)
            res = estimate_airport_and_sector_radio_workload(
                ds, airport_icao=airport_icao.strip().upper(), date_str=date or None
            )
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to estimate radio workload: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: scan_airspace_conflicts
    # -------------------------------------------------------------------------
    @server.tool()
    def scan_airspace_conflicts(
        date: str = "",
        hour: int = 12,
        horiz_nm: float = 5.0,
        vert_fl: float = 10.0,
        period: str = "202510",
        limit: int = 20,
    ) -> str:
        """Scan European airspace for 4D radar proximity events (<5 NM, <1000 ft) and sector complexity index.

        Args:
            date: Target date YYYY-MM-DD (optional).
            hour: Target UTC hour (0-23, default: 12).
            horiz_nm: Horizontal proximity threshold in NM (default: 5.0).
            vert_fl: Vertical proximity threshold in FL (default: 10.0 = 1,000 ft).
            period: Eurocontrol archive month (e.g. '202510').
            limit: Maximum conflict events to return (default: 20).
        """
        try:
            from flightml.geo.conflict_scanner import scan_airspace_proximity_events
            ds = get_dataset(period)
            res = scan_airspace_proximity_events(
                ds,
                date_str=date or None,
                hour=hour,
                horiz_sep_nm=horiz_nm,
                vert_sep_fl=vert_fl,
                limit=limit,
            )
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to scan airspace conflicts: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: get_jet_fuel_prices
    # -------------------------------------------------------------------------
    @server.tool()
    def get_jet_fuel_prices(period: str = "202510") -> str:
        """Retrieve weekly European Jet-A spot fuel prices from IATA / Platts benchmarks in Euros (€/kg, €/tonne).

        Args:
            period: Eurocontrol archive month (e.g. '202510').
        """
        try:
            from flightml.eval.jet_fuel_prices import get_weekly_fuel_price
            price = get_weekly_fuel_price(date_val=None, period_hint=period)
            return _safe_json(price)
        except Exception as exc:
            return json.dumps({"error": f"Failed to retrieve jet fuel prices: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: locate_fir
    # -------------------------------------------------------------------------
    @server.tool()
    def locate_fir(lat: float, lon: float, flight_level: float = 350.0) -> str:
        """Determine which European FIR/UIR airspace contains a given (lat, lon, FL) coordinate.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            flight_level: Pressure altitude in Flight Level (e.g. 350.0 = 35,000 ft).
        """
        try:
            from flightml.geo.fir_engine import FIREngine
            engine = FIREngine()
            firs = engine.locate_fir(lat=lat, lon=lon, flight_level=flight_level)
            return json.dumps({
                "latitude": lat,
                "longitude": lon,
                "flight_level": flight_level,
                "enclosing_firs": firs or []
            }, indent=2)
        except Exception as exc:
            return json.dumps({"error": f"Failed to locate FIR: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: calculate_runway_winds
    # -------------------------------------------------------------------------
    @server.tool()
    def calculate_runway_winds(
        airport: str = "EGLL",
        wind_dir: float = 270.0,
        wind_speed: float = 18.0,
        gust: float = 0.0,
    ) -> str:
        """Calculate headwind, crosswind, and tailwind components for all available runways at an airport.

        Args:
            airport: Aerodrome ICAO code (e.g. 'EGLL', 'EDDF', 'LFPG').
            wind_dir: Wind direction in degrees true (0-360).
            wind_speed: Steady wind speed in knots.
            gust: Gust speed in knots (optional).
        """
        try:
            from flightml.weather.wind_calc import find_optimal_runways
            ds = get_dataset()
            apt = airport.strip().upper()
            runways_df = ds.query("SELECT * FROM raw_runways WHERE airport_ident = ?", [apt])
            if runways_df.empty:
                return json.dumps({"error": f"No runway data found for {apt} in catalog."}, indent=2)
            res = find_optimal_runways(
                runways_df.to_dict(orient="records"),
                wind_dir_deg=wind_dir,
                wind_speed_kts=wind_speed,
                gust_kts=gust if gust and gust > 0 else None,
            )
            return _safe_json(res)
        except Exception as exc:
            return json.dumps({"error": f"Failed to calculate runway winds: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # TOOL: predict_flight_delay
    # -------------------------------------------------------------------------
    @server.tool()
    def predict_flight_delay(
        adep: str,
        ades: str,
        operator: str = "BAW",
        market_segment: str = "Traditional Scheduled",
        requested_fl: float = 350.0,
        distance_nm: float = 500.0,
        hour: int = 14,
        day_of_week: int = 2,
    ) -> str:
        """Predict flight arrival delay using trained gradient-boosted ML delay models.

        Args:
            adep: Departure aerodrome ICAO (e.g. 'EGLL').
            ades: Destination aerodrome ICAO (e.g. 'EDDF').
            operator: Airline operator ICAO (e.g. 'BAW', 'DLH').
            market_segment: Market segment (e.g. 'Traditional Scheduled', 'Low Cost').
            requested_fl: Cruising flight level (e.g. 350.0).
            distance_nm: Flown distance in nautical miles.
            hour: Departure hour (0-23 UTC).
            day_of_week: Day of week (0=Mon, 6=Sun).
        """
        try:
            from flightml.models.delay_predictor import DelayPredictor, MODEL_DIR
            model_path = MODEL_DIR / "arr_delay_min_model.pkl"
            if model_path.exists():
                predictor = DelayPredictor.load(model_path)
            else:
                ds = get_dataset()
                predictor = DelayPredictor()
                df = predictor.prepare_data(ds, sample_size=40000)
                predictor.train(df)
                predictor.save(model_path)

            pred = predictor.predict({
                "adep": adep.strip().upper(),
                "ades": ades.strip().upper(),
                "ac_operator": operator.strip().upper(),
                "market_segment": market_segment,
                "requested_fl": requested_fl,
                "actual_distance_nm": distance_nm,
                "dep_hour": hour,
                "dep_dayofweek": day_of_week,
                "filed_air_time_min": (distance_nm / 450.0) * 60.0,
            })
            risk = "HIGH" if pred > 15 else "MODERATE" if pred > 5 else "LOW"
            return json.dumps({
                "adep": adep.strip().upper(),
                "ades": ades.strip().upper(),
                "operator": operator.strip().upper(),
                "predicted_arr_delay_min": round(float(pred), 1),
                "delay_risk_category": risk,
            }, indent=2)
        except Exception as exc:
            return json.dumps({"error": f"Failed to predict delay: {exc}"}, indent=2)

    # -------------------------------------------------------------------------
    # RESOURCE: Assumptions Glossary
    # -------------------------------------------------------------------------
    @server.resource("flightml://assumptions/marl_glossary")
    def get_assumptions_glossary() -> str:
        """Infilled operational parameters glossary for ATFM slot gaming."""
        return json.dumps(ASSUMPTIONS_GLOSSARY, indent=2)

    # -------------------------------------------------------------------------
    # RESOURCE: Airport Runway Capacity Profile
    # -------------------------------------------------------------------------
    @server.resource("flightml://airports/{icao}/capacity")
    def get_airport_resource(icao: str) -> str:
        """Nominal hourly runway movement capacity for an ICAO airport."""
        apt = icao.strip().upper()
        cap = NOMINAL_CAPACITIES.get(apt, 40)
        return json.dumps({
            "airport_icao": apt,
            "nominal_hourly_capacity": cap,
        }, indent=2)

    # -------------------------------------------------------------------------
    # PROMPT: Executive Airspace Disruption Briefing
    # -------------------------------------------------------------------------
    @server.prompt()
    def analyze_airport_delay_shock(airport: str = "EGLL", delay_min: int = 60) -> str:
        """Prompt template guiding an LLM on investigating and briefing an airport disruption."""
        return (
            f"You are an expert Air Traffic Flow Management (ATFM) specialist. "
            f"Please investigate an injected delay shock of {delay_min} minutes at {airport}. "
            f"Call the `simulate_delay_contagion` tool to assess how the disruption propagates "
            f"across European hub airports and downstream turnaround rotations. "
            f"Provide an executive operational summary highlighting: "
            f"1) Runway saturation at {airport}, 2) Top downstream hubs at risk, "
            f"3) Aircraft tail rotations with broken turnaround buffers, and 4) Recommended ATC flow measures."
        )

    # -------------------------------------------------------------------------
    # TOOL: get_fra_boundary_rankings
    # -------------------------------------------------------------------------
    @server.tool()
    def get_fra_boundary_rankings(
        period: str = "202510",
        min_flights: int = 1000,
        limit: int = 25,
        cross_border_only: bool = True,
    ) -> str:
        """Analyze Free Route Airspace (FRA) FIR boundary inefficiencies, detect existing FRA, and rank priority candidates.

        Args:
            period: Eurocontrol archive month (e.g. '202510' or '202410').
            min_flights: Minimum boundary crossing traffic volume threshold (default: 1000).
            limit: Number of top-ranked boundaries to return (default: 25).
            cross_border_only: Filter strictly to cross-border international boundaries (default: True).
        """
        ds = get_dataset(period)
        report = analyze_fir_boundary_inefficiencies(
            dataset=ds,
            min_flights=min_flights,
            limit=limit,
            cross_border_only=cross_border_only,
        )
        return _safe_json(report.to_dict())

    # -------------------------------------------------------------------------
    # TOOL: diagnose_fra_boundary
    # -------------------------------------------------------------------------
    @server.tool()
    def diagnose_fra_boundary(
        boundary_pair: str = "EGTTUIR <-> LFFFUIR",
        period: str = "202510",
    ) -> str:
        """Detailed operational diagnostic for a specific FIR boundary pair (e.g. 'EGTTUIR <-> LFFFUIR').

        Args:
            boundary_pair: Canonical FIR boundary pair (e.g. 'EGTTUIR <-> LFFFUIR' or 'LECBUIR <-> LFFFUIR').
            period: Eurocontrol archive month (e.g. '202510').
        """
        ds = get_dataset(period)
        detail = get_boundary_detail(dataset=ds, boundary_pair=boundary_pair)
        return _safe_json(detail)

    # -------------------------------------------------------------------------
    # TOOL: get_fra_worst_flights
    # -------------------------------------------------------------------------
    @server.tool()
    def get_fra_worst_flights(
        boundary_pair: str = "EGTTUIR <-> LFFFUIR",
        period: str = "202510",
        limit: int = 10,
        max_distance_nm: float | None = None,
        min_flight_level: float | None = 305.0,
    ) -> str:
        """Query the most detoured flights across a specific FIR boundary interface vs direct Free Route Airspace.
        Filters flights cruising at or above `min_flight_level` (default: FL305 per European CP1 mandate).

        Args:
            boundary_pair: Canonical FIR boundary pair (e.g. 'EGTTUIR <-> LFFFUIR' or 'EDUUUIR <-> LOVVFIR').
            period: Eurocontrol archive month (e.g. '202510' or '202410').
            limit: Maximum flights to return (default: 10).
            max_distance_nm: Optional filter to focus on intra-European flights (e.g. 2500.0).
            min_flight_level: Minimum cruising flight level (default: 305.0 per CP1 mandate).
        """
        ds = get_dataset(period)
        flights = get_worst_flights_for_boundary(
            dataset=ds,
            boundary_pair=boundary_pair,
            limit=limit,
            max_gc_dist_nm=max_distance_nm,
            include_trajectories=False,
            min_fl=min_flight_level,
        )
        return _safe_json(flights)

    # -------------------------------------------------------------------------
    # TOOL: forecast_fra_benefits
    # -------------------------------------------------------------------------
    @server.tool()
    def forecast_fra_benefits(
        period: str = "202510",
        year_baseline: str = "2025",
        baseline_mode: str = "full_year",
        use_seasonal_synthesis: bool = True,
        annual_growth_rate_pct: float = 2.0,
        realization_rate_pct: float = 85.0,
        scope: str = "prime_only",
        jet_a_price_eur_per_kg: float = 0.6421,
        fuel_growth_optimistic_pct: float = 1.5,
        fuel_growth_median_pct: float = 3.5,
        fuel_growth_shock_pct: float = 6.5,
    ) -> str:
        """Forecast cumulative multi-horizon benefits (1, 5, and 20 years) of Free Route Airspace (FRA) deployment.

        Features:
        - 12-Month Empirical Full-Year Baseline: Uses the contiguous 12-month Eurocontrol operational cycle (202411 to 202510).
        - Multi-Season Synthesis across Spring (02), Summer (06), and Autumn (10) to capture vacation surge and convective detours.
        - Dynamic 20-Year Fuel Price Escalation under Optimistic, Median Expected, and Geopolitical Shock scenarios.

        Args:
            period: Eurocontrol archive month fallback (e.g. '202510').
            year_baseline: Baseline archive year ('2025' or '2024') for seasonal synthesis.
            baseline_mode: Baseline mode ('full_year' for 12 contiguous empirical months, 'seasonal_synthesis', or 'single_period').
            use_seasonal_synthesis: Whether to blend Spring, Summer, Autumn into a weighted annual baseline (default: True).
            annual_growth_rate_pct: Projected compound annual air traffic growth rate (default: 2.0% STATFOR baseline).
            realization_rate_pct: Directness realization factor (default: 85.0%).
            scope: Boundary subset to model ('prime_only', 'prime_and_medium', 'all_cross_border', 'top_10', 'top_25').
            jet_a_price_eur_per_kg: Jet-A fuel benchmark price in €/kg (default: 0.6421 €/kg).
            fuel_growth_optimistic_pct: Annual fuel price growth in optimistic scenario (default: 1.5%/yr).
            fuel_growth_median_pct: Annual fuel price growth in median scenario (default: 3.5%/yr).
            fuel_growth_shock_pct: Annual fuel price growth in shock scenario (default: 6.5%/yr).
        """
        ds = get_dataset(period)
        report = analyze_fir_boundary_inefficiencies(ds, min_flights=500, cross_border_only=True)
        forecast = calculate_fra_long_term_forecast(
            boundary_metrics=report.boundaries,
            dataset=ds,
            year_baseline=year_baseline,
            baseline_mode=baseline_mode,
            use_seasonal_synthesis=use_seasonal_synthesis,
            annual_growth_rate=annual_growth_rate_pct / 100.0,
            realization_rate=realization_rate_pct / 100.0,
            horizons=[1, 5, 20],
            jet_a_eur_per_kg=jet_a_price_eur_per_kg,
            fuel_growth_optimistic=fuel_growth_optimistic_pct / 100.0,
            fuel_growth_median=fuel_growth_median_pct / 100.0,
            fuel_growth_shock=fuel_growth_shock_pct / 100.0,
            scope=scope,
        )
        return _safe_json(forecast)

    return server


# Default server instance
mcp = create_mcp_server()


def main() -> None:
    """CLI entrypoint for running the FlightML MCP server."""
    parser = argparse.ArgumentParser(description="FlightML Model Context Protocol (MCP) Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport mechanism (default: stdio for desktop LLMs like LM Studio, Claude Desktop, Cursor)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on when using SSE or streamable-http transport (default: 8000)",
    )
    args, unknown = parser.parse_known_args()

    print(f"[FlightML MCP] Starting FlightML Aviation Intelligence Server (transport={args.transport})...", file=sys.stderr)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
