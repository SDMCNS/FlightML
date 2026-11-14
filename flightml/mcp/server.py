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
