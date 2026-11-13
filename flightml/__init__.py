"""
FlightML - Global Aviation Machine Learning, 4D Trajectory Analytics & Weather Fusion.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Core Data Access & Ingestion
from flightml.data.dataset import FlightDataset
from flightml.data.archive_ingest import EurocontrolArchiveIngestor, list_available_periods
from flightml.data.external_sources import compile_all_external_sources

# Geospatial & 4D Trajectory
from flightml.geo.trajectory import analyze_trajectory, TrajectoryMetrics
from flightml.geo.fir_engine import FIREngine
from flightml.geo.airac_diff import compare_airac_cycles, AIRACDiffReport
from flightml.geo.fra_analyzer import (
    analyze_flight_fra_adherence,
    compare_fra_evolution,
    FRAReport,
)
from flightml.geo.conflict_scanner import (
    scan_airspace_proximity_events,
    compute_sector_complexity_index,
    compute_hourly_grid_saturation,
    ProximityReport,
)
from flightml.geo.rnp_navaid_analyzer import (
    analyze_flight_navaids_and_rnp,
    query_unified_navaids,
    FlightNavaidRNPReport,
    ActiveRouteNavaid,
    TrajectoryRNPPoint,
    compute_rlos_nm,
)

# Weather & Environmental Modeling
from flightml.weather.metar_client import MetarClient
from flightml.weather.upper_air_client import UpperAirClient, UpperAirPointState
from flightml.weather.runway_safety import assess_flight_runway_safety, RunwaySafetyReport
from flightml.weather.delay_weather import analyze_delayed_flight_weather
from flightml.weather.network_weather_impact import analyze_network_weather_impact
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions, EnvironmentalImpact
from flightml.eval.jet_fuel_prices import get_weekly_fuel_price, get_all_weekly_fuel_prices_df, FuelPriceRecord
from flightml.eval.crco_economics import (
    compute_flight_crco_charges,
    detect_route_charge_avoidance,
    CRCOChargeReport,
)
from flightml.eval.contrail_climate import (
    analyze_flight_wind_and_contrails,
    UpperAirImpactReport,
)

# Efficiency & Performance Benchmarking
from flightml.eval.flight_twins import compare_twin_flights, TwinFlightComparison
from flightml.eval.cdo_analyzer import (
    analyze_cdo_descent,
    CDOReport,
    analyze_cco_climb,
    CCOReport,
    analyze_vertical_profile_efficiency,
)
from flightml.eval.pru_kpis import (
    compute_flight_pru_kpis,
    analyze_network_level_capping,
    PRUFlightScorecard,
)
from flightml.eval.recurring_series import analyze_recurring_flight_series, RecurringSeriesReport
from flightml.eval.macro_comparison import compare_aviation_periods, PeriodComparison
from flightml.eval.diagnostics import diagnose_flight_bottlenecks, DelayAttribution
from flightml.eval.surface_taxi import (
    analyze_airport_taxi_out_performance,
    analyze_flight_taxi_out,
    FlightTaxiOutReport,
)

# Airspace Bottlenecks & En-Route Sectors
from flightml.features.fir_saturation import (
    calculate_fir_hourly_saturation,
    calculate_flight_fir_dwell_inefficiency,
)
from flightml.features.aua_saturation import (
    calculate_aua_hourly_saturation,
    calculate_flight_aua_dwell_inefficiency,
)
from flightml.features.radio_workload import estimate_airport_and_sector_radio_workload
from flightml.features.sector_bottlenecks import analyze_fir_bottlenecks

# Simulation & Stress Testing
from flightml.models.airline_stress import stress_test_airline, StressTestResult
from flightml.models.gnn_delay import AirportGraphDelayModel, HeterogeneousAirspaceGNN
from flightml.sim.what_if import simulate_turnaround_mitigation, simulate_runway_weather_scenario
from flightml.sim.gdp_optimizer import optimize_ground_delay_program, GDPResult
from flightml.sim.tail_swap_optimizer import (
    optimize_hub_tail_swaps,
    TailSwapOptimizationResult,
)

# Visualization
from flightml.viz.map_builder import (
    create_trajectory_map,
    create_grid_saturation_map,
    create_conflict_map,
    create_world_navaids_map,
)

__all__ = [
    "__version__",
    # Data
    "FlightDataset",
    "EurocontrolArchiveIngestor",
    "list_available_periods",
    "compile_all_external_sources",
    # Geo
    "analyze_trajectory",
    "TrajectoryMetrics",
    "FIREngine",
    "compare_airac_cycles",
    "AIRACDiffReport",
    "analyze_flight_fra_adherence",
    "compare_fra_evolution",
    "FRAReport",
    "scan_airspace_proximity_events",
    "compute_sector_complexity_index",
    "compute_hourly_grid_saturation",
    "ProximityReport",
    "analyze_flight_navaids_and_rnp",
    "query_unified_navaids",
    "FlightNavaidRNPReport",
    "ActiveRouteNavaid",
    "TrajectoryRNPPoint",
    "compute_rlos_nm",
    # Weather & Fuel
    "MetarClient",
    "UpperAirClient",
    "UpperAirPointState",
    "assess_flight_runway_safety",
    "RunwaySafetyReport",
    "analyze_delayed_flight_weather",
    "analyze_network_weather_impact",
    "compute_flight_fuel_and_emissions",
    "EnvironmentalImpact",
    "get_weekly_fuel_price",
    "get_all_weekly_fuel_prices_df",
    "FuelPriceRecord",
    "compute_flight_crco_charges",
    "detect_route_charge_avoidance",
    "CRCOChargeReport",
    "analyze_flight_wind_and_contrails",
    "UpperAirImpactReport",
    # Evaluation
    "compare_twin_flights",
    "TwinFlightComparison",
    "analyze_cdo_descent",
    "CDOReport",
    "analyze_cco_climb",
    "CCOReport",
    "analyze_vertical_profile_efficiency",
    "compute_flight_pru_kpis",
    "analyze_network_level_capping",
    "PRUFlightScorecard",
    "analyze_recurring_flight_series",
    "RecurringSeriesReport",
    "compare_aviation_periods",
    "PeriodComparison",
    "diagnose_flight_bottlenecks",
    "DelayAttribution",
    "analyze_airport_taxi_out_performance",
    "analyze_flight_taxi_out",
    "FlightTaxiOutReport",
    # Features
    "calculate_fir_hourly_saturation",
    "calculate_flight_fir_dwell_inefficiency",
    "calculate_aua_hourly_saturation",
    "calculate_flight_aua_dwell_inefficiency",
    "estimate_airport_and_sector_radio_workload",
    "analyze_fir_bottlenecks",
    # Models & Simulation
    "stress_test_airline",
    "StressTestResult",
    "AirportGraphDelayModel",
    "HeterogeneousAirspaceGNN",
    "simulate_turnaround_mitigation",
    "simulate_runway_weather_scenario",
    "optimize_ground_delay_program",
    "GDPResult",
    "optimize_hub_tail_swaps",
    "TailSwapOptimizationResult",
    # Viz
    "create_trajectory_map",
    "create_grid_saturation_map",
    "create_conflict_map",
    "create_world_navaids_map",
]
