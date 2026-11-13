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

# Weather & Environmental Modeling
from flightml.weather.metar_client import MetarClient
from flightml.weather.delay_weather import analyze_delayed_flight_weather
from flightml.weather.network_weather_impact import analyze_network_weather_impact
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions, EnvironmentalImpact
from flightml.eval.jet_fuel_prices import get_weekly_fuel_price, get_all_weekly_fuel_prices_df, FuelPriceRecord

# Efficiency & Performance Benchmarking
from flightml.eval.flight_twins import compare_twin_flights, TwinFlightComparison
from flightml.eval.cdo_analyzer import analyze_cdo_descent, CDOReport
from flightml.eval.recurring_series import analyze_recurring_flight_series, RecurringSeriesReport
from flightml.eval.macro_comparison import compare_aviation_periods, PeriodComparison
from flightml.eval.diagnostics import diagnose_flight_bottlenecks, DelayAttribution

# Airspace Bottlenecks & En-Route Sectors
from flightml.features.fir_saturation import (
    calculate_fir_hourly_saturation,
    calculate_flight_fir_dwell_inefficiency,
)
from flightml.features.sector_bottlenecks import analyze_fir_bottlenecks

# Simulation & Stress Testing
from flightml.models.airline_stress import stress_test_airline, StressTestResult
from flightml.sim.what_if import simulate_turnaround_mitigation, simulate_runway_weather_scenario
from flightml.sim.gdp_optimizer import optimize_ground_delay_program, GDPResult

# Visualization
from flightml.viz.map_builder import create_trajectory_map

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
    # Weather & Fuel
    "MetarClient",
    "analyze_delayed_flight_weather",
    "analyze_network_weather_impact",
    "compute_flight_fuel_and_emissions",
    "EnvironmentalImpact",
    "get_weekly_fuel_price",
    "get_all_weekly_fuel_prices_df",
    "FuelPriceRecord",
    # Evaluation
    "compare_twin_flights",
    "TwinFlightComparison",
    "analyze_cdo_descent",
    "CDOReport",
    "analyze_recurring_flight_series",
    "RecurringSeriesReport",
    "compare_aviation_periods",
    "PeriodComparison",
    "diagnose_flight_bottlenecks",
    "DelayAttribution",
    # Features
    "calculate_fir_hourly_saturation",
    "calculate_flight_fir_dwell_inefficiency",
    "analyze_fir_bottlenecks",
    # Models & Simulation
    "stress_test_airline",
    "StressTestResult",
    "simulate_turnaround_mitigation",
    "simulate_runway_weather_scenario",
    "optimize_ground_delay_program",
    "GDPResult",
    # Viz
    "create_trajectory_map",
]
