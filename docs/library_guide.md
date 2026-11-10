# Python Library Guide 🐍

FlightML is designed as a first-class Python library that can be embedded into Jupyter notebooks, batch machine learning pipelines, Airflow DAGs, and custom microservices.

```python
import flightml as fml
```

All primary classes, evaluation routines, simulation engines, and data models are exported at the root namespace.

---

> [!IMPORTANT]
> ### 🔒 Data Access & Eurocontrol OneSky Online Compliance Notice
> **Eurocontrol Research Data Access Prerequisite**:
> The underlying aeronautical traffic, flight plans, and 4D radar trajectory archives supported by FlightML (such as Eurocontrol DDR2 / PRISME / R-NEST datasets) are **proprietary and not for public consumption or redistribution**.
>
> To access or download raw historical datasets, researchers and organizations **must have an approved user account on Eurocontrol's OneSky Online portal** ([https://www.eurocontrol.int/](https://www.eurocontrol.int/)) with institutional access granted to the **Demand Data Repository (DDR)** under Eurocontrol's Data Dissemination Policy and standard Research Agreement.
>
> **Data Privacy & Redaction Policy**:
> In accordance with Eurocontrol data dissemination terms, operational commercial confidentiality, and GDPR requirements, **specific flight identifiers, physical aircraft tail registrations, and operational callsigns in this documentation and example snippets have been anonymized / redacted (e.g., `29126XXXX`, `N686XX`, `BAWXXXX`)** to prevent disclosing sensitive airline operational data.

---

## 📑 Table of Contents

1. [The Columnar Dataset (`FlightDataset`)](#1-the-columnar-dataset-flightdataset)
2. [4D Trajectory & Holding Loop Analytics](#2-4d-trajectory--holding-loop-analytics)
3. [Continuous Descent Operations (CDO)](#3-continuous-descent-operations-cdo)
4. [Weekly Jet Fuel Spot Pricing & Emissions](#4-weekly-jet-fuel-spot-pricing--emissions)
5. [Twin-Flight Efficiency Benchmarking](#5-twin-flight-efficiency-benchmarking)
6. [Longitudinal Recurring Flight Series & Diurnal Analysis](#6-longitudinal-recurring-flight-series--diurnal-analysis)
7. [Airspace En-Route FIR Saturation & Dwell Inefficiency](#7-airspace-en-route-fir-saturation--dwell-inefficiency)
8. [Airline Turnaround Shock Stress Testing](#8-airline-turnaround-shock-stress-testing)
9. [Ground Delay Program (GDP) Slot Optimization](#9-ground-delay-program-gdp-slot-optimization)
10. [Macro Period Comparison & AIRAC Diffing](#10-macro-period-comparison--airac-diffing)

---

## 1. The Columnar Dataset (`FlightDataset`)

The [`FlightDataset`](file:///C:/Projects/data/FlightML/flightml/data/dataset.py) class manages connections to DuckDB and Apache Parquet tables for a specific aeronautical period (e.g. `"202510"`, `"201611"`).

### Initializing and Inspecting

```python
from flightml import FlightDataset

# 1. List available ingested periods
periods = FlightDataset.list_available_periods()
print("Available periods:", periods)  # ['201611', '202510']

# 2. Connect to the October 2025 modern archive
ds = FlightDataset(period="202510")

# 3. High-level summary metrics
summary = ds.get_summary()
print(f"Total flights: {summary['total_flights']:,}")
print(f"Total airports: {summary['total_airports']:,}")
print(f"Date range: {summary['date_min']} to {summary['date_max']}")
```

### Multi-Criteria Flight Search

Use `search_flights()` to query flights using business filters:

```python
flights_df = ds.search_flights(
    adep="EGLL",             # Origin ICAO
    ades="KORD",             # Destination ICAO
    operator="BAW",          # Airline 3-letter ICAO code
    start_date="2025-10-01", # Optional YYYY-MM-DD
    end_date="2025-10-07",   # Optional YYYY-MM-DD
    min_distance_nm=3000,    # Minimum Great Circle / Actual distance
    max_fl=390,              # Maximum cruise Flight Level (e.g. FL390 = 39,000 ft)
    limit=10
)

# Returns a pandas DataFrame with human-readable labels:
# ectrl_id, flight_label, adep, ades, operator, callsign, ac_type, actual_distance_nm, flt_duration_min
for _, row in flights_df.iterrows():
    print(f"[{row['ectrl_id']}] {row['flight_label']} - Duration: {row['flt_duration_min']} min")
```

### Thread-Safe Columnar Queries

`FlightDataset` provides a thread-safe `cursor` property that spawns dedicated DuckDB connection cursors, preventing concurrency conflicts when queried across multi-threaded applications or Streamlit reruns:

```python
# Execute arbitrary SQL queries safely:
cursor = ds.cursor
df = cursor.execute("""
    SELECT operator, COUNT(*) AS flight_count, AVG(arr_delay) AS avg_delay
    FROM flights
    WHERE arr_delay IS NOT NULL
    GROUP BY operator
    HAVING COUNT(*) > 500
    ORDER BY avg_delay DESC
    LIMIT 5
""").df()

print(df)
```

---

## 2. 4D Trajectory & Holding Loop Analytics

FlightML analyzes 4D radar points (Time, Latitude, Longitude, Altitude/FL) to identify terminal holding loops, racetrack patterns, and trajectory extension inefficiency.

```python
from flightml import analyze_trajectory, create_trajectory_map

# Retrieve radar points for a specific flight (example ID redacted with XXXX for privacy)
traj_points = ds.get_trajectory(ectrl_id=29126XXXX)

# Analyze trajectory metrics
metrics = analyze_trajectory(traj_points)

print(f"Actual Distance: {metrics.actual_distance_nm:.1f} nm")
print(f"Great Circle Distance: {metrics.great_circle_distance_nm:.1f} nm")
print(f"Inefficiency Ratio: {metrics.inefficiency_ratio:.3f}x")
print(f"Holding Loops: {metrics.holding_loops}")
print(f"Holding Duration: {metrics.holding_duration_minutes:.1f} minutes")
print(f"Average Groundspeed: {metrics.avg_groundspeed_knots:.1f} kts")

# Generate an interactive Leaflet HTML map with unblocked open tile layers
map_obj = create_trajectory_map(
    trajectory_points=traj_points,
    holding_segments=metrics.holding_segments,
    flight_info=ds.get_flight(29126XXXX)
)
map_obj.save("flight_trajectory_map.html")
```

---

## 3. Continuous Descent Operations (CDO)

The Continuous Descent Operations ([`CDOAnalyzer`](file:///C:/Projects/data/FlightML/flightml/eval/cdo_analyzer.py)) assesses the descent phase from Top of Descent (TOD) to landing. Stepped staircases and intermediate level-offs burn significant excess fuel compared to smooth idle-thrust descents.

```python
from flightml import analyze_cdo_descent

cdo = analyze_cdo_descent(ds, ectrl_id=29126XXXX)

print(f"CDO Compliance Score: {cdo.cdo_compliance_score:.1f}%")
print(f"Top of Descent: FL{cdo.tod_fl} ({cdo.tod_time})")
print(f"Intermediate Level-offs: {cdo.level_off_count} occurrences")
print(f"Total Level-off Time: {cdo.total_level_off_minutes:.1f} minutes")
print(f"Excess Descent Fuel: {cdo.excess_fuel_kg:.1f} kg")

# Inspect level-off altitude segments
for seg in cdo.level_off_segments:
    print(f"  - Level-off at FL{seg['fl']} for {seg['duration_min']:.1f} min")
```

---

## 4. Weekly Jet Fuel Spot Pricing & Emissions

FlightML connects fuel burn physics (BADA aerodynamic mass and drag approximations) directly with historical weekly spot prices from official **U.S. Energy Information Administration (EIA Series `EER_EPJK_PF4_RGC_DPG`)** and **IATA Jet Fuel Price Monitor** benchmarks.

```python
from flightml import (
    compute_flight_fuel_and_emissions,
    get_weekly_fuel_price,
    get_all_weekly_fuel_prices_df
)

# 1. Inspect historical weekly spot prices table
prices_df = get_all_weekly_fuel_prices_df()
print(prices_df.tail())

# 2. Get price for a specific flight departure date
fuel_price = get_weekly_fuel_price("2025-10-01")
print(f"Weekly Spot Price: ${fuel_price.price_usd_per_gallon:.3f}/gal (${fuel_price.price_usd_per_kg:.3f}/kg)")
print(f"Benchmark Week: {fuel_price.week_label} (Source: {fuel_price.source})")

# 3. Calculate full fuel and environmental impact for a flight
impact = compute_flight_fuel_and_emissions(ds, ectrl_id=29126XXXX)

print(f"Total Fuel Burned: {impact.total_fuel_kg:,.1f} kg")
print(f"CO2 Emissions: {impact.co2_emissions_kg:,.1f} kg")
print(f"Detour Waste: {impact.detour_excess_fuel_kg:,.1f} kg")
print(f"Airborne Holding Waste: {impact.holding_excess_fuel_kg:,.1f} kg")
print(f"Weekly Spot Fuel Price: ${impact.fuel_price_usd_per_gallon:.3f}/gal")
print(f"Total Fuel Cost: ${impact.total_fuel_cost_usd:,.2f}")
print(f"Holding Financial Penalty: ${impact.holding_fuel_cost_usd:,.2f}")
```

---

## 5. Twin-Flight Efficiency Benchmarking

[`compare_twin_flights`](file:///C:/Projects/data/FlightML/flightml/eval/flight_twins.py) pairs flights sharing the identical city pair and aircraft type within the same historical period to identify the root causes of trajectory variance.

```python
from flightml import compare_twin_flights

# Compare flights between Seoul Incheon (RKSI) and London Heathrow (EGLL) on Boeing 777-300ER (B77W)
twins = compare_twin_flights(ds, adep="RKSI", ades="EGLL", ac_type="B77W")

print(f"Efficient Flight: {twins.efficient_flight['callsign']} ({twins.efficient_flight['flight_date']})")
print(f"  - Distance: {twins.efficient_distance_nm:.0f} nm | Air Time: {twins.efficient_air_time_min:.0f} min")
print(f"  - Fuel Burn: {twins.efficient_fuel_kg:,.0f} kg")

print(f"Inefficient Flight: {twins.inefficient_flight['callsign']} ({twins.inefficient_flight['flight_date']})")
print(f"  - Distance: {twins.inefficient_distance_nm:.0f} nm | Air Time: {twins.inefficient_air_time_min:.0f} min")
print(f"  - Fuel Burn: {twins.inefficient_fuel_kg:,.0f} kg")

print(f"Route Detour Divergence: +{twins.detour_delta_nm:.0f} nm")
print(f"Excess Fuel Burn: +{twins.excess_fuel_kg:,.0f} kg (+{twins.excess_fuel_percent:.1f}%)")
print(f"Excess Cost Penalty: +${twins.excess_fuel_cost_usd:,.2f}")
```

---

## 6. Longitudinal Recurring Flight Series & Diurnal Analysis

[`analyze_recurring_flight_series`](file:///C:/Projects/data/FlightML/flightml/eval/recurring_series.py) evaluates route performance over time and breaks down flights by departure time-of-day.

```python
from flightml import analyze_recurring_flight_series

report = analyze_recurring_flight_series(
    ds,
    adep="EGLL",
    ades="KORD",
    operator="BAW",
    ac_type="B772"
)

print(f"Total Flights Evaluated: {report.total_flights}")
print(f"Average Duration: {report.avg_duration_min:.1f} min (StdDev: {report.std_duration_min:.1f} min)")
print(f"Average Delay: {report.avg_arrival_delay_min:.1f} min")

# Time-of-day (diurnal) distribution
print("\nDiurnal Departure Breakdown:")
for period, count in report.diurnal_breakdown.items():
    print(f"  - {period:20s}: {count:3d} flights")

# Inspect daily series DataFrame
print(report.series_df[["flight_date", "callsign", "flt_duration_min", "arr_delay", "diurnal_period"]].head())
```

---

## 7. Airspace En-Route FIR Saturation & Dwell Inefficiency

Analyze network bottlenecks and flight sector efficiency across European Flight Information Regions (FIRs).

```python
from flightml import (
    calculate_fir_hourly_saturation,
    calculate_flight_fir_dwell_inefficiency,
    analyze_fir_bottlenecks
)

# 1. Identify peak saturated FIR sectors
saturated_firs = calculate_fir_hourly_saturation(ds, threshold_flights_per_hour=40, limit=5)
print(saturated_firs[["fir_id", "fir_name", "hour_slot", "hourly_traffic", "saturation_ratio"]])

# 2. Sector-by-sector dwell analysis for a specific flight
dwell_report = calculate_flight_fir_dwell_inefficiency(ds, ectrl_id=29126XXXX)
print(f"Flight traversed {dwell_report.total_firs_traversed} FIR sectors.")
for sector in dwell_report.sectors:
    print(f"  FIR {sector['fir_id']} ({sector['fir_name']}):")
    print(f"    Actual: {sector['actual_dwell_min']:.1f}m | GCD: {sector['gcd_dwell_min']:.1f}m | Inefficiency: {sector['dwell_inefficiency_ratio']:.2f}x")
```

---

## 8. Airline Turnaround Shock Stress Testing

[`stress_test_airline`](file:///C:/Projects/data/FlightML/flightml/models/airline_stress.py) simulates cascading delay propagation across physical aircraft tail numbers (`ac_reg`) when subject to initial operational shocks.

```python
from flightml import stress_test_airline

result = stress_test_airline(
    ds,
    operator="RYR",              # Ryanair
    start_date="2025-10-01",
    end_date="2025-10-04",
    shock_minutes=[15, 30, 45, 60, 90]
)

print(f"Tails Analyzed: {result.total_tails} | Total Flights: {result.total_flights}")
print("Shock Contagion Curve:")
for shock_min, metrics in result.shock_results.items():
    print(f"  +{shock_min:2d} min shock -> Fleet Infection Rate: {metrics['infection_rate_pct']:.1f}% | Avg Propagated Delay: {metrics['avg_propagated_delay_min']:.1f} min")
```

---

## 9. Ground Delay Program (GDP) Slot Optimization

Simulate Ration-by-Schedule (RBS) ground delay program optimization to replace airborne holding loops with ground gate holds during severe aerodrome capacity collapses.

```python
from flightml import optimize_ground_delay_program

gdp = optimize_ground_delay_program(
    ds,
    airport="EGLL",
    start_time="2025-10-01T06:00:00",
    end_time="2025-10-01T12:00:00",
    target_capacity_per_hour=25  # Severe fog capacity limit
)

print(f"Total Impacted Flights: {gdp.flights_delayed}")
print(f"Total Ground Delay Assigned: {gdp.total_delay_minutes:,.0f} minutes")
print(f"Airborne Holding Prevented: {gdp.airborne_holding_prevented_minutes:,.0f} minutes")
print(f"Jet-A Fuel Saved: {gdp.fuel_saved_kg:,.1f} kg")
print(f"Financial Cost Savings: ${gdp.fuel_cost_saved_usd:,.2f}")
print(f"CO2 Emissions Prevented: {gdp.co2_saved_kg:,.1f} kg")
```

---

## 10. Macro Period Comparison & AIRAC Diffing

Compare multi-year macro evolution across Eurocontrol archives and inspect aeronautical navigation changes across AIRAC cycles.

```python
from flightml import compare_aviation_periods, compare_airac_cycles

# Macro comparison between 2016 baseline and 2025 modern archive
macro = compare_aviation_periods("201611", "202510")
print(f"Total Traffic Growth: {macro.traffic_growth_percent:+.1f}%")
print(f"Average Delay Delta: {macro.avg_delay_delta_minutes:+.2f} minutes")
print(f"Ryanair (RYR) Growth: {macro.top_carrier_growth.get('RYR', 0):+.1f}%")

# Compare contiguous AIRAC navigation cycles
diff = compare_airac_cycles("2509", "2510")
print(f"Added Airway Segments: {len(diff.added_routes)}")
print(f"Modified FIR Boundary Sectors: {len(diff.modified_firs)}")
```

