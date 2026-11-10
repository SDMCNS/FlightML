# FlightML Documentation ✈️

Welcome to the **FlightML** documentation. FlightML is a high-performance Python machine learning library, interactive simulation platform, REST API, and command-line interface (CLI) for **Global Airspace Network Intelligence, 4D Trajectory Analytics, and Weather Fusion**.

FlightML enables researchers, airlines, air navigation service providers (ANSPs), and data scientists to ingest massive aeronautical datasets, uncover en-route airspace chokepoints, audit flight-level fuel and carbon inefficiencies, benchmark identical twin routes, and stress-test airline schedule resilience.

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

## 📚 Documentation Index

- [**Quickstart Guide**](file:///C:/Projects/data/FlightML/docs/quickstart.md): Step-by-step installation, dataset configuration, dashboard launching, and first steps.
- [**Python Library Guide**](file:///C:/Projects/data/FlightML/docs/library_guide.md): Comprehensive developer guide with copy-pasteable Python code snippets for all core modules.
- [**CLI Reference Manual**](file:///C:/Projects/data/FlightML/docs/cli_reference.md): Detailed reference guide for all 32 `flightml` terminal commands and options.
- [**REST API Reference**](file:///C:/Projects/data/FlightML/docs/rest_api.md): Complete catalog of FastAPI endpoints, request parameters, response schemas, and Swagger docs.
- [**Architecture & Methodology**](file:///C:/Projects/data/FlightML/docs/architecture.md): Deep dive into DuckDB columnar query execution, thread-safe cursor handling, BADA fuel burn physics, and weather fusion.

---

## ⚡ What Does FlightML Do?

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FLIGHTML ECOSYSTEM                                     │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│    DATA INGESTION &      │   4D TRAJECTORY &        │   NETWORK INTELLIGENCE &         │
│    COLUMNAR ENGINE       │   EFFICIENCY EVALUATION  │   SIMULATION ENGINE              │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ • DuckDB Columnar Store  │ • 4D Spatial Trajectory  │ • En-Route FIR Saturation Rates  │
│ • Zero-Copy Arrow Memory │ • Airborne Holding Loops │ • Sector Dwell Inefficiency      │
│ • Eurocontrol Ingestion  │ • CDO Staircase Analysis │ • Airline Shock Stress Testing   │
│ • Multi-Period Switching │ • Twin-Flight Detours    │ • Ground Delay Program (GDP) RBS │
│ • Historical EIA Jet-A   │ • Weekly Spot Jet Fuel   │ • Turnaround Delay Propagation   │
│ • IEM ASOS Weather       │ • Excess CO2 & Burn Math │ • AIRAC Cycle Diff Engine        │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

1. **Massive-Scale Aeronautical Data Ingestion**:
   - Ingests and normalizes multi-gigabyte Eurocontrol archives (e.g. November 2016 baseline of 697,000+ flights; October 2025 modern archive of 934,000+ flights).
   - Partitions flight records, airport layouts, runway magnetic headings, FIR airspace geometries, and 4D trajectory radar points into compressed Apache Parquet tables queried via an embedded DuckDB engine.

2. **4D Radar Trajectory Analytics & Holding Detection**:
   - Evaluates true 4D (Latitude, Longitude, Altitude/FL, Timestamp) radar tracks against planned filed flight plans.
   - Computes path extension ratios ($\text{Actual Distance} / \text{Great Circle Distance}$) and automatically identifies airborne holding stacks (repetitive 360° loops, racetrack patterns, extended vectoring).

3. **Continuous Descent Operations (CDO) Analyzer**:
   - Assesses terminal arrival phase descents from Top of Descent (TOD) to touchdown.
   - Detects inefficient stepped level-offs and altitude reversals, calculating CDO compliance scores and excess fuel burned during non-optimal descent profiles.

4. **Dynamic Weekly Jet Fuel Pricing & Carbon Accounting**:
   - Integrates official historical weekly jet fuel spot prices from the **U.S. Energy Information Administration (EIA Series `EER_EPJK_PF4_RGC_DPG`)** and **IATA Jet Fuel Price Monitor / Rotterdam Platts ARA benchmark**.
   - Resolves spot fuel prices dynamically according to flight departure dates, converting fuel burn kilograms into exact financial costs and carbon dioxide emissions ($3.16\text{ kg CO}_2\text{ per kg Jet-A}$).

5. **Twin-Flight Efficiency Benchmarking**:
   - Pairs identical flights (same origin ADEP, destination ADES, and aircraft type within the same period) to compare the most efficient trajectory against the least efficient trajectory.
   - Quantifies detour nautical miles, holding duration, air time penalties, fuel burn variance, and monetary cost differentials.

6. **Longitudinal Recurring Flight Series & Diurnal Congestion**:
   - Analyzes recurring flights across days and weeks on a specific route.
   - Evaluates time-of-day (diurnal) congestion patterns across Early Morning, Morning Peak, Midday, Afternoon/Evening Peak, and Night departure windows.

7. **Airspace En-Route FIR Saturation & Chokepoints**:
   - Calculates hourly traffic saturation across European Flight Information Regions (FIRs).
   - Analyzes individual flight dwell times within each sector against optimal great-circle transit times to isolate ATC vectoring and holding bottlenecks.

8. **Aviation Simulation & Stress Testing**:
   - **Scenario A (Turnaround Buffer Scheduling)**: Simulates the addition of 5 to 25 minutes of turnaround slack to evaluate reactionary delay reduction across an airline fleet.
   - **Scenario B (Runway Capacity Collapse / Severe Weather)**: Simulates ground delay programs (GDP) using Ration-by-Schedule (RBS) slot allocation to prevent airborne holding fuel waste.
   - **Scenario C (Fleet Operational Shock Stress Testing)**: Injects cascading delay shocks (+15m to +90m) into airline hub schedules to identify fleet fragility curves and infection thresholds.

---

## 🚀 Quick Code Example

```python
import flightml as fml

# 1. Connect to the ingested dataset (period defaults to modern '202510')
ds = fml.FlightDataset(period="202510")

# 2. Inspect high-level metrics
print(f"Loaded {ds.total_flights:,} flights from Eurocontrol archive.")

# 3. Search flights with multi-criteria filtering
flights = ds.search_flights(
    adep="EGLL",
    ades="KORD",
    operator="BAW",
    limit=5
)
print(flights[["ectrl_id", "adep", "ades", "flight_label", "actual_distance_nm", "flt_duration_min"]])

# 4. Compute fuel burn, carbon emissions, and weekly spot pricing
fuel_impact = fml.compute_flight_fuel_and_emissions(ds, ectrl_id=29126XXXX)
print(f"Total Fuel: {fuel_impact.total_fuel_kg:,.1f} kg | CO2: {fuel_impact.co2_emissions_kg:,.1f} kg")
print(f"Fuel Price: ${fuel_impact.fuel_price_usd_per_gallon:.3f}/gal | Cost: ${fuel_impact.total_fuel_cost_usd:,.2f}")

# 5. Analyze Continuous Descent Operations (CDO)
cdo = fml.analyze_cdo_descent(ds, ectrl_id=29126XXXX)
print(f"CDO Compliance Score: {cdo.cdo_compliance_score:.1f}% (Level-offs: {cdo.level_off_count})")
```

---

## 🌐 Three Operating Modalities

| Modality | Target Use Case | Primary Command |
| :--- | :--- | :--- |
| **Python Library** | Machine learning pipelines, custom analytics, Jupyter notebooks | `import flightml` |
| **Interactive Dashboard** | Visual airspace exploration, what-if simulations, interactive maps | `flightml dashboard` |
| **Command-Line Interface** | Automation scripts, CI/CD, fast terminal queries | `flightml <command>` |
| **REST API Server** | Microservice integration, enterprise web backends | `flightml serve` |

---

## 📖 Next Steps

- Jump to [**Quickstart Guide**](file:///C:/Projects/data/FlightML/docs/quickstart.md) to set up and run FlightML in under 5 minutes.
- Explore the [**Python Library Guide**](file:///C:/Projects/data/FlightML/docs/library_guide.md) to integrate FlightML into your data science stack.

