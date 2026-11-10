# FlightML ✈️

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DuckDB Powered](https://img.shields.io/badge/Storage-DuckDB%20%2B%20Apache%20Parquet-orange.svg)](https://duckdb.org/)
[![Documentation](https://img.shields.io/badge/Docs-Complete%20Guide-green.svg)](docs/index.md)

**A high-performance Python machine learning library, interactive simulation platform, REST API, and CLI for Global Airspace Network Intelligence, 4D Trajectory Analytics, and ASOS/METAR Weather Fusion.**

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

## 📖 Complete Documentation

Detailed guides and manuals are available in the [**`docs/`**](docs/index.md) directory:

- 🚀 [**Quickstart Guide**](docs/quickstart.md): Installation, OneSky data setup, dashboard launching, and first steps.
- 🐍 [**Python Library Guide**](docs/library_guide.md): Developer guide with complete Python code recipes for all modules.
- 💻 [**CLI Reference Manual**](docs/cli_reference.md): Exhaustive guide for all 32 `flightml` terminal commands and sample outputs.
- 🌐 [**REST API Reference**](docs/rest_api.md): Full FastAPI endpoints catalog, request/response schemas, and Swagger docs.
- 🏛️ [**Architecture & Methodology**](docs/architecture.md): Columnar DuckDB execution, BADA fuel burn physics, and weather fusion.

---

## ⚡ What is FlightML?

FlightML ingests multi-gigabyte aeronautical datasets across historical and modern Eurocontrol archives (November 2016 baseline with 697,000+ flights; October 2025 modern archive with 934,000+ flights) and couples them with historical IEM ASOS/METAR weather observations and official weekly Jet-A spot fuel prices.

It provides a unified platform to:
- **Analyze 4D Radar Trajectories**: Detect airborne holding stacks, compute path extension ratios, and export interactive maps.
- **Audit Fuel & Carbon Inefficiency**: Compute aerodynamic fuel burn and financial costs based on official weekly spot prices from the **U.S. EIA Series `EER_EPJK_PF4_RGC_DPG`** and **IATA Platts ARA** benchmarks.
- **Evaluate Continuous Descent Operations (CDO)**: Detect stepped staircase level-offs during arrival and quantify excess descent fuel waste.
- **Benchmark Twin Flights**: Pair identical city pairs and aircraft types (e.g. `RKSI` $\to$ `EGLL` on `B77W`) to quantify detour divergence and fuel waste.
- **Analyze Longitudinal Recurring Series**: Track daily performance and evaluate departure time-of-day (diurnal) congestion patterns.
- **Map FIR Airspace Saturation**: Identify European en-route sector chokepoints and measure flight-level dwell inefficiency.
- **Stress-Test Airline Resilience**: Injects operational delay shocks (+15m to +90m) into airline turnaround schedules to generate fleet fragility curves.
- **Optimize Ground Delay Programs (GDP)**: Run Ration-by-Schedule (RBS) slot allocation during severe weather/fog capacity collapses.

---

## 📦 Installation

```bash
git clone https://github.com/flightml/flightml.git
cd FlightML
pip install -e .
```

### Optional Extras

```bash
# Graph Neural Network support (PyTorch)
pip install -e ".[gnn]"

# Testing & development dependencies
pip install -e ".[dev]"
```

---

## 🖥️ CLI Snippets & Terminal Outputs

FlightML formats terminal output using Rich tables and color-coded status badges. Below are representative snippets with operational fields redacted (`XXXX`).

### 1. Global Dataset Summary (`flightml summary`)

```bash
flightml summary --period 202510
```

```text
                       ✈️  FlightML Dataset Overview                        
┌─────────────────────────────┬────────────────────────────────────────────┐
│ Metric                      │ Value                                      │
├─────────────────────────────┼────────────────────────────────────────────┤
│ Total Processed Flights     │ 934,141                                    │
│ Unique Aircraft Operators   │ 814                                        │
│ Origin Airports (ADEP)      │ 1,622                                      │
│ Destination Airports (ADES) │ 1,631                                      │
│ Date Coverage               │ 2025-09-29 22:15:00 to 2025-11-01 02:40:00 │
│ Mean Departure Delay        │ 4.36 min                                   │
│ Mean Arrival Delay          │ 4.16 min                                   │
│ Airports Catalog            │ 86,336                                     │
│ Runways Catalog             │ 48,405                                     │
│ FIR Airspaces Catalog       │ 170                                        │
└─────────────────────────────┴────────────────────────────────────────────┘
```

---

### 2. Multi-Criteria Flight Search (`flightml search-flights`)

```bash
flightml search-flights --period 202510 --adep EGLL --ades KORD --limit 3
```

```text
               Flight Search Results (Period: 202510, 3 matches)               
┌──────────────────────────────────┬──────────┬───────┬───────────┬───────────┐
│ Flight Label & Metadata          │ Distance │    FL │ Dep Delay │ Arr Delay │
├──────────────────────────────────┼──────────┼───────┼───────────┼───────────┤
│ EGLL ➔ KORD (2025-10-01 07:23)  │  3517 nm │ FL320 │       -7m │       -1m │
│ [XXX B763 N686XX] ... Flight ID: │          │       │           │           │
│ 29127XXXX                        │          │       │           │           │
│ EGLL ➔ KORD (2025-10-01 08:09)  │  3597 nm │ FL340 │       -6m │       -2m │
│ [YYY B789 N848XX] ... Flight ID: │          │       │           │           │
│ 29127XXXX                        │          │       │           │           │
│ EGLL ➔ KORD (2025-10-01 10:17)  │  3521 nm │ FL320 │      +12m │      +10m │
│ [ZZZ B763 N657XX] ... Flight ID: │          │       │           │           │
│ 29127XXXX                        │          │       │           │           │
└──────────────────────────────────┴──────────┴───────┴───────────┴───────────┘
```

---

### 3. Fuel Burn, Spot Pricing & Carbon Assessment (`flightml fuel-impact`)

```bash
flightml fuel-impact 29126XXXX --period 202510
```

```text
Environmental & Fuel Impact: ECTRL 29126XXXX (B772) - Period 202510
                      Fuel & Carbon Inefficiency Analysis                      
┌────────────────────────────┬────────────────────────────────────────────────┐
│ Metric                     │ Value                                          │
├────────────────────────────┼────────────────────────────────────────────────┤
│ Excess Distance Flown      │ 16.9 nm                                        │
│ Terminal Holding Duration  │ 0.0 min                                        │
│ Holding Stack Fuel Burned  │ 0.0 kg                                         │
│ Route Detour Fuel Burned   │ 232.2 kg                                       │
│ Total Excess Fuel Burned   │ 232.2 kg                                       │
│ Total Excess CO₂ Emitted   │ 733.8 kg (0.73 metric tons)                    │
│ Weekly Jet Fuel Price      │ $2.112/gal ($0.6941/kg)                        │
│ Price Week & Benchmark     │ 2025-W40 (2025-09-29 to 2025-10-05) [EIA       │
│                            │ Weekly Spot Price (USGC/IATA)]                 │
│ Estimated Excess Fuel Cost │ $161.18 USD                                    │
└────────────────────────────┴────────────────────────────────────────────────┘
```

---

### 4. Continuous Descent Operations (CDO) Scoring (`flightml cdo-descent`)

```bash
flightml cdo-descent 29126XXXX --period 202510
```

```text
Continuous Descent Operations (CDO) Report: ECTRL 29126XXXX
Route: KORD ➔ EGLL | Aircraft: B772 | Cruise FL: 390
CDO Compliance Score: 94.1% | Total Descent Time: 45.0 min
Level-Off Inefficiencies: 1 events (2.7 min un-optimized level flight)
Excess Descent Fuel Burn: 192.9 kg Jet-A | Excess CO₂: 609.5 kg

             Detected Descent Level-Off & Altitude Reversal Events             
┌────────────┬────────────┬─────────────┬──────────┬────────────┬─────────────┐
│ Event Type │ Start UTC  │ End UTC     │ Duration │      Level │ Location    │
├────────────┼────────────┼─────────────┼──────────┼────────────┼─────────────┤
│ LEVEL_OFF  │ XX:XX:37   │ XX:XX:17    │  2.7 min │      FL140 │ 51.XX°,     │
│            │            │             │          │            │ -1.XX°      │
└────────────┴────────────┴─────────────┴──────────┴────────────┴─────────────┘
```

---

### 5. European FIR Airspace Saturation (`flightml fir-saturation`)

```bash
flightml fir-saturation --period 202510 --threshold 40 --limit 3
```

```text
   🌐 European FIR Sector Saturation (Period: 202510, Threshold: ≥40 flt/hr)   
┌─────────┬──────────┬──────────┬───────────┬──────────┬───────────┬──────────┐
│         │    Total │     Peak │       Avg │    Dwell │           │ Saturat… │
│ FIR ID  │ Travers… │  Flts/Hr │   Flts/Hr │    (min) │ Hours ≥40 │        % │
├─────────┼──────────┼──────────┼───────────┼──────────┼───────────┼──────────┤
│ LTBBFIR │  159,166 │      328 │     210.3 │     22.1 │       747 │    98.7% │
│ LBSRFIR │  116,038 │      283 │     153.3 │     18.9 │       746 │    98.5% │
│ LTAAFIR │  121,562 │      235 │     161.0 │     52.7 │       746 │    98.8% │
└─────────┴──────────┴──────────┴───────────┴──────────┴───────────┴──────────┘
```

---

## 🐍 Python Library Quickstart

FlightML is designed to be easily embedded into Python pipelines, notebooks, and applications:

```python
import flightml as fml

# 1. Connect to the ingested dataset (period '202510')
ds = fml.FlightDataset(period="202510")
print(f"Dataset contains {ds.total_flights:,} flights.")

# 2. Search flights with multi-criteria business filtering
flights = ds.search_flights(
    adep="EGLL",
    ades="KORD",
    operator="BAW",
    limit=5
)
# Returns DataFrame with human-readable labels:
# ectrl_id, flight_label, actual_distance_nm, flt_duration_min

# 3. Dynamic weekly fuel pricing and emissions computation
impact = fml.compute_flight_fuel_and_emissions(ds, ectrl_id=29126XXXX)
print(f"Total Fuel: {impact.total_fuel_kg:,.1f} kg (${impact.total_fuel_cost_usd:,.2f})")
print(f"Weekly Spot Price: ${impact.fuel_price_usd_per_gallon:.3f}/gal ({impact.fuel_price_week})")

# 4. Continuous Descent Operations (CDO) descent scoring
cdo = fml.analyze_cdo_descent(ds, ectrl_id=29126XXXX)
print(f"CDO Compliance Score: {cdo.cdo_compliance_score:.1f}% (Level-offs: {cdo.level_off_count})")

# 5. Twin flight efficiency comparison
twins = fml.compare_twin_flights(ds, adep="RKSI", ades="EGLL", ac_type="B77W")
print(f"Detour Delta: +{twins.detour_delta_nm:.0f} nm | Excess Fuel: +{twins.excess_fuel_kg:,.0f} kg")
```

---

## 🖥️ Interactive Simulation Dashboard

Launch the browser-based simulation platform with a single command:

```bash
flightml dashboard
```

Opens at **`http://localhost:8501`** featuring **9 interactive tabs**:
1. 🌐 **Network Overview**: Busiest airports, delay heatmaps, and FIR saturation.
2. ✈️ **4D Trajectory & Holdings**: Trajectory inspector, holding stacks, and unblocked Leaflet maps.
3. 📅 **Recurring Flight Series**: Corridor tracking, diurnal departure distribution, and delay volatility.
4. 📉 **CDO Continuous Descent**: Descent profiles, level-off staircases, and fuel waste.
5. 🔍 **Delay Diagnostics Lab**: Multi-factor root-cause attribution (rotation, holding, wind, ceiling).
6. 🌱 **Fuel, Carbon & Costs**: Weekly EIA spot fuel prices, emissions, and holding penalty.
7. 👥 **Twin-Flight Benchmarking**: Side-by-side efficiency benchmarking on identical routes.
8. ⚡ **What-If & Airline Stress Tests**: Multi-operator turnaround buffer curves and fleet shock stress tests.
9. 📈 **Macro Evolution & AIRAC Diff**: Traffic growth YoY/MoM, airline market share, and AIRAC diffing.

---

## 🌐 REST API Service

Launch the FastAPI web service:

```bash
flightml serve --port 8000
```

- **Base URL**: `http://localhost:8000/api/v1`
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **Redoc Documentation**: `http://localhost:8000/redoc`

---

## 🧪 Testing

Run the test suite with `pytest`:

```bash
pytest tests -v
```

All 37 unit and integration tests pass cleanly.

---

## 📄 License

FlightML software code is licensed under the Apache License 2.0. Note that access to third-party aeronautical datasets remains subject to Eurocontrol's data dissemination terms and agreements.
