# FlightML ✈️

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DuckDB Powered](https://img.shields.io/badge/Storage-DuckDB%20%2B%20Apache%20Parquet-orange.svg)](https://duckdb.org/)
[![Documentation](https://img.shields.io/badge/Docs-Complete%20Guide-green.svg)](docs/index.md)

**A high-performance Python aviation intelligence library, interactive simulation workbench, REST API, and CLI for Global Airspace Network Analytics, 4D Trajectory Physics, Environmental & Eurocontrol CRCO Cost Accounting (`€ EUR`), and Prescriptive Operations Research.**

---

> [!IMPORTANT]
> ### 🔒 Data Access & Eurocontrol OneSky Online Compliance Notice
> **Eurocontrol Research Data Access Prerequisite**:
> The underlying aeronautical traffic, flight plans, and 4D radar trajectory archives supported by FlightML (such as Eurocontrol DDR2 / PRISME / R-NEST datasets) are **proprietary and not for public consumption or redistribution**.
>
> To access or download raw historical datasets, researchers and organizations **must have an approved user account on Eurocontrol's OneSky Online portal** ([https://www.eurocontrol.int/](https://www.eurocontrol.int/)) with institutional access granted to the **Demand Data Repository (DDR)** under Eurocontrol's Data Dissemination Policy and standard Research Agreement.
>
> **Synthetic Example Outputs Policy**:
> In accordance with Eurocontrol data dissemination terms, commercial confidentiality, and GDPR requirements, **all sample output tables, flight identifiers (e.g., `900104215`), aircraft tail registrations (e.g., `G-SYNA`, `EI-SYNB`), and aggregate statistics shown in this documentation are realistic synthetic records** generated to illustrate table schemas and platform capabilities without exposing proprietary archive rows.

---

## 📖 Documentation Suite

Detailed manuals and developer guides are located in the [**`docs/`**](docs/index.md) directory:

- 🚀 [**Quickstart Guide**](docs/quickstart.md): Installation, dataset directory layout, dashboard launch, and first steps.
- 🐍 [**Python Library Guide**](docs/library_guide.md): Developer guide with complete Python code recipes for all modules.
- 💻 [**CLI Reference Manual**](docs/cli_reference.md): Exhaustive guide for all 45+ `flightml` terminal commands and synthetic sample outputs.
- 🌐 [**REST API Reference**](docs/rest_api.md): Complete FastAPI endpoints catalog, request/response schemas, and Swagger docs.
- 🏛️ [**Architecture & Methodology**](docs/architecture.md): DuckDB columnar execution, ICAO GANP & Eurocontrol PRU formulas, OpenAP & ICAO EDB v32B physics, Eurocontrol CRCO charges (`€`), and operations research models.

---

## ✨ Platform Capabilities

FlightML fuses multi-gigabyte 4D trajectories, global aeronautical navigation databases (**86,280 5-letter RNAV waypoints**, **17,090 VHF navaids**, **48,405 runways**, **30,192 VHF radio frequencies**), aircraft performance models (**OpenAP** & **ICAO Engine Emissions Databank v32B**), **Eurocontrol CRCO en-route unit rates (`€`)**, and historical **IEM ASOS/METAR & Open-Meteo ERA5 upper-air meteorology**:

- **🗺️ $5^\circ \times 5^\circ$ Hourly Airspace Saturation & 4D Conflict Maps**: Period-averaged hourly flight density, peak traffic load, vertical climb/descent share, Dynamic Complexity Index, and 1-minute interpolated 4D co-altitude proximity detection (`< 5 nm` / `< 1,000 ft`).
- **📏 ICAO GANP & Eurocontrol PRU KPI Suite**: Complete implementation of **KPI02** (Additional Taxi-Out Time `ATOT`), **KPI04** (`KEP` Filed En-Route Extension), **KPI05** (`KEA` Actual En-Route Extension & Tactical ATC Attribution), **KPI08** (40 nm `ASMA` Terminal Transit Time), **KPI17** (Continuous Climb Operations `CCO`), and **KPI19** (Continuous Descent Operations `CDO` & Cruise Level-Capping).
- **🛣️ Free Route Airspace (`FRA`) & VHF Comms Workload**: Snaps 4D trajectories against published ATS airways and 5-letter fixes to quantify `FRA / DCT` directness vs. fixed airway adherence, and models ATC controller-pilot VHF voice channel occupancy (`%`).
- **🌱 OpenAP Fuel Burn, ICAO EDB v32B LTO Emissions & Contrail Physics**: Quantifies trajectory detour fuel waste, terminal holding penalties, ground taxi queue $\text{NO}_x / \text{CO} / \text{HC} / \text{nvPM}$ soot emissions, and **Schmidt-Appleman persistent contrail (`ISSR`)** radiative forcing (`CO₂e`).
- **💶 Eurocontrol CRCO Route Charges & Weekly Jet-A Spot Pricing (`€ EUR`)**: Computes exact FIR-by-FIR Service Units (`SU`) and en-route navigation fees (`€`), weekly EIA/Platts Jet-A spot fuel costs in Euros (`€`), and Corridor Route-Charge Avoidance vs. Fuel Burn trade-offs.
- **🔮 Prescriptive Operations Research & Stress Testing**: Includes **Network-Wide Hub Tail-Swap Recovery (`Tail A ⇄ Tail B`)**, **Ration-by-Schedule Ground Delay Program (`GDP` / ATFM Slot) Optimization**, **Turnaround Buffer Sensitivity**, **Runway Braking & Crosswind Collapse**, and **Multi-Operator Fleet Shock Contagion Curves**.

---

## 📦 Installation

```bash
git clone https://github.com/SDMCNS/FlightML.git
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

## 🖥️ Synthetic CLI Output Examples

Below are representative terminal outputs illustrating FlightML's Rich tables using **synthetic sample data** (all costs in **Euros `€`**).

### 1. Global Dataset Summary (`flightml summary`)

```bash
flightml summary --period 202510
```

```text
                       ✈️  FlightML Dataset Overview (Synthetic Sample)
┌─────────────────────────────┬────────────────────────────────────────────┐
│ Metric                      │ Value                                      │
├─────────────────────────────┼────────────────────────────────────────────┤
│ Total Processed Flights     │ 842,500                                    │
│ Unique Aircraft Operators   │ 740                                        │
│ Origin Airports (ADEP)      │ 1,580                                      │
│ Destination Airports (ADES) │ 1,595                                      │
│ Date Coverage               │ 2025-10-01 00:00:00 to 2025-10-31 23:59:00 │
│ Mean Departure Delay        │ 5.12 min                                   │
│ Mean Arrival Delay          │ 4.85 min                                   │
│ Airports Catalog            │ 86,336                                     │
│ Runways Catalog             │ 48,405                                     │
│ FIR Airspaces Catalog       │ 170                                        │
└─────────────────────────────┴────────────────────────────────────────────┘
```

---

### 2. ICAO GANP & Eurocontrol PRU Scorecard (`flightml pru-kpis`)

```bash
flightml pru-kpis 900104215 --period 202510
```

```text
     📏 ICAO GANP & Eurocontrol PRU Scorecard: ECTRL 900104215 (Synthetic)
┌──────────────────────────────────┬───────────────────────────────────────────┐
│ Indicator                        │ Value & Operational Verdict               │
├──────────────────────────────────┼───────────────────────────────────────────┤
│ Route & Aircraft                 │ EGLL ➔ LIRF (SYN A320, Reg: G-SYNA)       │
│ Great-Circle O-D Distance        │ 780.4 nm                                  │
│ KPI04 (KEP Filed Extension >40nm)│ 4.18% (Filed En-Route: 731.2 nm)          │
│ KPI05 (KEA Actual Extension >40n)│ 2.64% (Actual En-Route: 720.4 nm)         │
│ Tactical ATC Delta (KEA - KEP)   │ -1.54% (-10.8 nm — ATC_TACTICAL_SHORTCUT) │
│ KPI08 ASMA (40nm Terminal Time)  │ 19.4 min (+3.2 min additional vs p20)     │
│ KPI19 Cruise Flight Level        │ Max FL350 vs Req FL370 (CAPPED -20 FL)    │
│ Cruise Level-Cap Fuel Penalty    │ +58.4 kg Jet-A (+184.5 kg CO₂)            │
└──────────────────────────────────┴───────────────────────────────────────────┘
```

---

### 3. Fuel Burn, Eurocontrol CRCO Charges & LTO Emissions (`flightml crco-cost` & `flightml taxi-out`)

```bash
flightml crco-cost 900104215 --period 202510
```

```text
   💶 Eurocontrol CRCO Route Charges & Direct Operating Cost: ECTRL 900104215
┌────────────────────────────┬────────────────────────────────────────────────┐
│ Metric                     │ Value (EUR €)                                  │
├────────────────────────────┼────────────────────────────────────────────────┤
│ Aircraft Type & MTOW       │ A320 (78.0 metric tons | Weight Factor: 1.249) │
│ Total CRCO Service Units   │ 18.04 SU (1,444.5 km across 3 FIRs)            │
│ Actual CRCO En-Route Fee   │ €1,584.20 EUR (Filed: €1,612.80 EUR)           │
│ Estimated Total Jet-A Fuel │ 4,620.0 kg (€2,966.50 EUR @ €0.6421/kg)        │
│ Excess Detour & Hold Cost  │ €142.80 EUR (222.4 kg excess Jet-A)            │
│ Total Direct Route Cost    │ €4,550.70 EUR (34.8% CRCO share)               │
└────────────────────────────┴────────────────────────────────────────────────┘
```

---

### 4. Prescriptive Hub Tail-Swap Recovery (`flightml tail-swaps`)

```bash
flightml tail-swaps --period 202510
```

```text
        🔄 Prescriptive Hub Tail-Swap Recovery Schedule (Synthetic Sample)
┌────────────┬──────┬──────┬──────┬───────────────────┬────────────┬───────────┬─────────────┐
│ Date       │ Hub  │ Oper │ AC   │ Tail Swap Pair    │ Base Delay │ Post-Swap │ Saved (€)   │
├────────────┼──────┼──────┼──────┼───────────────────┼────────────┼───────────┼─────────────┤
│ 2025-10-04 │ EGSS │ SYN  │ B738 │ EI-SYNA ⇄ EI-SYNB │    118 min │     0 min │ €10,384 EUR │
│ 2025-10-09 │ EGLL │ SYN  │ A320 │ G-SYNC  ⇄ G-SYND  │     84 min │     6 min │  €6,864 EUR │
│ 2025-10-14 │ EDDF │ SYN  │ A321 │ D-SYNE  ⇄ D-SYNF  │     72 min │     0 min │  €6,336 EUR │
└────────────┴──────┴──────┴──────┴───────────────────┴────────────┴───────────┴─────────────┘
```

---

### 5. 4D Co-Altitude Airspace Proximity & Complexity Scan (`flightml airspace-conflicts`)

```bash
flightml airspace-conflicts --period 202510 --hour 10 --sep-nm 5.0
```

```text
    ⚠️ 4D Co-Altitude Airspace Proximity Scan (10:00–10:59 UTC, Synthetic Sample)
┌──────────┬───────────────────────────┬───────────────────────────┬──────────┬──────────┬──────────────┐
│ Epoch    │ Severity                  │ Aircraft Pair             │ Horiz    │ Vert     │ Coordinates  │
├──────────┼───────────────────────────┼───────────────────────────┼──────────┼──────────┼──────────────┤
│ 10:15:00 │ LOSS_OF_SEPARATION_ALERT  │ SYN #900104215 ⇄ SYN #908 │  2.14 nm │   200 ft │ 51.24°, 1.85°│
│ 10:30:00 │ HIGH_PROXIMITY            │ SYN #900112640 ⇄ SYN #914 │  3.82 nm │   400 ft │ 48.92°, 2.41°│
│ 10:45:00 │ HIGH_PROXIMITY            │ SYN #900119004 ⇄ SYN #921 │  4.45 nm │   600 ft │ 50.11°, 8.68°│
└──────────┴───────────────────────────┴───────────────────────────┴──────────┴──────────┴──────────────┘
```

---

## 🖥️ Interactive Simulation Workbench (`flightml dashboard`)

Launch the 10-tab Streamlit simulation workbench:

```bash
flightml dashboard
```

Opens at **`http://localhost:8501`** with **10 interactive tabs**:
1. 🌐 **Network Overview & Airspace Map**: Interactive **$5^\circ \times 5^\circ$ European Airspace Hourly Saturation & Complexity Map** (period average by hour of day `00:00`–`23:00 UTC`), FIR & AUA (`TMA`/`CTA`/`UTA`) saturation, `KPI02 ATOT` surface taxi delays, and VHF radio occupancy.
2. 🗺️ **4D Trajectory & Holdings**: Single-flight 4D radar track vs. filed plan map, `KPI04 KEP` / `KPI05 KEA` / `KPI08 ASMA` / `KPI19` scorecard, `FRA` vs. ATS airway snapping, and FIR/AUA sector dwell tables.
3. 📅 **Recurring Series & Diurnal Trends**: Longitudinal day-over-day route tracking and diurnal departure bank congestion curves.
4. 📉 **Vertical Profile: CDO & CCO**: Descent `CDO` and Climb `CCO` level-off detection, vertical profile charts, and network-wide `KPI19` cruise level-capping corridors.
5. 🩺 **Delay & Runway Diagnostics**: Multi-factor delay root-cause attribution and arrival runway braking action, density altitude, and stopping safety margins.
6. 🌱 **Fuel, Carbon & Route Charges (`€`)**: OpenAP excess fuel burn, `TAXI_OUT` ICAO EDB v32B idle emissions ($\text{NO}_x, \text{CO}, \text{nvPM}$), Eurocontrol CRCO fees (`€`), upper-air contrail `ISSR` forcing, and Corridor CRCO Route-Charge vs. Fuel Burn trade-offs.
7. 👯 **Twin-Flight Benchmark**: Side-by-side comparison and interactive **Dual-Trajectory Map** (`Green` vs. `Red`) with lateral divergence pinpointing.
8. 🔮 **What-If & Stress Tests**:
   - **Scenario A**: Multi-Operator Turnaround Buffer Sensitivity (`+5m` to `+25m`)
   - **Scenario B**: Adverse Weather & Runway Crosswind Capacity Collapse
   - **Scenario C**: Multi-Operator Fleet Operational Shock Contagion Curves (`+15m` to `+90m`)
   - **Scenario D**: Network-Wide Prescriptive Hub Tail-Swap Recovery Optimizer (`Tail A ⇄ Tail B`, `€`)
   - **Scenario E**: 4D Co-Altitude Airspace Proximity & $5^\circ \times 5^\circ$ Complexity Map Scanner
   - **Scenario F**: Ration-by-Schedule Ground Delay Program (`GDP` / ATFM Slot) Optimizer (`€`)
9. 📈 **Macro Evolution & AIRAC Diff**: Period-over-period traffic growth, `FRA` directness evolution, European weather-delay correlation, and AIRAC cycle airway/FIR diffing.
10. 📖 **Methodology & KPI Reference**: Comprehensive in-app mathematical reference for every KPI, physics model, and optimization algorithm.

---

## 🌐 REST API Service

```bash
flightml serve --port 8000
```

- **Base URL**: `http://localhost:8000/api/v1`
- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **Redoc Documentation**: `http://localhost:8000/redoc`

---

## 🧪 Testing

```bash
pytest tests -v
```

All 56 unit and integration tests pass cleanly.

---

## 📄 License

FlightML software code is licensed under the Apache License 2.0. Access to third-party aeronautical datasets remains subject to Eurocontrol's data dissemination terms and agreements.
