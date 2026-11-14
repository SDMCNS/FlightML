# Quickstart Guide 🚀

This guide walks you through setting up **FlightML**, installing dependencies, configuring datasets, launching the 10-tab interactive simulation workbench, and running your first analytical workflows.

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

## 🛠️ System Prerequisites

- **Operating System**: Linux, macOS, or Windows (10/11 / Server).
- **Python**: Version `3.10`, `3.11`, or `3.12`.
- **Memory**: 8 GB RAM minimum (16 GB recommended for multi-period Parquet processing).
- **Storage**: ~5 GB free disk space for unpacked Parquet datasets and 4D trajectory files.

---

## 📦 Installation

```bash
git clone https://github.com/SDMCNS/FlightML.git
cd FlightML
pip install -e .
```

### Optional Extras

```bash
# Install GNN capabilities (PyTorch)
pip install -e ".[gnn]"

# Install development & testing tools (pytest, httpx)
pip install -e ".[dev]"
```

---

## 📥 Data Ingestion & Initial Setup

FlightML stores its analytical tables in compressed, columnar Apache Parquet queried on the fly by DuckDB. Before running analyses, initialize external aeronautical references and ingest your historical traffic periods:

### Step 1: Compile Aeronautical References (`flightml external-data`)

Compiles bundled open aeronautical references (OpenAP aircraft drag polars, turbofan engine performance envelopes, ICAO Engine Emissions Databank v32B, global 5-letter RNAV waypoints, radio navaids, and Eurocontrol CRCO FIR charging rates) into DuckDB Parquet tables:

```bash
flightml external-data
```

### Step 2: Ingest Monthly Traffic Archives (`flightml ingest-archive`)

Ingest raw Eurocontrol monthly traffic zip archives (e.g., `202510.zip`, `201611.zip`) in a single automated command:

```bash
flightml ingest-archive data/202510.zip
```

The ingestion pipeline automatically:
1. Extracts the archive into staging.
2. Identifies and parses flight plans, 4D radar tracks, FIR boundary crossings, and airport pairs.
3. Converts raw CSV tables into partitioned, compressed Apache Parquet under `flightml/data/parquet/periods/{period_id}`.

*(Alternative: If raw CSV files are already placed in `flightml/data/`, run `flightml convert --include-trajectories`.)*

### Step 3: Verify Ingested Periods (`flightml list-periods`)

```bash
flightml list-periods
```

```text
      Available Aviation Data Periods      
┌───────────┬─────────────────────────────┐
│ Period ID │ Status                      │
├───────────┼─────────────────────────────┤
│ 201611    │ Ready (Parquet Partitioned) │
│ 202510    │ Ready (Parquet Partitioned) │
└───────────┴─────────────────────────────┘
```

---

## 📂 Data Setup & Directory Layout

FlightML organizes its data assets under `data/` (git-ignored for proprietary archives) and `flightml/data/` (bundled open reference catalogs):

```text
flightml/data/
├── external/                             # Auto-compiled Parquet reference catalogs
│   ├── openap_aircraft.parquet           # OpenAP aircraft MTOW, drag & fuel models
│   ├── openap_engines.parquet            # OpenAP turbofan thrust, BPR, and fuel flow rates
│   ├── icao_edb_gaseous.parquet          # ICAO EDB v32B NOx, CO, and HC emission indices
│   ├── icao_edb_nvpm.parquet             # ICAO EDB v32B non-volatile PM soot emission indices
│   ├── crco_unit_rates.parquet           # Eurocontrol CRCO national unit rates (€/SU)
│   ├── nav_fixes.parquet                 # 119,720 global 5-letter RNAV fixes (fix.dat)
│   └── nav_aids_xplane.parquet           # 26,767 global VOR/DME/ILS navaids (nav.dat)
├── fuel/
│   └── weekly_jet_fuel_prices.csv        # Weekly Jet-A spot prices converted to Euros (€)
└── parquet/periods/                      # Partitioned Parquet tables per traffic period
    ├── 201611/                           # Baseline historical period (Nov 2016)
    └── 202510/                           # Modern traffic period (Oct 2025)
        ├── flights.parquet               # Flight headers, delays, aircraft type & operator
        ├── trajectory_actual.parquet     # 4D actual radar coordinates, groundspeed & FL
        ├── trajectory_planned.parquet    # 4D filed flight plan coordinates & requested FL
        ├── flight_firs_actual.parquet    # En-route FIR & TAXI_OUT traversals
        └── flight_auas_actual.parquet    # ATC Unit Airspace (TMA/CTA/UTA) traversals
```

---

## 🖥️ Launching the Interactive Workbench

```bash
flightml dashboard
```

Opens at **`http://localhost:8501`** organized into **4 primary operational categories** across 10 specialized views:

### 1. Airspace & Trajectories
1. **🌐 Network Overview & Airspace Map**: Interactive **$5^\circ \times 5^\circ$ European Airspace Hourly Saturation & Complexity Map** (`00:00`–`23:00 UTC`), FIR & AUA (`TMA`/`CTA`/`UTA`) saturation, `KPI02 ATOT` surface taxi delays, and VHF radio occupancy.
2. **🗺️ 4D Trajectory & Holdings**: Flight-level 4D trajectory viewer, `KPI04 KEP` / `KPI05 KEA` / `KPI08 ASMA` / `KPI19` scorecard, `FRA` vs. ATS airway snapping, and FIR/AUA sector dwell tables.
3. **📅 Recurring Series & Diurnal Trends**: Multi-day tracking of recurring flight pairs and diurnal departure bank congestion curves.
4. **👯 Twin-Flight Benchmark**: Side-by-side comparison and interactive **Dual-Trajectory Map** (`Green` vs. `Red`) with lateral divergence pinpointing.

### 2. Operations & Diagnostics
5. **📉 Vertical Profile: CDO & CCO**: Continuous Descent Operations (`CDO`) and Continuous Climb Operations (`CCO`) intermediate level-off detection, vertical profile charts, and network-wide cruise level-capping corridors.
6. **🩺 Delay & Runway Diagnostics**: Multi-factor delay root-cause attribution and arrival runway braking action, density altitude, and stopping safety margins.
7. **🔮 What-If & Stress Tests**: Turnaround buffer curves, runway crosswind collapse, fleet shock contagion, network-wide **Hub Tail-Swap Recovery (`Tail A ⇄ Tail B`)**, **4D Co-Altitude Conflict & Saturation Map**, and **Ration-by-Schedule GDP Optimizer (`€`)**.

### 3. Economics, Fuel & Climate
8. **🌱 Fuel, Carbon & Route Charges (`€`)**:
   - **3-Way Cruise Altitude Optimization**: Ideal airframe/engine aerodynamic flight level vs. filed requested vs. actual radar flown, with dispatch planning vs. tactical ATC penalty attribution.
   - **Upper-Air Winds & ESAD**: Upper airspace wind vector integration, Equivalent Still Air Distance (`ESAD`), and wind routing efficiency verdicts with local-first caching.
   - **Eurocontrol CRCO Route Charges & DOC**: FIR-by-FIR navigation service units (`SU`), unit rates, direct flight operating cost (`Fuel € + CRCO Navigation €`), and historical spot vs. modern into-plane price comparisons.
   - **OpenAP Multi-Dimensional Emissions & LTO**: En-route fuel burn, terminal holding burn, and `TAXI_OUT` ICAO Engine Emissions Databank (EDB v32B) idle pollutants ($\text{NO}_x, \text{CO}, \text{nvPM}$).

### 4. Strategic Analytics & Reference
9. **📈 Macro Evolution & AIRAC Diff**: Period-over-period traffic growth, `FRA` directness evolution, European weather-delay correlation, and AIRAC cycle diffing.
10. **📖 Methodology, KPI & PBN Reference**: Full mathematical reference for all 21 ICAO/Eurocontrol KPIs, plus dedicated **PBN / RNAV vs. RNP Specifications & Operational Requirements** navigation guide.

---

## ⚡ First Steps with the CLI (Synthetic Sample Outputs)

### 1. View Dataset Summary
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

### 2. Search Flights on a Corridor
```bash
flightml search-flights --period 202510 --adep EGLL --ades KORD --limit 3
```

```text
          Flight Search Results (Period: 202510, Synthetic Sample)
┌──────────────────────────────────┬──────────┬───────┬───────────┬───────────┐
│ Flight Label & Metadata          │ Distance │    FL │ Dep Delay │ Arr Delay │
├──────────────────────────────────┼──────────┼───────┼───────────┼───────────┤
│ EGLL ➔ KORD (2025-10-03 08:15)   │  3495 nm │ FL360 │       -4m │       -2m │
│ [SYN B77W G-SYNA] ... Flight ID: │          │       │           │           │
│ 900104215                        │          │       │           │           │
│ EGLL ➔ KORD (2025-10-03 10:40)   │  3542 nm │ FL340 │       +8m │      +11m │
│ [SYN B789 G-SYNB] ... Flight ID: │          │       │           │           │
│ 900108492                        │          │       │           │           │
│ EGLL ➔ KORD (2025-10-03 14:20)   │  3510 nm │ FL350 │       +2m │       +0m │
│ [SYN A359 G-SYNC] ... Flight ID: │          │       │           │           │
│ 900112640                        │          │       │           │           │
└──────────────────────────────────┴──────────┴───────┴───────────┴───────────┘
```

### 3. Calculate Fuel Burn & Weekly Spot Price Penalty in Euros (`€`)
```bash
flightml fuel-impact 900104215 --period 202510
```

```text
Environmental & Fuel Impact: ECTRL 900104215 (B77W) - Period 202510 (Synthetic)
                      Fuel & Carbon Inefficiency Analysis                      
┌────────────────────────────┬────────────────────────────────────────────────┐
│ Metric                     │ Value                                          │
├────────────────────────────┼────────────────────────────────────────────────┤
│ Excess Distance Flown      │ 24.0 nm                                        │
│ Terminal Holding Duration  │ 4.5 min                                        │
│ Holding Stack Fuel Burned  │ 315.0 kg                                       │
│ Route Detour Fuel Burned   │ 288.0 kg                                       │
│ Total Excess Fuel Burned   │ 603.0 kg                                       │
│ Total Excess CO₂ Emitted   │ 1,905.5 kg (1.91 metric tons)                  │
│ Weekly Jet Fuel Price (€)  │ €1.954/gal (€0.6421/kg)                        │
│ Price Week & Benchmark     │ 2025-W40 [EIA / IATA Platts Converted to EUR]  │
│ Estimated Excess Fuel Cost │ €387.19 EUR                                    │
└────────────────────────────┴────────────────────────────────────────────────┘
```
