# Command-Line Interface (CLI) Reference 💻

FlightML includes an extensive, feature-rich command-line tool accessible via the `flightml` binary. All commands support rich terminal formatting, color-coded tables, and progress indicators.

```bash
flightml --help
```

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

## 📑 Command Categories

1. [Dashboard & Web Services](#1-dashboard--web-services)
2. [Data Ingestion & Management](#2-data-ingestion--management)
3. [Search & Flight Filtering](#3-search--flight-filtering)
4. [4D Trajectory, Holdings & CDO](#4-4d-trajectory-holdings--cdo)
5. [En-Route FIR Airspace & Bottlenecks](#5-en-route-fir-airspace--bottlenecks)
6. [Fuel, Carbon & Spot Pricing](#6-fuel-carbon--spot-pricing)
7. [Weather & Environmental Diagnostics](#7-weather--environmental-diagnostics)
8. [Machine Learning & Delay Diagnostics](#8-machine-learning--delay-diagnostics)
9. [Aviation Simulation & Stress Testing](#9-aviation-simulation--stress-testing)
10. [Macro Period & AIRAC Cycle Analysis](#10-macro-period--airac-cycle-analysis)

---

## 1. Dashboard & Web Services

### `flightml dashboard`
Launch the interactive Streamlit simulation dashboard in your default browser.

```bash
flightml dashboard [--port PORT]
```

- `--port`: Port to bind the Streamlit server (default: `8501`).

### `flightml serve`
Launch the production FastAPI REST web service with uvicorn.

```bash
flightml serve [--host HOST] [--port PORT] [--workers WORKERS]
```

- `--host`: Host IP to bind (default: `127.0.0.1`).
- `--port`: Port number (default: `8000`).
- `--workers`: Number of worker processes (default: `1`).

---

## 2. Data Ingestion & Management

### `flightml ingest-archive`
Extract, classify schemas, partition, and ingest a monthly Eurocontrol archive zip into compressed Parquet tables.

```bash
flightml ingest-archive <archive_path> [--data-dir DIR]
```

- `<archive_path>`: Path to the Eurocontrol zip file (e.g. `data/archive/202510.zip`).
- `--data-dir`: Base directory for processed data (default: `data`).

### `flightml list-periods`
List all ingested data periods available for queries.

```bash
flightml list-periods [--data-dir DIR]
```

### `flightml convert`
Convert legacy CSV datasets into high-performance Apache Parquet tables.

```bash
flightml convert <source_csv_dir> [--output-dir DIR]
```

### `flightml summary`
Display global dataset metrics, flight counts, date ranges, and airport coverage.

```bash
flightml summary [--period PERIOD] [--data-dir DIR]
```

- `--period`: Aeronautical period to inspect (default: `"202510"`).

**Example Output:**
```text
                       ✈️  FlightML Dataset Overview                        
┌─────────────────────────────┬────────────────────────────────────────────┐
│ Metric                      │ Value                                      │
├─────────────────────────────┼────────────────────────────────────────────┤
│ Total Processed Flights     │ 9XX,XXX                                    │
│ Unique Aircraft Operators   │ 8XX                                        │
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

## 3. Search & Flight Filtering

### `flightml search-flights`
Search flights using multiple criteria with human-readable flight labels.

```bash
flightml search-flights [--period PERIOD] [--adep ICAO] [--ades ICAO] [--operator CODE] [--date YYYY-MM-DD] [--limit N]
```

**Example Output (Identified parameters redacted with `XXXX`):**
```text
$ flightml search-flights --period 202510 --adep EGLL --ades KORD --limit 3

               Flight Search Results (Period: 202510, 3 matches)               
┌──────────────────────────────────┬──────────┬───────┬───────────┬───────────┐
│ Flight Label & Metadata          │ Distance │    FL │ Dep Delay │ Arr Delay │
├──────────────────────────────────┼──────────┼───────┼───────────┼───────────┤
│ EGLL ➔ KORD (2025-10-01 07:23)   │  3517 nm │ FL320 │       -7m │       -1m │
│ [XXX B763 N686XX] ... Flight ID: │          │       │           │           │
│ 29127XXXX                        │          │       │           │           │
│ EGLL ➔ KORD (2025-10-01 08:09)   │  3597 nm │ FL340 │       -6m │       -2m │
│ [YYY B789 N848XX] ... Flight ID: │          │       │           │           │
│ 29127XXXX                        │          │       │           │           │
│ EGLL ➔ KORD (2025-10-01 10:17)   │  3521 nm │ FL320 │      +12m │      +10m │
│ [ZZZ B763 N657XX] ... Flight ID: │          │       │           │           │
│ 29127XXXX                        │          │       │           │           │
└──────────────────────────────────┴──────────┴───────┴───────────┴───────────┘
```

### `flightml recurring-series`
Analyze recurring flights on a corridor day-by-day with time-of-day (diurnal) congestion breakdown.

```bash
flightml recurring-series --adep ICAO --ades ICAO [--operator CODE] [--ac TYPE] [--period PERIOD]
```

Example:
```bash
flightml recurring-series --adep EGLL --ades KORD --operator BAW --period 202510
```

### `flightml top-airports`
List the busiest airports ranked by total departures and average departure delay.

```bash
flightml top-airports [--period PERIOD] [--limit N]
```

### `flightml top-delays`
Identify network routes with the highest average arrival delays.

```bash
flightml top-delays [--period PERIOD] [--limit N] [--min-flights N]
```

- `--min-flights`: Minimum number of flights to avoid low-sample outliers (default: `50`).

---

## 4. 4D Trajectory, Holdings & CDO

### `flightml trajectory`
Analyze 4D trajectory radar points, calculating path extension ratio and detecting airborne holding loops.

```bash
flightml trajectory <ectrl_id> [--period PERIOD]
```

### `flightml map`
Generate an interactive HTML Leaflet map displaying the 4D radar track and highlighted holding stacks.

```bash
flightml map <ectrl_id> [--output PATH] [--period PERIOD]
```

- `--output`: Filepath to save the interactive HTML map (default: `flight_<id>_map.html`).

### `flightml cdo-descent`
Analyze Continuous Descent Operations (CDO) descent profile, intermediate level-offs, and descent fuel waste.

```bash
flightml cdo-descent <ectrl_id> [--period PERIOD]
```

**Example Output (Redacted with `XXXX`):**
```text
$ flightml cdo-descent 29126XXXX --period 202510

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

### `flightml compare-twins`
Compare the most and least efficient flights sharing the same city pair and aircraft type.

```bash
flightml compare-twins --adep ICAO --ades ICAO --ac TYPE [--period PERIOD] [--map PATH]
```

- `--map`: Optional HTML path to export dual-trajectory visual comparison.

---

## 5. En-Route FIR Airspace & Bottlenecks

### `flightml fir-saturation`
Calculate hourly traffic saturation rates across European Flight Information Regions (FIRs) and identify airspace chokepoints.

```bash
flightml fir-saturation [--period PERIOD] [--threshold N] [--limit N]
```

- `--threshold`: Hourly flight count threshold to flag as saturated (default: `40`).

**Example Output:**
```text
$ flightml fir-saturation --period 202510 --threshold 40 --limit 3

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

### `flightml flight-fir-dwell`
Analyze sector-by-sector en-route dwell efficiency for a single flight, detecting holding, vectoring, or shortcuts.

```bash
flightml flight-fir-dwell <ectrl_id> [--period PERIOD]
```

### `flightml bottlenecks`
Rank the most congested FIR airspaces across the entire network period.

```bash
flightml bottlenecks [--period PERIOD] [--limit N]
```

### `flightml fir-locate`
Identify which FIR airspace sector contains a given 3D coordinate.

```bash
flightml fir-locate --lat FLOAT --lon FLOAT [--fl FL] [--period PERIOD]
```

---

## 6. Fuel, Carbon & Spot Pricing

### `flightml fuel-prices`
Display official EIA and IATA historical weekly jet fuel spot prices across dataset periods.

```bash
flightml fuel-prices [--period PERIOD]
```

### `flightml fuel-impact`
Calculate excess fuel burn, CO2 emissions, and financial penalty from trajectory detour and airborne holding.

```bash
flightml fuel-impact <ectrl_id> [--period PERIOD]
```

**Example Output (Redacted with `XXXX`):**
```text
$ flightml fuel-impact 29126XXXX --period 202510

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

## 7. Weather & Environmental Diagnostics

### `flightml weather-delayed`
Fetch historical IEM ASOS/METAR observations for a delayed flight and perform multi-factor attribution.

```bash
flightml weather-delayed <ectrl_id> [--period PERIOD]
```

### `flightml weather-winds`
Calculate runway crosswind and headwind components and determine optimal active runways.

```bash
flightml weather-winds --airport ICAO [--time TIMESTAMP]
```

### `flightml weather-impact`
Analyze European macro weather history and statistical correlation with network delays.

```bash
flightml weather-impact --start YYYY-MM-DD --end YYYY-MM-DD
```

---

## 8. Machine Learning & Delay Diagnostics

### `flightml diagnose`
Perform comprehensive multi-factor delay root-cause attribution (Reactionary, Holding, Crosswind, Ceiling).

```bash
flightml diagnose <ectrl_id> [--period PERIOD]
```

### `flightml predict-delay`
Predict arrival delay using a trained gradient-boosted machine learning model.

```bash
flightml predict-delay --adep ICAO --ades ICAO --operator CODE --sched-dep TIMESTAMP [--ac TYPE]
```

### `flightml reactionary`
Trace aircraft tail rotation history and quantify downstream delay propagation.

```bash
flightml reactionary --tail REGISTRATION [--date YYYY-MM-DD] [--period PERIOD]
```

### `flightml train-gnn`
Train the Spatio-Temporal Graph Neural Network (ST-GNN) on international airport connectivity graphs.

```bash
flightml train-gnn [--epochs N] [--lr FLOAT] [--batch-size N]
```

---

## 9. Aviation Simulation & Stress Testing

### `flightml stress-test`
Stress-test an airline's fleet schedule under operational delay shock waves to build fragility curves.

```bash
flightml stress-test --operator CODE [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--period PERIOD]
```

### `flightml simulate-buffer`
Simulate how expanding turnaround buffers (+5m to +25m) reduces reactionary delays across an airline's schedule.

```bash
flightml simulate-buffer --operator CODE [--buffer-min N] [--period PERIOD]
```

### `flightml gdp`
Run Ration-by-Schedule (RBS) Ground Delay Program (GDP) slot allocation to prevent airborne holding during weather collapses.

```bash
flightml gdp --airport ICAO --start TIMESTAMP --end TIMESTAMP --capacity FLIGHTS_PER_HOUR [--period PERIOD]
```

---

## 10. Macro Period & AIRAC Cycle Analysis

### `flightml compare-periods`
Deep comparative macro analysis between two Eurocontrol monthly archives (traffic growth, delay drift, airline shifts).

```bash
flightml compare-periods --p1 PERIOD1 --p2 PERIOD2
```

Example:
```bash
flightml compare-periods --p1 201611 --p2 202510
```

### `flightml airac-diff`
Compare contiguous AIRAC cycle aeronautical configurations (airways, waypoints, FIR boundaries).

```bash
flightml airac-diff --c1 CYCLE1 --c2 CYCLE2
```

Example:
```bash
flightml airac-diff --c1 2509 --c2 2510
```

