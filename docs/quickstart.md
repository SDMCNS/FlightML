# Quickstart Guide 🚀

This guide walks you through setting up **FlightML**, installing dependencies, ingesting data, launching the interactive simulation dashboard, and running your first analytical workflows.

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

## 🛠️ System Prerequisites

- **Operating System**: Linux, macOS, or Windows (10/11 / Server).
- **Python**: Version `3.10`, `3.11`, or `3.12`.
- **Memory**: 8 GB RAM minimum (16 GB recommended for multi-period Parquet processing).
- **Storage**: ~5 GB free disk space for unpacked Parquet datasets and 4D trajectory files.

---

## 📦 Installation

### Standard Installation

Clone the repository and install FlightML in editable mode:

```bash
git clone https://github.com/flightml/flightml.git
cd FlightML
pip install -e .
```

### Optional Dependencies

FlightML provides optional extras for Graph Neural Networks (GNN) and development/testing tools:

```bash
# Install GNN capabilities (PyTorch)
pip install -e ".[gnn]"

# Install development & testing tools (pytest, httpx)
pip install -e ".[dev]"

# Install all extras
pip install -e ".[gnn,dev]"
```

Verify your installation:

```bash
flightml --help
```

---

## 📂 Data Setup & Directory Layout

FlightML organizes its data assets under a centralized `data/` directory.

```
data/
├── archive/                      # Raw or downloaded Eurocontrol archive zips
│   ├── 201611.zip                # November 2016 baseline archive
│   └── 202510.zip                # October 2025 modern archive
├── fuel/
│   ├── weekly_jet_fuel_prices.parquet  # Historical weekly spot prices (EIA / IATA)
│   └── weekly_jet_fuel_prices.csv
├── weather/                      # Cached ASOS/METAR weather observations
├── 201611/                       # Ingested Parquet tables for period 201611
│   ├── flights.parquet           # Flight header & metadata records
│   ├── trajectories.parquet      # 4D trajectory coordinates & flight levels
│   ├── airports.parquet          # Aerodrome layouts & runway configurations
│   ├── firs.parquet              # European FIR airspace boundaries
│   └── metadata.json             # AIRAC cycle and schema metadata
└── 202510/                       # Ingested Parquet tables for period 202510
    ├── flights.parquet
    ├── trajectories.parquet
    ├── airports.parquet
    ├── firs.parquet
    └── metadata.json
```

### Ingesting Eurocontrol Archives

To ingest a new monthly Eurocontrol archive zip file:

```bash
# Ingest an archive file
flightml ingest-archive data/archive/202510.zip

# List all available ingested periods
flightml list-periods
```

Output:
```
Available FlightML Data Periods:
- 202510 [MODERN] (AIRAC Cycle 2510, ~XXXk flights)
- 201611 [HISTORICAL] (AIRAC Cycle 1611, ~XXXk flights)
```

---

## 🖥️ Launching the Interactive Dashboard

Launch the browser-based simulation and analytics dashboard:

```bash
flightml dashboard
```

The dashboard will open automatically at **`http://localhost:8501`** and features **9 dedicated analytics tabs**:

1. **🌐 Network Overview**: Live metrics, busiest airports, delay heatmaps, and FIR sector saturation.
2. **✈️ 4D Trajectory & Holdings**: Flight-level trajectory viewer, airborne holding loop detection, sector dwell inefficiency, and unblocked Leaflet maps.
3. **📅 Recurring Flight Series**: Multi-day tracking of recurring flight pairs, departure time-of-day (diurnal) distribution, and delay volatility.
4. **📉 CDO Continuous Descent**: Descent profile visualization, level-off staircases, CDO compliance scoring, and descent fuel waste.
5. **🔍 Delay Diagnostics Lab**: Multi-factor root-cause attribution (reactionary rotation, holding, runway crosswinds, weather ceiling) with actionable ATC recommendations.
6. **🌱 Fuel, Carbon & Costs**: Dynamic weekly Jet-A spot fuel pricing, fuel burn calculations, CO2 emissions, and financial penalty breakdown.
7. **👥 Twin-Flight Benchmarking**: Side-by-side comparison of the most and least efficient flights on identical city pairs and aircraft types.
8. **⚡ What-If & Airline Stress Tests**:
   - **Scenario A**: Multi-operator turnaround buffer expansion (+5m to +25m) side-by-side curve.
   - **Scenario B**: Ground Delay Program (GDP) Ration-by-Schedule slot rationing under severe weather.
   - **Scenario C**: Multi-operator fleet shock stress test (+15m to +90m delay contagion).
9. **📈 Macro Evolution & AIRAC Diff**: Period-over-period macro traffic growth, airline market share shifts, fleet modernization, and AIRAC cycle airspace configuration diffing.

---

## 🌐 Launching the REST API

FlightML includes a high-performance FastAPI service:

```bash
# Start the API server on port 8000
flightml serve --port 8000
```

- **API Base URL**: `http://localhost:8000/api/v1`
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`
- **Redoc Documentation**: `http://localhost:8000/redoc`

Test the API via curl:

```bash
curl http://localhost:8000/api/v1/summary
```

---

## ⚡ First Steps with the CLI

Here are a few quick commands to test your setup, complete with expected terminal outputs (with operational flight IDs and callsigns redacted with `XXXX` for compliance):

### 1. View Dataset Summary
```bash
flightml summary --period 202510
```

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

### 2. Search Flights on a City Pair
```bash
flightml search-flights --period 202510 --adep EGLL --ades KORD --limit 3
```

```text
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

### 3. Calculate Fuel Burn & Weekly Spot Price Financial Penalty
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

## 🐍 First Steps in Python

Create a Python script `demo.py`:

```python
import flightml as fml

# Initialize connection to the modern dataset period
ds = fml.FlightDataset(period="202510")

# Fetch summary
summary = ds.get_summary()
print(f"Airports: {summary['total_airports']:,} | Flights: {summary['total_flights']:,}")

# Look up an anonymized flight by Eurocontrol ID
flight = ds.get_flight(29126XXXX)
print(f"Flight: {flight['operator']} {flight['callsign']} ({flight['adep']} -> {flight['ades']})")

# Evaluate fuel burn and weekly jet fuel pricing
impact = fml.compute_flight_fuel_and_emissions(ds, 29126XXXX)
print(f"Fuel: {impact.total_fuel_kg:,.1f} kg (${impact.total_fuel_cost_usd:,.2f})")
print(f"Price Benchmark: ${impact.fuel_price_usd_per_gallon:.3f}/gal (Week: {impact.fuel_price_week})")
```

Run it:

```bash
python demo.py
```

