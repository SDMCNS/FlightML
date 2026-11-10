# REST API Reference 🌐

FlightML includes a production-ready, asynchronous **FastAPI** web service providing REST endpoints for aeronautical data queries, 4D trajectory analytics, weather fusion, weekly jet fuel pricing, and simulation stress testing.

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

## 🚀 Starting the API Service

```bash
flightml serve --host 127.0.0.1 --port 8000
```

- **Base URL**: `http://localhost:8000/api/v1`
- **Interactive Swagger UI**: `http://localhost:8000/docs`
- **Redoc UI**: `http://localhost:8000/redoc`

---

## 📑 Endpoints Catalog

### 1. System & Discovery

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service health status and link to `/docs`. |
| `GET` | `/api/v1/periods` | List all ingested Eurocontrol historical periods (`201611`, `202510`). |
| `GET` | `/api/v1/summary` | Global dataset statistics (total flights, airports, routes, dates). |

#### Example: `GET /api/v1/periods`
```json
{
  "available_periods": ["201611", "202510"]
}
```

---

### 2. Flight Search & Trajectory

#### `GET /api/v1/flights/search`
Search flights across any period with rich human-readable labels.

**Query Parameters:**
- `period` *(string, default: "201611")*: Period ID (`201611`, `202510`).
- `adep` *(string, optional)*: Origin ICAO (e.g. `EGLL`).
- `ades` *(string, optional)*: Destination ICAO (e.g. `KORD`).
- `operator` *(string, optional)*: Airline ICAO code (e.g. `BAW`).
- `ac_type` *(string, optional)*: Aircraft type (e.g. `B772`).
- `date_from` *(string, optional)*: `YYYY-MM-DD`.
- `date_to` *(string, optional)*: `YYYY-MM-DD`.
- `min_dist` *(float, optional)*: Minimum distance in nautical miles.
- `max_dist` *(float, optional)*: Maximum distance in nautical miles.
- `min_fl` / `max_fl` *(float, optional)*: Flight level range (e.g. `310` to `390`).
- `limit` *(int, default: 25)*: Max records to return.

#### `GET /api/v1/flights/{ectrl_id}`
Retrieve complete flight details, filed flight plan, actual arrival delay, and 4D trajectory holding metrics.

#### `GET /api/v1/flights/{ectrl_id}/map`
Returns an interactive Leaflet HTML map rendering the 4D radar track, waypoints, and holding loops using unblocked Humanitarian OSM and USGS Topo tile layers.

#### `GET /api/v1/flights/recurring`
Longitudinal daily recurring series analysis on an origin-destination corridor with diurnal (time-of-day) congestion breakdown.

---

### 3. Fuel, Carbon & Weekly Spot Pricing

#### `GET /api/v1/fuel/prices`
Retrieve official historical weekly jet fuel spot prices (EIA / IATA Rotterdam Platts benchmark) in USD per gallon, USD per kg, and USD per metric ton.

**Query Parameters:**
- `period` *(string, optional)*: Filter by period ID (`201611`, `202510`).

**Response Schema:**
```json
[
  {
    "period_id": "202510",
    "week_label": "2025-W40 (Sep 29 - Oct 05)",
    "price_usd_per_gallon": 2.112,
    "price_usd_per_kg": 0.694,
    "price_usd_per_metric_ton": 694.0,
    "source": "U.S. EIA Series EER_EPJK_PF4_RGC_DPG / IATA Platts ARA"
  }
]
```

#### `GET /api/v1/flights/{ectrl_id}/fuel`
Calculate aerodynamic fuel burn, carbon dioxide emissions ($3.16\text{ kg CO}_2\text{ per kg Jet-A}$), trajectory detour waste, holding loop waste, and weekly spot price financial penalties.

**Query Parameters:**
- `period` *(string, default: "201611")*: Data period.
- `price_per_kg` *(float, optional)*: Manual price override in USD/kg.

**Response Schema (Redacted with `XXXX`):**
```json
{
  "ectrl_id": 29126XXXX,
  "departure_date": "2025-10-01",
  "total_fuel_kg": 46820.4,
  "co2_emissions_kg": 147952.5,
  "detour_excess_fuel_kg": 182.1,
  "holding_excess_fuel_kg": 0.0,
  "fuel_price_usd_per_gallon": 2.112,
  "fuel_price_usd_per_kg": 0.694,
  "fuel_price_week": "2025-W40 (Sep 29 - Oct 05)",
  "total_fuel_cost_usd": 32493.36,
  "detour_fuel_cost_usd": 126.38,
  "holding_fuel_cost_usd": 0.0
}
```

---

### 4. Continuous Descent Operations (CDO) & Twin Flights

#### `GET /api/v1/flights/{ectrl_id}/cdo`
Evaluate the terminal arrival descent profile, identify stepped level-offs, calculate CDO compliance score, and quantify excess descent fuel waste.

#### `GET /api/v1/twins/compare`
Benchmark the fastest vs slowest flights of identical complexity sharing the same city pair and aircraft type.

**Query Parameters:**
- `adep` *(string, required)*: Departure ICAO (e.g. `RKSI`).
- `ades` *(string, required)*: Destination ICAO (e.g. `EGLL`).
- `ac_type` *(string, default: "B77W")*: Aircraft ICAO type.
- `period` *(string, default: "201611")*: Historical period.

---

### 5. Airspace, En-Route FIR & Sector Dwell

#### `GET /api/v1/fir/saturation`
Calculate hourly traffic saturation rates across European Flight Information Regions (FIRs) and identify airspace chokepoints.

**Query Parameters:**
- `period` *(string, default: "202510")*: Data period.
- `threshold` *(int, default: 40)*: Hourly traffic threshold.
- `top_n` *(int, default: 15)*: Maximum top records.

#### `GET /api/v1/flights/{ectrl_id}/fir-dwell`
Assess sector-by-sector en-route dwell efficiency for a single flight, comparing actual sector dwell time against great-circle transit time to isolate holding or vectoring.

#### `GET /api/v1/airac/diff`
Diff contiguous AIRAC aeronautical navigation cycles (airways, waypoints, FIR boundary coordinates).

---

### 6. Delay Diagnostics, Weather & Machine Learning

#### `GET /api/v1/flights/{ectrl_id}/diagnostics`
Synthesize trajectory holding, aircraft tail rotation history, aerodrome crosswinds, and cloud ceilings to assign percentage delay attribution.

#### `GET /api/v1/flights/{ectrl_id}/weather`
Fetch on-demand historical IEM ASOS/METAR weather observations for the departure and arrival airports at the exact flight times.

#### `POST /api/v1/predict/delay`
Predict arrival delay using a trained gradient-boosted machine learning model.

**Request Body:**
```json
{
  "adep": "EGLL",
  "ades": "KORD",
  "ac_operator": "BAW",
  "market_segment": "Traditional Scheduled",
  "requested_fl": 350.0,
  "actual_distance_nm": 3400.0,
  "dep_hour": 10,
  "dep_dayofweek": 2,
  "filed_air_time_min": 480.0
}
```

---

### 7. Simulation & Airline Stress Testing

#### `GET /api/v1/stress-test`
Simulate cascading delay shock waves across an airline fleet schedule to generate fragility curves.

**Query Parameters:**
- `operator` *(string, default: "RYR")*: Airline ICAO code.
- `start_date` *(string, default: "2016-11-01")*: `YYYY-MM-DD`.
- `end_date` *(string, default: "2016-11-04")*: `YYYY-MM-DD`.
- `period` *(string, default: "201611")*: Data period.

#### `GET /api/v1/periods/compare`
Run macro comparative analytics between two Eurocontrol periods, evaluating traffic growth, delay drift, airline market share, and fleet modernization.

