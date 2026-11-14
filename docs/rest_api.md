# REST API Reference 🌐

FlightML includes an asynchronous **FastAPI** web service providing REST endpoints for aeronautical data queries, $5^\circ \times 5^\circ$ airspace saturation, 4D trajectory analytics, ICAO GANP & Eurocontrol PRU KPIs, Eurocontrol CRCO route charges (`€ EUR`), and prescriptive operations research.

---

> [!IMPORTANT]
> ### 🔒 Synthetic Example Outputs Policy
> All sample JSON responses, flight identifiers (e.g., `900104215`), and aircraft tail registrations (e.g., `G-SYNA`, `EI-SYNB`) shown below are realistic synthetic records generated to illustrate API schemas without exposing proprietary archive rows. All monetary fields are reported in **Euros (`€`, `EUR`)**.

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

### 1. System, Discovery & Search

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service health status and link to `/docs`. |
| `GET` | `/api/v1/periods` | List all ingested Eurocontrol periods (`201611`, `202510`). |
| `GET` | `/api/v1/summary` | Global dataset statistics (total flights, operators, airports, delays). |
| `GET` | `/api/v1/flights/search` | Multi-criteria flight search with human-readable labels. |
| `GET` | `/api/v1/flights/{ectrl_id}` | Retrieve flight metadata, filed vs. actual route, and 4D holding summary. |
| `GET` | `/api/v1/flights/{ectrl_id}/map` | Render interactive HTML Leaflet map of 4D trajectory and holding stacks. |

---

### 2. Airspace Saturation, AUA Dwell & 4D Conflict Detection

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/airspace/grid-saturation` | Period-average hourly flight density & Dynamic Complexity Index per $5^\circ \times 5^\circ$ square (`hour=0..23`). |
| `GET` | `/api/v1/airspace/conflicts` | 1-minute interpolated 4D co-altitude proximity scanner (`< 5 nm` / `< 1,000 ft`) and sector complexity. |
| `GET` | `/api/v1/fir/saturation` | Hourly traffic saturation across European Flight Information Regions (`FIRs`). |
| `GET` | `/api/v1/flights/{ectrl_id}/fir-dwell` | Single-flight FIR entry/exit dwell efficiency vs. filed flight plan. |
| `GET` | `/api/v1/aua/saturation` | Hourly saturation across granular ATC Unit Airspaces (`TMA`, `CTA`, `UTA`). |
| `GET` | `/api/v1/flights/{ectrl_id}/aua-dwell` | Single-flight terminal (`TMA`) vs. en-route (`CTA`/`UTA`) dwell breakdown. |
| `GET` | `/api/v1/airports/{ident}/radio-workload` | Estimate ATC controller-pilot VHF voice frequency occupancy (`%`). |

#### Synthetic Example: `GET /api/v1/airspace/grid-saturation?hour=10&period=202510`
```json
{
  "period": "202510",
  "hour_utc": 10,
  "total_active_squares": 65,
  "critical_squares_count": 9,
  "high_density_squares_count": 14,
  "max_avg_flights_per_hour": 312.4,
  "squares": [
    {
      "cell_lat": 45,
      "cell_lon": 5,
      "cell_label": "45°N to 50°N, 5°E to 10°E",
      "active_days": 31,
      "avg_flights_per_hour": 312.4,
      "peak_flights_per_hour": 368,
      "vertical_transition_pct": 34.2,
      "avg_flight_level": 330.0,
      "dynamic_complexity_score": 472.7,
      "saturation_status": "CRITICAL_SATURATION"
    }
  ]
}
```

---

### 3. ICAO GANP & Eurocontrol PRU KPIs, CDO/CCO & Free Route Airspace (`FRA`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/kpis/multi-level` | Multi-tier KPI suite across Flight, Fleet, Airport, and Network levels with timeframe filtering (`start_date`, `end_date`, `operator`, `airport`). |
| `GET` | `/api/v1/flights/{ectrl_id}/pru-kpis` | Compute `KPI04 KEP`, `KPI05 KEA`, `KPI08 ASMA`, and `KPI19` level-capping scorecard. |
| `GET` | `/api/v1/flights/{ectrl_id}/cdo` | Continuous Descent Operations (`KPI19 CDO`) compliance and level-off events. |
| `GET` | `/api/v1/flights/{ectrl_id}/cco` | Continuous Climb Operations (`KPI17 CCO`) compliance and climb level-offs. |
| `GET` | `/api/v1/flights/{ectrl_id}/vertical-profile` | Unified 3-phase vertical efficiency report (`CCO` + `Cruise Cap` + `CDO`). |
| `GET` | `/api/v1/pru/level-capping` | Top corridors affected by systematic `KPI19` cruise flight-level capping. |
| `GET` | `/api/v1/flights/{ectrl_id}/fra` | Snap 4D waypoints to ATS airways and 5-letter fixes to compute `FRA` directness. |
| `GET` | `/api/v1/fra/evolution` | Compare Free Route Airspace (`FRA`) directness shift across two archives. |

---

### 4. Fuel Burn, LTO Surface Emissions, Contrails & Eurocontrol CRCO (`€ EUR`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/fuel/prices` | Historical weekly Jet-A spot prices in Euros (`price_eur_per_gallon`, `price_eur_per_kg`). |
| `GET` | `/api/v1/flights/{ectrl_id}/fuel` | OpenAP excess fuel burn, $\text{CO}_2$, and weekly spot cost in Euros (`€`). Supports `engine_override` query parameter to evaluate specific engine variant efficiencies. |
| `GET` | `/api/v1/surface/taxi-out` | Airport `KPI02 ATOT` queue delays and ICAO EDB v32B LTO idle emissions. |
| `GET` | `/api/v1/flights/{ectrl_id}/taxi-out` | Single-flight surface taxi-out `ATOT` delay and 7% idle $\text{NO}_x / \text{CO} / \text{nvPM}$ emissions. |
| `GET` | `/api/v1/flights/{ectrl_id}/crco-cost` | Eurocontrol CRCO Service Units (`SU`), FIR fees (`€`), and Total Direct Operating Cost (`€`). |
| `GET` | `/api/v1/economics/route-avoidance` | Corridor CRCO Route-Charge vs. Extra Fuel Burn economic trade-off analysis (`€`). |
| `GET` | `/api/v1/flights/{ectrl_id}/contrails` | Upper-air wind optimality (`TAS` vs `GS`), Equivalent Still Air Distance (`esad_nm`), wind routing verdict, and persistent contrail `ISSR` forcing (`CO₂e`). |

#### Synthetic Example: `GET /api/v1/flights/900104215/fuel?period=202510`
```json
{
  "ectrl_id": 900104215,
  "ac_type": "B77W",
  "optimal_cruise_fl": 340.0,
  "requested_fl": 320.0,
  "actual_cruise_fl": 320.0,
  "excess_distance_nm": 24.0,
  "holding_duration_min": 4.5,
  "holding_fuel_kg": 315.0,
  "detour_fuel_kg": 288.0,
  "excess_altitude_fuel_kg": 412.5,
  "total_excess_fuel_kg": 1015.5,
  "excess_co2_kg": 3208.98,
  "fuel_price_eur_per_gallon": 1.954,
  "fuel_price_eur_per_kg": 0.6421,
  "fuel_price_week": "2025-W40 (2025-09-29 to 2025-10-05)",
  "estimated_excess_fuel_cost_eur": 652.05
}
```

---

### 5. Diagnostics, Safety & Prescriptive Operations Research

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/flights/{ectrl_id}/diagnostics` | Multi-factor delay root-cause attribution (tail rotation, holding, crosswinds, ceiling). |
| `GET` | `/api/v1/flights/{ectrl_id}/runway-safety` | Arrival runway braking action, density altitude, and stopping distance margin (`LDA`). |
| `GET` | `/api/v1/sim/tail-swaps` | Prescriptive hub tail-swap recovery optimizer (`Tail A ⇄ Tail B`, `€ EUR` savings). |
| `GET` | `/api/v1/stress-test` | Multi-shock airline fleet delay contagion and fragility threshold analysis. |
| `GET` | `/api/v1/twins/compare` | Benchmark most vs. least efficient twin flights on the same city pair and aircraft type. |
| `GET` | `/api/v1/periods/compare` | Macro traffic growth, delay drift, and fleet modernization across archives. |
| `GET` | `/api/v1/airac/diff` | Diff airway segments and FIR boundaries across contiguous AIRAC cycles. |
