# FlightML v3 Evolution — Strategic Roadmap & Sprint Specification 🚀

> **Version**: 3.0.0-dev  
> **Status**: Active Working Specification  
> **Target Branch**: `v3` (V2 baseline secured on `v2` / `main`)  
> **Scope**: Multi-Period Longitudinal Intelligence, Predictive Airspace Modeling, High-Resolution Atmospheric Reanalysis, and Airline Behavioral Economics.

---

## 🧭 Executive Summary & V3 Vision

While **FlightML v2** established a high-performance aeronautical analytics engine capable of processing single-month European flight archives with exact physics (OpenAP fuel/CO₂, ICAO EDB LTO emissions, CRCO navigation charges, and FIR/AUA saturation), **FlightML v3** transforms the platform from a *retrospective snapshot analyzer* into a **longitudinal network intelligence and predictive foresight engine**.

With **7 full Eurocontrol archives** now ingested (>5.7M flights spanning `201611`, `202402`, `202406`, `202410`, `202502`, `202506`, `202510`), FlightML v3 leverages repeating seasonal snapshots (Winter Trough, Summer Peak, Autumn Shoulder) and controlled Year-over-Year (YoY) pairings to:
1. **Disentangle secular trends from seasonal weather & leisure oscillations**.
2. **Forecast airspace chokepoints and capacity breaches months in advance**.
3. **Decode GRIB2 upper-air winds to compute true Equivalent Still Air Distance (ESAD) and Schmidt-Appleman contrail climate forcing**.
4. **Model airline behavioral economics and strategic FIR charge avoidance (€)**.
5. **Simulate network-wide delay contagion and systemic resilience across sequential aircraft rotations**.

---

## 📊 Ingested Data Matrix (V3 Baseline)

The platform operates across an indexed columnar Parquet corpus:

| Period | Season / Archetype | Total Flights | YoY Growth (2024 ➔ 2025) | Analytical Significance |
| :--- | :--- | :---: | :---: | :--- |
| **`201611`** | Historical Baseline | 697,326 | — | Pre-COVID, pre-conflict airspace closures, legacy fleet baseline |
| **`202402`** | Winter Trough (Leap) | 641,952 | — | Structural demand floor, winter de-icing/storm delay regimes |
| **`202406`** | Summer Peak | 935,934 | — | Peak holiday surge (+45.8% over Feb), convective storm reroutes |
| **`202410`** | Autumn Shoulder | 891,648 | — | High business travel, transatlantic transition, calm weather baseline |
| **`202502`** | Winter Trough | 656,458 | **+2.26%** | Structural growth baseline under cold-front meteorology |
| **`202506`** | Summer Peak | 957,457 | **+2.30%** | All-time network load peak, maximum ATFM slot delays |
| **`202510`** | Autumn Shoulder | 934,141 | **+4.77%** | Strong late-season extension, long-haul fleet modernization |

---

## 🏗️ V3 Work Breakdown Structure (Epics & Tasks)

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                FLIGHTML V3 WORK STREAMS                                │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ Epic 1: Longitudinal &   │ Epic 2: GRIB2 Weather &  │ Epic 3: Predictive Airspace &    │
│ YoY Comparative Engine   │ Contrail Climate Forcing │ Capacity Forecasting             │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ Epic 4: Behavioral Route │ Epic 5: Aircraft Tail    │ Epic 6: V3 Dashboard Views &     │
│ Economics & CRCO Elastic.│ Delay Contagion Graph    │ REST API Extensions              │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

---

### Epic 1: Multi-Period Longitudinal & YoY Comparative Engine [COMPLETED]
**Goal**: Build automated analytical modules in `flightml/eval/longitudinal.py` that quantify trends across multiple periods.

- [x] **Task 1.1: Automated Seasonal Decomposition**:
  - Implemented classical three-season network decomposition (Winter Trough, Summer Peak, Autumn Shoulder) via `LongitudinalEngine.decompose_seasonality(year)`.
  - Establishes seasonal scaling factors: measured the +40.9% daily flight surge between February and June 2024, +220.1% delay surge, and a 5.38x delay-to-demand elasticity.
  - Detects top surging leisure resort airports (e.g. Ibiza +351.7%, Antalya +340.8%, Menorca +277.3%) and surging Mediterranean FIRs (e.g. Athens FIR +285.2%).
- [x] **Task 1.2: Matched Year-Over-Year (YoY) Differential Metrics**:
  - Pairwise comparative pipelines: `202402` vs `202502`, `202406` vs `202506`, `202410` vs `202510` via `LongitudinalEngine.compare_yoy(baseline, target)`.
  - Calculates key performance deltas:
    - Traffic volume growth rate (+4.77% in October 2024 ➔ 2025).
    - Mean arrival delay and on-time performance (OTP-15) deltas.
    - Flown distance and flight duration deltas.
    - Top growing airlines and aircraft fleet transition shifts.
- [x] **Task 1.3: Fleet Modernization & Engine Transition Tracker**:
  - Longitudinal fleet evolution tracking across all 7 archives via `LongitudinalEngine.get_fleet_evolution_timeline()`.
  - Tracks adoption rates of A320neo, A321neo, B737 MAX, A220, A350, and B787 vs classic CEO/NG types.
  - Measured European new-generation fleet penetration rising from 1.55% in 2016 to 24.64% in October 2024 and 28.17% in October 2025 (+3.53 percentage points in 12 months).
  - CLI integration: added `flightml longitudinal-timeline`, `flightml seasonality`, and `flightml compare-yoy`. Unit tests in `tests/test_longitudinal.py` (7/7 passing).

---

### Epic 2: High-Resolution GRIB2 Atmospheric Integration & Contrail Forcing [ON HOLD]
> [!NOTE]
> **Status: ON HOLD** (Pending acquisition of a reliable, continuous high-resolution GRIB data source). Work is deferred while Epic 1 and downstream epics proceed.

**Goal**: Integrate numerical weather prediction (NWP) reanalysis grids into the trajectory evaluation pipeline.

- [ ] **Task 2.1: GRIB/GRIB2 Atmospheric Decoder** *(Deferred)*:
  - Build `flightml/weather/grib_loader.py` leveraging `cfgrib` / `eccodes` or lightweight raster processing.
  - Ingest October 2025 100m wind fields (`data/202510.grib`) and upper-air pressure levels (FL300–FL400 $u, v$ wind vectors, temperature $T$, relative humidity $RH$).
- [ ] **Task 2.2: 4D Trajectory Wind Integration & ESAD** *(Deferred)*:
  - Interpolate 4D radar trajectory points $(t, \text{lat}, \text{lon}, \text{alt})$ onto atmospheric wind grids.
  - Compute instantaneous ground speed vs true airspeed (TAS) and compute **Equivalent Still Air Distance (ESAD)**:
    $$\text{ESAD} = \int \frac{\text{TAS}}{\text{TAS} + V_{\text{tail}}} \, ds$$
  - Quantify wind-induced fuel penalties and tailwind benefits on transatlantic and trans-European routes.
- [ ] **Task 2.3: Schmidt-Appleman Criterion & Persistent Contrail Radiative Forcing** *(Deferred)*:
  - Implement the Schmidt-Appleman criterion using cruise-level temperature and water vapor saturation.
  - Calculate Ice-Supersaturated Region (ISSR, $RH_i \ge 100\%$) penetration for each flight.
  - Output persistent contrail warming potential (RF, $\text{mW/m}^2$) alongside engine CO₂ to assess the net climate trade-off of vertical avoidance.

---

### Epic 3: Predictive Airspace Saturation & Bottleneck Forecasting [COMPLETED]
**Goal**: Model and forecast emerging capacity bottlenecks across European airspace before peak traffic periods.

- [x] **Task 3.1: Variable-Resolution Spatial Grid Engine**:
  - Implemented dynamic spatial grid densities via `AirspaceForecaster.compute_variable_grid_densities()` supporting variable resolutions from $5.0^\circ \times 5.0^\circ$ down to $0.5^\circ \times 0.5^\circ$.
  - Optimized multi-period spatial joins with DuckDB vector operations, achieving sub-second response times across 15M+ radar trajectory points.
- [x] **Task 3.2: Airspace Sector Growth Projection**:
  - Spatio-temporal forward projection modeling (`project_sector_growth()`): blends secular network growth scenarios (e.g. +3.5% for Summer 2026) with historical 2024 ➔ 2025 sector-specific elasticity.
  - Predicts future peak hourly flight rates and daily sector throughput across all European FIRs and UACs.
- [x] **Task 3.3: Critical Sector Capacity Breach Predictor**:
  - Evaluated against Eurocontrol sustained sector capacities; classifies sectors into `NOMINAL`, `ELEVATED`, `SEVERE`, and `BREACH` categories.
  - Computes non-linear ATFM delay minutes generated and formulates prescriptive operational mitigations (sector de-combining, Free Route Airspace mandate, military airspace TSA release).
  - Production CLI command `flightml forecast-airspace` added. Unit tests in `tests/test_airspace_forecaster.py` (4/4 passing). Full test suite (80/80 passing).

---

### Epic 4: Airline Behavioral Economics & CRCO Route Charge Elasticity [COMPLETED]
**Goal**: Analyze and predict how airlines optimize trajectories to balance fuel costs against national en-route navigation charges.

- [x] **Task 4.1: Cross-Border FIR Detour Detection**:
  - Implemented network-wide systematic corridor scanner `AirlineEconomicsEngine.scan_network_route_charge_avoidance()`.
  - Flags city pairs where airlines deliberately fly dogleg extensions to avoid expensive FIR charging zones (saving € in CRCO fees at the cost of additional fuel burn and CO₂).
- [x] **Task 4.2: Total Direct Operating Cost (DOC) Optimization Engine**:
  - Implemented full multi-objective airline cost function via `evaluate_flight_doc()`:
    $$\text{DOC} = C_{\text{fuel}} \cdot m_{\text{fuel}} + C_{\text{time}} \cdot t_{\text{flight}} + \sum_{k} \left( \frac{d_k}{100} \cdot \sqrt{\frac{\text{MTOW}}{50}} \cdot U_k \right)$$
  - Quantifies component allocations (Fuel % vs Crew/Time % vs CRCO Navigation %) and the marginal avoidance incentive ratio (€ saved per NM).
- [x] **Task 4.3: Policy Impact Assessment (Single European Sky Harmonized Pricing)**:
  - Built `simulate_harmonized_eurocontrol_rate()` modeling a unified European unit rate (e.g. €72.50/SU).
  - Simulates the elimination of ~635,000 detour flight legs, saving over 230,000 tonnes of Jet-A fuel and mitigating 730,000+ tonnes of CO₂ annually, returning direct traffic to German DFS and Swiss Skyguide airspaces.
  - Production CLI commands `flightml flight-doc`, `flightml scan-avoidance`, and `flightml simulate-harmonization` added. Unit tests in `tests/test_airline_economics.py` (3/3 passing).

---

### Epic 5: Aircraft Tail-Tracking & Delay Contagion Graph [COMPLETED]
**Goal**: Track sequential aircraft rotations across the network to model delay propagation and systemic network vulnerability.

- [x] **Task 5.1: Multi-Leg Airframe Rotation Reconstruction**:
  - Implemented `DelayContagionGraphEngine.reconstruct_airframe_rotations()` grouping flights by aircraft tail registration across operating days.
  - Reconstructs sequential rotation legs (Leg 1 $\to$ Turnaround Buffer $\to$ Leg 2 $\to$ Leg 3), measuring turnaround absorption capacity.
- [x] **Task 5.2: Delay Infection & Recovery Graph**:
  - Implemented `identify_super_spreaders()` tracking directed delay transmission along airframe chains.
  - Pinpoints morning departure delays that infect 2+ downstream rotations, calculating cumulative downstream delay hours and economic reactionary delay cost (€105/min standard).
- [x] **Task 5.3: Network Resilience & Disruption Simulator**:
  - Built `simulate_network_shock()` for airport runways and FIR airspace sectors.
  - Quantifies direct delayed flights, reactionary infected rotations, cumulative flight delay hours, passenger misconnections, and economic damage in € EUR.
  - Production CLI commands `flightml delay-spreaders` and `flightml simulate-shock` added. Unit tests in `tests/test_delay_contagion_graph.py` (4/4 passing). Full test suite (84/84 passing).

---

### Epic 6: V3 Dashboard & REST API Evolution [COMPLETED]
**Goal**: Expose all longitudinal, predictive, and meteorological capabilities in the interactive UI and programmatic API.

- [x] **Task 6.1: Dedicated "Longitudinal & Seasonal Trends" View**:
  - Implemented top-level category `"📈 Longitudinal & Predictive Trends"` in `flightml/dashboard/app.py`.
  - Sub-tab 1: `📊 Multi-Period Timeline & Fleet Evolution` with cross-sectional metric KPIs, tabular data across all 7 historical archives, and interactive line charts for daily traffic volume, neo/MAX eco-fleet share, arrival delay, and flown distance.
  - Sub-tab 2: `🔄 Year-over-Year (YoY) & Seasonality` with pairwise period delta benchmarking (daily volume, arrival delay, distance, CO₂, fleet penetration) and 3-season surge curves (Winter Feb vs Summer Jun vs Autumn Oct) identifying top leisure destination surges.
- [x] **Task 6.2: Side-by-Side Period Comparison Tool & Policy Economics**:
  - Built pairwise YoY comparative analyzer identifying top expanding and contracting airline carriers and aircraft models.
  - Sub-tab 4: `💶 Airline Economics & Delay Contagion` integrating Direct Operating Cost (DOC) breakdown, route charge avoidance detour detection, Single European Sky (SES) rate harmonization impact simulation, and airframe delay contagion super-spreader detection.
- [x] **Task 6.3: High-Resolution Grid Map & Bottleneck Forecasting Controls**:
  - Sub-tab 3: `🔮 Predictive Airspace Saturation Forecaster` with interactive sliders for forecast horizon year (2026-2028), season (summer/winter/autumn), compound growth rate (1.0%–8.0%), spatial grid resolution (0.5°, 1.0°, 2.0°, 5.0°), and sector capacity threshold.
  - Identifies forward sector capacity breaches and top congested spatial density hotspots.
- [x] **Task 6.4: FastAPI v3 Endpoints**:
  - Implemented REST API endpoints in `flightml/api/app.py`:
    - `GET /api/v3/longitudinal/timeline`: Multi-period historical metrics & fleet shares.
    - `GET /api/v3/longitudinal/compare`: Pairwise YoY delta benchmarking.
    - `GET /api/v3/longitudinal/seasonality`: 3-season traffic & delay surge decomposition.
    - `POST /api/v3/predict/airspace`: Sector capacity breach and spatial saturation forecasting.
    - `GET /api/v3/economics/flight-doc/{ectrl_id}`: Flight DOC component breakdown (€ EUR).
    - `GET /api/v3/economics/avoidance-corridors`: Detection of route charge avoidance detours.
    - `POST /api/v3/economics/simulate-harmonization`: European flat rate counterfactual simulation.
    - `GET /api/v3/delay/spreaders`: Airframe rotation delay super-spreader detection.
    - `POST /api/v3/sim/network-shock`: Runway and sector network resilience shock simulator.
  - Full automated test suite in `tests/test_api_v3.py` (8/8 passing).

---

## 🏆 V3 Evolution Summary & Status

| Epic | Description | Status | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Epic 1** | Multi-Period Longitudinal & YoY Comparative Engine | **COMPLETED** | `flightml/eval/longitudinal.py`, `tests/test_longitudinal.py`, CLI commands |
| **Epic 2** | High-Resolution GRIB2 Atmospheric Reanalysis | **ON HOLD** | Deferred per user instruction (continuous GRIB source unavailable) |
| **Epic 3** | Predictive Airspace Saturation & Bottleneck Forecaster | **COMPLETED** | `flightml/models/airspace_forecaster.py`, `tests/test_airspace_forecaster.py`, CLI |
| **Epic 4** | Airline Behavioral Economics & Route Charge Elasticity | **COMPLETED** | `flightml/eval/airline_economics.py`, `tests/test_airline_economics.py`, CLI |
| **Epic 5** | Aircraft Tail-Tracking & Delay Contagion Graph | **COMPLETED** | `flightml/models/delay_contagion_graph.py`, `tests/test_delay_contagion_graph.py`, CLI |
| **Epic 6** | V3 Dashboard & REST API Evolution | **COMPLETED** | `flightml/dashboard/app.py`, `flightml/api/app.py`, `tests/test_api_v3.py` |
