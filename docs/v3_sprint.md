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

### Epic 1: Multi-Period Longitudinal & YoY Comparative Engine
**Goal**: Build automated analytical modules in `flightml/eval/longitudinal.py` that quantify trends across multiple periods.

- [ ] **Task 1.1: Automated Seasonal Decomposition**:
  - Implement classical three-component time-series decomposition (Trend, Seasonal, Residual) across recurring snapshots.
  - Establish seasonal scaling factors: quantify the regular ~46% peak-to-trough surge between February and June.
- [ ] **Task 1.2: Matched Year-Over-Year (YoY) Differential Metrics**:
  - Pairwise comparative pipelines: `202402` vs `202502`, `202406` vs `202506`, `202410` vs `202510`.
  - Calculate key performance deltas:
    - Traffic volume growth rate ($\% \Delta$).
    - Route extension inefficiency delta ($\Delta$ KPI19 actual vs planned distance).
    - Average en-route flight duration and taxi-out duration changes.
    - Total network fuel burn and CO₂ emissions intensity per flight-km.
- [ ] **Task 1.3: Fleet Modernization & Engine Transition Tracker**:
  - Track market share shift of new-generation airframes (e.g. A320neo, A321neo, B737 MAX, A350, B787) vs legacy types (A320ceo, B737NG, B777-200ER).
  - Compute empirical fuel burn reduction achieved by fleet renewal on identical city-pairs (e.g. EGLL ➔ LFPG, EHAM ➔ LEMD).

---

### Epic 2: High-Resolution GRIB2 Atmospheric Integration & Contrail Forcing
**Goal**: Integrate numerical weather prediction (NWP) reanalysis grids into the trajectory evaluation pipeline.

- [ ] **Task 2.1: GRIB/GRIB2 Atmospheric Decoder**:
  - Build `flightml/weather/grib_loader.py` leveraging `cfgrib` / `eccodes` or lightweight raster processing.
  - Ingest October 2025 100m wind fields (`data/202510.grib`) and upper-air pressure levels (FL300–FL400 $u, v$ wind vectors, temperature $T$, relative humidity $RH$).
- [ ] **Task 2.2: 4D Trajectory Wind Integration & ESAD**:
  - Interpolate 4D radar trajectory points $(t, \text{lat}, \text{lon}, \text{alt})$ onto atmospheric wind grids.
  - Compute instantaneous ground speed vs true airspeed (TAS) and compute **Equivalent Still Air Distance (ESAD)**:
    $$\text{ESAD} = \int \frac{\text{TAS}}{\text{TAS} + V_{\text{tail}}} \, ds$$
  - Quantify wind-induced fuel penalties and tailwind benefits on transatlantic and trans-European routes.
- [ ] **Task 2.3: Schmidt-Appleman Criterion & Persistent Contrail Radiative Forcing**:
  - Implement the Schmidt-Appleman criterion using cruise-level temperature and water vapor saturation.
  - Calculate Ice-Supersaturated Region (ISSR, $RH_i \ge 100\%$) penetration for each flight.
  - Output persistent contrail warming potential (RF, $\text{mW/m}^2$) alongside engine CO₂ to assess the net climate trade-off of vertical avoidance.

---

### Epic 3: Predictive Airspace Saturation & Bottleneck Forecasting
**Goal**: Model and forecast emerging capacity bottlenecks across European airspace before peak traffic periods.

- [ ] **Task 3.1: Variable-Resolution Spatial Grid Engine**:
  - Extend the spatial grid generator to dynamically support resolutions from $5.0^\circ \times 5.0^\circ$ down to $0.5^\circ \times 0.5^\circ$.
  - Optimize spatial binning using DuckDB vector operations for sub-second query response.
- [ ] **Task 3.2: Airspace Sector Growth Projection**:
  - Train spatio-temporal autoregressive models on historical 2024 ➔ 2025 seasonal growth rates.
  - Project forward traffic densities for upcoming seasons (e.g. Summer 2026 under $+2.5\%$ to $+4.0\%$ growth scenarios).
- [ ] **Task 3.3: Critical Sector Capacity Breach Predictor**:
  - Combine sector geometry with declared hourly capacity thresholds.
  - Flag sectors projected to exceed $90\%$ and $100\%$ saturation, identifying where ATFM regulations will bind and where Free Route Airspace (FRA) expansions are urgent.

---

### Epic 4: Airline Behavioral Economics & CRCO Route Charge Elasticity
**Goal**: Analyze and predict how airlines optimize trajectories to balance fuel costs against national en-route navigation charges.

- [ ] **Task 4.1: Cross-Border FIR Detour Detection**:
  - Identify city-pairs where actual trajectories systematically deviate from the Great Circle or minimum-wind route to avoid high-fee airspaces (e.g., German DFS, Swiss Skyguide).
- [ ] **Task 4.2: Total Direct Operating Cost (DOC) Optimization Engine**:
  - Model the complete multi-objective airline cost function:
    $$\text{DOC} = C_{\text{fuel}} \cdot m_{\text{fuel}} + C_{\text{time}} \cdot t_{\text{flight}} + \sum_{k} \left( d_k \cdot \sqrt{\frac{\text{MTOW}}{50}} \cdot U_k \right)$$
  - Quantify the economic indifference curve: how much lateral detour (extra fuel burn and CO₂) an airline is willing to fly to save $€100$ in CRCO charges.
- [ ] **Task 4.3: Policy Impact Assessment**:
  - Model the environmental cost (unnecessary CO₂ emissions) resulting from fragmented European unit rates vs a single harmonized Eurocontrol rate.

---

### Epic 5: Aircraft Tail-Tracking & Delay Contagion Graph
**Goal**: Track sequential aircraft rotations across the network to model delay propagation and systemic network vulnerability.

- [ ] **Task 5.1: Multi-Leg Airframe Rotation Reconstruction**:
  - Group flight records by aircraft tail registration / airframe identifier across each operating day.
  - Order chronological legs: Leg 1 $\to$ Turnaround $\to$ Leg 2 $\to$ Turnaround $\to$ Leg 3.
- [ ] **Task 5.2: Delay Infection & Recovery Graph**:
  - Construct a directed delay transmission graph.
  - Measure how ground turnarounds buffer or amplify inbound departure delays.
  - Identify "super-spreader" flights and choke hubs (e.g., morning delays at LHR/FRA/CDG spreading to 4+ subsequent destinations).
- [ ] **Task 5.3: Network Resilience & Disruption Simulator**:
  - Provide a "What-If" simulator: if a major airport runway or FIR experiences a 2-hour closure, how many total network flights and passengers are impacted?

---

### Epic 6: V3 Dashboard & REST API Evolution
**Goal**: Expose all longitudinal, predictive, and meteorological capabilities in the interactive UI and programmatic API.

- [ ] **Task 6.1: Dedicated "Longitudinal & Seasonal Trends" View**:
  - Add an interactive Streamlit page allowing multi-period selection and side-by-side YoY trendlines.
  - Visualizers: Seasonal traffic curves, fleet neo/MAX share progression, and network emissions intensity.
- [ ] **Task 6.2: Side-by-Side Period Comparison Tool**:
  - Direct comparative analyzer: select Period A (e.g. `202410`) and Period B (e.g. `202510`) to view differential heatmaps of airspace saturation and delay deltas.
- [ ] **Task 6.3: High-Resolution Grid Map Controls**:
  - Incorporate the $0.5^\circ$ to $5.0^\circ$ spatial grid resolution selector directly on the saturation map.
- [ ] **Task 6.4: FastAPI v3 Endpoints**:
  - `/api/v3/longitudinal/trends`: Historical and seasonal traffic/emissions time-series.
  - `/api/v3/longitudinal/compare`: Pairwise period delta comparison.
  - `/api/v3/predict/saturation`: Forward sector saturation projections.

---

## 📅 Immediate Next Steps

1. **Sprint Kickoff Implementation**:
   - Begin **Epic 1 (Task 1.1 & 1.2)**: Implement `flightml/eval/longitudinal.py` with multi-period comparative aggregators (YoY deltas, seasonal volume indices, and fleet share evolution).
   - Write comprehensive unit tests for longitudinal comparison in `tests/test_longitudinal.py`.
2. **Dashboard Integration**:
   - Add the "Longitudinal & Multi-Period Trends" tab in `flightml/dashboard/app.py`.
3. **Meteorological Pipeline**:
   - Begin **Epic 2 (Task 2.1)**: Build the GRIB2 wind field reader to decode `data/202510.grib` and couple with October 2025 flights.
