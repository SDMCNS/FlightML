# Architecture & Methodology 🏛️

This document outlines the system architecture, mathematical methodologies, international aviation KPI standards (**ICAO GANP**, **Eurocontrol PRU**, **Eurocontrol CRCO**), physics models (**OpenAP**, **ICAO EDB v32B**, **Schmidt-Appleman ISSR**), and columnar data designs powering **FlightML**.

---

> [!IMPORTANT]
> ### 🔒 Data Access & Eurocontrol OneSky Online Compliance Notice
> **Eurocontrol Research Data Access Prerequisite**:
> The underlying aeronautical traffic, flight plans, and 4D radar trajectory archives supported by FlightML (such as Eurocontrol DDR2 / PRISME / R-NEST datasets) are **proprietary and not for public consumption or redistribution**.
>
> **Synthetic Example Outputs Policy**:
> All sample output tables, flight identifiers (e.g., `900104215`), and aircraft tail registrations (e.g., `G-SYNA`) shown in this documentation are realistic synthetic records generated to illustrate table schemas and formulas without exposing proprietary archive rows.

---

## 🏗️ High-Level System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                PRESENTATION & APIS                                     │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ Streamlit Dashboard      │ FastAPI REST Server      │ Typer / Rich CLI                 │
│ (4 Categorized Tiers)    │ (Swagger / OpenAPI)      │ (45+ Analytical Commands)        │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
                                    ▲
                                    │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              ANALYTICAL & SIMULATION LAYER                             │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ • 5°×5° Saturation Maps  │ • OpenAP Engine Fuel & CO₂│ • Hub Tail-Swap Optimizer (MIP) │
│ • ICAO GANP & PRU KPIs   │ • 3-Way Cruise FL Model  │ • Ration-by-Schedule GDP Slots   │
│ • Multi-Tier KPI Suite   │ • ICAO EDB LTO Emissions │ • Turnaround & Shock Contagion   │
│ • FRA & PBN Navigation   │ • Eurocontrol CRCO (€)   │ • Runway Braking & Crosswinds    │
│ • 4D Conflict Scanner    │ • Upper Winds & ESAD     │ • Contrail ISSR Radiative Multi. │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
                                    ▲
                                    │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                COLUMNAR STORAGE ENGINE                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Embedded DuckDB Engine  •  Thread-Safe Connection Cursors  •  Apache Parquet Tables    │
│   flights  •  trajectories  •  firs  •  auas  •  openap  •  icao_edb  •  crco_rates    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Columnar Storage & Thread-Safe DuckDB Cursors

Monthly Eurocontrol archives contain hundreds of thousands of flights and tens of millions of 4D trajectory points. FlightML stores all ingested tables in **Apache Parquet** and queries them via **DuckDB** using thread-safe isolated cursors (`self.con.cursor()`) so concurrent Streamlit dashboard sessions and FastAPI workers never contend on connection locks.

---

## 📏 ICAO GANP & Eurocontrol PRU Indicators

| KPI Code | Indicator Name | Mathematical Formulation |
| :--- | :--- | :--- |
| **KPI02 (`ATOT`)** | **Additional Taxi-Out Time** | $\text{ATOT} = \max(0,\; T_{\text{taxi, actual}} - \text{UTOT}_{p20})$, where $\text{UTOT}_{p20}$ is the 20th-percentile unimpeded taxi-out baseline for that `(airport, wake category)` pair. |
| **KPI04 (`KEP`)** | **Horizontal En-Route Efficiency (Filed)** | $\text{KEP (\%)} = \left(\frac{L_{\text{filed, >40nm}} - D_{\text{GCD, >40nm}}}{D_{\text{GCD, >40nm}}}\right) \times 100$ outside the 40 nm origin/destination cylinders. |
| **KPI05 (`KEA`)** | **Horizontal En-Route Efficiency (Actual)** | $\text{KEA (\%)} = \left(\frac{L_{\text{actual, >40nm}} - D_{\text{GCD, >40nm}}}{D_{\text{GCD, >40nm}}}\right) \times 100$. Tactical attribution is $\Delta_{\text{ATC}} = \text{KEA} - \text{KEP}$. |
| **KPI08 (`ASMA`)** | **Additional Terminal Airspace Time (40 nm)** | $\text{ASMA}_{\text{add}} = \max(0,\; T_{\text{40nm}\to\text{ALDT}} - T_{\text{unimpeded, p20}})$. |
| **KPI17 (`CCO`)** | **Continuous Climb Operations** | Detects intermediate level-offs ($\lvert\text{ROCD}\rvert < 200\text{ ft/min}$ for $\ge 60\text{ s}$) below Top of Climb (`TOC`). |
| **KPI19 (`CDO` & Level-Cap)** | **Continuous Descent & Cruise Level-Capping** | Detects descent level-offs below Top of Descent (`TOD`) and flags flights capped $\ge 20\text{ FL}$ ($2,000\text{ ft}$) below their filed requested flight level ($\text{FL}_{\text{req}}$). |

---

## 📊 Multi-Tier KPI Aggregation Framework

FlightML aggregates flight-level performance across four operational tiers with flexible date-windowing:
1. **Flight Tier**: Granular per-flight metrics (`KEP`, `KEA`, `ATOT`, `CDO/CCO`, CRCO route charges, and fuel burn).
2. **Fleet / Airline Tier**: Fleet-wide averages filtered by airline operator (`ac_operator`), wake category, or aircraft family.
3. **Airport Tier**: Hub-specific departure taxi-out queues (`KPI02 ATOT`), arrival terminal sequencing (`KPI08 ASMA`), and runway crosswind disruption.
4. **Network Tier**: Pan-European macro indicators, FIR sector saturation, and total excess carbon across the airspace.

---

## ✈️ 3-Way Cruise Altitude Optimization Physics

Fuel burn is strongly coupled to cruise altitude and engine thermodynamic efficiency:
- **Optimal Aerodynamic Cruise Flight Level ($\text{FL}_{\text{opt}}$)**: Resolved per airframe MTOW and turbofan engine type (bypass ratio, overall pressure ratio).
- **Altitude Burn Penalty ($\Delta \text{Burn}_{\%}$)**:
  $$\Delta \text{Burn}_{\%} = 0.015 \times \left(\frac{\Delta \text{FL}}{10}\right)^2 + 0.008 \times \left(\frac{\Delta \text{FL}}{10}\right)$$
- **Disaggregated Inefficiency Attribution**:
  1. **Flight Planning Inefficiency**: $\Delta \text{FL}_{\text{plan}} = \text{FL}_{\text{req}} - \text{FL}_{\text{opt}}$ (airline dispatch choice).
  2. **Tactical ATC Inefficiency**: $\Delta \text{FL}_{\text{ATC}} = \text{FL}_{\text{act}} - \text{FL}_{\text{req}}$ (tactical level capping by air traffic control).

---

## 🌬️ Upper-Air Winds & Equivalent Still Air Distance (ESAD)

True fuel efficiency cannot be judged by ground distance alone when strong jet stream winds exist:
- **Equivalent Still Air Distance ($\text{ESAD}$)**:
  $$\text{ESAD} = D_{\text{ground}} \times \left(\frac{\text{TAS}}{\text{GS}}\right) = D_{\text{ground}} \times \left(\frac{\text{TAS}}{\text{TAS} + W_{\text{tailwind}}}\right)$$
- **Local-First Caching**: Upper-air wind vectors are fetched on demand from the Open-Meteo ERA5 / GFS API and cached locally in SQLite for sub-millisecond retrieval.
- **Wind Routing Efficiency Verdict**:
  - `SMART_WIND_RIDING`: Airline successfully leveraged tailwinds ($\text{ESAD} < D_{\text{ground}}$).
  - `SEVERE_HEADWIND_PENALTY`: Flight incurred aerodynamic resistance ($\text{ESAD} > D_{\text{ground}}$).

---

## 🧭 Performance-Based Navigation (PBN): RNAV vs. RNP

FlightML evaluates navigation compliance and lateral waypoint adherence across PBN specifications:
- **RNAV Specifications (RNAV 10, RNAV 5, RNAV 1/2)**: Area navigation without mandatory on-board performance monitoring and alerting (OPMA).
- **RNP Specifications (RNP 4, RNP 2, RNP 1, RNP APCH, RNP AR APCH)**: On-board performance monitoring and alerting required. RNP AR APCH enables curved radius-to-fix (`RF`) legs with lateral containment as tight as $\pm 0.1\text{ nm}$.

---

## 🌱 OpenAP Fuel Burn, ICAO EDB v32B LTO Emissions & Contrail Physics

1. **Aerodynamic Fuel Burn & Carbon (`OpenAP`)**: Maps aircraft types to OpenAP mass, drag polar, and thrust-specific fuel consumption profiles ($3.16\text{ kg CO}_2$ per $\text{kg Jet-A}$).
2. **Surface Taxi-Out LTO Idle Emissions (`ICAO EDB v32B`)**: Applies empirical 7% LTO idle fuel flow and emission indices ($\text{NO}_x$, $\text{CO}$, $\text{HC}$, and $\text{nvPM}$ soot mass) from the ICAO Aircraft Engine Emissions Databank v32B during excess `KPI02 ATOT` departure queues.
3. **Upper-Air Winds & Persistent Contrail (`ISSR`) Forcing**: Cross-references 4D cruise segments (`FL280`–`FL430`) against upper-air pressure levels and the **Schmidt-Appleman criterion** ($T < -40^\circ\text{C}$, $\text{RH}_{\text{ice}} \ge 100\%$) to identify persistent contrail segments and radiative forcing multipliers (`CO₂e`).

---

## 💶 Eurocontrol CRCO Route Charges & Fuel Economics (`€ EUR`)

En-route navigation charges across European Flight Information Regions are billed in **Euros (`€`)** using the official Eurocontrol Central Route Charges Office (CRCO) formula:
$$\text{CRCO Charge (€)} = \sum_{i} \left(\frac{D_{i,\text{km}}}{100}\right) \times \sqrt{\frac{\text{MTOW (tons)}}{50}} \times u_i$$
where $D_{i,\text{km}}$ is the great-circle entry-to-exit distance in charging zone $i$ and $u_i$ is the national unit rate in `€/SU`.
- **Fuel Pricing Benchmark**: Weekly Jet-A fuel costs are resolved from historical EIA/IATA spot prices converted into **Euros (`€`)**, with contextual benchmarking against modern into-plane airline rates (~€0.85/kg).

---

## 🔮 Prescriptive Operations Research & Behavioral Economics Models

- **Network-Wide Hub Tail-Swap Optimizer**: Evaluates same-operator, same-aircraft-type turnaround pairs (`Delayed Tail A ⇄ Donor Tail B`) on the ground simultaneously at the same hub and recommends conflict-free swaps that minimize total reactionary delay minutes and Euro (`€88/min`) delay costs.
- **Ration-by-Schedule Ground Delay Program (`GDP`) Optimizer**: Allocates arrival slots under an Airport Arrival Rate (`AAR`) cap to shift high-burn airborne terminal holding stacks onto the departure ground at 7% idle/gate burn. Supports exact date windowing dynamically bounded to dataset dates (`get_dataset_date_bounds()`).
- **Contemporary YoY Cohort Filtering**: Enforces strict seasonal pair matching across multi-year archives (e.g. Winter vs Winter, Summer vs Summer, Autumn vs Autumn) to eliminate seasonal leisure distortions and isolates distant historical baselines as independent reference anchors rather than distorting short-term CAGR growth trajectories.
- **Airspace Sector Capacity Breach Forecaster**: Blends compound secular traffic growth scenarios with historical sector elasticity to project peak hourly flight counts, classifying sectors into `NOMINAL`, `ELEVATED`, `SEVERE`, and `BREACH` regimes.
- **Airline Behavioral Economics & Total Direct Operating Cost (DOC)**:
  $$\text{DOC} = C_{\text{fuel}} \cdot m_{\text{fuel}} + C_{\text{time}} \cdot t_{\text{flight}} + \sum_{k} \left( \frac{d_k}{100} \cdot \sqrt{\frac{\text{MTOW}}{50}} \cdot U_k \right)$$
  Calculates the avoidance incentive ratio (€ saved per extra nautical mile flown) to model airline decisions to dogleg around expensive national FIRs.
- **Delay Contagion Graph & Repeat Offender Scanner**:
  Reconstructs tail-specific flight rotations $(L_1 \to \Delta_{\text{turn}} \to L_2 \to \dots \to L_n)$ in DuckDB windowed queries. Identifies single-date super-spreader root flights and period-wide chronic repeat offender routes and operators that systematically export delays across the European network (€105/min standard delay cost).

---

## 🧪 Deep Learning, Neural Surrogates & Multi-Agent Simulation (v4 AI Lab)

FlightML features an experimental machine learning and operations research suite designed to solve high-dimensional aviation problems beyond conventional empirical lookups:

### 1. Flight2Vec Trajectory Foundation Embeddings
- **Architecture**: A self-supervised projection neural network mapping 16 kinematic and operational features (actual distance, departure/arrival delays, requested and cruise flight levels, climb energy, airframe and operator hashes, diurnal departure hour, route charge avoidance indicator, heavy wake class, and tactical reroute flags) into a dense 128-dimensional unit hypersphere $\mathbb{S}^{127}$.
- **Seeded Determinism**: Uses fixed random seeds (`torch.manual_seed(42)`) and 32-bit CRC32 deterministic hashing to guarantee identical embedding coordinates across Python process restarts.
- **Operational Aircraft Taxonomy Filtering**: Categorizes aircraft into strict operational cohorts (`COMMERCIAL_JET_NARROWBODY`, `COMMERCIAL_JET_WIDEBODY`, `REGIONAL_TURBOPROP`, `BUSINESS_JET`, `HELICOPTER`, `GENERAL_AVIATION`). Twin search filters out incompatible airframe categories (e.g. preventing an S-92 helicopter from matching an A320 commercial jet).
- **High-Speed Vector Cache**: Pre-vectorizes flights in batches into an on-disk binary cache (`_v2`), enabling sub-millisecond similarity scans over thousands of candidate flights.
- **Explainable Twin Similarity**: Generates human-readable explanations detailing why two flights are operational twins (range delta %, airframe drag polar, RVSM cruise band alignment, punctuality correlation, operator fleet policy).

### 2. Physics-Informed Neural Network (PINN) for Aircraft Mass Inversion
- **Aerodynamic Conservation Constraint**: Inverts unknown Takeoff Gross Weight (TGW) and commercial payload directly from initial climb radar points (below FL220) by embedding the longitudinal equation of motion:
  $$T(M, h) - D(C_L, C_D, m) - mg \sin \gamma = m \frac{dV}{dt}$$
- **Output Metrics**: Inferred Takeoff Gross Weight (kg), Operating Empty Weight (OEW), commercial payload (kg), Passenger Load Factor (PLF %), climb rate (fpm), acceleration ($\text{m/s}^2$), and physics residual error.

### 3. Microsecond Contrail Neural Surrogate Engine
- **Thermodynamic Basis**: Enforces the Schmidt-Appleman criterion ($T \le -39.5^\circ\text{C}$) and Ice Super-Saturated Region (ISSR) condition ($\text{RH}_{\text{ice}} \ge 100\%$).
- **Sub-Millisecond Inference**: Evaluates non-linear optical depth $\tau$ and instantaneous radiative forcing ($\text{mW/m}^2$) in $<15\,\mu\text{s}$ per waypoint, facilitating dynamic en-route re-clearing.

### 4. Spatio-Temporal Graph WaveNet (ST-GNN) Delay Contagion & Runway Saturation
- **Network Graph**: Models European airspace as a directed weighted graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, W)$ with nodes representing major hubs (EGLL, LFPG, EHAM, EDDF, LEMD, LIRF, etc.) and high-density FIR sectors (EDUUUIR, LFFFFIR, EGTTFIR, LSASFIR).
- **Graph Diffusion**: Combines spatial graph convolution $A \cdot X$ with temporal gated causal projections to forecast delay shockwave propagation over multi-horizon lookaheads ($t = 0\text{h}, 1\text{h}, 2\text{h}, 4\text{h}, 6\text{h}$).
- **Physical Runway Capacity Limits & Point-Queue Bottlenecks**: Incorporates hourly physical runway movement throughput limits ($C_i$). Computes empirical runway saturation $U = \text{Movements} / C_i$ and spare recovery capacity buffer ($C_i - \text{Movements}$). For saturated hubs like Heathrow ($U \approx 99\%$, spare buffer $\approx 1.0\text{ mvts/hr}$), departures and arrivals cannot outstrip runway limits, causing point-queue delay to persist across multi-hour horizons ($t=4\text{h}$ and $t=6\text{h}$).
- **Rotational Reactionary Delay Contagion**: Models physical aircraft tail rotations (`ac_reg`), tracking inbound delays exceeding turnaround buffers and transferring reactionary delay downstream to subsequent flights.
- **Temporal Event Controls**: Accepts explicit shock date, UTC start hour, and disruption duration.

### 5. 4D Trajectory Counterfactual & FIR Closure Simulator
- **Airspace Disruption Model**: Simulates the temporary closure of a Flight Information Region (e.g. `LSASFIR`, `EDUUUIR`, `LFFFFIR`) with user-specified closure date, UTC start time, and duration.
- **Detour Diversion Corridors**: Automatically identifies transiting flights and maps them into adjacent bypass FIRs, computing extra track miles, additional cruise fuel burn, $\text{CO}_2$ emissions, and airline direct costs (`€ EUR`).
- **Post-Reopening Recovery Dynamics**: Solves non-linear queue drainage when the closed sector reopens, modeling backlog flight decay against surge sector throughput:
  $$T_{\text{recovery}} = \frac{\text{Backlog Flights}}{\mu_{\text{surge}} - \mu_{\text{nominal}}}$$

### 6. Multi-Agent Reinforcement Learning (MARL) ATFM Slot Gaming & Nash Equilibrium
- **Decentralized Game Theory**: Models strategic competition between heterogeneous airline archetypes under ATFM arrival rate restrictions:
  - *Network Legacy Carrier*: Prioritizes connecting passenger value, swaps slots to protect long-haul arrivals, and speeds up flights at higher fuel burn.
  - *Low-Cost Carrier (LCC)*: Prioritizes fuel conservation and turnaround punctuality, absorbs ground delay at gate with engines off, and engages in bilateral trades for compensation credits.
  - *Hybrid European Carrier*: Balances slot swaps on corporate business routes with strategic flight cancellations.
  - *Express Cargo Operator*: Tactically defers arrival slots to off-peak daytime windows with zero passenger impact, earning peak-period fee credits.
- **Infilled Domain Assumptions**: Raw Eurocontrol DDR2 radar tracks lack passenger manifests and crew rosters; airline agents infill operational parameters:
  - *Connecting Passengers*: 35%–50% on long-haul widebodies (~80–140 pax), 20%–35% on European feeders (~25–50 pax), 0%–4% on LCCs.
  - *EU261 Compensation Exposure*: Delays $\ge 180\text{ min}$ incur €250 (<1,500 km), €400 (1,500–3,500 km), or €600 (>3,500 km).
  - *EASA Crew Duty Limits*: 35–55 min remaining FDP buffer on tail-end long-haul rotations (critical timeout risk).
  - *Missed Connection Care*: €220 per missed connecting passenger.
  - *Speedup Fuel Burn*: CI 30 $\to$ CI 80 burns ~280–340 kg Jet-A at €0.85/kg (~€240–€290 EUR).
- **Blow-by-Blow Autonomous Decision Log**: 10-step chronological trace of autonomous decisions (regulation issuance, assumption infilling, prioritization, tactical cancellation, intra-fleet swaps, bilateral slot trading, speedup, gate hold, slot deferral, and Nash convergence).
- **Financial Balance Sheet**: Rigorously computes baseline unmitigated costs vs post-gaming costs, proving net economic payoff ($\text{Baseline Cost} - \text{Post-Gaming Cost} = \text{Net Economic Payoff}$) in Euros (`€ EUR`).

### 7. Model Context Protocol (MCP) Server Architecture
- **FastMCP Protocol Bridge**: Implements the Model Context Protocol over standard input/output (`stdio`) and Server-Sent Events (`SSE`), enabling any local LLM runner (LM Studio, Claude Desktop, Cursor) to invoke FlightML engines as structured tools.
- **Tool Suite**: Exposes 9 tools (`get_flight_details`, `search_flights`, `find_operational_twins`, `simulate_delay_contagion`, `simulate_fir_closure`, `simulate_marl_slot_gaming`, `evaluate_contrail_climate_forcing`, `get_airport_capacity`, `assess_passenger_itinerary`).
- **Dynamic Resources & Prompts**: Exposes airport capacity profiles (`flightml://airports/{icao}/capacity`), MARL assumptions glossary (`flightml://assumptions/marl_glossary`), and pre-engineered disruption briefing prompts.

### 8. Passenger Journey & Public Travel Delay Risk Architecture
- **Intelligent Period Auto-Selection**: Evaluates target flight dates against available Eurocontrol monthly archives (`202402`, `202406`, `202410`, `202502`, `202506`, `202510`, `201611`). Calculates circular seasonal distance $\Delta M = \min(|M_T - M_P|, 12 - |M_T - M_P|)$, year distance $|Y_T - Y_P|$, and penalizes pre-COVID archives (>5 years old) when modern 2024/2025 archives exist:
  $$\text{Score} = (\Delta M \times 10) + |Y_T - Y_P| + \text{LegacyPenalty}$$
- **Empirical Flight Sampling & Commercial Filtering**: Filters out high-frequency overnight cargo haulers (`BCS` DHL, `FDX`, `UPS`, `CLX`, `BOX`, `GEC`) and private jet operators (`NJE`, `AHO`, `FYG`, `VJT`) to guarantee punctuality distributions represent scheduled commercial passenger operations.
- **Meteorological NWP Coupling**: Interrogates Open-Meteo live numerical weather prediction models (ECMWF / DWD ICON Ensemble) at airport coordinates, extracting temperature, wind speed, gusts, and precipitation probabilities. Projects wind vectors onto runway geometry to calculate primary crosswind components.
- **Transfer Hub Minimum Connection Time (MCT) Buffer Dynamics**: Implements a published European hub MCT database (`EGLL`, `EHAM`, `EDDF`, `LFPG`, `EBBR`, `LEMD`, `EDDM`, `EIDW`, `EINN`). Dynamically distinguishes between same-terminal transfers (e.g. Heathrow T5 $\to$ T5) and inter-terminal airside shuttle transfers (e.g. T2 $\to$ T5). Enforces strict airline biometric flight connection gate conformance cutoffs (20 minutes before departure), accounting for weather-induced tarmac transit delays, arrival holding delays, and luggage sortation misconnect probabilities.
- **Data Provenance & Diagnostic Audit Framework**: When `--debug` is requested, outputs a full accounting trail detailing DuckDB SQL queries, sample flight records, distribution quantiles, NWP model endpoints, and buffer calculations.

### 9. Multimodal Aeronautical Copilot Architecture (Epic 6 - Specification)
- **Document Text Corpus**: Designed to ingest Eurocontrol Route Availability Document (RAD), national Aeronautical Information Publications (AIPs), active NOTAMs, and METAR/TAF weather briefings into a vector database with hybrid BM25 + dense neural retrieval.
- **Implementation Status**: Architectural specification defined; implementation deferred pending authoritative raw document text pipelines.

