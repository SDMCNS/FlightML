# ATC Digital Twin: Pan-European 4D Trajectory Rebuild & Network Optimization Engine

## Overview

The **ATC Digital Twin** is an autonomous prescriptive airspace simulator within FlightML that reconstructs an **Ideal 4D Flight Trajectory** for commercial flights across the European Civil Aviation Conference (ECAC) airspace.

While Eurocontrol historical archives provide both pre-departure filed flight plans (**Planned**) and radar tracking logs (**Actual Flown**), historical flights routinely suffer from tactical ATC vectoring, military airspace doglegs, suboptimal cruise altitudes, airborne holding stacks, and reactionary delay contagion. The ATC Digital Twin simulates an omniscient, safety-first Air Traffic Control system that rebuilds flights in fully safe trajectories, minimizing delays, flight times, congestion, and environmental impact.

```
       [Filed Flight Plan (Planned)]
                     │
                     ▼
  [ATC Digital Twin 4D Rebuilder Engine] ◄─── [Airport Weather & Capacity Matrix]
                     │                         (>5 delays/hr = weather; <20 moves/hr = unconstrained)
                     ▼
       [Ideal 4D Flight Trajectory]
                     │
  ┌──────────────────┴──────────────────┐
  ▼                                     ▼
[4D Trajectory Map & Telemetry]      [Network-Wide Optimization Scorecard]
(Planned vs Actual vs Ideal)         (Delay, Fuel, CO2 & € EUR Savings)
```

---

## 1. Airport Weather & Capacity Classification Rules

To avoid external meteorological API dependencies and prevent fragile network lookups, local weather disruptions and airport throughput constraints are inferred directly from empirical flight delay clustering:

### Axiom 1: The Small Airport Capacity Axiom
$$\text{movements}(A, H) < 20 \implies C_{\text{eff}}(A, H) = \infty$$
* If an airport processes fewer than 20 total movements (departures + arrivals) during an hour, it has ample runway and terminal airspace capacity. Weather lookup is bypassed, and no capacity bottleneck is declared.

### Axiom 2: The Delay Cluster Weather Axiom
For airports with $\ge 20$ movements per hour:
$$\text{delayed\_flights}(A, H) > 5 \implies \text{Weather Impacted}$$
$$\text{delayed\_flights}(A, H) \le 5 \implies \text{Operational / Procedural / Turnaround Inefficiency}$$
* A flight is delayed if departure delay $\ge 15$ minutes or arrival delay $\ge 15$ minutes.
* **Weather-Impacted**: When $> 5$ flights suffer delays in the same hour, the disruption is environmental (severe convection, low visibility CAT III, gusting crosswinds). The ATC Digital Twin **respects** this real-world constraint by capping airport throughput to the observed acceptance rate:
  $$AAR_{\text{ideal}}(A, H) = \text{movements}_{\text{actual}}(A, H)$$
  Flights scheduled into these hours are metered smoothly via ground delay programs, converting high-burn airborne holding into zero-burn gate holds.
* **Non-Weather Inefficiency**: When $\le 5$ flights are delayed, the delay was caused by non-weather procedural friction, tactical vectoring, or turnaround buffer deficit. The ATC Digital Twin **resolves** these delays, restoring on-time pushback and direct routing.

---

## 2. Ideal Trajectory Reconstruction Physics

Each flight's ideal trajectory is reconstructed across four operational phases:

### Phase 1: Departure Scheduling & Gate Operations
* **Pushback**: Scheduled on-time ($t_{\text{ideal}} = t_{\text{sched}}$) unless the departure hub is in a weather-restricted hour, in which case departure is scheduled into the next available runway slot.
* **Taxi-Out**: Modeled with zero queue holding, saving $50-150\text{ kg}$ of ground idle fuel.

### Phase 2: Lateral Free Route Airspace (FRA) Geodesy
* Replaces legacy airway doglegs and zig-zags with an orthodromic **Great Circle Geodesic** from departure fix to arrival fix:
  $$s = 2 R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos \phi_1 \cos \phi_2 \sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
* Intermediate 4D coordinates are synthesized at regular intervals using spherical linear interpolation (Slerp), preserving continuous spatial fidelity.

### Phase 3: Vertical Profile (Continuous Climb & Descent)
* **Continuous Climb Operations (CCO)**: Uninterrupted climb to optimal cruising flight level ($FL_{\text{opt}}$) determined by aircraft category (e.g., A320/B737 at FL360–FL390; widebodies at FL390–FL410), respecting RVSM cruising directions (Eastbound Odd, Westbound Even).
* **Continuous Descent Operations (CDO)**: Unpowered idle descent directly from Top of Descent ($ToD$) to the runway threshold.
* **Airborne Holding Elimination**: Holding duration is reduced to $0.0\text{ min}$ by metering destination arrival times before descent.

### Phase 4: Strategic 4D Deconfliction & Separation Assurance
* All flights are cross-checked against a 4D spatial separation grid:
  $$\Delta d_{\text{horizontal}} \ge 5.0\text{ NM} \quad \text{or} \quad |\Delta z| \ge 1,000\text{ ft RVSM}$$
* Tactical deconfliction resolves crossing conflicts via altitude step assignments or minor speed control adjustments ($\pm 0.02\text{ Mach}$).

---

## 3. Multi-Objective Performance Accounting

For every flight, the ATC Digital Twin calculates the three-way differential:

$$\mathbf{Planned} \longleftrightarrow \mathbf{Actual\ Flown} \longleftrightarrow \mathbf{Ideal\ Digital\ Twin}$$

| Performance Metric | Formulation | Notes |
| :--- | :--- | :--- |
| **Air Time Saved** | $\Delta t_{\text{air}} = t_{\text{air, actual}} - t_{\text{air, ideal}}$ | Minutes saved by eliminating route extensions and holding |
| **Delay Prevented** | $\Delta t_{\text{delay}} = \max(0, \text{delay}_{\text{actual}} - \text{delay}_{\text{ideal}})$ | Total departure and arrival delay minutes eliminated |
| **Fuel Burn Saved** | $\Delta F = F_{\text{actual}} - F_{\text{ideal}}$ | Kinematic calculation via OpenAP WRAP / ICAO Engine Databank |
| **CO₂ Emissions Avoided** | $\Delta \text{CO}_2 = \Delta F \times 3.15$ | Standard ICAO Doc 9889 carbon factor |
| **Fuel Cost Saved (€)** | $\Delta C_{\text{fuel}} = \Delta F \times P_{\text{fuel, EIA}}(\text{period})$ | EIA weekly jet fuel spot price converted to EUR |
| **Passenger EU261 Savings** | Flights with delay $\ge 180\text{ min}$ eliminated $\times €350$ | Statutory passenger compensation avoided |
| **CRCO Route Cost Savings** | Eurocontrol CRCO formula delta | Reduction in en-route navigation distance |

---

## 4. CLI Execution: `flightml atc-twin`

The CLI provides long-run processing of Eurocontrol traffic cycles:

```bash
# Run ATC Digital Twin on peak summer period (202507) across all flights
flightml atc-twin --period 202507

# Run on a specific date with a batch limit for rapid inspection
flightml atc-twin --period 202507 --date 2025-07-15 --max-flights 500

# Inspect generated dataset output
flightml atc-twin --period 202507 --summary
```

### Generated Datasets
* `data/digital_twin/atc_twin_{period}.parquet`: Comprehensive flight-level dataset containing Planned, Actual, and Ideal metrics for each flight.
* `data/digital_twin/atc_twin_trajectories_{period}.parquet`: High-fidelity 4D waypoint coordinates for ideal flights.
* `data/digital_twin/atc_twin_summary_{period}.json`: Network-wide executive summary and airport weather capacity matrix.

---

## 5. Dashboard Visualizations

### 5.1 4D Trajectory & RNP Profile Tab (`Single Flight`)
* Toggle: **✨ Show ATC Digital Twin: Ideal Trajectory**.
* Displays the **Ideal 4D Trajectory (solid emerald green)** alongside the **Actual Flown (solid purple)** and **Filed Planned (dashed blue)** routes on the interactive Folium map.
* Side-by-side comparison table: Planned vs. Actual vs. Ideal for Distance, Flight Time, Delay, Fuel, CO₂, and Total Cost.

### 5.2 Network Level Tab (`Multi-Level KPI Suite`)
* Dedicated section: **🤖 ATC Digital Twin: Pan-European Airspace Rebuild & Optimization**.
* Executive KPI scorecards displaying total delay minutes saved, fuel tons saved, CO₂ tons avoided, and net economic savings.
* Interactive chart: **Airport Demand Smoothing & Weather Capacity Bounds**, highlighting hours where airports were weather-impacted ($>5$ delays/hr) versus non-weather inefficiencies.
