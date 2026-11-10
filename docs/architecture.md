# Architecture & Methodology 🏛️

This document outlines the system architecture, mathematical methodologies, physics models, and columnar data designs powering **FlightML**.

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

## 🏗️ High-Level System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                PRESENTATION & APIS                                     │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ Streamlit Dashboard      │ FastAPI REST Server      │ Typer / Rich CLI                 │
│ (9 Interactive Tabs)     │ (Swagger / OpenAPI)      │ (32 Analytical Commands)         │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
                                    ▲
                                    │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              ANALYTICAL & SIMULATION LAYER                             │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ 4D Trajectory & Holding  │ Fuel Burn & Emissions    │ Airline Turnaround Cascades      │
│ CDO Descent Profile      │ Weekly EIA Jet-A Pricing │ Ground Delay Program (GDP)       │
│ FIR Saturation & Dwell   │ IEM METAR Weather Fusion │ AIRAC Diff & Macro Evolution     │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
                                    ▲
                                    │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                COLUMNAR STORAGE ENGINE                                 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Embedded DuckDB Engine  •  Thread-Safe Connection Cursors  •  Apache Parquet Tables    │
│            flights.parquet  •  trajectories.parquet  •  firs.parquet                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Columnar Storage & Thread Safety

### Why DuckDB and Apache Parquet?

Aeronautical trajectory datasets are massive:
- Monthly Eurocontrol archives exceed hundreds of megabytes compressed and several gigabytes uncompressed.
- A single monthly archive contains nearly 1,000,000 flight records and millions of 4D radar trajectory points.

FlightML utilizes **Apache Parquet** with Snappy compression for disk persistence and embeds **DuckDB** for vectorized, zero-copy in-memory querying. This provides:
1. **Sub-second aggregations** over millions of rows without loading entire datasets into Python heap memory.
2. **Column projection & predicate pushdown**: Queries for a single flight ID only read the specific bytes necessary from disk.

### Thread-Safe Cursor Architecture

DuckDB connection handles are single-threaded for query execution. When running Streamlit dashboards with concurrent user sessions or multithreaded evaluation scripts, querying a shared connection directly triggers `_duckdb.ResourceInUseException`.

FlightML solves this by implementing an isolated cursor pattern in [`FlightDataset`](file:///C:/Projects/data/FlightML/flightml/data/dataset.py):

```python
@property
def cursor(self) -> duckdb.DuckDBPyConnection:
    """Spawn an isolated DuckDB cursor sharing memory buffers safely across threads."""
    return self.con.cursor()
```

Every read operation in `FlightDataset` (`get_flight`, `search_flights`, `get_trajectory`, `get_top_airports`) executes through `self.cursor.execute(...)`.

---

## 📐 4D Trajectory & Holding Loop Analytics

### Spatial Geometry & Path Extension

For any flight with trajectory waypoints $P_1, P_2, \dots, P_n$ where $P_i = (\text{lat}_i, \text{lon}_i, \text{alt}_i, t_i)$:

1. **Actual Flown Distance ($D_{\text{actual}}$)**: Sum of Haversine segment distances:
   $$D_{\text{actual}} = \sum_{i=1}^{n-1} d_{\text{haversine}}(P_i, P_{i+1})$$

2. **Great Circle Distance ($D_{\text{gcd}}$)**: Orthodromic distance from $P_1$ to $P_n$.

3. **Inefficiency / Detour Ratio ($R_{\text{detour}}$)**:
   $$R_{\text{detour}} = \frac{D_{\text{actual}}}{D_{\text{gcd}}}$$

### Holding Loop Detection

Aircraft placed into terminal holding patterns execute standardized 360° loops or racetrack circuits. FlightML detects holdings by:
1. Calculating instantaneous bearing changes $\Delta \theta_i$ between successive track vectors.
2. Tracking cumulative angular heading changes $\Theta = \sum |\Delta \theta_i|$ within rolling time windows ($t \in [t_0, t_0 + \Delta T]$).
3. If $\Theta \ge 300^\circ$ and the spatial bounding box remains constrained within $15\text{ nm} \times 15\text{ nm}$, the sequence is flagged as an active holding stack.

---

## 📉 Continuous Descent Operations (CDO)

In an optimal Continuous Descent Operation (CDO), an aircraft descends continuously from its cruising flight level (Top of Descent, TOD) to the runway threshold at minimum engine thrust:

- **Stepped Descents (Non-CDO)**: Air Traffic Control (ATC) frequently stops descents at intermediate flight levels (e.g. FL150, FL100, FL070) for sector coordination.
- **Level-off Detection**: FlightML filters vertical speed $V_z = \frac{\Delta \text{Altitude}}{\Delta t}$. Any period where $|V_z| < 250\text{ ft/min}$ lasting longer than $1.5\text{ minutes}$ during the descent phase is classified as a level-off.
- **CDO Compliance Score**:
  $$\text{Score}_{\text{CDO}} = \max\left(0, 100 - (\text{Level-off Count} \times 15) - (\text{Level-off Duration (min)} \times 2)\right)$$

---

## ⛽ Fuel Burn Physics & Weekly Spot Pricing

### Aerodynamic Fuel Burn Formulation

FlightML models fuel consumption across flight phases using BADA (Base of Aircraft Data) aerodynamic mass and drag principles:

1. **Cruise Phase**:
   $$\dot{m}_{\text{fuel, cruise}} = C_{\text{cr}} \times \left(\frac{W}{\text{L/D}}\right) \times \text{TSFC}$$
   Where $W$ is aircraft gross weight, $\text{L/D}$ is lift-to-drag ratio ($\approx 16$ to $18$), and $\text{TSFC}$ is thrust-specific fuel consumption.

2. **Holding Phase**:
   Holding loops flown at low altitudes (FL070 - FL140) in clean or flap-extended configurations burn fuel at significantly higher rates ($35\text{ to }65\text{ kg/min}$ for wide-bodies) than high-altitude cruise.

3. **Carbon Dioxide Conversion**:
   $$\text{CO}_2 \text{ (kg)} = \text{Fuel Burn (kg)} \times 3.16$$

### Dynamic Weekly Jet Fuel Spot Pricing

Rather than applying a static, outdated price per gallon, FlightML integrates the official historical weekly spot prices:
- **Source**: **U.S. Energy Information Administration (EIA Series `EER_EPJK_PF4_RGC_DPG`)** and **IATA Jet Fuel Price Monitor / Rotterdam Platts ARA**.
- **Density Conversion**: Jet-A kerosene density of $0.804\text{ kg/L} \approx 3.0434\text{ kg/gal}$.
- **Resolution**: Given a flight departure timestamp $T_{\text{dep}}$, FlightML matches the ISO calendar week ($YYYY\text{-W}ww$) to assign the exact prevailing spot price.

---

## 🌦️ Weather Integration (IEM ASOS / METAR)

FlightML queries historical weather from the **Iowa Environmental Mesonet (IEM)** automated ASOS archive:
- Fetches METAR observations (`tmpf`, `dwpf`, `drct`, `sknt`, `gust`, `vsby`, `skyc1`, `skyl1`, `wxcodes`).
- **Runway Crosswind Decomposition**: For each runway heading $\alpha_{\text{rwy}}$ and wind $(\theta_{\text{wind}}, V_{\text{wind}})$:
  $$V_{\text{cross}} = V_{\text{wind}} \cdot |\sin(\theta_{\text{wind}} - \alpha_{\text{rwy}})|$$
  $$V_{\text{head}} = V_{\text{wind}} \cdot \cos(\theta_{\text{wind}} - \alpha_{\text{rwy}})$$
- Evaluates operational crosswind limits ($>20\text{ kts}$) and low ceiling / low visibility (CAT II/III IFR conditions) to attribute aerodrome capacity constraints.

---

## ⚡ Network Simulation & Airline Stress Testing

### Reactionary Delay Propagation

Airlines operate tightly linked aircraft rotation networks where an airframe executes consecutive flight legs:
$$\text{Leg}_1 \xrightarrow{\text{Turnaround Buffer}} \text{Leg}_2 \xrightarrow{\text{Turnaround Buffer}} \text{Leg}_3$$

If $\text{Arrival Delay}_1 > \text{Scheduled Turnaround Buffer}$, the remaining delay propagates to $\text{Leg}_2$ as **Reactionary Delay**:
$$\text{Dep Delay}_2 = \max(0, \text{Arrival Delay}_1 - \text{Scheduled Turnaround Buffer})$$

### Shock Stress Testing

The stress testing engine injects synthetic initial delay shocks ($+15\text{m}, +30\text{m}, +45\text{m}, +60\text{m}, +90\text{m}$) into the first morning leg of every aircraft tail across an airline's fleet and computes:
- **Fleet Infection Rate (%)**: Percentage of downstream flights delayed $>15\text{ min}$.
- **Fragility Threshold**: The shock size at which infection exceeds 50%.

