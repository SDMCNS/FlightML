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

## 🔮 Prescriptive Operations Research Models

- **Network-Wide Hub Tail-Swap Optimizer**: Evaluates same-operator, same-aircraft-type turnaround pairs (`Delayed Tail A ⇄ Donor Tail B`) on the ground simultaneously at the same hub and recommends conflict-free swaps that minimize total reactionary delay minutes and Euro (`€88/min`) delay costs.
- **Ration-by-Schedule Ground Delay Program (`GDP`) Optimizer**: Allocates arrival slots under an Airport Arrival Rate (`AAR`) cap to shift high-burn airborne terminal holding stacks onto the departure ground at 7% idle/gate burn.
