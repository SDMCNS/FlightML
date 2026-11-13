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
│ (10 Interactive Tabs)    │ (Swagger / OpenAPI)      │ (45+ Analytical Commands)        │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
                                    ▲
                                    │
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              ANALYTICAL & SIMULATION LAYER                             │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ • 5°×5° Saturation Maps  │ • OpenAP Fuel & Carbon   │ • Hub Tail-Swap Optimizer (MIP)  │
│ • ICAO GANP & PRU KPIs   │ • ICAO EDB LTO Emissions │ • Ration-by-Schedule GDP Slots   │
│ • FRA vs. ATS Airways    │ • Eurocontrol CRCO (€)   │ • Turnaround & Shock Contagion   │
│ • 4D Conflict Scanner    │ • ERA5 Contrail ISSRs    │ • Runway Braking & Crosswinds    │
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
| **KPI17 (`CCO`)** | **Continuous Climb Operations** | Detects intermediate level-offs ($|\text{ROCD}| < 200\text{ ft/min}$ for $\ge 60\text{ s}$) below Top of Climb (`TOC`). |
| **KPI19 (`CDO` & Level-Cap)** | **Continuous Descent & Cruise Level-Capping** | Detects descent level-offs below Top of Descent (`TOD`) and flags flights capped $\ge 20\text{ FL}$ ($2,000\text{ ft}$) below their filed requested flight level ($\text{FL}_{\text{req}}$). |

---

## 🗺️ $5^\circ \times 5^\circ$ Airspace Saturation & 4D Conflict Detection

For any selected hour of the day $h \in [0, 23]$ UTC, FlightML aggregates 4D trajectory points across all active days in the archive into $5^\circ \times 5^\circ$ geographic squares:
$$\text{Dynamic Complexity Index} = \bar{N}_{h} \times \left(1 + 1.5 \cdot \frac{N_{\text{vert, } |\Delta \text{FL}| \ge 5}}{N_{\text{total pts}}}\right)$$
where $\bar{N}_{h}$ is the period-average hourly flight count in that square. Simultaneously, 4D trajectories are linearly interpolated at synchronized 1-minute epochs (`:00`, `:15`, `:30`, `:45`) to detect co-altitude aircraft pairs within $< 1,000\text{ ft}$ vertical and $\le 5.0\text{ nm}$ horizontal separation.

---

## 🌱 OpenAP Fuel Burn, ICAO EDB v32B LTO Emissions & Contrail Physics

1. **Aerodynamic Fuel Burn & Carbon (`OpenAP`)**: Maps aircraft types to OpenAP mass, drag polar, and thrust-specific fuel consumption profiles ($3.16\text{ kg CO}_2$ per $\text{kg Jet-A}$).
2. **Surface Taxi-Out LTO Idle Emissions (`ICAO EDB v32B`)**: Applies empirical 7% LTO idle fuel flow and emission indices ($\text{NO}_x$, $\text{CO}$, $\text{HC}$, and $\text{nvPM}$ soot mass) from the ICAO Aircraft Engine Emissions Databank v32B during excess `KPI02 ATOT` departure queues.
3. **Upper-Air Winds & Persistent Contrail (`ISSR`) Forcing**: Cross-references 4D cruise segments (`FL280`–`FL430`) against **Open-Meteo ERA5 Upper-Air Pressure Levels (`300/250/200/150 hPa`)** and the **Schmidt-Appleman criterion** ($T < -40^\circ\text{C}$, $\text{RH}_{\text{ice}} \ge 100\%$) to identify persistent contrail segments and radiative forcing multipliers (`CO₂e`).

---

## 💶 Eurocontrol CRCO Route Charges & Fuel Economics (`€ EUR`)

En-route navigation charges across European Flight Information Regions are billed in **Euros (`€`)** using the official Eurocontrol Central Route Charges Office (CRCO) formula:
$$\text{CRCO Charge (€)} = \sum_{i} \left(\frac{D_{i,\text{km}}}{100}\right) \times \sqrt{\frac{\text{MTOW (tons)}}{50}} \times u_i$$
where $D_{i,\text{km}}$ is the great-circle entry-to-exit distance in charging zone $i$ and $u_i$ is the national unit rate in `€/SU`. Weekly Jet-A fuel costs are resolved from historical EIA/IATA spot prices converted into **Euros (`€`)**.

---

## 🔮 Prescriptive Operations Research Models

- **Network-Wide Hub Tail-Swap Optimizer**: Evaluates same-operator, same-aircraft-type turnaround pairs (`Delayed Tail A ⇄ Donor Tail B`) on the ground simultaneously at the same hub and recommends conflict-free swaps that minimize total reactionary delay minutes and Euro (`€88/min`) delay costs.
- **Ration-by-Schedule Ground Delay Program (`GDP`) Optimizer**: Allocates arrival slots under an Airport Arrival Rate (`AAR`) cap to shift high-burn airborne terminal holding stacks onto the departure ground at 7% idle/gate burn.
