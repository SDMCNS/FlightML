# FlightML Documentation ✈️

Welcome to the **FlightML** documentation. FlightML is a high-performance Python aviation intelligence library, interactive simulation workbench, REST API, and command-line interface (CLI) for **Global Airspace Network Analytics, 4D Trajectory Physics, Environmental & Eurocontrol CRCO Cost Accounting (`€ EUR`), and Prescriptive Operations Research**.

---

> [!IMPORTANT]
> ### 🔒 Data Access & Eurocontrol OneSky Online Compliance Notice
> **Eurocontrol Research Data Access Prerequisite**:
> The underlying aeronautical traffic, flight plans, and 4D radar trajectory archives supported by FlightML (such as Eurocontrol DDR2 / PRISME / R-NEST datasets) are **proprietary and not for public consumption or redistribution**.
>
> To access or download raw historical datasets, researchers and organizations **must have an approved user account on Eurocontrol's OneSky Online portal** ([https://www.eurocontrol.int/](https://www.eurocontrol.int/)) with institutional access granted to the **Demand Data Repository (DDR)** under Eurocontrol's Data Dissemination Policy and standard Research Agreement.
>
> **Synthetic Example Outputs Policy**:
> In accordance with Eurocontrol data dissemination terms, commercial confidentiality, and GDPR requirements, **all sample output tables, flight identifiers (e.g., `900104215`), aircraft tail registrations (e.g., `G-SYNA`, `EI-SYNB`), and aggregate statistics shown in this documentation are realistic synthetic records** generated to illustrate table schemas and platform capabilities without exposing proprietary archive rows.

---

## 📚 Documentation Index

- [**Quickstart Guide**](quickstart.md): Step-by-step installation, dataset directory layout, dashboard launching, and first steps.
- [**Python Library Guide**](library_guide.md): Developer guide with copy-pasteable Python code recipes across all modules.
- [**CLI Reference Manual**](cli_reference.md): Detailed reference guide for all 45+ `flightml` terminal commands and synthetic output tables.
- [**REST API Reference**](rest_api.md): Complete catalog of FastAPI endpoints, query parameters, response schemas (`€ EUR`), and Swagger docs.
- [**Architecture & Methodology**](architecture.md): Deep dive into DuckDB columnar execution, ICAO GANP & Eurocontrol PRU formulas, OpenAP & ICAO EDB v32B physics, Eurocontrol CRCO charges (`€`), and operations research models.

---

## ⚡ Ecosystem Overview

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FLIGHTML ECOSYSTEM                                     │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│    DATA INGESTION &      │   4D TRAJECTORY, KPI &   │   AIRSPACE MAPS, CRCO (€) &      │
│    COLUMNAR ENGINE       │   CLIMATE PHYSICS        │   OPERATIONS RESEARCH            │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ • DuckDB Columnar Store  │ • ICAO GANP & PRU KPIs   │ • 5°×5° Hourly Saturation Maps   │
│ • Zero-Copy Parquet      │ • KPI02/04/05/08/17/19   │ • 4D Co-Altitude Conflict Scan   │
│ • 119k Fixes & 26k Navs  │ • Multi-Tier KPI Suite   │ • Eurocontrol CRCO Route Fees €  │
│ • OpenAP + ICAO EDB v32B │ • 3-Way Cruise FL Model  │ • Hub Tail-Swap Recovery (MIP)   │
│ • Weekly Jet-A Prices €  │ • FRA vs. ATS Airways    │ • Ration-by-Schedule GDP Slots   │
│ • IEM METAR + Upper Wind │ • Upper Winds & ESAD     │ • Fleet Shock Contagion Curves   │
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

---

## 🚀 Quick Python Example (Synthetic IDs & Euro Pricing)

```python
import flightml as fml

# 1. Connect to the ingested dataset
ds = fml.FlightDataset(period="202510")

# 2. Compute ICAO GANP & Eurocontrol PRU KPIs for a flight
pru = fml.compute_flight_pru_kpis(ds, ectrl_id=900104215)
print(f"KPI04 KEP: {pru.kpi04_kep_pct:.2f}% | KPI05 KEA: {pru.kpi05_kea_pct:.2f}% ({pru.enroute_routing_verdict})")

# 3. Compute fuel burn, 3-way cruise altitude optimization & Eurocontrol CRCO route charges (€ EUR)
impact = fml.compute_flight_fuel_and_emissions(ds, ectrl_id=900104215)
print(f"Optimal: FL{impact.optimal_cruise_fl:.0f} vs Actual: FL{impact.actual_cruise_fl:.0f} | Excess Fuel: {impact.total_excess_fuel_kg:,.1f} kg")

crco = fml.compute_flight_crco_charges(ds, ectrl_id=900104215)
print(f"CRCO Fee: €{crco.actual_crco_charge_eur:,.2f} | Total Direct Cost: €{crco.total_direct_route_cost_eur:,.2f}")

# 4. Run network-wide prescriptive hub tail-swap recovery
swaps = fml.optimize_hub_tail_swaps(ds, operator=None)
print(f"Found {swaps.recommended_swaps_count} tail swaps saving €{swaps.estimated_cost_saved_eur:,.0f} EUR")
```
