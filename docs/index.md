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
- [**Model Context Protocol (MCP) Server Guide**](mcp_server.md): Exposing FlightML as an AI tool to local LLMs (LM Studio, Claude Desktop, Cursor).
  - [**LM Studio + Qwen 2.5 Coder Example Transcript**](mcp_example_lmstudio.md): End-to-end multi-turn transcript querying flight telemetry, calculating punctuality, and rendering markdown tables.
- [**Passenger Travel Copilot & Multi-Leg Journey Predictor**](passenger_copilot.md): Dedicated public-facing journey intelligence assessing direct and multi-leg connections, automatic seasonal archive selection, Minimum Connection Times (MCT), live 7-day weather forecasts, statutory EU261 compensation entitlements, and full `--debug` data provenance.

---

## ⚡ Ecosystem Overview

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FLIGHTML ECOSYSTEM                                     │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│    DATA INGESTION &      │   4D TRAJECTORY, KPI &   │   LONGITUDINAL, ECONOMICS &      │
│    COLUMNAR ENGINE       │   CLIMATE PHYSICS        │   OPERATIONS RESEARCH            │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ • DuckDB Columnar Store  │ • ICAO GANP & PRU KPIs   │ • Contemporary YoY Matching      │
│ • Zero-Copy Parquet      │ • KPI02/04/05/08/17/19   │ • 3-Season Elasticity Curves     │
│ • 119k Fixes & 26k Navs  │ • Multi-Tier KPI Suite   │ • Airspace Breach Forecaster     │
│ • OpenAP + ICAO EDB v32B │ • 3-Way Cruise FL Model  │ • Airline DOC & FIR Detours €    │
│ • Weekly Jet-A Prices €  │ • FRA vs. ATS Airways    │ • SES Flat-Rate Counterfactual   │
│ • Multi-Period Ingestion │ • Upper Winds & ESAD     │ • Delay Contagion & Super-Spread │
│ • IEM METAR + Upper Wind │ • Contrail ISSR Radiance │ • Repeat Offender Routes & Ops   │
│ • Thread-Safe Cursors    │ • Dynamic Complexity     │ • Hub Tail-Swaps & Date GDP Slots│
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│    NEURAL AIRSPACE       │   COUNTERFACTUAL &       │   LOCAL LLM AGENTS &             │
│    INTELLIGENCE (v4)     │   GAME THEORY SIMULATION │   MODEL CONTEXT PROTOCOL (MCP)   │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ • 128D Flight2Vec Twins  │ • 4D Airspace Closure    │ • FastMCP Server Architecture    │
│ • Deterministic Weights  │ • Detour NM & Fuel Waste │ • LM Studio, Claude & Cursor     │
│ • Commercial Taxonomy    │ • Non-Linear Backlog     │ • 9 Tools & Travel Copilot       │
│ • PINN Mass Inversion    │ • MARL ATFM Slot Gaming  │ • Zero-Hallucination Radar Search│
│ • Contrail Fast Surrogate│ • 10-Step Decision Trace │ • Multi-Turn Context Retention   │
│ • ST-GNN Delay Contagion │ • Infilled Assumptions   │ • Disruption Briefing Prompts    │
│ • Runway Saturation Queues│ • Balance Sheet (€ EUR) │ • JSON-RPC stdio & SSE Transports│
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

---

## 🚀 Quick Python Example (Synthetic IDs & Euro Pricing)

```python
import flightml as fml
from flightml.eval.longitudinal import LongitudinalEngine
from flightml.models.delay_contagion_graph import DelayContagionGraphEngine

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

# 4. Multi-period contemporary YoY benchmarking
long_engine = LongitudinalEngine()
yoy = long_engine.compare_yoy("202410", "202510")
print(f"YoY Growth: {yoy.total_flights_growth_pct:+.2f}% | NewGen Fleet Penetration: {yoy.newgen_share_delta_pct:+.2f}%")

# 5. Delay contagion graph & repeat offender detection
contagion = DelayContagionGraphEngine(ds)
top_offenders = contagion.scan_repeat_offender_routes(min_primary_delay_min=25.0, min_incidents=3, top_n=3)
for r in top_offenders:
    print(f"Repeat Offender {r.route}: {r.incident_count} days, {r.total_downstream_legs_infected} legs infected (€{r.total_reactionary_cost_eur:,.0f} EUR)")

# 6. Public passenger travel copilot & connection delay risk
from flightml.eval.passenger_journey import PassengerJourneyEngine
passenger_engine = PassengerJourneyEngine(period="auto")
itinerary = passenger_engine.assess_itinerary(
    origin="EINN", dest="EBBR", hub="EGLL", day_of_week="tuesday", time_of_day="morning", scheduled_layover_min=75
)
print(f"Connection Slack: {itinerary.connection.effective_slack_buffer_min:+.0f}m | Missed Risk: {itinerary.connection.missed_connection_prob*100:.1f}% | EU261: €{itinerary.passenger_rights.statutory_compensation_amount_eur:.0f} EUR")
```
