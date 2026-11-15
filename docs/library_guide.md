# Python Library Guide 🐍

FlightML is designed as a modular Python library that can be embedded into Jupyter notebooks, batch analytics pipelines, and custom microservices.

```python
import flightml as fml
```

---

> [!IMPORTANT]
> ### 🔒 Synthetic Example Outputs Policy
> All sample output tables, flight identifiers (e.g., `900104215`), and aircraft tail registrations (e.g., `G-SYNA`) shown in this guide are realistic synthetic records generated to illustrate API usage without exposing proprietary archive rows. All monetary outputs are in **Euros (`€`, `EUR`)**.

---

## 1. Columnar Dataset & Multi-Criteria Search (`FlightDataset`)

```python
from flightml import FlightDataset

ds = FlightDataset(period="202510")
summary = ds.get_summary()
print(f"Loaded {summary['total_flights']:,} flights across {summary['unique_dep_airports']:,} departure airports.")

# Multi-criteria search
flights_df = ds.search_flights(
    adep="EGLL",
    ades="KORD",
    ac_operator="BAW",
    min_distance_nm=3000,
    limit=5,
)
print(flights_df[["ectrl_id", "label", "actual_distance_nm", "arr_delay_min"]])
```

---

## 2. $5^\circ \times 5^\circ$ Airspace Saturation Maps & 4D Conflict Detection

```python
from flightml import (
    compute_hourly_grid_saturation,
    scan_airspace_proximity_events,
    create_grid_saturation_map,
)

# 1. Compute period-average hourly density & complexity across 5°x5° squares at 10:00 UTC
grid_data = compute_hourly_grid_saturation(ds, hour_utc=10)
print(f"Active 5°x5° squares: {grid_data['total_active_squares']} (Critical: {grid_data['critical_squares_count']})")

# 2. Scan 1-minute epochs for co-altitude proximity pairs (< 5 nm horizontal, < 1,000 ft vertical)
prox_rep = scan_airspace_proximity_events(ds, hour=10, horiz_sep_nm=5.0)
print(f"Detected {prox_rep.events_detected} proximity pairs ({prox_rep.loss_of_sep_alerts} LoS alerts)")

# 3. Render interactive Folium map combining 5°x5° saturation rectangles and conflict pins
m = create_grid_saturation_map(grid_data, proximity_events=prox_rep.events)
m.save("airspace_saturation_map.html")
```

---

## 3. ICAO GANP & Eurocontrol PRU Scorecard (`KPI02`–`KPI19`)

```python
from flightml import (
    compute_flight_pru_kpis,
    analyze_cdo_descent,
    analyze_cco_climb,
    analyze_network_level_capping,
)

# Single-flight PRU scorecard
pru = compute_flight_pru_kpis(ds, ectrl_id=900104215)
print(f"KPI04 KEP (Filed): {pru.kpi04_kep_pct:.2f}% | KPI05 KEA (Actual): {pru.kpi05_kea_pct:.2f}%")
print(f"KPI08 ASMA (40nm): {pru.asma_40nm_actual_min:.1f} min (+{pru.kpi08_asma_40nm_additional_min:.1f}m additional)")
print(f"KPI19 Cruise FL: Max FL{pru.max_actual_fl:.0f} vs Req FL{pru.requested_fl:.0f} (Capped: {pru.is_level_capped})")

# Vertical CDO (Descent) and CCO (Climb)
cdo = analyze_cdo_descent(ds, ectrl_id=900104215)
cco = analyze_cco_climb(ds, ectrl_id=900104215)
print(f"CDO Score: {cdo.cdo_compliance_score:.1f}% | CCO Score: {cco.cco_compliance_score:.1f}%")
```

---

## 4. Fuel Burn, ICAO EDB v32B Surface Emissions, Contrails & CRCO Costs (`€ EUR`)

```python
from flightml import (
    compute_flight_fuel_and_emissions,
    analyze_flight_taxi_out,
    compute_flight_crco_charges,
    analyze_flight_wind_and_contrails,
)

# 1. OpenAP excess fuel burn & weekly Jet-A spot cost (€)
impact = compute_flight_fuel_and_emissions(ds, ectrl_id=900104215)
print(f"Excess Fuel: {impact.total_excess_fuel_kg:,.1f} kg (€{impact.estimated_excess_fuel_cost_eur:,.2f} EUR)")

# 2. Surface taxi-out (KPI02 ATOT) & ICAO EDB v32B 7% idle pollutants
taxi = analyze_flight_taxi_out(ds, ectrl_id=900104215)
print(f"Additional ATOT: +{taxi.additional_atot_min:.1f} min | Excess NOx: {taxi.excess_nox_g:,.0f} g | Cost: €{taxi.excess_fuel_cost_eur:,.2f}")

# 3. Eurocontrol CRCO En-Route Navigation Charges (€)
crco = compute_flight_crco_charges(ds, ectrl_id=900104215)
print(f"Service Units: {crco.total_service_units:.2f} SU | CRCO Fee: €{crco.actual_crco_charge_eur:,.2f} EUR")

# 4. Upper-air winds & Equivalent Still Air Distance (ESAD)
contrail = analyze_flight_wind_and_contrails(ds, ectrl_id=900104215)
print(f"Net Wind: {contrail.avg_tailwind_component_kts:+.1f} kts | Air Distance (ESAD): {contrail.esad_nm:,.0f} nm ({contrail.wind_efficiency_verdict})")
```

---

## 5. 3-Way Cruise Altitude Optimization & Engine Efficiency

```python
from flightml.eval.aircraft_engine_perf import (
    resolve_aircraft_engine_perf,
    compute_cruise_altitude_efficiency,
)

# 1. Resolve turbofan engine bypass ratio and aerodynamic optimal flight level
eng_perf = resolve_aircraft_engine_perf("B77W", engine_override="GE90-115B")
print(f"Engine: {eng_perf.engine_name} | BPR: {eng_perf.bypass_ratio:.1f} | Opt FL: FL{eng_perf.optimum_cruise_fl:.0f}")

# 2. Evaluate planning vs. tactical ATC vertical efficiency penalties
vert_eff = compute_cruise_altitude_efficiency(
    ac_type="B77W",
    actual_cruise_fl=320.0,
    requested_fl=340.0,
    cruise_distance_nm=3200.0,
    engine_override="GE90-115B",
)
print(f"Optimal FL{vert_eff.optimal_cruise_fl:.0f} vs Actual FL{vert_eff.actual_cruise_fl:.0f}")
print(f"Fuel Penalty: +{vert_eff.actual_burn_penalty_pct:.1f}% ({vert_eff.excess_fuel_burn_kg:.1f} kg)")
```

---

## 6. Multi-Tier KPI Suite (Flight, Fleet, Airport, Network)

```python
from flightml.eval.multi_level_kpis import query_multi_level_kpis

# Query aggregated KPIs for British Airways (BAW) during early October 2025
fleet_kpis = query_multi_level_kpis(
    ds,
    level="fleet",
    start_date="2025-10-01",
    end_date="2025-10-07",
    operator="BAW",
)
print(f"Analyzed {fleet_kpis.flights_analyzed:,} BAW flights:")
print(f"Mean Dep Delay: {fleet_kpis.mean_dep_delay_min:+.1f}m | Mean Arr Delay: {fleet_kpis.mean_arr_delay_min:+.1f}m")
print(f"Total Excess Fuel: {fleet_kpis.total_excess_fuel_kg:,.0f} kg (€{fleet_kpis.total_excess_fuel_cost_eur:,.2f} EUR)")
```

---

## 7. Prescriptive Hub Tail-Swap Recovery & GDP Slot Optimization (`€ EUR`)

```python
from flightml import optimize_hub_tail_swaps, optimize_ground_delay_program

# 1. Scan network-wide across all operators, hubs, and dates for viable tail swaps
swaps = optimize_hub_tail_swaps(ds, operator=None)
for s in swaps.swaps[:5]:
    print(
        f"[{s.swap_date} @ {s.hub_airport}] {s.operator} ({s.ac_type}): "
        f"{s.delayed_tail_reg} ⇄ {s.donor_tail_reg} | Saved {s.net_delay_saved_min:.0f} min (€{s.estimated_cost_saved_eur:,.0f} EUR)"
    )

# 2. Ration-by-Schedule Ground Delay Program (GDP) slot optimization (with date selection)
gdp = optimize_ground_delay_program(
    ds,
    target_airport="EGLL",
    program_date="2025-10-15",
    start_hour=8,
    end_hour=14,
    degraded_capacity_rate_per_hr=36,
)
print(f"Regulated Flights: {gdp.total_flights_regulated:,} | Net Fuel Cost Saved: €{gdp.total_fuel_cost_saved_eur:,.2f} EUR")
```

---

## 8. Longitudinal Intelligence, Forecasting, Behavioral Economics & Delay Contagion (`v3`)

### 1. Multi-Period Longitudinal & Contemporary YoY Comparison
```python
from flightml.eval.longitudinal import LongitudinalEngine

engine = LongitudinalEngine()

# Multi-period timeline filtered to contemporary YoY cohort (2024-2025)
timeline_df = engine.get_timeline_dataframe(cohort="yoy")
print(timeline_df[["period_id", "season", "daily_flights", "avg_arr_delay_min", "newgen_share_pct"]])

# 3-season decomposition (Winter Trough, Summer Peak, Autumn Shoulder)
decomp = engine.decompose_seasonality(year=2024)
print(f"Delay Surge Elasticity: {decomp.delay_demand_elasticity:.2f}x (Summer surge: +{decomp.summer_surge_pct:.1f}%)")

# Controlled YoY comparison between matched seasons (Autumn 2024 vs Autumn 2025)
yoy = engine.compare_yoy(baseline_period="202410", target_period="202510")
print(f"YoY Growth: {yoy.daily_flights_growth_pct:+.2f}% | NewGen Fleet Penetration: {yoy.newgen_share_delta_pct:+.2f}%")
```

### 2. Predictive Airspace Saturation & Bottleneck Forecaster
```python
from flightml.models.airspace_forecaster import AirspaceForecaster

forecaster = AirspaceForecaster()
forecast = forecaster.generate_comprehensive_forecast(
    forecast_year=2026,
    season="summer",
    growth_scenario_pct=3.5,
    grid_size_deg=1.0,
)
print(f"Network Risk Index: {forecast.network_atfm_risk_index:.1f}/10.0")
print(f"Capacity Breaches Projected: {forecast.breached_sectors_count} sectors")
for b in forecast.top_capacity_breaches[:3]:
    print(f"  Sector {b.sector_id}: {b.projected_peak_hourly}/hr vs nominal {b.nominal_capacity_hourly}/hr ({b.projected_saturation_pct:.1f}%)")
```

### 3. Airline Behavioral Economics & Direct Operating Cost (DOC)
```python
from flightml.eval.airline_economics import AirlineEconomicsEngine

econ_engine = AirlineEconomicsEngine(ds)

# 1. Evaluate Direct Operating Cost (DOC) component allocation for a flight
doc = econ_engine.evaluate_flight_doc(ectrl_id=900104215, cost_of_time_eur_per_hour=850.0)
print(f"Total DOC: €{doc.total_doc_eur:,.2f} EUR (Fuel: {doc.fuel_cost_share_pct:.1f}%, CRCO: {doc.crco_cost_share_pct:.1f}%)")
print(f"Route Charge Avoidance Incentive: €{doc.avoidance_incentive_ratio:.2f}/NM")

# 2. Scan network for FIR charge avoidance detour corridors
detours = econ_engine.scan_network_route_charge_avoidance(min_corridor_flights=50, top_n=5)
for d in detours:
    print(f"Corridor {d.route}: {d.avoidance_share_pct:.1f}% detours saving €{d.mean_crco_savings_eur:.2f} fee (+{d.mean_extra_distance_nm:.1f} NM)")

# 3. Simulate Single European Sky (SES) unified flat-rate policy
ses = econ_engine.simulate_harmonized_eurocontrol_rate(harmonized_rate_eur=72.50)
print(f"Harmonized Rate: €{ses.harmonized_unit_rate_eur:.2f}/SU | Eliminated Detours: {ses.eliminated_detour_flights:,} flights/yr")
print(f"Annual Airline Savings: €{ses.estimated_annual_airline_cost_saved_eur:,.2f} EUR")
```

### 4. Delay Contagion Graph, Super-Spreaders & Period-Wide Repeat Offenders
```python
from flightml.models.delay_contagion_graph import DelayContagionGraphEngine

contagion_engine = DelayContagionGraphEngine(ds)

# 1. Detect single-date delay super-spreader flights
spreaders = contagion_engine.identify_super_spreaders(date_str="2025-10-15", min_primary_delay_min=20.0, top_n=5)
for s in spreaders:
    print(f"Spreader {s.ectrl_id} ({s.ac_reg}): +{s.primary_delay_min:.0f}m delay infected {s.infected_downstream_legs} legs (€{s.downstream_economic_cost_eur:,.0f} EUR)")

# 2. Scan full monthly period for chronic repeat offender routes
offender_routes = contagion_engine.scan_repeat_offender_routes(min_primary_delay_min=25.0, min_incidents=3, top_n=5)
for r in offender_routes:
    print(f"Route {r.route}: {r.incident_count} contagion days, {r.total_downstream_legs_infected} legs infected (€{r.total_reactionary_cost_eur:,.0f} EUR)")

# 3. Scan full monthly period for repeat offender airline operators
offender_ops = contagion_engine.scan_repeat_offender_operators(min_primary_delay_min=25.0, top_n=5)
for o in offender_ops:
    print(f"Airline {o.operator}: {o.total_root_spreader_flights} spreader flights, {o.downstream_infection_multiplier:.2f}x multiplier (€{o.total_reactionary_cost_eur:,.0f} EUR)")

# 4. Simulate network capacity shock
shock = contagion_engine.simulate_network_shock(
    disruption_target="EGLL",
    target_type="AIRPORT",
    duration_hours=2.0,
    capacity_cut_pct=50.0,
    date_str="2025-10-15",
)
print(f"Shock Result: {shock.total_impacted_flights:,} flights impacted | Cumulative Delay: {shock.total_delay_hours_incurred:.1f} hrs (€{shock.total_economic_delay_cost_eur:,.2f} EUR)")
```

---

## 8. Neural Airspace Intelligence, Counterfactual Simulators & FastMCP

### A. Flight2Vec 128D Trajectory Embeddings & Operational Twins
```python
from flightml.models.flight2vec import Flight2VecEngine

f2v = Flight2VecEngine(ds)

# 1. Pre-vectorize flights into high-speed on-disk cache
cached_count = f2v.build_vector_cache(limit=1000)
print(f"Pre-vectorized {cached_count} flights into local vector cache.")

# 2. Find operational twins for a commercial flight (with airframe taxonomy filtering)
twins = f2v.find_operational_twins(target_ectrl_id=901045971, top_k=3)
for t in twins:
    print(f"Twin {t.ectrl_id} ({t.route}, {t.ac_type}): Cosine Similarity = {t.cosine_similarity:.4f} ({t.similarity_pct:.1f}%)")
    print(f"   Explanation: {t.explanation}")
```

### B. Spatio-Temporal Graph WaveNet (ST-GNN) Delay Contagion
```python
from flightml.models.st_gnn import SpatioTemporalDelayEngine

st_engine = SpatioTemporalDelayEngine(ds, period="202510")

# Simulate a 60-min delay shock at London Heathrow under high runway saturation
prop_res = st_engine.simulate_delay_propagation(
    shock_origin="EGLL",
    injected_delay_min=60.0,
    shock_date="2025-10-15",
    shock_time_hour=10,
    shock_duration_hours=2.0,
)

print(f"Origin Hub: {prop_res.shock_origin_node} (Runway Saturation: {prop_res.origin_saturation_pct:.1f}%, Spare Buffer: {prop_res.origin_spare_capacity:.1f} mvts/hr)")
print(f"Total Network Reactionary Delay: {prop_res.total_network_reactionary_delay_hours:,.1f} delay hours across {prop_res.total_flights_impacted:,} flights.")
print(f"Peak Downstream Hub: {prop_res.top_vulnerable_nodes[1].node_id} (+{prop_res.top_vulnerable_nodes[1].reactionary_delay_hours:.1f} reactionary hrs)")
```

### C. 4D Airspace (FIR) Closure Counterfactual Simulator
```python
from flightml.sim.airspace_closure import AirspaceClosureSimulator

fir_sim = AirspaceClosureSimulator(ds, period="202510")

# Simulate temporary 3-hour closure of Swiss airspace (LSASFIR)
closure_res = fir_sim.simulate_fir_closure(
    target_fir="LSASFIR",
    closure_start_hour=10,
    closure_duration_hours=3.0,
    date_str="2025-10-15",
)

print(f"Impacted Flights: {closure_res.total_flights_impacted:,} | Tactical Reroutes: {closure_res.rerouted_flights_count:,}")
print(f"Detour Distance: +{closure_res.total_extra_distance_flown_nm:,.0f} NM | Extra Fuel: +{closure_res.total_extra_fuel_burned_tonnes:,.1f} tonnes")
print(f"Total Airline Operating Loss: €{closure_res.total_airline_economic_loss_eur:,.2f} EUR")
print(f"Post-Reopening Queue Recovery Time: {closure_res.post_reopening_recovery_hours:.1f} hours")
```

### D. Multi-Agent Reinforcement Learning (MARL) ATFM Slot Gaming
```python
from flightml.sim.marl_atfm import MultiAgentATFMSimulator

marl_sim = MultiAgentATFMSimulator(ds, period="202510")

# Simulate decentralized slot gaming under a 40% arrival rate reduction at Heathrow
marl_res = marl_sim.run_marl_simulation(
    regulated_airport="EGLL",
    capacity_cut_pct=40.0,
    date_str="2025-10-15",
    start_time_hour=10,
    duration_hours=3.0,
)

print(f"System Delay Saved: {marl_res.total_network_delay_saved_min:,.0f} min | Net Economic Payoff: €{marl_res.total_net_economic_benefit_eur:,.2f} EUR")
print(f"Cooperative vs Competitive Efficiency Gap: {marl_res.cooperative_vs_competitive_efficiency_gap_pct:.1f}%")

# Inspect airline balance sheets
for strat in marl_res.airline_strategies:
    print(f"Agent: {strat.airline_name} ({strat.flights_regulated} flights)")
    print(f"   Baseline Cost: €{strat.baseline_cost_no_gaming_eur:,.0f} EUR -> Post-Gaming Cost: €{strat.post_gaming_cost_eur:,.0f} EUR (Net Payoff: €{strat.net_economic_payoff_eur:,.0f} EUR)")
    print(f"   Dominant Strategy: {strat.dominant_strategy}")

# Inspect first 3 chronological steps of the blow-by-blow decision log
for step in marl_res.decision_log[:3]:
    print(f"Step #{step.step_number} [{step.timestamp_rel}] {step.agent_name} -> {step.action_type}: {step.flight_id} ({step.route})")
```

### E. Model Context Protocol (FastMCP) Integration
```python
from flightml.mcp.server import create_mcp_server

# Create and configure the FastMCP server
mcp_server = create_mcp_server()

# Run via stdio transport for local desktop LLMs (LM Studio, Claude Desktop, Cursor)
# mcp_server.run(transport="stdio")
```

---

## 9. Passenger Journey & Travel Risk Copilot (`flightml.eval.passenger_journey`)

The `PassengerJourneyEngine` brings Eurocontrol DDR2 ATM big data, live Open-Meteo numerical weather predictions, terminal transfer dynamics, and EU261/UK261 consumer law directly to the travelling public.

### A. Direct Flight Delay Assessment & Live Weather Impact

```python
from flightml.eval.passenger_journey import PassengerJourneyEngine

# Initialize the engine (uses auto period selection by default)
engine = PassengerJourneyEngine(period="auto")

# Assess a direct flight from Dublin (EIDW) to Amsterdam (EHAM)
itinerary = engine.assess_itinerary(
    origin="EIDW",
    dest="EHAM",
    day_of_week="friday",
    time_of_day="evening",
    flight_date="2026-11-20",
)

leg = itinerary.legs[0]
print(f"Direct Route: {leg.origin} ➔ {leg.dest} ({leg.operator} {leg.aircraft_type})")
print(f"Historical Delay > 15m : {leg.historical_delay_prob * 100:.1f}%")
print(f"Weather-Adjusted Delay : {leg.weather_adjusted_delay_prob * 100:.1f}%")
print(f"Destination Weather    : {leg.dest_weather.weather_summary}")
print(f"EU261 Compensation     : €{itinerary.passenger_rights.statutory_compensation_amount_eur:.0f} EUR")
```

### B. Multi-Leg Connecting Itinerary & Missed Connection Risk

```python
# Assess a multi-leg connection: Shannon (EINN) to Brussels (EBBR) via London Heathrow (EGLL)
itinerary = engine.assess_itinerary(
    origin="EINN",
    dest="EBBR",
    hub="EGLL",
    day_of_week="tuesday",
    time_of_day="morning",
    scheduled_layover_min=75,
    operator="BAW",
    flight_date="2026-11-17",
    debug=True,
)

# Leg details
for idx, leg in enumerate(itinerary.legs, 1):
    print(f"Leg #{idx}: {leg.origin} ➔ {leg.dest} | P90 Delay: +{leg.p90_delay_min:.1f}m | Risk: {leg.weather_adjusted_delay_prob*100:.1f}%")

# Hub connection metrics
conn = itinerary.connection
print(f"Hub Airport: {conn.hub_airport} ({conn.hub_name})")
print(f"Effective Slack Buffer  : {conn.effective_slack_buffer_min:+.1f} min (Nominal MCT: {conn.minimum_connection_time_min}m)")
print(f"Missed Connection Risk  : {conn.missed_connection_prob * 100:.1f}% ({conn.connection_risk_rating})")
print(f"Checked Luggage Risk    : {conn.luggage_risk_level}")

# Statutory passenger rights
rights = itinerary.passenger_rights
print(f"Distance: {rights.great_circle_distance_km:,.0f} km ({rights.distance_tier})")
print(f"Statutory Compensation  : €{rights.statutory_compensation_amount_eur:,.0f} EUR (Delay Threshold: ≥{rights.delay_threshold_hours:.0f}h)")
print(f"Duty of Care (Meals)    : After {rights.duty_of_care_threshold_hours:.0f} hours delay")
```

### C. Diurnal Departure Wave Recommendations ("When Should I Fly?")

```python
# Inspect delay risk across diurnal departure windows for the first leg
print(f"Departure Wave Recommendations for {itinerary.legs[0].origin} ➔ {itinerary.legs[0].dest}:")
for wave in itinerary.legs[0].departure_waves:
    print(f"  {wave.wave_name:12s} ({wave.time_window}): Avg +{wave.avg_delay_min:4.1f}m | P90 +{wave.p90_delay_min:4.1f}m | >15m Risk: {wave.prob_delay_over_15m*100:4.1f}%")
```

### D. Full Provenance & Diagnostic Audit (`debug=True`)

```python
# Inspect underlying historical records and calculation trace
dbg = itinerary.debug_info
print(f"Audit Baseline Period Selected: {dbg['baseline_period']}")
print(f"Leg 1 Records Sampled         : {dbg['leg_1_audit']['records_sampled']}")
print(f"Leg 1 SQL Filter              : {dbg['leg_1_audit']['sql_filter']}")
for sample in dbg['leg_1_audit']['sample_records'][:3]:
    print(f"  Sample: ECTRL {sample['ectrl_id']} | Date: {sample['date']} | AC: {sample['aircraft_type']} ({sample['ac_reg']}) | Delay: +{sample['arr_delay_min']:.1f}m")
```

### E. Available Commercial Passenger Airlines Discovery

```python
# Discover active commercial carriers on a city pair (automatically filtering out cargo freighters)
operators = engine.get_available_operators(origin="LEMD", dest="EGLL")
for op in operators:
    print(f"Airline: {op['operator']} | Historical Flights: {op['flight_count']} | Fleet: {op['top_aircraft']}")
```
