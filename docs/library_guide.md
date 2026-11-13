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

# 4. Upper-air winds & persistent contrail (ISSR) climate forcing
contrail = analyze_flight_wind_and_contrails(ds, ectrl_id=900104215)
print(f"Net Wind: {contrail.avg_tailwind_component_kts:+.1f} kts | ISSR Contrail: {contrail.persistent_contrail_distance_nm:.1f} nm")
```

---

## 5. Prescriptive Hub Tail-Swap Recovery & GDP Slot Optimization (`€ EUR`)

```python
from flightml import optimize_hub_tail_swaps, optimize_ground_delay_program

# 1. Scan network-wide across all operators, hubs, and dates for viable tail swaps
swaps = optimize_hub_tail_swaps(ds, operator=None)
for s in swaps.swaps[:5]:
    print(
        f"[{s.swap_date} @ {s.hub_airport}] {s.operator} ({s.ac_type}): "
        f"{s.delayed_tail_reg} ⇄ {s.donor_tail_reg} | Saved {s.net_delay_saved_min:.0f} min (€{s.estimated_cost_saved_eur:,.0f} EUR)"
    )

# 2. Ration-by-Schedule Ground Delay Program (GDP) slot optimization
gdp = optimize_ground_delay_program(ds, airport_icao="EGLL", declared_aar_per_hour=36)
print(f"Slotted Flights: {gdp.slotted_flights_count} | Net Fuel Cost Saved: €{gdp.net_fuel_cost_saved_eur:,.2f} EUR")
```
