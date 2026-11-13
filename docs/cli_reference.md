# Command-Line Interface (CLI) Reference 💻

FlightML includes a comprehensive command-line interface accessible via `flightml`. All commands output Rich-formatted tables and status badges, with all monetary figures reported in **Euros (`€`, `EUR`)**.

```bash
flightml --help
```

---

> [!IMPORTANT]
> ### 🔒 Synthetic Example Outputs Policy
> In accordance with Eurocontrol data dissemination terms, commercial confidentiality, and GDPR requirements, **all sample output tables, flight identifiers (e.g., `900104215`), aircraft tail registrations (e.g., `G-SYNA`, `EI-SYNB`), and aggregate statistics shown below are realistic synthetic records** generated to illustrate CLI output schemas without exposing proprietary archive rows.

---

## 1. Dashboard & Web Services

### `flightml dashboard`
Launch the 10-tab interactive Streamlit simulation workbench (`http://localhost:8501`).
```bash
flightml dashboard [--port 8501]
```

### `flightml serve`
Launch the FastAPI REST server (`http://127.0.0.1:8000`).
```bash
flightml serve [--host 127.0.0.1] [--port 8000]
```

---

## 2. Dataset Summary & Search

### `flightml summary`
```bash
flightml summary --period 202510
```

**Synthetic Output Table:**
```text
                       ✈️  FlightML Dataset Overview (Synthetic Sample)
┌─────────────────────────────┬────────────────────────────────────────────┐
│ Metric                      │ Value                                      │
├─────────────────────────────┼────────────────────────────────────────────┤
│ Total Processed Flights     │ 842,500                                    │
│ Unique Aircraft Operators   │ 740                                        │
│ Origin Airports (ADEP)      │ 1,580                                      │
│ Destination Airports (ADES) │ 1,595                                      │
│ Date Coverage               │ 2025-10-01 00:00:00 to 2025-10-31 23:59:00 │
│ Mean Departure Delay        │ 5.12 min                                   │
│ Mean Arrival Delay          │ 4.85 min                                   │
│ Airports Catalog            │ 86,336                                     │
│ Runways Catalog             │ 48,405                                     │
│ FIR Airspaces Catalog       │ 170                                        │
└─────────────────────────────┴────────────────────────────────────────────┘
```

### `flightml search-flights`
```bash
flightml search-flights --period 202510 --adep EGLL --ades KORD --limit 3
```

**Synthetic Output Table:**
```text
          Flight Search Results (Period: 202510, Synthetic Sample)
┌──────────────────────────────────┬──────────┬───────┬───────────┬───────────┐
│ Flight Label & Metadata          │ Distance │    FL │ Dep Delay │ Arr Delay │
├──────────────────────────────────┼──────────┼───────┼───────────┼───────────┤
│ EGLL ➔ KORD (2025-10-03 08:15)   │  3495 nm │ FL360 │       -4m │       -2m │
│ [SYN B77W G-SYNA] ... Flight ID: │          │       │           │           │
│ 900104215                        │          │       │           │           │
│ EGLL ➔ KORD (2025-10-03 10:40)   │  3542 nm │ FL340 │       +8m │      +11m │
│ [SYN B789 G-SYNB] ... Flight ID: │          │       │           │           │
│ 900108492                        │          │       │           │           │
│ EGLL ➔ KORD (2025-10-03 14:20)   │  3510 nm │ FL350 │       +2m │       +0m │
│ [SYN A359 G-SYNC] ... Flight ID: │          │       │           │           │
│ 900112640                        │          │       │           │           │
└──────────────────────────────────┴──────────┴───────┴───────────┴───────────┘
```

---

## 3. ICAO GANP & Eurocontrol PRU KPI Commands

### `flightml pru-kpis <ectrl_id>`
Compute single-flight ICAO GANP & Eurocontrol PRU indicators (`KPI04 KEP`, `KPI05 KEA`, `KPI08 ASMA`, `KPI19` level-capping).
```bash
flightml pru-kpis 900104215 --period 202510
```

### `flightml cdo-descent <ectrl_id>` & `flightml cco-climb <ectrl_id>`
Analyze Continuous Descent Operations (`KPI19 CDO`) and Continuous Climb Operations (`KPI17 CCO`) level-off staircases and excess fuel burn.
```bash
flightml cdo-descent 900104215 --period 202510
flightml cco-climb 900104215 --period 202510
```

### `flightml level-capping`
Identify corridors with systematic `KPI19` cruise flight-level capping ($\ge 2,000\text{ ft}$ below requested FL).
```bash
flightml level-capping --period 202510 --top-n 5
```

**Synthetic Output Table:**
```text
           📏 Network KPI19 Cruise Level-Capping Corridors (Synthetic Sample)
┌───────────────┬─────────┬────────┬───────────┬──────────┬──────────┬──────────────┐
│ Corridor      │ Flights │ Capped │ Capped %  │ Avg Req  │ Avg Act  │ Mean Deficit │
├───────────────┼─────────┼────────┼───────────┼──────────┼──────────┼──────────────┤
│ EGLL ➔ EHAM   │     420 │    168 │     40.0% │    FL290 │    FL240 │     -50.0 FL │
│ LFPG ➔ EDDF   │     385 │    131 │     34.0% │    FL310 │    FL260 │     -50.0 FL │
│ EGKK ➔ LEMD   │     310 │     62 │     20.0% │    FL360 │    FL330 │     -30.0 FL │
└───────────────┴─────────┴────────┴───────────┴──────────┴──────────┴──────────────┘
```

---

## 4. Airspace Saturation, AUA Dwell, FRA & VHF Radio Comms

### `flightml fir-saturation` & `flightml aua-saturation`
```bash
flightml fir-saturation --period 202510 --threshold 40 --limit 3
flightml aua-saturation --period 202510 --type TMA --threshold 30
```

**Synthetic Output Table (`aua-saturation`):**
```text
      📡 ATC Unit Airspace (AUA) Hourly Saturation (Synthetic Sample)
┌──────────┬──────────┬───────────┬──────────┬──────────┬───────────┬──────────────┐
│ AUA ID   │ Type     │ Traversals│ Peak/Hr  │ Avg/Hr   │ Avg Dwell │ Status       │
├──────────┼──────────┼───────────┼──────────┼──────────┼───────────┼──────────────┤
│ EGTTTMA  │ TMA      │    48,200 │      112 │     64.8 │  16.4 min │ CRITICAL_SAT │
│ LFFFTMA  │ TMA      │    41,900 │       98 │     56.3 │  14.9 min │ HIGH_DENSITY │
│ EDDGTMA  │ TMA      │    35,100 │       84 │     47.2 │  13.8 min │ HIGH_DENSITY │
└──────────┴──────────┴───────────┴──────────┴──────────┴───────────┴──────────────┘
```

### `flightml flight-fra <ectrl_id>` & `flightml fra-evolution`
Snap 4D trajectory waypoints to published ATS airways and 5-letter RNAV fixes (`fix.dat` / `nav.dat`) to measure Free Route Airspace (`FRA`) directness.
```bash
flightml flight-fra 900104215 --period 202510
flightml fra-evolution --p1 201611 --p2 202510
```

### `flightml radio-workload`
Estimate ATC controller-pilot VHF voice channel occupancy (`%`) across Tower, Approach, and Area Control positions.
```bash
flightml radio-workload --airport EGLL --period 202510
```

---

## 5. Fuel Burn, Surface LTO Emissions, Contrails & CRCO Costs (`€ EUR`)

### `flightml fuel-impact <ectrl_id>`
Compute excess detour and holding fuel burn, $\text{CO}_2$, and weekly Jet-A spot fuel cost in Euros (`€`).

### `flightml taxi-out` & `flightml flight-taxi-out <ectrl_id>`
Quantify `KPI02 ATOT` departure runway queue delays and **ICAO EDB v32B** 7% LTO idle pollutants ($\text{NO}_x$, $\text{CO}$, $\text{HC}$, $\text{nvPM}$ soot) and Euro fuel cost.
```bash
flightml taxi-out --period 202510 --limit 3
```

**Synthetic Output Table:**
```text
    🛞 Airport Surface Taxi-Out (KPI02 ATOT) & LTO Idle Emissions (Synthetic)
┌─────────┬─────────┬───────────┬──────────┬───────────┬───────────┬──────────────┐
│ Airport │ Flights │ Mean Taxi │ UTOT p20 │ Add. ATOT │ Excess CO₂│ Queue Cost € │
├─────────┼─────────┼───────────┼──────────┼───────────┼───────────┼──────────────┤
│ EGLL    │  19,400 │  22.8 min │ 13.5 min │  +9.3 min │  742.0 t  │ €150,820 EUR │
│ LFPG    │  18,100 │  19.4 min │ 12.0 min │  +7.4 min │  538.5 t  │ €109,450 EUR │
│ EHAM    │  17,650 │  17.2 min │ 11.2 min │  +6.0 min │  412.0 t  │  €83,740 EUR │
└─────────┴─────────┴───────────┴──────────┴───────────┴───────────┴──────────────┘
```

### `flightml crco-cost <ectrl_id>` & `flightml route-avoidance`
Compute Eurocontrol CRCO Service Units (`SU`), national FIR charges (`€`), and Corridor Route-Charge Avoidance vs. Fuel Burn trade-offs.
```bash
flightml crco-cost 900104215 --period 202510
flightml route-avoidance --adep EGLL --ades LIRF --ac A320 --period 202510
```

### `flightml contrails <ectrl_id>`
Analyze upper-air tailwinds (`TAS` vs `GS`) and **Schmidt-Appleman persistent contrail (`ISSR`)** formation using Open-Meteo ERA5 upper-air meteorology.
```bash
flightml contrails 900104215 --period 202510
```

---

## 6. Safety, Conflict Detection & Prescriptive Operations Research

### `flightml airspace-conflicts`
Scan 1-minute interpolated 4D trajectories for co-altitude proximity pairs (`< 5 nm` horizontal, `< 1,000 ft` vertical) and $5^\circ \times 5^\circ$ airspace complexity.
```bash
flightml airspace-conflicts --period 202510 --hour 10 --sep-nm 5.0
```

### `flightml runway-safety <ectrl_id>`
Assess arrival runway braking action, density altitude, crosswinds, and landing distance safety margin (`LDA`).
```bash
flightml runway-safety 900104215 --period 202510
```

### `flightml tail-swaps`
Discover concrete same-operator, same-aircraft-type tail swaps (`Tail A ⇄ Tail B`) at hubs to absorb reactionary delays (`€88/min`).
```bash
flightml tail-swaps --period 202510
```

### `flightml gdp`
Run Ration-by-Schedule (`RBS`) Ground Delay Program slot optimization (`€ EUR` fuel savings).
```bash
flightml gdp --airport EGLL --aar 36 --period 202510
```
