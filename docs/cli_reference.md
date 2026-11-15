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

## 1. Data Ingestion & System Management

### `flightml external-data`
Compile bundled open aeronautical references (OpenAP performance envelopes, ICAO Engine Emissions Databank v32B, RNAV waypoints, radio navaids, and Eurocontrol CRCO unit rates) into DuckDB Parquet tables.
```bash
flightml external-data [--rebuild]
```

### `flightml ingest-archive <archive_path>`
Extract, classify, and ingest a monthly Eurocontrol traffic zip archive into partitioned Apache Parquet under `flightml/data/parquet/periods/{period_id}`.
```bash
flightml ingest-archive data/202510.zip
```

### `flightml convert`
Convert raw CSV files placed in `flightml/data/` to compressed, columnar Apache Parquet.
```bash
flightml convert [--include-trajectories]
```

### `flightml list-periods`
List all ingested and available historical aviation periods ready for analysis.
```bash
flightml list-periods
```

---

## 2. Dashboard & Web Services

### `flightml dashboard`
Launch the 4-tier interactive Streamlit simulation workbench (`http://localhost:8501`).
```bash
flightml dashboard [--port 8501]
```

### `flightml serve`
Launch the FastAPI REST server (`http://127.0.0.1:8000`).
```bash
flightml serve [--host 127.0.0.1] [--port 8000]
```

---

## 3. Dataset Summary & Search

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

## 4. ICAO GANP & Eurocontrol PRU KPI Commands

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

## 5. Airspace Saturation, AUA Dwell, FRA & VHF Radio Comms

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

## 6. Fuel Burn, Surface LTO Emissions, Contrails & CRCO Costs (`€ EUR`)

### `flightml fuel-impact <ectrl_id>`
Compute excess detour, holding, and vertical cruise altitude penalty fuel burn, $\text{CO}_2$, and weekly Jet-A spot fuel cost in Euros (`€`).
```bash
flightml fuel-impact 900104215 --period 202510
```

### `flightml fuel-prices`
Display official EIA / IATA historical weekly jet fuel spot prices (converted to Euros EUR) across all supported periods.
```bash
flightml fuel-prices
```

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
Analyze upper-air tailwinds (`TAS` vs `GS`), Equivalent Still Air Distance (`ESAD`), and **Schmidt-Appleman persistent contrail (`ISSR`)** formation using on-demand Open-Meteo upper-air meteorology.
```bash
flightml contrails 900104215 --period 202510
```

---

## 7. Safety, Conflict Detection & Prescriptive Operations Research

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
Run Ration-by-Schedule (`RBS`) Ground Delay Program slot optimization (`€ EUR` fuel savings) with an optional custom regulated arrival date.
```bash
flightml gdp --airport EGLL --aar 36 --date 2025-10-15 --period 202510
```

**Synthetic Output Table:**
```text
Ground Delay Program (GDP) Analysis: EGLL (Rate: 36 movements/hr)
              GDP Optimization Impact (2025-10-15 08:00 to 2025-10-15 14:00)              
┌──────────────────────────────────┬─────────────────────────────────────────────────────┐
│ Metric                           │ Value                                               │
├──────────────────────────────────┼─────────────────────────────────────────────────────┤
│ Total Flights Regulated          │ 214                                                 │
│ Ground Delay Assigned (at Origin)│ 2,420.0 min                                         │
│ Airborne Holding Delay Prevented │ 2,420.0 min                                         │
│ Total Jet-A Fuel Saved           │ 77,440.0 kg (77.4 metric tons)                      │
│ Total CO₂ Emissions Prevented    │ 244,710.4 kg (244.7 metric tons)                    │
│ Direct Fuel Cost Saved           │ €49,724.22 EUR                                      │
└──────────────────────────────────┴─────────────────────────────────────────────────────┘
```

---

## 8. Longitudinal Trends, Predictive Airspace, Airline Behavioral Economics & Delay Contagion (`v3`)

### `flightml longitudinal-timeline`
Display European airspace timeline & multi-period trends across all ingested archives (filtered by contemporary YoY cohort or historical baseline).
```bash
flightml longitudinal-timeline
```

**Synthetic Output Table:**
```text
                    European ATM Longitudinal Macro Evolution (Synthetic Sample)
┌─────────┬──────────────────────┬──────┬───────────────┬───────────────┬───────────┬────────────┬──────────────┐
│ Period  │ Season / Archetype   │ Days │ Total Flights │ Daily Flights │ Arr Delay │ OTP (≤15m) │ NewGen Share │
├─────────┼──────────────────────┼──────┼───────────────┼───────────────┼───────────┼────────────┼──────────────┤
│ 201611  │ Historical Baseline  │   30 │       697,326 │       23,244.2│  4.12 min │      85.4% │         1.5% │
│ 202402  │ Winter Trough        │   29 │       641,952 │       22,136.3│  3.84 min │      87.2% │        21.8% │
│ 202406  │ Summer Peak          │   30 │       935,934 │       31,197.8│ 12.31 min │      68.4% │        23.4% │
│ 202410  │ Autumn Shoulder      │   31 │       891,648 │       28,762.8│  5.18 min │      83.9% │        24.6% │
│ 202502  │ Winter Trough        │   28 │       656,458 │       23,444.9│  4.02 min │      86.5% │        25.7% │
│ 202506  │ Summer Peak          │   30 │       957,457 │       31,915.2│ 13.85 min │      65.1% │        27.2% │
│ 202510  │ Autumn Shoulder      │   31 │       934,141 │       30,133.6│  5.74 min │      82.1% │        28.2% │
└─────────┴──────────────────────┴──────┴───────────────┴───────────────┴───────────┴────────────┴──────────────┘
```

### `flightml seasonality`
Analyze 3-season annual network decomposition (Winter Trough, Summer Peak, Autumn Shoulder) and delay-to-demand elasticity.
```bash
flightml seasonality --year 2024
```

**Synthetic Output Table:**
```text
Top Surging Leisure Airports in Summer 2024 (vs Winter) (Synthetic Sample)
┌──────────────┬────────────────┬────────────────┬───────────────┬──────────────────┐
│ Airport ICAO │ Winter Flights │ Summer Flights │ Volume Delta  │ Summer Surge (%) │
├──────────────┼────────────────┼────────────────┼───────────────┼──────────────────┤
│ LEIB (Ibiza) │          1,840 │          8,312 │        +6,472 │          +351.7% │
│ LTAI (Antalya│          2,450 │         10,800 │        +8,350 │          +340.8% │
│ LEMH (Menorca│          1,120 │          4,225 │        +3,105 │          +277.2% │
│ LGIR (Herakl.│          1,950 │          7,150 │        +5,200 │          +266.7% │
│ LGKO (Kos)   │            890 │          3,120 │        +2,230 │          +250.6% │
└──────────────┴────────────────┴────────────────┴───────────────┴──────────────────┘
```

### `flightml compare-yoy`
Conduct controlled Year-over-Year (YoY) comparative analysis between matched seasonal archives (`202402 ➔ 202502`, `202406 ➔ 202506`, `202410 ➔ 202510`).
```bash
flightml compare-yoy --baseline 202410 --target 202510
```

**Synthetic Output Table:**
```text
           Macro Performance & Efficiency Comparison (YoY 202410 ➔ 202510)
┌───────────────────────────┬───────────────────┬──────────────────┬──────────────────────┐
│ Indicator                 │ 202410 (Baseline) │ 202510 (Target)  │ YoY Evolution        │
├───────────────────────────┼───────────────────┼──────────────────┼──────────────────────┤
│ Total Flights             │ 891,648           │ 934,141          │ +42,493 (+4.77%)     │
│ Daily Flight Rate         │ 28,762.8          │ 30,133.6         │ +1,370.8 (+4.77%)    │
│ Mean Arrival Delay        │ 5.18 min          │ 5.74 min         │ +0.56 min            │
│ On-Time (OTP-15)          │ 83.9%             │ 82.1%            │ -1.8%                │
│ New-Gen Fleet Penetration │ 24.64%            │ 28.17%           │ +3.53%               │
└───────────────────────────┴───────────────────┴──────────────────┴──────────────────────┘
```

### `flightml forecast-airspace`
Predict European airspace saturation, sector bottlenecks, and capacity breaches for future peak seasons.
```bash
flightml forecast-airspace --year 2026 --season summer --growth 3.5 --grid 1.0
```

**Synthetic Output Table:**
```text
         Projected Critical Airspace Sector Capacity Breaches (2026, Synthetic)
┌──────────────────┬─────────────┬───────────┬───────────┬─────────────────┬──────────────┬───────────────────────┐
│ Sector / FIR ID  │ Nominal Cap │ Hist Peak │ Proj Peak │ Utilization (%) │ Breach Level │ Est. Daily ATFM Delay │
├──────────────────┼─────────────┼───────────┼───────────┼─────────────────┼──────────────┼───────────────────────┤
│ EDUUUIR (Rhine)  │      140/hr │    138/hr │    152/hr │          108.6% │ BREACH       │             3,850 min │
│ LFFFFIR (Paris)  │      130/hr │    126/hr │    141/hr │          108.5% │ BREACH       │             3,420 min │
│ EGTTFIR (London) │      150/hr │    144/hr │    159/hr │          106.0% │ BREACH       │             3,110 min │
│ EDMMFIR (Munich) │      110/hr │    105/hr │    116/hr │          105.5% │ BREACH       │             2,480 min │
│ LIRRFIR (Rome)   │      100/hr │     98/hr │    107/hr │          107.0% │ SEVERE       │             1,950 min │
└──────────────────┴─────────────┴───────────┴───────────┴─────────────────┴──────────────┴───────────────────────┘
```

### `flightml flight-doc <ectrl_id>`
Evaluate Total Direct Operating Cost ($\text{DOC} = \text{Fuel } € + \text{Crew/Time } € + \text{CRCO Charges } €$) and route charge avoidance incentive ratio.
```bash
flightml flight-doc 900104215 --period 202510 --time-cost 850.0
```

**Synthetic Output Table:**
```text
                    Direct Operating Cost (€ EUR) Allocation (Synthetic Sample)
┌───────────────────────────────────┬───────────────┬────────────────┬─────────────────────────────────────────┐
│ Cost Component                    │  Cost (€ EUR) │  DOC Share (%) │ Economic Operational Driver             │
├───────────────────────────────────┼───────────────┼────────────────┼─────────────────────────────────────────┤
│ 1. Jet-A Fuel Burn Cost           │  €2,966.50    │          47.2% │ 4,620.0 kg Jet-A burned                 │
│ 2. Flight Time & Crew Cost        │  €1,732.50    │          27.6% │ 2.04 flight hours @ €850/hr             │
│ 3. Eurocontrol CRCO Route Charges │  €1,584.20    │          25.2% │ En-route navigation service charges     │
│ Total Direct Operating Cost       │  €6,283.20    │         100.0% │ Avoidance Incentive Ratio: €1.84/NM     │
└───────────────────────────────────┴───────────────┴────────────────┴─────────────────────────────────────────┘
```

### `flightml scan-avoidance`
Scan European network city pairs for systematic FIR route charge avoidance detours.
```bash
flightml scan-avoidance --period 202510 --min-flights 50 --detour-nm 15.0 --top 5
```

**Synthetic Output Table:**
```text
                 Detected Route Charge Avoidance Corridors (Synthetic Sample)
┌──────────┬──────────┬─────────┬───────────┬────────────┬───────────┬───────────┬─────────────┐
│ Corridor │ Analyzed │ Detours │ Share (%) │ Extra Dist │ Fee Saved │ Extra CO₂ │ Avoided FIR │
├──────────┼──────────┼─────────┼───────────┼────────────┼───────────┼───────────┼─────────────┤
│ EGLL-LIRF│      480 │     142 │     29.6% │   +28.4 NM │   €148.20 │   +295 kg │ LSAS (Swiss)│
│ EHAM-LEMD│      390 │      98 │     25.1% │   +22.1 NM │   €112.50 │   +228 kg │ LFFF (Paris)│
│ EDDB-LEPA│      310 │      64 │     20.6% │   +19.8 NM │    €86.40 │   +192 kg │ LFRR (Brest)│
│ EBBR-LEBL│      285 │      52 │     18.2% │   +18.2 NM │    €74.80 │   +174 kg │ LFFF (Paris)│
│ EGKK-AGKK│      190 │      28 │     14.7% │   +16.5 NM │    €62.10 │   +158 kg │ EISN (Shan.)│
└──────────┴──────────┴─────────┴───────────┴────────────┴───────────┴───────────┴─────────────┘
```

### `flightml simulate-harmonization`
Simulate Single European Sky (SES) unified CRCO en-route navigation flat-rate policy impact.
```bash
flightml simulate-harmonization --period 202510 --rate 72.50
```

**Synthetic Output Table:**
```text
Single European Sky (SES) Harmonized Rate Policy Simulation (Synthetic Sample)
Analyzed Period: 202510 | Harmonized Unit Rate: €72.50 / SU
Eliminated Detour Flights: 635,000 flights/yr | Annual Fuel Saved: 232,000 tonnes Jet-A
Annual CO₂ Mitigated: 733,120 tonnes CO₂ | Airline Net Direct Cost Benefit: €248,500,000.00 EUR
```

### `flightml delay-spreaders`
Identify super-spreader flights where morning departure delays infected downstream aircraft rotations on a given date.
```bash
flightml delay-spreaders --period 202510 --date 2025-10-15 --min-delay 20.0 --top 3
```

**Synthetic Output Table:**
```text
      Flights Generating Severe Downstream Reactionary Delay Contagion (Synthetic Sample)
┌──────────┬─────────────────┬────────────┬───────────────┬───────────────┬──────────────────┬─────────────────┬──────────────────────┐
│ Flight ID│ Tail / Airline  │ Origin Hub │ Initial Delay │ Infected Legs │ Downstream Delay │ Economic Cost € │ Infected Destination │
├──────────┼─────────────────┼────────────┼───────────────┼───────────────┼──────────────────┼─────────────────┼──────────────────────┤
│ 900104215│ G-SYNA (SYN)    │ EGLL ➔ EHAM│       +52 min │        3 legs │          4.8 hrs │     €30,240 EUR │ EHAM, EDDF, LEBL     │
│ 900112640│ EI-SYNB (SYN)   │ EIDW ➔ EGSS│       +46 min │        3 legs │          4.1 hrs │     €25,830 EUR │ EGSS, LEMD, LFPG     │
│ 900119004│ D-SYNE (SYN)    │ EDDF ➔ LIRF│       +38 min │        2 legs │          2.9 hrs │     €18,270 EUR │ LIRF, LGAV           │
└──────────┴─────────────────┴────────────┴───────────────┴───────────────┴──────────────────┴─────────────────┴──────────────────────┘
```

### `flightml delay-offenders`
Scan an entire reporting period for chronic repeat offender routes and operators causing systemic delay contagion.
```bash
# Repeat offender routes
flightml delay-offenders --period 202510 --type routes --top 3

# Repeat offender airline operators
flightml delay-offenders --period 202510 --type operators --top 3
```

**Synthetic Output Table (`--type routes`):**
```text
  City-Pair Routes Triggering Systemic Downstream Reactionary Delay Contagion (Synthetic Sample)
┌───────────┬────────────────────┬───────────────────┬──────────────────────────┬───────────────────┬─────────────────────┐
│ Route     │ Contagion Incident │ Avg Initial Delay │ Infected Downstream Legs │ Reactionary Delay │ Reactionary Cost €  │
├───────────┼────────────────────┼───────────────────┼──────────────────────────┼───────────────────┼─────────────────────┤
│ EIDW-EHAM │ 36 days            │ 48.6 min          │ 68 legs                  │ 27.1 hrs          │ €170,625 EUR        │
│ EDDB-LEPA │ 20 days            │ 57.5 min          │ 34 legs                  │ 10.6 hrs          │  €66,780 EUR        │
│ EGLL-EDDF │ 18 days            │ 42.1 min          │ 31 legs                  │  9.4 hrs          │  €59,220 EUR        │
└───────────┴────────────────────┴───────────────────┴──────────────────────────┴───────────────────┴─────────────────────┘
```

**Synthetic Output Table (`--type operators`):**
```text
   Airline Operators Triggering Systemic Downstream Delay Cascades (Synthetic Sample)
┌─────────┬──────────────────┬───────────────────┬───────────────┬─────────────────────┬───────────────────┬────────────────────┐
│ Airline │ Spreader Flights │ Avg Initial Delay │ Infected Legs │ Infection Multiplier│ Reactionary Delay │ Reactionary Cost € │
├─────────┼──────────────────┼───────────────────┼───────────────┼─────────────────────┼───────────────────┼────────────────────┤
│ RYR     │ 450              │ 34.2 min          │ 782 legs      │ 1.74x               │ 257.3 hrs         │ €1,620,990 EUR     │
│ EZY     │ 220              │ 36.8 min          │ 396 legs      │ 1.80x               │ 132.1 hrs         │   €832,230 EUR     │
│ KLM     │ 217              │ 38.5 min          │ 378 legs      │ 1.74x               │ 128.8 hrs         │   €811,440 EUR     │
└─────────┴──────────────────┴───────────────────┴───────────────┴─────────────────────┴───────────────────┴────────────────────┘
```

### `flightml simulate-shock`
Simulate systemic network delay contagion resulting from an airport runway or FIR airspace capacity shock.
```bash
flightml simulate-shock EGLL --type AIRPORT --duration 2.0 --cut 50.0 --date 2025-10-15 --period 202510
```

**Synthetic Output Table:**
```text
Network Shock & Systemic Delay Contagion Simulation: EGLL (AIRPORT) (Synthetic Sample)
Simulated Disruption: 2.0 hours @ 50% capacity cut (2025-10-15)
Directly Delayed Flights: 72 | Reactionary Infected Rotations: 148 | Total Impacted: 220 flights
Cumulative Delay: 184.5 flight hours | Economic Delay Cost: €1,162,350.00 EUR
Passenger Misconnections: 1,840 passengers | Cancellation Risk Flights: 14 flights
```

---

## 9. Model Context Protocol (MCP) Server

### `flightml mcp`
Launch the FlightML Model Context Protocol (MCP) server over standard I/O (`stdio`) or Server-Sent Events (`sse`), exposing FlightML trajectory analytics and simulation engines to local LLM hosts (LM Studio, Claude Desktop, Cursor).

```bash
# Standard I/O transport (default for LM Studio, Claude Desktop, Cursor)
flightml mcp

# Server-Sent Events (SSE) mode on port 8000
flightml mcp --transport sse --port 8000
```

**Options:**
- `--transport, -t`: Transport mechanism (`stdio`, `sse`, `streamable-http`) [default: `stdio`]
- `--port, -p`: Port for SSE transport [default: `8000`]
- `--help`: Show command documentation and exit.

---

## 10. Public Travel Copilot & Itinerary Risk

### `flightml trip-risk` (alias: `flightml passenger-itinerary`)
Assess multi-leg flight journey delay risk, missed connection probability, live 7-day meteorological forecast impact, EU261/UK261 passenger rights, and departure wave optimization for the general travelling public.

```bash
flightml trip-risk --origin EINN --dest EBBR --hub EGLL --day tuesday --time morning --layover 75 --date 2026-11-17 --debug
```

**Options:**
- `--origin, -o`: Departure airport 4-letter ICAO code [default: `EINN`]
- `--dest, -d`: Arrival airport 4-letter ICAO code [default: `EBBR`]
- `--hub`: Optional transfer connection hub 4-letter ICAO code [default: `None` for direct flights]
- `--day`: Day of week (`tuesday`, `monday`, `friday`, etc.) [default: `tuesday`]
- `--time`: Departure wave (`morning`, `midday`, `afternoon`, `evening`) [default: `morning`]
- `--layover, -l`: Scheduled layover duration at transfer hub in minutes [default: `75`]
- `--date`: Target flight date in `YYYY-MM-DD` format or relative keywords (`today`, `tomorrow`, `next tuesday`) [default: upcoming date]
- `--period`: Eurocontrol baseline data period (`202402`, `202406`, `202410`, `202502`, `202506`, `202510`, `201611`) [default: `auto` to automatically select the closest seasonal and most modern contemporary archive]
- `--debug`: Flag to print detailed diagnostic accounting of all historical records, DuckDB SQL queries, NWP weather endpoints, and MCT buffer calculations.

**Standard Advisory Output:**
```text
┌────────────────────── FlightML Public Travel Advisory ──────────────────────┐
│ 🧳 Passenger Journey & Delay Risk Copilot                                   │
│ Route: EINN ➔ EGLL (Hub) ➔ EBBR | Flight Date: 2026-11-17 (Tuesday) | Wave: │
│ Morning                                                                     │
│ Data Period: 202510 (Closest seasonal match (autumn shoulder season:        │
│ October 2025, archive 202510) for flight date 2026-11-17.)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                 ✈️ Flight Legs & Delay Risk (Tuesday morning)                 
┌─────────┬─────────┬─────────┬─────────┬────────┬─────────┬────────┬─────────┐
│ Leg     │ Operator│ Aircraft│ Avg/Med │ P90 Del│ Del>15m │ Weather│ Adjusted│
├─────────┼─────────┼─────────┼─────────┼────────┼─────────┼────────┼─────────┤
│ EINN➔   │ EIN     │ A20N    │ +5.8m / │ +32.2m │   22.5% │ Drizzle│   25.8% │
│ EGLL    │         │         │   +1.0m │        │         │ (G20kt)│         │
│ EGLL➔   │ BAW     │ A20N    │ +8.2m / │ +23.8m │   25.0% │ Overcast│  28.5% │
│ EBBR    │         │         │   +5.5m │        │         │ (G4kt) │         │
└─────────┴─────────┴─────────┴─────────┴────────┴─────────┴────────┴─────────┘
       ⏳ Connection Hub & Missed Flight Risk (London Heathrow Airport)        
┌──────────┬─────────┬──────────┬──────────┬───────────┬──────────┬───────────┐
│ Hub      │ Layover │ MCT      │ Slack    │ Misconnect│ Luggage  │ Status    │
├──────────┼─────────┼──────────┼──────────┼───────────┼──────────┼───────────┤
│ EGLL     │  75 min │   91 min │  -19 min │     99.0% │      78% │ CRITICAL  │
└──────────┴─────────┴──────────┴──────────┴───────────┴──────────┴───────────┘
⚠️ Inter-Terminal Transfer Required: EIN (T2) ➔ BAW (T5). Heathrow inter-terminal 
transfers require Flight Connections shuttle buses and transit security.
🌧️ Hub Weather Impact: Inbound Holding: ~2 min delay; Outbound taxi delays will 
NOT hold the gate (boarding closes strictly 20 minutes before departure).

         💶 Passenger Rights & Statutory Compensation (EU261 / UK261)          
┌───────────────┬──────────────┬───────────────┬──────────────┬───────────────┐
│ Route Distance│ Distance Tier│ Compensation  │ Delay Thresh │ Duty of Care  │
├───────────────┼──────────────┼───────────────┼──────────────┼───────────────┤
│        942 km │ Short-haul   │          €250 │ ≥ 3 hours    │ After 2 hours │
└───────────────┴──────────────┴───────────────┴──────────────┴───────────────┘

                      🕒 When Should I Fly? (EINN ➔ EGLL)                      
┌────────────┬─────────────┬───────────┬───────────┬────────────┬─────────────┐
│ Departure  │ Time Window │ Avg Delay │ P90 Delay │    On-Time │ Recommenda… │
├────────────┼─────────────┼───────────┼───────────┼────────────┼─────────────┤
│ Early Morn │ 06:00-09:00 │ +23.0 min │ +38.4 min │      50.0% │ CONGESTED   │
│ Midday Wave│ 09:00-13:00 │+214.0 min │+214.0 min │       0.0% │ ⚠️ HIGHEST  │
│ Afternoon  │ 13:00-17:00 │ -10.8 min │  -6.2 min │     100.0% │ CONGESTED   │
│ Evening    │ 17:00-22:00 │ -16.4 min │ -10.6 min │     100.0% │ 🌟 BEST (100│
└────────────┴─────────────┴───────────┴───────────┴────────────┴─────────────┘
```

**Diagnostic Audit Output (`--debug`):**
```text
================================================================================
🛠️ DETAILED FLIGHTML DIAGNOSTIC & DATA PROVENANCE AUDIT (--debug)
================================================================================

                             🗄️ Eurocontrol Period Selection Scoring Audit                             
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Period ┃ Year / Month     ┃ Season          ┃  Δ Month ┃  Δ Year ┃ Legacy Penalty ┃ Total Score ┃ Status    ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 201611 │ 2016 / November  │ autumn shoulder │       0m │     10y │            +50 │          60 │ Candidate │
│ 202402 │ 2024 / February  │ winter          │       3m │      2y │             +0 │          32 │ Candidate │
│ 202406 │ 2024 / June      │ summer peak     │       5m │      2y │             +0 │          52 │ Candidate │
│ 202410 │ 2024 / October   │ autumn shoulder │       1m │      2y │             +0 │          12 │ Candidate │
│ 202502 │ 2025 / February  │ winter          │       3m │      1y │             +0 │          31 │ Candidate │
│ 202506 │ 2025 / June      │ summer peak     │       5m │      1y │             +0 │          51 │ Candidate │
│ 202510 │ 2025 / October   │ autumn shoulder │       1m │      1y │             +0 │          11 │ 🏆 WINNER │
└────────┴━━━━━━━━━━━━━━━━━━┴━━━━━━━━━━━━━━━━━┴━━━━━━━━━━┴━━━━━━━━━┴━━━━━━━━━━━━━━━━┴━━━━━━━━━━━━━┴━━━━━━━━━━━┘

╭────────────────────────────────────── ✈️ Historical Records Audit: Leg 1 ──────────────────────────────────────╮
│ Leg 1: EINN ➔ EGLL (Shannon Airport to London Heathrow Airport)                                              │
│ Sampling Stage: Exact Day-of-Week and Departure Window (Commercial Filtered) | Matched Records: 89           │
│ DuckDB SQL: SELECT ectrl_id, ac_operator, ac_type, ac_reg, actual_off_block_time, actual_arrival_time,       │
│ arr_delay_min, dep_delay_min FROM flights WHERE adep = 'EINN' AND ades = 'EGLL' ...                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                     Sample Historical Flight Executions (EINN ➔ EGLL)                      
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┓
┃  ECTRL ID ┃ Off-Block (UTC)     ┃ Arrival (UTC)       ┃ Operator          ┃ Type ┃ Reg   ┃ Dep Delay ┃ Arr Delay ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━┩
│ 291268739 │ 2025-10-01 06:41:00 │ 2025-10-01 08:08:00 │ EIN (Aer Lingus)  │ A21N │ EILRA │    +1.0m  │   +19.0m  │
│ 291316499 │ 2025-10-07 06:33:00 │ 2025-10-07 07:56:00 │ EIN (Aer Lingus)  │ A21N │ EILRC │    -7.0m  │    +7.0m  │
│ 291364239 │ 2025-10-14 06:45:00 │ 2025-10-14 07:59:00 │ EIN (Aer Lingus)  │ A21N │ EILRA │    +5.0m  │   +10.0m  │
└━━━━━━━━━━━┴━━━━━━━━━━━━━━━━━━━━━┴━━━━━━━━━━━━━━━━━━━━━┴━━━━━━━━━━━━━━━━━━━┴━━━━━━┴━━━━━━━┴━━━━━━━━━━━┴━━━━━━━━━━━┘

                     🌦️ Live NWP Weather Observation & API Provenance Audit                     
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Airport        ┃ Role     ┃ Coordinates      ┃ NWP Provider / Endpoint        ┃ Matched UTC      ┃ Live/NWP ┃ Temp ┃ Winds (kts)         ┃ Rain % ┃ Conditions    ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ EINN (Shannon) ┃ Dep.     ┃ 52.70°, -8.92°   ┃ open-meteo.com (ECMWF/ICON)    ┃ 2026-11-17T08:00 ┃ LIVE API ┃ 9.9° │ 254° @ 10kt (G20kt) │    68% │ Light drizzle │
│ EGLL (Heathrow)┃ Hub      ┃ 51.47°, -0.46°   ┃ open-meteo.com (ECMWF/ICON)    ┃ 2026-11-17T10:00 ┃ LIVE API ┃ 8.5° │ 179° @ 04kt (G04kt) │    62% │ Overcast      │
│ EBBR (Brussels)┃ Dest.    ┃ 50.90°, 4.48°    ┃ open-meteo.com (ECMWF/ICON)    ┃ 2026-11-17T12:00 ┃ LIVE API ┃ 9.2° │ 178° @ 05kt (G10kt) │    35% │ Overcast      │
└━━━━━━━━━━━━━━━━┴━━━━━━━━━━┴━━━━━━━━━━━━━━━━━━┴━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┴━━━━━━━━━━━━━━━━━━┴━━━━━━━━━━┴━━━━━━┴━━━━━━━━━━━━━━━━━━━━━┴━━━━━━━━┴━━━━━━━━━━━━━━━┘
```

