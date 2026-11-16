# Test 1: Trajectory Physics & PRU Scorecards 🔬

**Domain Area:** Flight Efficiency & Trajectory Physics  
**Evaluation Date:** 2026-11-16 11:47:47 UTC  

## 1. Test Scenario Prompt
> Analyze the en-route horizontal and vertical flight efficiency for synthetic flight 900104215 (SYN A320, EGLL to LIRF, Great-Circle Distance: 780.4 NM). Filed en-route distance >40nm is 731.2 NM, and actual radar en-route distance >40nm is 720.4 NM. Compute Eurocontrol PRU KPI04 (KEP) and KPI05 (KEA), explain tactical ATC intervention, and evaluate terminal holding.

## 2. Five-Stage Comparative Results

### Stage 1: Deterministic Truth (FlightML CLI / Ground Truth Engine)
**Command Executed:** `"C:\Python314\python.exe" -m flightml.cli pru-scorecard 291262109 --json`  
```json
{
  "ectrl_id": 291262109,
  "period": "202510",
  "adep": "EYVI",
  "ades": "LGAV",
  "ac_operator": "AEE",
  "ac_type": "A20N",
  "great_circle_od_nm": 1004.2,
  "enroute_achieved_dist_nm": 926.1,
  "enroute_filed_dist_nm": 980.7,
  "enroute_actual_dist_nm": 977.4,
  "kpi04_kep_pct": 5.9,
  "kpi05_kea_pct": 5.54,
  "tactical_kea_minus_kep_pct": -0.36,
  "tactical_distance_delta_nm": -3.3,
  "enroute_routing_verdict": "HIGH_EFFICIENCY_ENROUTE",
  "asma_40nm_actual_min": 14.25,
  "asma_40nm_unimpeded_min": 9.15,
  "kpi08_asma_40nm_additional_min": 5.1,
  "asma_100nm_actual_min": 22.46,
  "arrival_tma_id": "LGAVTMA",
  "arrival_tma_dwell_min": 11.3,
  "requested_fl": 300.0,
  "max_actual_fl": 370.0,
  "median_cruise_fl": 370.0,
  "level_cap_delta_fl": 0.0,
  "is_level_capped": false,
  "lower_fir_dwell_min": 105.2,
  "upper_uir_dwell_min": 51.6,
  "level_cap_excess_fuel_kg": 0.0,
  "level_cap_excess_co2_kg": 0.0,
  "optimal_fl": 380.0
}
```

### Stage 2: Qwen 2.5 7B Base (Zero-Shot, No Tools) — Score: 60/100 (ACCEPTABLE)
**Latency:** 20.05s  

To analyze the en-route horizontal and vertical flight efficiency for synthetic flight 900104215 (SYN A320, EGLL to LIRF, Great-Circle Distance: 780.4 NM), we need to compute the Key Performance Indicators (KPIs) as defined by Eurocontrol PRU. Specifically, we will calculate KPI04 (KEP - En-Route Efficiency) and KPI05 (KEA - En-Route Accuracy).

### Step 1: Calculate KPI04 (KEP - En-Route Efficiency)

KEP is calculated using the formula:
\[ \text{KEP} = \frac{\text{Actual En-Route Distance}}{\text{Filed En-Route Distance}} \times 100 \]

Given:
- Filed en-route distance >40nm: 731.2 NM
- Actual radar en-route distance >40nm: 720.4 NM

\[ \text{KEP} = \frac{720.4 \text{ NM}}{731.2 \text{ NM}} \times 100 \]
\[ \text{KEP} = 0.9846 \times 100 \]
\[ \text{KEP} = 98.46\% \]

### Step 2: Calculate KPI05 (KEA - En-Route Accuracy)

KEA is calculated using the formula:
\[ \text{KEA} = \left( 1 - \left| \frac{\text{Actual En-Route Distance} - \text{Filed En-Route Distance}}{\text{Filed En-Route Distance}} \right| \right) \times 100 \]

Given:
- Filed en-route distance >40nm: 731.2 NM
- Actual radar en-route distance >40nm: 720.4 NM

\[ \text{KEA} = \left( 1 - \left| \frac{720.4 \text{ NM} - 731.2 \text{ NM}}{731.2 \text{ NM}} \right| \right) \times 100 \]
\[ \text{KEA} = \left( 1 - \left| \frac{-10.8 \text{ NM}}{

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 4/5 terms matched (20/25 pts)
- Mathematical Precision: 0/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 3: EuroAvia-7B (Fine-Tuned QLoRA, No Tools) — Score: 76/100 (GOOD)
**Latency:** 39.93s  

<thought>
1. Flight Identification:
   - ECTRL ID: 900104215 (SYN A320, registration G-SYNA)
   - City Pair: EGLL ➔ LIRF (Great-Circle Distance H = 780.4 NM)
   - En-Route Terminal Boundary (40 NM buffer deducted from each aerodrome): H_enroute = 700.4 NM
2. Horizontal En-Route Flight Efficiency Calculations:
   - Filed En-Route Distance (L): 731.2 NM
   - KPI04 (KEP): ((L - H_enroute) / H_enroute) * 100 = ((731.2 - 700.4) / 700.4) * 100 = 1.54%
   - Actual Radar Flown En-Route Distance (A): 720.4 NM
   - KPI05 (KEA): ((A - H_enroute) / H_enroute) * 100 = ((720.4 - 700.4) / 700.4) * 100 = 0.77%
3. Tactical ATC Attribution:
   - Delta = KEA - KEP = 0.77% - 1.54% = -0.77%
   - Distance Delta: -10.8 NM
   - Operational Verdict: ATC_DIRECT_SHORTCUT
4. Terminal & Vertical Efficiency:
   - KPI08 ASMA (40 NM Terminal Inbound): 10.9 min (+4.9 min additional delay holding vs unimpeded baseline)
   - KPI19 Cruise Level: Requested FL340 vs Max Actual FL350 (Capped: False)
</thought>
### Flight Efficiency & Eurocontrol PRU Scorecard: Flight #900104215

**Route Overview:** EGLL ➔ LIRF | **Operator:** SYN | **Aircraft:** A320 (`G-SYNA`)
**Great-Circle Distance:** 780.4 NM (En-Route Reference >40 NM: 700.4 NM)

#### 1. Horizontal En-Route Efficiency (ICAO GANP)
- **KPI0

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 5/5 terms matched (25/25 pts)
- Mathematical Precision: 11/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 4: Qwen 2.5 7B + FastMCP Live Tools — Score: 65/100 (ACCEPTABLE)
**Latency:** 16.55s | **Tool Called:** `get_pru_scorecard`  

Given the data provided, we will analyze the en-route horizontal and vertical flight efficiency for synthetic flight 900104215 (SYN A320, EGLL to LIRF, Great-Circle Distance: 780.4 NM) using Eurocontrol PRU KPIs (KEP and KEA). We will also discuss potential tactical ATC interventions and evaluate terminal holding based on the provided metrics.

### Step 1: Calculate KPI04 (KEP - En-Route Efficiency)

KEP (En-Route Efficiency) is calculated as follows:
\[ \text{KEP} = \frac{\text{Actual Radar En-Route Distance}}{\text{Filed En-Route Distance}} \times 100 \]

Given:
- Filed en-route distance >40nm: 731.2 NM
- Actual radar en-route distance >40nm: 720.4 NM

\[ \text{KEP} = \frac{720.4}{731.2} \times 100 \approx 98.46\% \]

### Step 2: Calculate KPI05 (KEA - En-Route Accuracy)

KEA (En-Route Accuracy) is calculated as follows:
\[ \text{KEA} = \left(1 - \left| \frac{\text{Actual Radar En-Route Distance} - \text{Filed En-Route Distance}}{\text{Filed En-Route Distance}} \right| \right) \times 100 \]

Given:
- Filed en-route distance >40nm: 731.2 NM
- Actual radar en-route distance >40nm: 720.4 NM

\[ \text{KEA} = \left(1 - \left| \frac{720.4 - 731.2}{731.2} \right| \right) \times 100 \]
\[ \text{KEA} = \left(1 - \left| \frac{-10.8}{731.2} \right| \right) \times 100 \]
\[ \text{KEA} = \left(1 - 0.0148 \right) \times 100 \]
\[ \text{KEA} \approx 98.

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 5/5 terms matched (25/25 pts)
- Mathematical Precision: 0/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 5: EuroAvia-7B + FastMCP Live Tools — Score: 76/100 (GOOD)
**Latency:** 35.96s | **Tool Called:** `get_pru_scorecard`  

<thought>
1. Flight Identification:
   - ECTRL ID: 900104215 (SYN A320, registration G-SYNA)
   - City Pair: EGLL ➔ LIRF (Great-Circle Distance H = 780.4 NM)
   - En-Route Terminal Boundary (40 NM buffer deducted from each aerodrome): H_enroute = 700.4 NM
2. Horizontal En-Route Flight Efficiency Calculations:
   - Filed En-Route Distance (L): 731.2 NM
   - KPI04 (KEP): ((L - H_enroute) / H_enroute) * 100 = ((731.2 - 700.4) / 700.4) * 100 = 1.54%
   - Actual Radar Flown En-Route Distance (A): 720.4 NM
   - KPI05 (KEA): ((A - H_enroute) / H_enroute) * 100 = ((720.4 - 700.4) / 700.4) * 100 = 0.77%
3. Tactical ATC Attribution:
   - Delta = KEA - KEP = 0.77% - 1.54% = -0.77%
   - Distance Delta: -10.8 NM
   - Operational Verdict: ATC_DIRECT_SHORTCUT
4. Terminal & Vertical Efficiency:
   - KPI08 ASMA (40 NM Terminal Inbound): 10.9 min (+4.9 min additional delay holding vs unimpeded baseline)
   - KPI19 Cruise Level: Requested FL340 vs Max Actual FL350 (Capped: False)
</thought>
### Flight Efficiency & Eurocontrol PRU Scorecard: Flight #900104215

**Route Overview:** EGLL ➔ LIRF | **Operator:** SYN | **Aircraft:** A320 (`G-SYNA`)
**Great-Circle Distance:** 780.4 NM (En-Route Reference >40 NM: 700.4 NM)

#### 1. Horizontal En-Route Efficiency (ICAO GANP)
- **KPI0

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 5/5 terms matched (25/25 pts)
- Mathematical Precision: 11/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

## 3. Critical Analytical Synthesis
- **Base Model Deviation:** Base Qwen 2.5 achieved 60/100 standalone, rising to 65/100 when augmented with live FastMCP telemetry.
- **EuroAvia Fine-Tuning Impact:** Standalone EuroAvia scored 76/100 without tools due to internalized European ATM domain knowledge and strict Euro formatting, and reached 76/100 when paired with live MCP data.
- **Currency & Standard Adherence:** EuroAvia strictly enforced Euro (`€ EUR`) calculations without foreign currency leakage.
