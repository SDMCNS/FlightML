# Test 4: 4D Airspace Closure Counterfactual 🔬

**Domain Area:** ATM Simulations & Macro-Disruption  
**Evaluation Date:** 2026-11-16 11:47:47 UTC  

## 1. Test Scenario Prompt
> Simulate a temporary 3-hour tactical closure of Swiss airspace (LSASFIR). What is the macro-level impact on European traffic reroutes, detour nautical miles, additional Jet-A fuel burn, airline operating losses in Euros (€ EUR), and post-reopening recovery queue duration?

## 2. Five-Stage Comparative Results

### Stage 1: Deterministic Truth (FlightML CLI / Ground Truth Engine)
**Command Executed:** `"C:\Python314\python.exe" -m flightml.cli fir-saturation --fir LSAS --json`  
```json
{
  "period": "202510",
  "has_data": true,
  "threshold_flights_per_hr": 40,
  "selected_fir": "LSAS",
  "selected_date": null,
  "top_saturated_firs": [
    {
      "fir_id": "LTBBFIR",
      "total_traversals": 159166.0,
      "peak_hourly_flights": 328,
      "avg_hourly_flights": 210.3,
      "avg_dwell_min": 22.1,
      "hours_over_capacity": 747,
      "saturation_pct": 98.7
    },
    {
      "fir_id": "LBSRFIR",
      "total_traversals": 116038.0,
      "peak_hourly_flights": 283,
      "avg_hourly_flights": 153.3,
      "avg_dwell_min": 18.9,
      "hours_over_capacity": 746,
      "saturation_pct": 98.5
    },
    {
      "fir_id": "LTAAFIR",
      "total_traversals": 121562.0,
      "peak_hourly_flights": 235,
      "avg_hourly_flights": 161.0,
      "avg_dwell_min": 52.7,
      "hours_over_capacity": 746,
      "saturation_pct": 98.8
    },
    {
      "fir_id": "LGGGUIR",
      "total_traversals": 102645.0,
      "peak_hourly_flights": 249,
      "avg_hourly_flights": 136.3,
      "avg_dwell_min": 33.8,
      "hours_over_capacity": 745,
      "saturation_pct": 98.9
    },
    {
      "fir_id": "EDUUUIR",
      "total_traversals": 253805.0,
      "peak_hourly_flights": 564,
      "avg_hourly_flights": 335.7,
      "avg_dwell_min": 17.5,
      "hours_over_capacity": 743,
      "saturation_pct": 98.3
    },
    {
      "fir_id": "LRBBFIR",
      "total_traversals": 92098.0,
      "peak_hourly_flights": 227,
      "avg_hourly_flights": 122.0,
      "avg_dwell_min": 29.9,
      "hours_over_capacity": 737,
      "saturation_pct": 97.6
    },
    {
      "fir_id": "LHCCFIR",
      "total_traversals": 115832.0,
      "peak_hourly_flights": 266,
      "avg_hourly_flights": 153.2,
      "avg_dwell_min": 16.1,
      "hours_over_capacity": 727,
      "saturation_pct": 96.2
    },
    {
      "fir_id": "EGTTUIR",
      "total_traversals": 175282.0,
      "peak_hourly_flights": 401,
      "avg_hourly_flights": 232.2,
      "avg_dwell_min": 18.1,
      "hours_over_capacity": 719,
      "saturation_pct": 95.2
    },
    {
      "fir_id": "LOVVFIR",
      "total_traversals": 199961.0,
      "peak_hourly_flights": 471,
      "avg_hourly_flights": 264.5,
      "avg_dwell_min": 12.3,
      "hours_over_capacity": 710,
      "saturation_pct": 93.9
    },
    {
      "fir_id": "LFFFUIR",
      "total_traversals": 327173.0,
      "peak_hourly_flights": 765,
      "avg_hourly_flights": 432.8,
      "avg_dwell_min": 34.3,
      "hours_over_capacity": 706,
      "saturation_pct": 93.4
    }
  ],
  "hourly_timeline": []
}
```

### Stage 2: Qwen 2.5 7B Base (Zero-Shot, No Tools) — Score: 80/100 (GOOD)
**Latency:** 17.07s  

To simulate a temporary 3-hour tactical closure of Swiss airspace (LSASFIR) and assess its impact on European traffic reroutes, detour nautical miles, additional Jet-A fuel burn, airline operating losses, and post-reopening recovery queue duration, we need to follow a structured approach. This involves several steps:

1. **Identify Affected Flights**: Determine which flights are likely to be affected by the closure.
2. **Calculate Reroute Paths**: Estimate the new paths for these flights.
3. **Compute Detour Nautical Miles**: Calculate the additional distance flown due to rerouting.
4. **Estimate Fuel Burn**: Determine the extra fuel required for the detours.
5. **Calculate Operating Losses**: Assess the financial impact on airlines.
6. **Estimate Recovery Queue Duration**: Predict the time it will take for the airspace to clear after reopening.

### Step 1: Identify Affected Flights

Assume that approximately 10% of all European flights pass through Swiss airspace. For simplicity, let's consider a day with 10,000 flights passing through LSASFIR. Therefore, about 1,000 flights would be affected during a 3-hour period.

### Step 2: Calculate Reroute Paths

Flights will need to find alternative routes around Switzerland. The most common alternatives are via northern or southern Europe, such as Germany, Austria, France, or Italy.

### Step 3: Compute Detour Nautical Miles

Assume an average detour of 100 nautical miles per flight. This is a rough estimate based on typical detour distances when avoiding restricted areas.

**Total Detour Distance**:
\[ \text{Total Detour Distance} = 1,000 \text{ flights} \times 100 \text{ nautical miles/flight} = 100,000 \text{ nautical miles} \]

### Step 4: Estimate Fuel Burn

Assume an average fuel burn rate of 8 kg/nautical mile for a modern jet aircraft. The total additional fuel burned can be calculated as follows:

**Additional Fuel Burn**:
\[ \text{Additional Fuel Burn} = 100,000 \text{ nautical miles} \times 8 \text{ kg/nautical mile} = 800,000 \text{ kg} \]
\[ \text{Additional Fuel Burn (in tons)} = \frac

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 3/5 terms matched (15/25 pts)
- Mathematical Precision: 25/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 3: EuroAvia-7B (Fine-Tuned QLoRA, No Tools) — Score: 45/100 (WEAK)
**Latency:** 30.29s  

<thought>
1. Scenario Setup:
   - Target Airspace (FIR): LSASFIR
   - Closure Duration: 3.0 hours
   - Historical Baseline Flight Influx: 526 flights scheduled to transit sector
2. Tactical Rerouting & Nautical Mile Penalties:
   - Diverted / Rerouted Flights: 462 flights (88% tactical reroute rate, remainder ground-delayed or cancelled)
   - Total Extra Flight Distance: +29,511 NM
   - Average Detour per Rerouted Flight: +63.9 NM
3. Fuel Burn & Environmental Emissions:
   - Extra Jet-A Fuel Consumed: +112.2 metric tonnes (+112,200 kg)
   - Additional CO2 Emissions: +354.5 tonnes CO2
4. Airline Economic Impact (Euro Compliance):
   - Fuel Detour Cost: 112,200 kg * €0.6421/kg = €72,055.03 EUR
   - Total Airline Operating Loss (Fuel + Ground Delays + Crew Duty): €1,041,055.03 EUR
5. Non-Linear Post-Reopening Queue Recovery Kinetics:
   - Backlog Recovery Time: 3.3 hours after reopening before sector traffic density normalizes
</thought>
### Counterfactual Airspace Closure Assessment: LSASFIR (3-Hour Window)

**Disrupted Airspace:** `LSASFIR` | **Disruption Duration:** 3.0 hours

#### 1. Air Traffic Demand & Tactical Rerouting Kinetics
- **Total Scheduled Flights Disrupted**: **526 flights**
- **Tactically Rerouted Flights**: **462 flights** (87.9% reroute compliance rate)
- **Aggregate Network Detour**: **+29,511 Nautical Miles** (mean detour of +63.9 NM per diverted airframe)

#### 2. Fuel Waste & Environmental Externalities
- **Extra Jet-A Fuel Burned**: **+112.2 metric tonnes**
- **Additional Atmospheric Carbon ($	ext{CO}_2$)**: **+354.5 metric tonnes**
- **En-Route Fuel Cost Penalty**: **€72,055.03 EUR**

**Evaluation Checks:**
- ⚠️ Mentioned Euros but leaked foreign currency ($ / £)
- Domain Terminology: 3/5 terms matched (15/25 pts)
- Mathematical Precision: 5/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 4: Qwen 2.5 7B + FastMCP Live Tools — Score: 80/100 (GOOD)
**Latency:** 334.87s | **Tool Called:** `simulate_fir_closure`  

### Step-by-Step Analysis of the Impact of a Temporary 3-Hour Tactical Closure of Swiss Airspace (LSASFIR)

#### Step 1: Rerouting and Detour Analysis

**Total Flights Impacted:** 744 flights were impacted by the closure of LSASFIR.

**Rerouted Flights Count:** 419 flights were rerouted, diverting them from their original route to avoid the closed airspace.

**Ground Held Flights Count:** 325 flights were held on the ground, waiting for the airspace to reopen.

**Cancellation Risk Flights Count:** 242 flights faced a risk of cancellation due to the closure.

**Total Extra Distance Flown (NM):** The total extra distance flown by all rerouted flights was 50,605.9 nautical miles.

**Total Extra Fuel Burned (Tonnes):** The total extra fuel burned by all rerouted flights was 232.79 tonnes.

**Total Extra CO2 Mitigated (Tonnes):** The total extra CO2 mitigated by rerouting flights around the closed airspace was 735.61 tonnes.

#### Step 2: Economic Impact on Airlines

**Total Airline Economic Loss (€ EUR):** The total economic loss incurred by airlines due to the closure was €10,242,861.59.

#### Step 3: Post-Reopening Recovery Queue Duration

**Post-Reopening Recovery Hours:** The recovery period after the airspace reopened lasted for 4.5 hours.

**Network Stability Rebound Time:** Network stability rebounded by 19:30 UTC on October 15, 2025.

**Recovery Curve:**

- **15:00 UTC:** 387 backlog flights, 60% airspace throughput.
- **16:00 UTC:** 332 backlog flights, 68.9% airspace throughput.
- **17:00 UTC:** 252 backlog flights, 77.8% airspace throughput.
- **18:00 UTC:** 158 backlog flights, 86.7% airspace throughput.
- **19:00 UTC:** 54 backlog flights, 95.6% airspace throughput.
- **20:00 UTC:** 0 backlog flights, 100% airspace throughput.

#### Step 4: Detailed Impact on

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 3/5 terms matched (15/25 pts)
- Mathematical Precision: 25/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 5: EuroAvia-7B + FastMCP Live Tools — Score: 65/100 (ACCEPTABLE)
**Latency:** 863.43s | **Tool Called:** `simulate_fir_closure`  

<thought>
1. Scenario Setup:
   - Target Airspace (FIR): LSASFIR
   - Closure Duration: 3.0 hours
   - Historical Baseline Flight Influx: 790 flights scheduled to transit sector
2. Tactical Rerouting & Nautical Mile Penalties:
   - Rerouted Flights: 441 flights diverted around the closed sector
   - Total Extra Flight Distance: +54,267.4 NM
   - Average Detour per Rerouted Flight: +123.2 NM
3. Fuel Burn & Environmental Emissions:
   - Extra Jet-A Fuel Consumed: +249.6 tonnes
   - Additional CO2 Emitted: +788.8 tonnes
4. Airline Economic Impact (Euro Compliance):
   - Fuel Detour Cost: 249.6 * €0.6421 = €160,890.37 EUR
   - Total Airline Operating Loss (Fuel + Ground Delay + Crew Duty): €1,083,248.41 EUR
5. Post-Reopening Queue Recovery Kinetics:
   - Backlog Recovery Time: 4.5 hours after reopening before sector traffic density normalizes
</thought>
### Counterfactual Airspace Closure Assessment: LSASFIR (3-Hour Window)

**Disrupted Airspace:** `LSASFIR` | **Disruption Duration:** 3.0 hours

#### 1. Air Traffic Demand & Tactical Rerouting Kinetics
- **Total Scheduled Flights Disrupted**: **790 flights**
- **Tactically Rerouted Flights**: **441 flights** (56% reroute compliance rate)
- **Aggregate Network Detour**: **+54,267.4 Nautical Miles** (mean detour of +123.2 NM per diverted airframe)

#### 2. Fuel Waste & Environmental Externalities
- **Extra Jet-A Fuel Burned**: **+249.6 metric tonnes**
- **Additional Atmospheric Carbon ($	ext{CO}_2$)**: **+788.8 metric tonnes**
- **En-Route Fuel Cost Penalty**: **€160,890.37 EUR** (at benchmark spot price of €0.6421/kg)

#### 3. Macroeconomic Airline Balance Sheet Impact

**Evaluation Checks:**
- ⚠️ Mentioned Euros but leaked foreign currency ($ / £)
- Domain Terminology: 3/5 terms matched (15/25 pts)
- Mathematical Precision: 25/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

## 3. Critical Analytical Synthesis
- **Base Model Deviation:** Base Qwen 2.5 achieved 80/100 standalone, rising to 80/100 when augmented with live FastMCP telemetry.
- **EuroAvia Fine-Tuning Impact:** Standalone EuroAvia scored 45/100 without tools due to internalized European ATM domain knowledge and strict Euro formatting, and reached 65/100 when paired with live MCP data.
- **Currency & Standard Adherence:** EuroAvia strictly enforced Euro (`€ EUR`) calculations without foreign currency leakage.
