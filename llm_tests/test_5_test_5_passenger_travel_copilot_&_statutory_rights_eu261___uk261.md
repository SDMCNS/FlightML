# Test 5: Passenger Travel Copilot & Statutory Rights (EU261 / UK261) 🔬

**Domain Area:** Passenger Rights & Multimodal Journey Risk  
**Evaluation Date:** 2026-11-16 11:47:47 UTC  

## 1. Test Scenario Prompt
> I am flying from Shannon (EINN) to Brussels (EBBR) via London Heathrow (EGLL) on a Tuesday morning with British Airways. I have a 75-minute scheduled layover at Heathrow T5. Assess my risk of missing the connection, evaluate the gate conformance rules at Heathrow, and explain my statutory compensation rights under EU261 / UK261 in Euros (€ EUR).

## 2. Five-Stage Comparative Results

### Stage 1: Deterministic Truth (FlightML CLI / Ground Truth Engine)
**Command Executed:** `"C:\Python314\python.exe" -m flightml.cli trip-risk --origin EINN --dest EBBR --hub EGLL --json`  
```json
{
  "itinerary_id": "EINN_EGLL_EBBR_Tue_2026-11-17_P202510",
  "origin": "EINN",
  "hub": "EGLL",
  "destination": "EBBR",
  "day_of_week": "Tuesday",
  "time_window": "morning",
  "target_date": "2026-11-17",
  "period": "202510",
  "period_selection_reason": "Explicitly configured period: 202510",
  "overall_delay_risk_score": 63.4,
  "overall_delay_risk_level": "ELEVATED",
  "legs": [
    {
      "origin": "EINN",
      "origin_name": "Shannon Airport",
      "destination": "EGLL",
      "destination_name": "London Heathrow Airport",
      "dominant_operator": "EIN",
      "operator_name": "Aer Lingus (operated as BAW codeshare)",
      "typical_aircraft": "A20N",
      "sample_size_flights": 89,
      "day_of_week": "Tuesday",
      "time_window": "Morning (06:00 - 11:00 UTC)",
      "mean_arr_delay_min": 5.8,
      "median_arr_delay_min": 1.0,
      "p80_arr_delay_min": 17.8,
      "p90_arr_delay_min": 32.2,
      "prob_delay_gt_15m_pct": 22.5,
      "prob_delay_gt_30m_pct": 13.5,
      "prob_delay_gt_60m_pct": 1.1,
      "cancellation_rate_pct": 1.2,
      "weather_risk_multiplier": 1.14,
      "weather_adjusted_prob_delay_pct": 25.6,
      "expected_delay_bracket": "15 - 35 min (Moderate delays likely)",
      "origin_weather": {
        "icao": "EINN",
        "airport_name": "Shannon Airport",
        "date_str": "2026-11-17",
        "temperature_c": 9.8,
        "wind_speed_kts": 9.5,
        "wind_gusts_kts": 19.1,
        "wind_dir_deg": 248.0,
        "precipitation_probability_pct": 88.0,
        "weather_description": "Overcast",
        "adverse_risk_score": 1.8,
        "crosswind_kts": 1.3,
        "risk_level": "LOW",
        "is_live_api": true,
        "api_endpoint": "https://api.open-meteo.com/v1/forecast (ECMWF / DWD ICON Ensemble)",
        "latitude_deg": 52.702,
        "longitude_deg": -8.92482,
        "forecast_iso_time": "2026-11-17T08:00",
        "weather_code": 3
      },
      "destination_weather": {
        "icao": "EGLL",
        "airport_name": "London Heathrow Airport",
        "date_str": "2026-11-17",
        "temperature_c": 12.0,
        "wind_speed_kts": 11.4,
        "wind_gusts_kts": 20.4,
        "wind_dir_deg": 242.0,
        "precipitation_probability_pct": 71.0,
        "weather_description": "Clear skies",
        "adverse_risk_score": 1.5,
        "crosswind_kts": 5.4,
        "risk_level": "LOW",
        "is_live_api": true,
        "api_endpoint": "https://api.open-meteo.com/v1/forecast (ECMWF / DWD ICON Ensemble)",
        "latitude_deg": 51.470748,
        "longitude_deg": -0.459909,
        "forecast_iso_time": "2026-11-17T10:00",
        "weather_code": 0
      }
    },
    {
      "origin": "EGLL",
      "origin_name": "London Heathrow Airport",
      "destination": "EBBR",
      "destination_name": "Brussels Airport",
      "dominant_operator": "BAW",
      "operator_name": "British Airways",
      "typical_aircraft": "A20N",
      "sample_size_flights": 12,
      "day_of_week": "Tuesd
... [truncated]
```

### Stage 2: Qwen 2.5 7B Base (Zero-Shot, No Tools) — Score: 48/100 (WEAK)
**Latency:** 18.37s  

To assess your risk of missing the connection from Shannon (EINN) to Brussels (EBBR) via London Heathrow (EGLL) with British Airways, we need to consider several factors including the flight schedules, gate conformance rules, and the potential for delays or cancellations.

### Step 1: Flight Schedules and Layover Time

First, let's assume you have the following flight details:
- **Shannon to Heathrow (EGLL):** BA101, departs EINN at 08:00, arrives at EGLL at 09:30.
- **Heathrow to Brussels (EBBR):** BA102, departs EGLL at 10:30, arrives at EBBR at 11:45.
- **Layover:** 75 minutes at EGLL.

Given these times, you would have exactly 75 minutes to transfer from one terminal to another at Heathrow. This is a very tight layover, especially considering the potential for delays due to baggage handling, security checks, and other transit procedures.

### Step 2: Gate Conformance Rules at Heathrow

Gate conformance rules at Heathrow dictate that airlines must ensure passengers can make their connecting flights within the published layover time. For Heathrow, the standard layover time for domestic-to-international connections is typically around 90 minutes, but this can vary based on the specific terminals and the airline's policies.

In your case, you have only 75 minutes, which is below the typical standard. This increases the risk of missing your connection significantly.

### Step 3: Risk Assessment

Given the tight layover time and the potential for delays, the risk of missing your connection is high. Factors such as:
- Baggage handling issues
- Security queue length
- Terminal congestion
- Airline operational issues (e.g., delayed or canceled flights)

All contribute to the likelihood of missing your connection.

### Step 4: Statutory Compensation Rights Under EU261/UK261

Under EU Regulation 261/2004 (EU261) and the UK equivalent, you have certain rights if you miss your connection due to the airline's fault. The airline must provide you with assistance, including meals and accommodation if necessary, until you reach your final destination.

If the delay is due to the airline's fault, you may be entitled to financial compensation

**Evaluation Checks:**
- ❌ Missing Euro currency citation
- Domain Terminology: 2/5 terms matched (10/25 pts)
- Mathematical Precision: 23/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 3: EuroAvia-7B (Fine-Tuned QLoRA, No Tools) — Score: 95/100 (EXCELLENT)
**Latency:** 33.64s  

<thought>
1. Itinerary Decomposition:
   - Leg 1: EINN ➔ EGLL (BAW, Day: Tuesday, Wave: morning)
   - Transfer Hub: EGLL (Scheduled Layover: 75 min, Terminal: T5)
   - Leg 2: EGLL ➔ EBBR
2. Hub Transfer & Minimum Connection Time (MCT) Analysis:
   - Hub: EGLL -> Nominal MCT = 60 min
   - Weather Transit Penalty: +0 min -> Effective MCT = 60 min
   - Effective Slack Buffer: Layover (75m) - Effective MCT (60m) = +15.0 min
   - Missed Connection Risk: 14.2%
3. Hub Terminal & Gate Conformance Nuances:
   - Terminal: T5 -> Internal Gates possible with 20 minutes remaining
   - External Gate transfer feasible with 45 minutes remaining
4. Statutory Passenger Rights (Regulation EC 261/2004 & UK261):
   - Great-Circle Distance: 912 km
   - Classification: Short-haul (≤ 1,500 km)
   - Statutory Cash Compensation: €250 EUR per passenger if arrival at final destination is delayed >= 3 hours
   - Duty of Care (Right to Care): Meals, refreshments, and communications after 2 hours delay
</thought>
### Passenger Travel Risk Advisory & Statutory Rights: EINN ➔ EGLL ➔ EBBR

**Route:** EINN ➔ EGLL (T5) ➔ EBBR | **Flight Window:** Tuesday Morning
**Scheduled Layover:** 75 minutes at EGLL | **Primary Carrier:** BAW

#### 1. Transfer Hub Feasibility & Connection Slack Analysis
- **Nominal Minimum Connection Time (MCT)**: **60 minutes** at EGLL
- **Effective Connection Slack**: **+15 minutes** buffer above MCT (75m layover − 60m MCT).
- **Missed Connection Probability**: **14.2%** (TIGHT connection with moderate misconnect risk).
- **Checked Luggage Transfer**: LOW RISK of luggage separation.

#### 2. Terminal Architecture & Gate Conformance Rules
> [!WARNING]
> **Terminal: T5** | **Gate

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 4/5 terms matched (20/25 pts)
- Mathematical Precision: 35/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 4: Qwen 2.5 7B + FastMCP Live Tools — Score: 95/100 (EXCELLENT)
**Latency:** 71.75s | **Tool Called:** `assess_passenger_itinerary`  

### Step-by-Step Analysis of Your Connection Risk and Statutory Compensation Rights

#### 1. **Assessment of Connection Risk**

**Leg 1: Shannon (EINN) to London Heathrow (EGLL)**
- **Expected Delay:** The average arrival delay at Heathrow is 5.8 minutes, with a p80 delay of 17.8 minutes.
- **Weather Impact:** The weather at Shannon is overcast with a high precipitation probability (88%), which could lead to delays due to adverse weather conditions.
- **Adverse Weather Score:** The Shannon airport has an adverse risk score of 1.8, indicating a low risk level.

**Leg 2: London Heathrow (EGLL) to Brussels (EBBR)**
- **Expected Delay:** The average arrival delay at Brussels is 8.2 minutes, with a p80 delay of 19.8 minutes.
- **Weather Impact:** The weather at Heathrow is described as dense drizzle with a high precipitation probability (86%), which could lead to delays.
- **Adverse Weather Score:** The Heathrow airport has an adverse risk score of 1.7, indicating a low risk level.

**Connection at Heathrow (EGLL):**
- **Scheduled Layover Time:** 75 minutes.
- **Minimum Connection Time (MCT):** 90 minutes.
- **Effective MCT:** 90.8 minutes (including weather penalties).
- **Slack Buffer:** -17.2 minutes (negative slack, indicating a tight connection).
- **Misconnect Probability:** 99%.
- **Luggage Transfer Risk:** 78%.
- **Gate Conformance:** British Airways operates biometric flight connections at Terminal 5 with a strict 20-minute gate conformance requirement.

**Conclusion:**
- The connection at Heathrow is highly risky, with a 99% misconnect probability and a negative slack of -17.2 minutes. The tight connection, combined with potential weather-related delays, significantly increases the risk of missing the connection.

#### 2. **Statutory Compensation Rights Under EU261 / UK261**

**Total Journey Distance:** 942 kilometers, classified as short-haul.

**Compensation Bracket:**
- **Compensation Amount:** €250.00 (under EU261/UK261).

**Delay Threshold:**
- **Delay Threshold:**

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 4/5 terms matched (20/25 pts)
- Mathematical Precision: 35/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 5: EuroAvia-7B + FastMCP Live Tools — Score: 95/100 (EXCELLENT)
**Latency:** 120.96s | **Tool Called:** `assess_passenger_itinerary`  

<thought>
1. Itinerary Decomposition:
   - Leg 1: EINN ➔ EGLL (Aer Lingus, A20N)
   - Transfer Hub: EGLL (Scheduled Layover: 75 min)
   - Leg 2: EGLL ➔ EBBR (British Airways, A20N)
2. Hub Transfer & Minimum Connection Time (MCT) Analysis:
   - Nominal MCT: 60 min
   - Weather Transit Penalty: +14 min (75m layover -> Effective MCT = 89 min)
   - Missed Connection Risk: 99.0%
   - Luggage Transfer Risk: 78.0%
3. Hub Terminal & Gate Conformance Nuances:
   - Heathrow T5 biometric connections enforce a strict 20-minute gate conformance cutoff.
   - Boarding closes 20 minutes before scheduled departure off-blocks.
   - Outbound taxi queue delay does NOT hold the gate open for connecting passengers.
4. Statutory Passenger Rights & Compensation Charter (EU261 / UK261):
   - Statutory Minimum Compensation: €250 EUR if arrival at final destination is delayed >= 3 hours, unless the delay was caused by extraordinary circumstances (e.g., unannounced ATC strikes or severe gale-force weather).
   - Right to Care: Free food, drink vouchers, 2 complimentary communications, and hotel accommodation if departure is delayed beyond 2 hours.
</thought>
### Passenger Travel Risk Advisory & Statutory Rights: EINN ➔ EGLL ➔ EBBR

**Route:** EINN ➔ EGLL (Hub) ➔ EBBR | **Flight Window:** Tuesday Morning
**Scheduled Layover:** 75 minutes at EGLL | **Primary Carrier:** British Airways

#### 1. Transfer Hub Feasibility & Connection Slack Analysis
- **Nominal Minimum Connection Time (MCT)**: **60 minutes** at EGLL
- **Effective Connection Slack**: **+29 minutes** buffer above nominal MCT (89 minutes total layover before gate conformance cutoff).
- **Missed Connection Probability**: **99.0%** (COMFORTABLE margin to miss connection under normal operating conditions).
- **Luggage Transfer Risk**: **78.0%** failure rate for luggage to change hands successfully.

#### 2. Terminal Architecture & Gate Conformance

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 4/5 terms matched (20/25 pts)
- Mathematical Precision: 35/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

## 3. Critical Analytical Synthesis
- **Base Model Deviation:** Base Qwen 2.5 achieved 48/100 standalone, rising to 95/100 when augmented with live FastMCP telemetry.
- **EuroAvia Fine-Tuning Impact:** Standalone EuroAvia scored 95/100 without tools due to internalized European ATM domain knowledge and strict Euro formatting, and reached 95/100 when paired with live MCP data.
- **Currency & Standard Adherence:** EuroAvia strictly enforced Euro (`€ EUR`) calculations without foreign currency leakage.
