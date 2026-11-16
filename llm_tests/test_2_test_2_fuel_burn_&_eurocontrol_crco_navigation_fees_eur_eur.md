# Test 2: Fuel Burn & Eurocontrol CRCO Navigation Fees (€ EUR) 🔬

**Domain Area:** Airline Economics & Navigation Charges  
**Evaluation Date:** 2026-11-16 11:47:47 UTC  

## 1. Test Scenario Prompt
> Calculate the Eurocontrol CRCO en-route navigation charges and direct fuel operating costs in Euros (€ EUR) for flight 900104215 (A320, MTOW 78.0 tonnes) operating from EGLL to LIRF across 1,445 km. Show the exact Service Unit formula and derive fuel cost at €0.6421/kg for 4,620 kg of Jet-A.

## 2. Five-Stage Comparative Results

### Stage 1: Deterministic Truth (FlightML CLI / Ground Truth Engine)
**Command Executed:** `"C:\Python314\python.exe" -m flightml.cli crco-cost 291262109 --json`  
```json
{
  "ectrl_id": 291262109,
  "period": "202510",
  "adep": "EYVI",
  "ades": "LGAV",
  "ac_operator": "AEE",
  "ac_type": "A20N",
  "mtow_tons": 79.0,
  "crco_weight_factor": 1.257,
  "actual_distance_nm": 1140.0,
  "actual_distance_km": 2111.3,
  "total_service_units": 26.54,
  "actual_crco_charge_eur": 1469.56,
  "filed_crco_charge_eur": 1590.93,
  "crco_charge_delta_eur": -121.37,
  "estimated_total_fuel_kg": 5350.4,
  "estimated_total_fuel_cost_eur": 3435.52,
  "excess_fuel_cost_eur": 38.47,
  "total_direct_route_cost_eur": 4905.08,
  "crco_share_of_direct_cost_pct": 30.0,
  "most_expensive_fir": "EPWWFIR",
  "fir_breakdown": [
    {
      "fir_id": "EYVLFIR",
      "zone_code": "EY",
      "country_name": "Lithuania",
      "dwell_min": 9.3,
      "estimated_distance_nm": 67.6,
      "estimated_distance_km": 125.2,
      "distance_factor": 1.252,
      "weight_factor": 1.257,
      "service_units": 1.574,
      "unit_rate_eur": 60.74,
      "charge_eur": 95.61
    },
    {
      "fir_id": "EYVLUIR",
      "zone_code": "EY",
      "country_name": "Lithuania",
      "dwell_min": 5.5,
      "estimated_distance_nm": 40.0,
      "estimated_distance_km": 74.1,
      "distance_factor": 0.741,
      "weight_factor": 1.257,
      "service_units": 0.931,
      "unit_rate_eur": 60.74,
      "charge_eur": 56.54
    },
    {
      "fir_id": "EPWWFIR",
      "zone_code": "EP",
      "country_name": "Poland",
      "dwell_min": 40.7,
      "estimated_distance_nm": 295.9,
      "estimated_distance_km": 548.0,
      "distance_factor": 5.48,
      "weight_factor": 1.257,
      "service_units": 6.889,
      "unit_rate_eur": 99.24,
      "charge_eur": 683.62
    },
    {
      "fir_id": "LZBBFIR",
      "zone_code": "LZ",
      "country_name": "Slovak Republic",
      "dwell_min": 6.9,
      "estimated_distance_nm": 50.2,
      "estimated_distance_km": 92.9,
      "distance_factor": 0.929,
      "weight_factor": 1.257,
      "service_units": 1.168,
      "unit_rate_eur": 76.22,
      "charge_eur": 89.01
    },
    {
      "fir_id": "LHCCFIR",
      "zone_code": "LH",
      "country_name": "Hungary",
      "dwell_min": 17.4,
      "estimated_distance_nm": 126.5,
      "estimated_distance_km": 234.3,
      "distance_factor": 2.343,
      "weight_factor": 1.257,
      "service_units": 2.945,
      "unit_rate_eur": 39.36,
      "charge_eur": 115.91
    },
    {
      "fir_id": "LRBBFIR",
      "zone_code": "LR",
      "country_name": "Romania",
      "dwell_min": 7.4,
      "estimated_distance_nm": 53.8,
      "estimated_distance_km": 99.6,
      "distance_factor": 0.996,
      "weight_factor": 1.257,
      "service_units": 1.252,
      "unit_rate_eur": 50.25,
      "charge_eur": 62.94
    },
    {
      "fir_id": "LYBAUIR",
      "zone_code": "LY",
      "country_name": "Serb.-Montenegro-KFOR",
      "dwell_min": 24.1,
      "estimated_distance_nm": 175.2,
      "estimated_distance_km": 324.5,
      "distance_factor": 3.245,
      "weight_factor": 1.257,
      "se
... [truncated]
```

### Stage 2: Qwen 2.5 7B Base (Zero-Shot, No Tools) — Score: 55/100 (ACCEPTABLE)
**Latency:** 16.81s  

To calculate the Eurocontrol CRCO en-route navigation charges and direct fuel operating costs for flight 900104215 (A320, MTOW 78.0 tonnes) operating from London Heathrow (EGLL) to Liège (LIRF) across 1,445 km, we will follow these steps:

### Step 1: Calculate En-Route Navigation Charges

The en-route navigation charge is calculated using the Service Unit (SU) formula provided by Eurocontrol. The formula is as follows:

\[ \text{En-Route Navigation Charge} = \text{Distance (km)} \times \text{Charge per SU} \]

For the A320, the charge per SU is €0.000025.

Given:
- Distance = 1,445 km
- Charge per SU = €0.000025

\[ \text{En-Route Navigation Charge} = 1,445 \, \text{km} \times 0.000025 \, \text{€/km} \]
\[ \text{En-Route Navigation Charge} = 0.036125 \, \text{€} \]

### Step 2: Calculate Direct Fuel Operating Costs

The direct fuel operating cost is calculated based on the fuel consumption and the price of jet fuel. The formula is:

\[ \text{Fuel Cost} = \text{Fuel Mass (kg)} \times \text{Price per kg} \]

Given:
- Fuel Mass = 4,620 kg
- Price per kg = €0.6421/kg

\[ \text{Fuel Cost} = 4,620 \, \text{kg} \times 0.6421 \, \text{€/kg} \]
\[ \text{Fuel Cost} = 2,978.062 \, \text{€} \]

### Summary

- **En-Route Navigation Charge**: €0.036125
- **Direct Fuel Operating Cost**: €2,978.062

Thus, the total direct fuel operating cost for the flight is €2,978.062, and the en-route navigation charge is €0.036125.

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 3/5 terms matched (15/25 pts)
- Mathematical Precision: 0/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 3: EuroAvia-7B (Fine-Tuned QLoRA, No Tools) — Score: 95/100 (EXCELLENT)
**Latency:** 40.57s  

<thought>
1. Flight & Airframe Parameters:
   - Aircraft: A320 (MTOW = 78.0 metric tonnes)
   - Route: EGLL ➔ LIRF (Actual Distance Flown = 1,445.0 km)
   - Jet-A Fuel Price: €0.6421 EUR/kg
2. Eurocontrol CRCO Service Unit (SU) Derivation:
   - Aircraft Weight Factor w = sqrt(MTOW / 50) = sqrt(78.0 / 50) = 1.2490
   - Distance Factor d = 1,445.0 km / 100 = 14.45
   - Total Service Units: SU = d * w = 18.04 SU
3. Navigation Charges:
   - Actual CRCO Fee: €1,282.52 EUR (Filed: €1,287.87 EUR, Delta: €-5.35 EUR)
4. Fuel Burn & Environmental Emissions (OpenAP & ICAO EDB v32B):
   - Total Fuel Burn: 4,620.0 kg -> €2,966.50 EUR
   - Detour & Holding Excess Fuel: 0.0 kg -> €0.00 EUR
   - CO2 Factor (3.16 kg CO2/kg fuel): 14,599.2 kg CO2
   - LTO NOx Emissions: 62.4 kg NOx
5. Total Direct Operating Cost (DOC) Allocation:
   - Total Route Cost = Fuel (€2,966.50) + CRCO (€1,282.52) = €4,249.02 EUR
   - CRCO Navigation Share: 30.1%
</thought>
### Flight Operating Economics & CRCO Cost Derivation: Flight #900104215

**City Pair:** EGLL ➔ LIRF | **Aircraft:** A320 (MTOW: 78.0 tonnes)
**Distance Traversed:** 1,445.0 km | **Jet-A Fuel Price:** €0.6421 EUR/kg

#### 1. Eurocontrol CRCO Navigation Charges Derivation
Eurocontrol route

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 5/5 terms matched (25/25 pts)
- Mathematical Precision: 30/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 4: Qwen 2.5 7B + FastMCP Live Tools — Score: 50/100 (ACCEPTABLE)
**Latency:** 17.7s | **Tool Called:** `get_crco_charges`  

Given that the flight data for flight 900104215 is not available in the specified period, I will provide a general method to calculate the Eurocontrol CRCO en-route navigation charges and direct fuel operating costs for an A320 aircraft with a maximum take-off weight (MTOW) of 78.0 tonnes, operating from London Heathrow (EGLL) to Liège (LIRF) over a distance of 1,445 km, using the provided fuel consumption and cost data.

### Step 1: Calculate Eurocontrol CRCO En-Route Navigation Charges

The CRCO (Chargeable Route Cost) for en-route navigation services is calculated using the following formula:

\[ \text{CRCO} = \text{Distance} \times \text{Unit Rate} \]

For en-route navigation charges, the unit rate is typically provided by Eurocontrol. As of the latest data, the en-route navigation charge per kilometer is €0.000125. Therefore, the calculation would be:

\[ \text{CRCO} = 1,445 \, \text{km} \times €0.000125/\text{km} \]
\[ \text{CRCO} = €0.180625 \]

Rounding to two decimal places, the CRCO is:

\[ \text{CRCO} = €0.18 \]

### Step 2: Calculate Direct Fuel Operating Costs

To calculate the direct fuel operating costs, we need to determine the amount of fuel consumed during the flight and then multiply it by the cost per kilogram of fuel.

#### Step 2.1: Determine Fuel Consumption

The fuel consumption can be estimated using the specific range of the aircraft and the distance flown. The specific range of an A320 is approximately 1,200 km per tonne of fuel. Given the MTOW of 78.0 tonnes, the fuel consumption for 1,445 km can be calculated as follows:

\[ \text{Fuel Consumption} = \frac{\text{Distance}}{\text{Specific Range}} \times \text{MTOW} \]
\[ \text{Fuel Consumption} = \frac{1,445 \, \text{km}}{1,200 \, \text{km/tonne}} \times 78.0 \, \text{

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 2/5 terms matched (10/25 pts)
- Mathematical Precision: 0/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

### Stage 5: EuroAvia-7B + FastMCP Live Tools — Score: 95/100 (EXCELLENT)
**Latency:** 37.96s | **Tool Called:** `get_crco_charges`  

<thought>
1. Flight & Airframe Parameters:
   - Aircraft: A320 (MTOW = 78.0 metric tonnes)
   - Route: EGLL ➔ LIRF (Actual Distance Flown = 1,445.0 km)
   - Jet-A Spot Price: €0.6421 EUR/kg
2. Eurocontrol CRCO Service Unit (SU) Derivation:
   - Aircraft Weight Factor w = sqrt(MTOW / 50) = sqrt(78.0 / 50) = 1.2490
   - Distance Factor d = 1,445.0 km / 100 = 14.45
   - Total Service Units: SU = d * w = 18.04 SU
3. Navigation Charges:
   - Actual CRCO Fee: €1,282.52 EUR (Filed: €1,287.81 EUR, Delta: €-5.29 EUR)
4. Fuel Burn & Environmental Emissions (OpenAP & ICAO EDB v32B):
   - Total Fuel Burn: 4,620.0 kg -> €2,966.50 EUR
   - Detour & Holding Excess Fuel: 0.0 kg -> €0.00 EUR
   - CO2 Factor (3.16 kg CO2/kg fuel): 14,599.2 kg CO2
   - LTO NOx Emissions: 62.4 kg NOx
5. Total Direct Operating Cost (DOC) Allocation:
   - Total Route Cost = Fuel (€2,966.50) + CRCO (€1,282.52) = €4,249.02 EUR
   - CRCO Navigation Share: 30.2%
</thought>
### Flight Operating Economics & CRCO Cost Derivation: Flight #900104215

**City Pair:** EGLL ➔ LIRF | **Aircraft:** A320 (MTOW: 78.0 tonnes)
**Distance Traversed:** 1,445.0 km | **Jet-A Fuel Price:** €0.6421 EUR/kg

#### 1. Eurocontrol CRCO Navigation Charges Derivation
Eurocontrol route

**Evaluation Checks:**
- ✓ 100% Strict Euro (€ EUR) Discipline (No foreign currency leakage)
- Domain Terminology: 5/5 terms matched (25/25 pts)
- Mathematical Precision: 30/35 pts based on exact ground truth numbers
- ✓ Rigorous step-by-step structure (15/15 pts)

## 3. Critical Analytical Synthesis
- **Base Model Deviation:** Base Qwen 2.5 achieved 55/100 standalone, rising to 50/100 when augmented with live FastMCP telemetry.
- **EuroAvia Fine-Tuning Impact:** Standalone EuroAvia scored 95/100 without tools due to internalized European ATM domain knowledge and strict Euro formatting, and reached 95/100 when paired with live MCP data.
- **Currency & Standard Adherence:** EuroAvia strictly enforced Euro (`€ EUR`) calculations without foreign currency leakage.
