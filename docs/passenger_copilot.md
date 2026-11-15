# Passenger Travel Copilot & Multi-Leg Journey Predictor 🧳

Welcome to the **FlightML Passenger Travel Copilot & Multi-Leg Journey Predictor** guide. This module translates millions of historical Eurocontrol DDR2 4D radar trajectories, real-time Open-Meteo numerical weather prediction (NWP) forecasts, airport hub terminal layouts, and European consumer aviation law (EU261 / UK261) into actionable, public-facing journey intelligence for the general travelling public.

---

> [!IMPORTANT]
> ### 🔒 Synthetic Example Outputs & Euro Currency Compliance Policy
> In accordance with Eurocontrol data dissemination terms, commercial confidentiality, and GDPR requirements:
> - **All sample flight records, tail registrations (e.g. `G-SYNA`, `EI-SYNB`), and aggregate punctuality statistics** illustrated in this guide are realistic synthetic records.
> - **All financial compensation figures and delay liabilities** are denominated exclusively in **Euros (`€`, `EUR`)**.

---

## 🧭 Why a Passenger Travel Copilot?

Traditional flight-search engines and airline status pages show static schedules (e.g., *"Departure: 06:45, Arrival: 08:05"*). Airlines frequently advertise aggregate on-time performance figures (e.g., *"85% on-time"*), which obscure critical real-world failure modes:

1. **Rotational Delay Accumulation (Tail-Tracking)**: Commercial narrowbody airliners (A320, B737) fly 4 to 8 consecutive legs per day. A delay incurred during a morning slot propagates down the entire rotational sequence as reactionary delay. An evening flight on an airframe that incurred delays in Frankfurt or Madrid will depart late, even if local weather at your departure airport is perfectly clear.
2. **Intermediate Hub Transfer Vulnerability**: Booking a tight connection (e.g., 75 minutes at London Heathrow or Paris Charles de Gaulle) carries hidden risks. Even if the inbound leg is delayed by only 25 minutes, strict **biometric gate conformance rules (closing 20 minutes before off-blocks)**, airside terminal shuttle bus rides, and transit security queues reduce effective slack to zero, resulting in a missed connection.
3. **Compound Hub Weather Vulnerability**: Adverse weather at an intermediate transfer hub has a compound triple impact:
   - **Inbound airborne holding** / speed vectoring delays arrival.
   - **Tarmac ramp transit** is slowed by slippery taxiways and safety speed restrictions on airside transfer buses (+5 to +15 min transit penalty).
   - **Outbound departure taxi queues** add ground delay, but gate agents will **not** hold the gate open past boarding closure.
4. **Checked Luggage Transfer Deficit**: Even when an agile passenger sprints through the terminal to make the connecting flight, automated baggage sorting systems and tarmac transfer tugs require a physical transfer window. If inbound delay encroaches on baggage transfer time, your luggage will be left behind.
5. **Opaque Consumer Rights (EU261 / UK261)**: Passengers rarely know their statutory compensation rights (€250 to €600) or Duty of Care entitlements (meals, refreshments, hotel accommodation) when disruptions occur.

FlightML’s Passenger Copilot unifies these disparate dimensions into a single decision engine.

---

## 🏛️ System Architecture & Mechanics

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      FLIGHTML PASSENGER JOURNEY ENGINE ARCHITECTURE                    │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│   ATM BIG DATA BASELINE  │   LIVE METEOROLOGY (NWP) │   HUB TERMINALS & EU261 CHARTER  │
├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
│ • DuckDB Columnar Store  │ • Open-Meteo 7-Day ECMWF │ • Minimum Connection Times (MCT) │
│ • Intelligent Period Auto│ • Surface Wind & Gusts   │ • Terminal Transfer Matrices     │
│ • Commercial PAX Filter  │ • Crosswind Component    │ • Heathrow T5 Biometric Conformance│
│ • Diurnal Wave Profiles │ • Convective Weather/Rain│ • Double Hub Weather Penalty     │
│ • Empirical Delay Curves │ • Ceiling & Visibility   │ • Geodesic EU261 / UK261 Rights €│
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATED JOURNEY DELAY ASSESSMENT                            │
├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
│ • Leg 1 & 2 Delay Probs  │ • Missed Connection Risk │ • Baggage Misconnect Probability │
│ • P50/P90 Quantile Delays│ • Effective Slack Buffer │ • Diurnal Wave Recommendations   │
│ • Weather Attribution    │ • Gate Conformance Alert │ • Full Provenance Audit (--debug)│
└──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

### 1. Intelligent Period Auto-Selection (`period="auto"`)

Eurocontrol archives are organized into discrete monthly periods (e.g., `202510`, `202506`, `202410`, `202402`, `201611`). When a passenger enters a target travel date (such as a flight in November 2026), the engine automatically selects the best matching historical archive using a multi-factor penalty scoring function:

$$\text{Score}(P) = \Delta M(P, \text{Target}) \times 10.0 + |\text{Year}(P) - \text{Year}(\text{Target})| \times 2.0 + \text{Penalty}_{\text{pre-covid}}$$

Where:
- **Circular Month Distance ($\Delta M$)**: Captures seasonal atmospheric conditions, convective storms, jet stream shifts, and seasonal scheduling banks:
  $$\Delta M(m_1, m_2) = \min(|m_1 - m_2|,\; 12 - |m_1 - m_2|)$$
  *(e.g., November ($m=11$) has $\Delta M = 1$ with October ($m=10$), making October the closest seasonal proxy).*
- **Year Distance**: Minimizes structural shifts in airline fleet modernization (e.g. A320neo / B737 MAX penetration) and airspace sector redesign.
- **Pre-COVID Penalty**: Adds $+50.0$ penalty points to archives prior to 2020 (such as `201611`) to ensure contemporary post-pandemic operational baselines are prioritized.

### 2. Commercial Passenger vs. Cargo & General Aviation Filtering

On major European trunk corridors, freight and logistics carriers operate significant night-time and scheduled cargo feeder flights (e.g. European Air Transport / DHL `BCS` operating A300/B757 freighters, UPS `UPS`, FedEx `FDX`, Star Air `SRR`).

When evaluating passenger journey risk, including cargo flights distorts delay distributions because freighter operations operate under different curfew constraints, slot priorities, and off-peak ground handling. FlightML automatically filters out known cargo, air ambulance, and general aviation operators, selecting the primary scheduled commercial passenger airlines (e.g. British Airways `BAW`, Aer Lingus `EIN`, Lufthansa `DLH`, Air France `AFR`, KLM `KLM`, Iberia `IBE`).

### 3. Multi-Leg Transfer Hubs & Connection Slack Dynamics

For connecting itineraries ($Leg_1 \to Hub \to Leg_2$), journey feasibility depends on the interaction between inbound arrival delay, nominal airport transfer time, weather-induced transit delays, and outbound boarding cutoffs:

$$\text{Effective Slack Buffer} = T_{\text{layover}} - \text{Eff MCT} - D_{\text{arr}, \text{inbound}}$$

Where:
- **$T_{\text{layover}}$**: Scheduled layover duration between scheduled inbound on-blocks and scheduled outbound off-blocks.
- **$\text{Eff MCT}$**: Effective Minimum Connection Time, computed as:
  $$\text{Eff MCT} = \text{MCT}_{\text{nominal}} + \Delta T_{\text{bus, weather}}$$
- **$D_{\text{arr}, \text{inbound}}$**: Expected or P90 arrival delay of the first leg.

#### Airport Nominal Minimum Connection Times (MCT):
| Airport ICAO | Name | Nominal MCT | Terminal Transfer Architecture |
| :--- | :--- | :--- | :--- |
| **EGLL** | London Heathrow | 60 min | T5 Biometric Conformance; Inter-terminal airside shuttle buses |
| **LFPG** | Paris Charles de Gaulle | 60 min | CDGVAL light rail / inter-terminal shuttle bus network |
| **EHAM** | Amsterdam Schiphol | 50 min | Single-terminal pier concept; long walking distances (Piers B–G) |
| **EDDF** | Frankfurt am Main | 45 min | Skyline elevated transit train between T1 and T2 |
| **LEMD** | Madrid Barajas | 45 min | Automated people mover (APM) connecting T4 and T4S satellite |
| **EIDW** | Dublin Airport | 45 min | Covered pedestrian link corridor between T1 and T2 |
| **Others** | European Standard | 45 min | Standard medium/large European hub transfer |

#### Strict Biometric Gate Conformance Rule:
At modern European hubs such as London Heathrow Terminal 5:
> [!WARNING]
> **Strict 20-Minute Gate Conformance Cutoff**:
> Boarding gates and biometric flight connections close strictly **20 minutes before scheduled departure off-blocks**. Even if the outbound flight incurs a 30-minute ground taxi queue before takeoff, gate agents will **not** hold the gate open. A passenger who arrives at the gate 19 minutes before departure is offloaded and their seat reallocated.

#### Compound Hub Weather Impact:
Adverse weather at an intermediate transfer hub exerts a compound triple impact:
1. **Inbound Airborne Holding Delay**: Strong surface winds, low visibility, or reduced arrival acceptance rates force air traffic control into holding stacks (e.g., BNN, LAM, BIG, OCK at Heathrow), adding $+5$ to $+20$ minutes of delay before landing.
2. **Ramp Transit / Shuttle Bus Penalty**: Wet, icy, or gale-force tarmac conditions enforce apron speed limits and ground service equipment (GSE) restrictions. Airside inter-terminal shuttle buses encounter delays, adding $+5$ to $+15$ minutes of transit penalty ($\Delta T_{\text{bus, weather}}$).
3. **Outbound Departure Taxi Queue**: While departure queues lengthen takeoff waiting time, boarding conformance does not extend.

#### Quantitative Luggage Misconnect Risk:
Luggage transfer requires baggage handlers to unload containers from the inbound aircraft, transfer them across the tarmac or through automated underground baggage tunnels, sort them, and load them onto the outbound aircraft.
- **Effective Luggage Buffer**: $\text{Buffer}_{\text{luggage}} = T_{\text{layover}} - D_{\text{arr}, \text{inbound}} - 35\text{ min}$.
- If $\text{Buffer}_{\text{luggage}} \ge 30\text{ min}$: **`LOW_RISK`** ($<15\%$ risk of baggage mishandling).
- If $10\text{ min} \le \text{Buffer}_{\text{luggage}} < 30\text{ min}$: **`MODERATE_RISK`** ($35\%–50\%$ risk of delayed baggage).
- If $\text{Buffer}_{\text{luggage}} < 10\text{ min}$: **`HIGH_RISK`** ($>75\%$ probability that luggage will miss the connecting flight).

### 4. Statutory Passenger Rights & Compensation Charter (EU261 / UK261)

FlightML includes a built-in regulatory compliance engine based on **Regulation (EC) No 261/2004** and **The Air Passenger Rights and Air Travel Organisers' Licensing (Amendment) (EU Exit) Regulations 2019 (UK261)**.

#### Distance Tiers & Cash Compensation:
Compensation is determined by the Great-Circle Geodesic distance between the original departure aerodrome and the final destination aerodrome:

| Distance Tier | Great-Circle Distance ($d$) | Statutory Cash Compensation | Delay Threshold at Final Destination |
| :--- | :--- | :--- | :--- |
| **Short-Haul** | $d \le 1,500\text{ km}$ | **€250.00 EUR** | Delay $\ge 3\text{ hours}$ |
| **Medium-Haul** | $1,500\text{ km} < d \le 3,500\text{ km}$ | **€400.00 EUR** | Delay $\ge 3\text{ hours}$ |
| **Long-Haul** | $d > 3,500\text{ km}$ | **€600.00 EUR** | Delay $\ge 4\text{ hours}$ (50% reduction to €300 if 3–4h) |

#### Duty of Care (Right to Care):
Regardless of whether delay is caused by extraordinary circumstances, airlines have an unconditional statutory Duty of Care:
- **Free Meals and Refreshments**: In reasonable relation to the waiting time:
  - After **2 hours delay** for flights $\le 1,500\text{ km}$.
  - After **3 hours delay** for flights between $1,500\text{ km}$ and $3,500\text{ km}$.
  - After **4 hours delay** for flights $> 3,500\text{ km}$.
- **Communications**: Two free telephone calls, telex or fax messages, or emails.
- **Hotel Accommodation & Transport**: Obligatory if an overnight stay or a stay additional to that intended by the passenger becomes necessary.

### 5. Diurnal Departure Wave Optimization ("When Should I Fly?")

Airline on-time performance follows a well-documented diurnal delay curve. FlightML breaks down delay distributions across five distinct departure waves:

| Departure Wave | Time Window (Local) | Rotational State | Typical Operational Characteristics |
| :--- | :--- | :--- | :--- |
| **Early Morning** | 05:00 – 08:59 | First Wave | Clean airframes departing after overnight maintenance. **Lowest delay risk.** |
| **Midday** | 09:00 – 12:59 | Wave 2 | Minor turnaround queues; morning slots absorbing minor weather holds. |
| **Afternoon** | 13:00 – 16:59 | Wave 3 | Convective summer weather build-up; European sector congestion peaks. |
| **Evening** | 17:00 – 20:59 | Peak Wave 4 | **Highest delay risk.** Accumulated reactionary delays from 3+ preceding legs. |
| **Night** | 21:00 – 04:59 | Curfew Wave | High cancellation risk if inbound delays encroach on airport night curfews. |

---

## 💻 Command-Line Interface (`flightml trip-risk`)

The Passenger Copilot can be invoked directly from the terminal via `flightml trip-risk` (or alias `flightml passenger-itinerary`).

### Syntax & Available Options

```bash
flightml trip-risk [OPTIONS]
```

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--origin` | `-o` | `EINN` | Departure airport 4-letter ICAO code. |
| `--dest` | `-d` | `EBBR` | Arrival destination airport 4-letter ICAO code. |
| `--hub` | | `None` | Optional intermediate transfer hub ICAO code (omit for direct flights). |
| `--day` | | `tuesday` | Day of week (`monday`, `tuesday`, `wednesday`, `thursday`, `friday`, `saturday`, `sunday`). |
| `--time` | | `morning` | Target departure wave (`morning`, `midday`, `afternoon`, `evening`, `night`). |
| `--layover` | `-l` | `75` | Scheduled layover duration at transfer hub in minutes. |
| `--operator` | | `None` | Preferred airline operator 3-letter ICAO code (e.g. `BAW`, `EIN`). If omitted, selects top commercial carrier. |
| `--date` | | `None` | Specific flight date (`YYYY-MM-DD`, `today`, `tomorrow`, `next tuesday`). Enables automatic seasonal archive selection. |
| `--period` | | `auto` | Historical Eurocontrol archive period (`auto`, `202510`, `202410`, etc.). |
| `--debug` | | `False` | Enables detailed audit trail showing sampled flight records, weather queries, and MCT formulas. |

---

### Example 1: Multi-Leg Itinerary (Shannon to Brussels via London Heathrow)

Evaluating an itinerary from Shannon (`EINN`) to Brussels (`EBBR`) via London Heathrow (`EGLL`) on Tuesday morning:

```bash
flightml trip-risk --origin EINN --dest EBBR --hub EGLL --day tuesday --time morning --layover 75 --date 2026-11-17
```

```text
╭─────────────────────────────────────── FlightML Public Travel Advisory ───────────────────────────────────────╮
│ 🧳 Passenger Journey & Delay Risk Copilot                                                                     │
│ Route: EINN ➔ EGLL (Hub) ➔ EBBR | Flight Date: 2026-11-17 (Tuesday) | Wave: Morning | Baseline: 202510       │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
                                  ✈️ Flight Legs & Delay Risk (Tuesday morning)                                  
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Leg         ┃ Operator ┃ Aircraft ┃ Avg / Med Delay ┃ P90 Delay ┃ Delay > 15m ┃ Weather Forecast ┃ Adj. Delay  ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ EINN ➔ EGLL │ BAW      │ A20N     │   +4.2m / +3.0m │     +9.0m │        6.8% │ Overcast (G14kt) │        7.1% │
│ EGLL ➔ EBBR │ BAW      │ A320     │   +5.1m / +4.0m │    +12.0m │        8.4% │ Clear (G11kt)    │        8.8% │
└─────────────┴──────────┴──────────┴─────────────────┴───────────┴─────────────┴──────────────────┴─────────────┘

                                ⏳ Connection Hub & Missed Flight Risk (London Heathrow)                           
┏━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Transfer Hub ┃ Layover ┃ Eff. MCT   ┃ Effective Slack ┃ Arr. Holding ┃ Taxi Queue ┃ Missed Risk ┃ Luggage Risk    ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ EGLL (LHR)   │  75 min │ 60 min     │         +15 min │       +0 min │     +0 min │       14.2% │ 30% (LOW_RISK)  │
└──────────────┴─────────┴────────────┴─────────────────┴──────────────┴────────────┴─────────────┴─────────────────┘
✅ Same Terminal Transfer: T5. Heathrow biometric flight connections at T5 with strict 20-minute gate conformance.

                                 💶 Passenger Rights & Statutory Compensation (EU261 / UK261)                     
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Route Distance ┃ Distance Tier           ┃ Statutory Compensation ┃ Delay Threshold          ┃ Duty of Care        ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│         912 km │ Short-haul (≤ 1,500 km) │                   €250 │ ≥ 3 hours at destination │ After 2 hours delay ┃
└────────────────┴─────────────────────────┴────────────────────────┴──────────────────────────┴─────────────────────┘

                                             🕒 When Should I Fly? (EINN ➔ EGLL)                                  
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Departure Wave ┃ Time Window       ┃ Avg Delay ┃ P90 Delay ┃ Probability Delay > 15m  ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Morning        │ 06:00 – 12:00     │     +4.2m │     +9.0m │                     6.8% │
│ Midday         │ 12:00 – 15:00     │     +6.8m │    +14.0m │                    12.4% │
│ Afternoon      │ 15:00 – 18:00     │     +9.4m │    +19.0m │                    18.1% │
│ Evening        │ 18:00 – 22:00     │    +13.1m │    +24.0m │                    24.6% │
└────────────────┴───────────────────┴───────────┴───────────┴──────────────────────────┘
```

---

### Example 2: Inspecting Data Provenance with Diagnostic Mode (`--debug`)

The `--debug` flag outputs a complete accounting of the historical radar records queried, DuckDB execution parameters, Open-Meteo coordinates, and mathematical formulas:

```bash
flightml trip-risk --origin LEMD --dest EGLL --operator BAW --time evening --debug
```

```text
================================================================================
🔬 PASSENGER COPILOT PROVENANCE & DIAGNOSTIC AUDIT REPORT
================================================================================

1. Baseline Archive & Engine Configuration:
   - Target Period Requested : auto
   - Baseline Period Selected: 202510
   - Historical Flights Total: 984,212 flights in dataset
   - Aircraft Performance DB : OpenAP v1.2 & ICAO EDB v32B compiled

2. Leg 1 Provenance Audit: LEMD ➔ EGLL
   - Operator Filter Applied : BAW (Passenger commercial priority)
   - Day of Week / Wave      : Tuesday / evening (18:00 - 22:00 UTC)
   - DuckDB SQL Query Filter : adep='LEMD' AND ades='EGLL' AND ac_operator='BAW'
   - Total Matching Records  : 27 historical flights in period
   - Average / Median Delay  : +6.0 min / +5.0 min (P90: +13.0 min)
   - Sample Historical Records Queried:
     * ECTRL 900114012 | 2025-10-07 18:45 | A20N (G-TTNB) | Delay: +4.0 min
     * ECTRL 900118224 | 2025-10-14 18:50 | A20N (G-TTNK) | Delay: +6.0 min
     * ECTRL 900122490 | 2025-10-21 18:40 | A21N (G-NEOP) | Delay: +8.0 min
   - Live Weather Station    : Lat 40.49°, Lon -3.57° (Madrid Barajas)
     * Surface Temperature   : 14.2°C | Wind: 230° @ 8 kt (Gusts: 13 kt)
     * Precipitation Rate    : 0.0 mm/h | Low Cloud Cover: 15%
     * Adjusted Delay Risk   : 11.1% historical -> 11.6% weather-adjusted

3. Passenger Rights Geodesic Calculation Audit:
   - Origin Coordinates      : LEMD (40.4983° N, 3.5676° W)
   - Destination Coordinates : EGLL (51.4700° N, 0.4543° W)
   - Great-Circle Distance   : 1,245.8 km (672.7 NM)
   - Statutory Classification: Short-haul (<= 1,500 km)
   - Entitled Compensation   : €250.00 EUR per passenger (Regulation EC 261/2004)
================================================================================
```

---

## 🐍 Python Library Recipes (`flightml.eval.passenger_journey`)

### Recipe 1: Direct Flight Risk Assessment

```python
from flightml.eval.passenger_journey import PassengerJourneyEngine

# Initialize copilot with automatic period selection
engine = PassengerJourneyEngine(period="auto")

# Assess direct flight from Dublin to Amsterdam
itinerary = engine.assess_itinerary(
    origin="EIDW",
    dest="EHAM",
    day_of_week="friday",
    time_of_day="evening",
    flight_date="2026-11-20",
)

leg = itinerary.legs[0]
print(f"Route: {leg.origin} -> {leg.dest} ({leg.operator} {leg.aircraft_type})")
print(f"Historical Delay > 15m: {leg.historical_delay_prob * 100:.1f}%")
print(f"Weather-Adjusted Delay: {leg.weather_adjusted_delay_prob * 100:.1f}%")
print(f"Destination Weather   : {leg.dest_weather.weather_summary}")
print(f"EU261 Compensation    : €{itinerary.passenger_rights.statutory_compensation_amount_eur:.0f} EUR")
```

---

### Recipe 2: Multi-Leg Connecting Flight with Diagnostic Provenance

```python
from flightml.eval.passenger_journey import PassengerJourneyEngine

engine = PassengerJourneyEngine(period="auto")

# Assess connection: Shannon to Brussels via London Heathrow
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

# Inspect connection risk
conn = itinerary.connection
print(f"Hub Airport: {conn.hub_airport} ({conn.hub_name})")
print(f"Effective Slack Buffer  : {conn.effective_slack_buffer_min:+.1f} min")
print(f"Missed Connection Risk  : {conn.missed_connection_prob * 100:.1f}%")
print(f"Luggage Transfer Risk   : {conn.luggage_risk_level}")
print(f"Connection Status Rating: {conn.connection_risk_rating}")

# Inspect Actionable Travel Recommendations
for rec in itinerary.actionable_recommendations:
    print(f"💡 {rec}")

# Print Provenance Audit Details
if itinerary.debug_info:
    print(f"\nAudit Baseline Period: {itinerary.debug_info['baseline_period']}")
    print(f"Leg 1 Records Sampled: {itinerary.debug_info['leg_1_audit']['records_sampled']}")
```

---

### Recipe 3: Querying Top Passenger Carriers for an Airport Pair

```python
from flightml.eval.passenger_journey import PassengerJourneyEngine

engine = PassengerJourneyEngine(period="auto")

# Find active commercial passenger operators between Madrid and Heathrow
operators = engine.get_available_operators(origin="LEMD", dest="EGLL")
for op in operators:
    print(f"Carrier: {op['operator']} | Historical Flights: {op['flight_count']} | Fleet: {op['top_aircraft']}")
```

---

## 🌐 REST API Endpoints

The FlightML REST API exposes the complete journey copilot functionality over HTTP.

### 1. `POST /api/v1/passenger/trip-risk`

Evaluates single-leg or connecting multi-leg itineraries.

#### Request Payload:
```json
{
  "origin": "EINN",
  "dest": "EBBR",
  "hub": "EGLL",
  "day_of_week": "tuesday",
  "time_of_day": "morning",
  "scheduled_layover_min": 75,
  "flight_date": "2026-11-17",
  "preferred_operator": "BAW",
  "period": "auto",
  "debug": true
}
```

#### Response Payload (Truncated):
```json
{
  "itinerary_id": "EINN-EGLL-EBBR-TUESDAY-MORNING",
  "is_connecting": true,
  "legs": [
    {
      "origin": "EINN",
      "dest": "EGLL",
      "operator": "BAW",
      "aircraft_type": "A20N",
      "avg_delay_min": 4.2,
      "median_delay_min": 3.0,
      "p90_delay_min": 9.0,
      "historical_delay_prob": 0.068,
      "weather_adjusted_delay_prob": 0.071
    },
    {
      "origin": "EGLL",
      "dest": "EBBR",
      "operator": "BAW",
      "aircraft_type": "A320",
      "avg_delay_min": 5.1,
      "median_delay_min": 4.0,
      "p90_delay_min": 12.0,
      "historical_delay_prob": 0.084,
      "weather_adjusted_delay_prob": 0.088
    }
  ],
  "connection": {
    "hub_airport": "EGLL",
    "hub_name": "London Heathrow Airport",
    "scheduled_layover_min": 75,
    "effective_mct_min": 60,
    "effective_slack_buffer_min": 15.0,
    "missed_connection_prob": 0.142,
    "luggage_risk_level": "LOW_RISK",
    "connection_risk_rating": "TIGHT"
  },
  "passenger_rights": {
    "statutory_compensation_amount_eur": 250.0,
    "distance_tier": "Short-haul (<= 1,500 km)",
    "delay_threshold_hours": 3.0,
    "duty_of_care_threshold_hours": 2.0
  },
  "actionable_recommendations": [
    "Boarding closes strictly 20 minutes before departure at Heathrow T5. Proceed directly through Flight Connections.",
    "Morning departure wave is optimal; delay risk increases significantly after 17:00 UTC."
  ]
}
```

### 2. `GET /api/v1/passenger/operators`

Returns top historical commercial airlines for a specified origin and destination airport pair.

```bash
curl -X GET "http://localhost:8000/api/v1/passenger/operators?origin=LEMD&dest=EGLL"
```

---

## 🤖 Model Context Protocol (FastMCP) Tool

FlightML includes a dedicated MCP tool enabling local LLMs (e.g. LM Studio, Claude Desktop, Cursor) to invoke the Passenger Copilot autonomously:

- **Tool Name**: `flightml_assess_passenger_trip`
- **Arguments**:
  - `origin` (string, required): 4-letter ICAO code.
  - `dest` (string, required): 4-letter ICAO code.
  - `hub` (string, optional): 4-letter ICAO transfer hub.
  - `day_of_week` (string, default: `"tuesday"`): Day of travel.
  - `time_of_day` (string, default: `"morning"`): Departure wave.
  - `flight_date` (string, optional): Target travel date (`YYYY-MM-DD`).
  - `scheduled_layover_min` (integer, default: `75`): Hub layover duration.
  - `preferred_operator` (string, optional): Airline ICAO code.
  - `debug` (boolean, default: `false`): Enable diagnostic provenance.

### Example Agent Prompt:
> *"I'm flying from Shannon (EINN) to Brussels (EBBR) via London Heathrow (EGLL) next Tuesday morning. I have a 75-minute layover. What is my risk of missing the connection, and what are my EU261 rights?"*

The agent calls `flightml_assess_passenger_trip` and renders a structured, cited markdown report including connection buffers, terminal advisories, and compensation rights.

---

## 🖥️ Streamlit Interactive Dashboard

The Passenger Copilot is integrated as **Category Tab 8** in the FlightML interactive simulation workbench:

```bash
flightml dashboard
```

Navigate to **`🧳 Passenger Travel Copilot & Itinerary Risk`**:
1. **Journey Builder Sidebar**: Input Origin, Destination, optional Transfer Hub, Day of Week, Departure Wave, Layover minutes, and Flight Date.
2. **Key Metric Tiles**: Immediate display of Leg 1 Risk, Leg 2 Risk, Missed Connection Probability, and Statutory EU261 Rights (`€ EUR`).
3. **Flight Leg Analysis Table**: Breakdown of historical delays, aircraft types, live weather forecasts, and weather-adjusted probabilities.
4. **Hub Connection Analyzer**: Visual gauges of effective slack buffer vs. minimum connection times and luggage risk warnings.
5. **Diurnal Wave Heatmap**: Interactive comparison table highlighting the best departure windows.
6. **Statutory Compensation Charter**: Breakdown of applicable distance tiers, cash entitlements, and airline duty-of-care obligations.
7. **Provenance & Audit Expander (`--debug`)**: Deep-dive inspection of raw historical flight rows, METAR reports, and intermediate mathematical calculations.
