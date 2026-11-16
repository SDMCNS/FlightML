# Free Route Airspace (FRA) Boundary Efficiency & Long-Term Forecaster

## Executive Summary

**Free Route Airspace (FRA)** is a cornerstone Single European Sky (SES / SESAR) operational concept that removes fixed ATS airway networks, allowing airspace users to plan and fly user-preferred direct trajectories (`DCT`) between defined entry and exit waypoints.

While many European ANSPs have successfully deployed domestic or regional FRA initiatives (e.g. **NEFRA** in Scandinavia/Baltics, **SECSI FRA** in South-East Europe, **SEE FRA** in the Balkans), significant lateral routing inefficiencies persist at major national Flight Information Region (FIR) boundary interfaces.

FlightML provides a deterministic analytics and forecasting engine to:
1. **Explore and quantify horizontal and vertical routing inefficiencies** at FIR boundaries using multi-period Eurocontrol radar trajectories.
2. **Identify boundaries likely already providing cross-border FRA** directness.
3. **Spotlight the 7 Prime Candidate FIR boundaries** for immediate FRA harmonization.
4. **Rank all European boundaries** via the Free Route Airspace Boundary Priority Index (**FRA-BPI**, 0–100).
5. **Model multi-horizon accumulated benefits** over **1-year, 5-year, and 20-year** investment horizons.
6. **Incorporate multi-season empirical synthesis** across Spring (02), Summer Peak (06), and Autumn (10).
7. **Reconcile with official Eurocontrol annual network statistics** (**11,124,247 flights** in 2025; **10,696,391 flights** in 2024).
8. **Track exact percentage routing and $\text{CO}_2$ deviations** alongside mandatory **EU ReFuelEU Aviation SAF blending mandates** (2% in 2025 ramping to 42% in 2045) and forward-looking energy price escalation regimes.

---

## The 7 Prime Bottleneck FIR Boundaries

Across the European network, 7 cross-border and cross-UAC boundary interfaces represent the highest concentration of airway funneling, dog-leg vectoring, and lateral detours:

| Boundary Pair | Sovereign Interface | Annual Crossing Flights (2025) | Mean Detour (NM) | Directness Index (%) | Detour % Dev | CO₂ Mitigation % Dev | Priority Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`EGTT_LFFF`** | UK (London) ↔ France (Paris) | 1,482,910 | 6.8 NM | 93.2% | +7.30% | -5.78% | 🚨 #1 Prime |
| **`EDUU_EDVV`** | Germany (Rhein UAC) ↔ Germany (Hannover) | 1,241,580 | 5.8 NM | 94.1% | +6.27% | -5.01% | 🚨 #2 Prime |
| **`LFFF_LFMM`** | France (Paris) ↔ France (Marseille) | 1,029,450 | 6.2 NM | 93.8% | +6.61% | -5.27% | 🚨 #3 Prime |
| **`LFFF_LSAS`** | France (Paris) ↔ Switzerland (Swiss UAC) | 884,120 | 6.5 NM | 93.5% | +6.95% | -5.52% | 🚨 #4 Prime |
| **`EGTT_EISN`** | UK (London) ↔ Ireland (Shannon) | 712,380 | 6.1 NM | 93.9% | +6.50% | -5.19% | 🚨 #5 Prime |
| **`LFMM_LIRR`** | France (Marseille) ↔ Italy (Rome) | 598,240 | 7.1 NM | 92.9% | +7.64% | -6.03% | 🚨 #6 Prime |
| **`LIRR_LIBB`** | Italy (Rome) ↔ Italy (Brindisi) | 477,271 | 6.9 NM | 93.1% | +7.41% | -5.86% | 🚨 #7 Prime |
| **Total (Prime 7)** | — | **6,425,951** | **6.4 NM** | **93.5%** | **+7.23%** | **-5.73%** | — |

---

## Flights vs. Boundary Crossings Reconciliation

A key distinction in European air traffic management is the relationship between **European Flights (Departures/Movements)** and **FIR Boundary Crossings**:

* **Total European Network Flights (2025)**: **11,124,247 flights** (+4.0% YoY vs 2024's 10,696,391).
* **Crossings Across All European FIR Interfaces**: **52,506,445 crossings** (averaging 4.72 boundary crossings per flight).
* **Crossings Across All Cross-Border Interfaces**: **44,941,957 crossings** (averaging 4.04 cross-border handoffs per flight).
* **Crossings Across Top 150 Tracked Boundaries**: **33,039,013 crossings** (averaging 2.97 crossings per flight).
* **Crossings Across the 7 Prime Bottlenecks**: **6,425,951 crossings** (averaging 0.58 bottleneck crossings per European flight).

---

## Baseline Extrapolation Modes: Contiguous 12-Month Full Year & Seasonal Synthesis

FlightML provides two deterministic baseline extrapolation modes for long-term benefit forecasting, selectable via the Streamlit dashboard (`Baseline Extrapolation Source`), CLI (`--baseline-mode`), or FastMCP (`baseline_mode`):

### 1. Contiguous 12-Month Full-Year Empirical Baseline (`202411-202510`) *(Default / Recommended)*
FlightML ingests **12 contiguous Eurocontrol monthly archives** (November 2024 through October 2025), representing **10,235,149 verified DDR2 radar flights** and **88,164,473 FIR traversals**:
* Each of the 12 individual months is computed deterministically from real 4D radar tracks and FIR boundary logs.
* Month-by-month scaling factors reconcile each archive month to Eurocontrol's official monthly traffic count ($11,124,247$ annual flights in 2025).
* Tracks the true European seasonal volatility: winter trough in January (€3.73M/mo saved) through the peak summer vacation surge in July/August (€8.65M–€9.21M/mo saved).

| Period | Month | Season Cluster | Eurocontrol Flights | Archive Flights | Coverage | Scoped Crossings | Fuel Saved (t) | CO₂ Mitigated (t) | Operating Value (€ EUR) | Mean Detour |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **202411** | Nov 2024 | Autumn Shoulder | 810,270 | 714,083 | 88.1% | 376,446 | 6,633.1 t | 20,960.5 t | €4,259,132 | +6.95% |
| **202412** | Dec 2024 | Winter Block | 827,049 | 732,074 | 88.5% | 394,408 | 6,672.1 t | 21,083.8 t | €4,284,148 | +6.68% |
| **202501** | Jan 2025 | Winter Block | 745,085 | 691,278 | 92.8% | 333,958 | 5,816.5 t | 18,380.0 t | €3,734,793 | +6.88% |
| **202502** | Feb 2025 | Winter Block | 713,160 | 656,458 | 91.9% | 402,318 | 7,231.6 t | 22,851.9 t | €4,643,393 | +7.11% |
| **202503** | Mar 2025 | Winter Block | 825,871 | 759,566 | 91.6% | 418,187 | 7,400.8 t | 23,386.5 t | €4,752,054 | +7.00% |
| **202504** | Apr 2025 | Autumn Shoulder | 918,900 | 848,673 | 92.4% | 538,707 | 9,190.6 t | 29,042.3 t | €5,901,277 | +6.74% |
| **202505** | May 2025 | Autumn Shoulder | 1,000,184 | 926,106 | 92.0% | 649,816 | 11,242.3 t | 35,525.6 t | €7,218,671 | +6.85% |
| **202506** | Jun 2025 | Summer Peak | 1,037,280 | 957,457 | 92.3% | 593,123 | 10,922.5 t | 34,515.2 t | €7,013,316 | +7.27% |
| **202507** | Jul 2025 | Summer Peak | 1,097,183 | 1,019,405 | 92.9% | 758,263 | 13,470.7 t | 42,567.5 t | €8,649,506 | +7.02% |
| **202508** | Aug 2025 | Summer Peak | 1,097,059 | 1,027,054 | 93.2% | 770,711 | 14,342.5 t | 45,322.2 t | €9,209,314 | +7.35% |
| **202509** | Sep 2025 | Summer Peak | 1,044,210 | 968,854 | 92.8% | 691,728 | 11,978.6 t | 37,852.3 t | €7,691,466 | +6.84% |
| **202510** | Oct 2025 | Autumn Shoulder | 1,007,996 | 934,141 | 92.4% | 600,356 | 10,123.5 t | 31,990.2 t | €6,500,299 | +6.66% |
| **TOTAL** | **Full Year** | **12 Months** | **11,124,247** | **10,235,149** | **92.0%** | **6,528,020** | **115,024.7 t** | **363,478.0 t** | **€73,857,363** | **+6.95%** |

### 2. Multi-Season Empirical Synthesis (3 Anchor Blocks)
For environments where only seasonal samples are indexed:
* **Spring (02)**: Low winter trough and shoulder operations (lower convective weather, standard airway routing).
* **Summer Peak (06)**: Severe seasonal traffic surges, holiday charter flows to Southern Europe, and extensive tactical convective storm avoidance.
* **Autumn Shoulder (10)**: High-density business trunk corridors, dense North-South European axis operations.

```text
Seasonal Cluster Mapping (11,124,247 Total Flights in 2025):
• Winter Block (Jan, Feb, Mar, Dec): 3,111,165 flights (27.97% of annual)
  Anchor: 202502 | Official Feb: 713,160 | Archive: 656,458 | Cluster Multiplier: 4.7393x
• Summer Peak Block (Jun, Jul, Aug, Sep): 4,275,732 flights (38.44% of annual)
  Anchor: 202506 | Official Jun: 1,037,280 | Archive: 957,457 | Cluster Multiplier: 4.4657x
• Autumn Shoulder Block (Apr, May, Oct, Nov): 3,737,350 flights (33.60% of annual)
  Anchor: 202510 | Official Oct: 1,007,996 | Archive: 934,141 | Cluster Multiplier: 4.0008x
```

Reconstructed Annual Total:
$$3{,}111{,}165 + 4{,}275{,}732 + 3{,}737{,}350 = \mathbf{11{,}124{,}247}\text{ flights (exact 0.00\% variance)}$$

---

## Routing & $\text{CO}_2$ Percentage Deviation Metrics

The engine tracks exact mathematical deviations from the geodesic Great Circle Distance (GCD):

### 1. Lateral Detour % Deviation
$$\% \Delta \text{detour} = \left(\frac{100}{\text{Directness Index \%}} - 1\right) \times 100\%$$
* Average Prime Candidate Detour Deviation: **$+7.23\%$** above GCD.

### 2. Cruise Excess $\text{CO}_2$ % Deviation
Because cruise fuel burn scales directly with distance flown ($3.20\text{ kg fuel/NM} \times 3.16\text{ kg CO}_2\text{/kg fuel}$), the baseline routing causes an excess carbon emissions deviation matching detour distance:
$$\% \Delta \text{CO}_2 \text{ (Excess)} = +7.23\%$$

### 3. Post-FRA $\text{CO}_2$ Mitigation % Deviation
At an 85% realization rate ($R=0.85$ factoring in convective weather avoidance and military airspace reserves):
$$\% \Delta \text{CO}_2 \text{ (Mitigated)} = -\frac{\% \Delta \text{detour} \times R}{1 + \% \Delta \text{detour} / 100} \times 100\% = \mathbf{-5.73\%}$$
Direct cruise crossing emissions are reduced by **$-5.73\%$**.

---

## ReFuelEU Aviation SAF Trajectory & Net Lifecycle Emissions

Under the **EU ReFuelEU Aviation Regulation (EU) 2023/2405**, aircraft operating from European airports must uplift increasing minimum blends of Sustainable Aviation Fuel (SAF):
* **Year 1 (2025)**: 2.0% SAF blend $\to$ **$-1.6\%$** lifecycle carbon intensity reduction
* **Year 5 (2030)**: 6.0% SAF blend $\to$ **$-4.8\%$** lifecycle carbon intensity reduction
* **Year 10 (2035)**: 20.0% SAF blend $\to$ **$-16.0\%$** lifecycle carbon intensity reduction
* **Year 20 (2045)**: 42.0% SAF blend $\to$ **$-33.6\%$** lifecycle carbon intensity reduction

FlightML models both:
1. **Gross Tailpipe $\text{CO}_2$ Mitigated**: Physical engine emissions eliminated by avoiding nautical miles.
2. **Net Lifecycle $\text{CO}_2$ Mitigated**: Factoring in the progressive decarbonization of the European aviation fuel mix.

---

## Multi-Horizon Investment Scorecards (1, 5, 20 Years)

Evaluated across the 7 Prime Candidates under **+2.0%/yr traffic growth**, **85% directness realization**, and **4 forward fuel pricing regimes** (Constant €0.6421/kg, Optimistic +1.5%/yr, Median Expected +3.5%/yr, Geopolitical Shock +6.5%/yr):

### 1-Year Horizon (Immediate)
* **Crossing Flights**: 6,554,470 (across ~11.35M European flights in Year 1)
* **Detour NM Avoided**: 36,371,223 NM
* **Jet-A Fuel Saved**: 116,388 tonnes
* **Gross $\text{CO}_2$ Mitigated**: 367,786 tonnes ($-5.73\%$ crossing deviation avoided)
* **Net Lifecycle $\text{CO}_2$**: 361,901 tonnes (2.0% SAF blend)
* **Financial Savings**:
  * 🟢 Optimistic (+1.5%): **€75.85M EUR**
  * 🟡 Median (+3.5%): **€77.35M EUR**
  * 🔴 Shock (+6.5%): **€79.59M EUR**

### 5-Year Horizon (Medium-Term)
* **Crossing Flights**: 34,109,725 (across ~59.0M European flights over 5 years)
* **Detour NM Avoided**: 189,277,303 NM
* **Jet-A Fuel Saved**: 605,687 tonnes
* **Gross $\text{CO}_2$ Mitigated**: 1,913,972 tonnes ($-5.73\%$ crossing deviation avoided)
* **Net Lifecycle $\text{CO}_2$**: 1,822,101 tonnes (6.0% SAF blend)
* **Financial Savings**:
  * 🟢 Optimistic (+1.5%): **€407.01M EUR**
  * 🟡 Median (+3.5%): **€432.29M EUR**
  * 🔴 Shock (+6.5%): **€472.83M EUR**

### 20-Year Horizon (Long-Term SESAR Vision)
* **Crossing Flights**: 159,256,381 (across ~275.7M European flights over 20 years)
* **Detour NM Avoided**: 883,725,046 NM
* **Jet-A Fuel Saved**: 2,827,920 tonnes
* **Gross $\text{CO}_2$ Mitigated**: 8,936,228 tonnes ($-5.73\%$ crossing deviation avoided)
* **Net Lifecycle $\text{CO}_2$**: 5,933,655 tonnes (42.0% SAF blend)
* **Financial Savings**:
  * 🟢 Optimistic (+1.5%): **€2.15 Billion EUR**
  * 🟡 Median (+3.5%): **€2.72 Billion EUR**
  * 🔴 Shock (+6.5%): **€3.91 Billion EUR**

---

## CLI Commands

All FRA commands support `--json` for automation and LLM tool calling:

```bash
# 1. Analyze FIR boundary inefficiencies across Europe
flightml fra-boundaries --period 202510 --min-flights 500 --cross-border

# 2. Compute 1, 5, and 20-year long-term multi-horizon forecasts
flightml fra-forecast --year 2025 --scope prime_only --growth 0.02 --realization 0.85

# 3. Retrieve the top detoured radar tracks for a specific boundary
flightml fra-worst-flights EGTT_LFFF --period 202510 --limit 10

# 4. Compare longitudinal FRA adoption between two periods (2024 vs 2025)
flightml fra-compare 202410 202510 --min-flights 500 --top 25
```
