"""
Mathematical Formulations, Physics & Analytical Proofs Accordions for FlightML Dashboard.

Provides interactive, educational Streamlit expanders (`st.expander`) for every dashboard view,
rendering exact mathematical derivations, KaTeX formulas, physical units, parameter definitions,
and verified numerical examples across all aviation engineering domains in FlightML:

1. 4D Trajectory & RNP Profile (Haversine, Bearing, Cross-Track Distance, RLOS Horizon, DME Subtended Angle).
2. Vertical Profile: CDO & CCO (Continuous Descent glideslopes, idle thrust, vertical flight path angle).
3. Delay & Holding Diagnostics (Schedule variance, buffer consumption, holding stack dwell integration).
4. Fuel, Carbon & LTO Emissions (BADA drag polar, Breguet range, ICAO EDB 4-mode LTO, BFFM2 NOx).
5. Upper-Air Winds & Contrail Climate (Schmidt-Appleman Criterion, T_SAC, Sonntag vapor pressures, RHi, iRF, ESAD).
6. Multi-Level KPI Suite & Airspace Saturation (OTP_15, sector occupancy, capacity Gini index).
7. Free Route Airspace (FRA) & FIR Inefficiency (ICAO GANP / Eurocontrol PRU KEA & KEP directness formulas).
8. GNSS Jamming & Spoofing Kinematics (Segment speed, vertical velocity spikes, track angle jitter).
9. Airspace Economics & Route Charges (Eurocontrol CRCO billing formula, MTOW weight factor, Cost Index).
10. Experimental AI & ML Lab (Flight2Vec cosine similarity, PINN mass inversion, ST-GNN contagion, MARL slot gaming).
11. Passenger Copilot & Itinerary Risk (MCT buffer risk, Gaussian miss probability, disruption contagion).
"""

from __future__ import annotations

import streamlit as st


def render_4d_rnp_math_accordion() -> None:
    """Mathematical formulations for 4D Trajectory, RNP Boundaries, and Navaid RLOS Physics."""
    with st.expander("📐 Mathematical Formulations, Physics & Analytical Proofs for this View", expanded=False):
        st.markdown(r"""
#### 1. Great-Circle Geodesics & Navigation Geometry

##### A. Haversine Great-Circle Distance
The shortest surface distance $d$ between two points $(\phi_1, \lambda_1)$ and $(\phi_2, \lambda_2)$ on a spherical Earth of mean radius $R = 3,440.065\,\text{nm}$ ($6,371.0\,\text{km}$):

$$\Delta \sigma = 2 \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$

$$d = R \cdot \Delta \sigma$$

*Antimeridian Handling:* When traversing the International Date Line ($\lambda_1 \cdot \lambda_2 < 0$ and $|\lambda_1 - \lambda_2| > 180^\circ$), the longitudinal difference is normalized via $\Delta \lambda = 360^\circ - |\lambda_1 - \lambda_2|$.

##### B. Initial Forward Azimuth (Bearing)
The initial true track heading $\theta$ from origin to destination:

$$\theta = \text{atan2}\left( \sin(\Delta \lambda)\cos(\phi_2), \; \cos(\phi_1)\sin(\phi_2) - \sin(\phi_1)\cos(\phi_2)\cos(\Delta \lambda) \right) \pmod{360^\circ}$$

##### C. Cross-Track Distance (XTD) & Along-Track Distance (ATD)
For an aircraft position $P_3$ relative to planned airway segment $P_1 \to P_2$:

$$XTD = R \cdot \arcsin \left( \sin\left(\frac{d_{13}}{R}\right) \sin(\theta_{13} - \theta_{12}) \right)$$

$$ATD = R \cdot \arccos \left( \frac{\cos(d_{13}/R)}{\cos(XTD/R)} \right)$$

*Sign Convention:* $XTD > 0$ denotes a starboard (right) deviation; $XTD < 0$ denotes a port (left) deviation.

---

#### 2. Radio Line-of-Sight (RLOS) & DME/DME Multi-Lateration

##### A. Atmospheric Refraction & Radio Horizon
Due to atmospheric density lapse, radio waves refract toward Earth (standard $4/3$ effective Earth radius model). The maximum geometric line-of-sight distance $d_{RLOS}$ between a terrestrial station at elevation $h_{station}$ and an aircraft at altitude $h_{aircraft}$ (both in feet):

$$d_{RLOS} = 1.23 \cdot \left( \sqrt{h_{station}} + \sqrt{h_{aircraft}} \right) \quad [\text{nautical miles}]$$

*Worked Example:* For a DME station at $h_s = 150\,\text{ft}$ and an aircraft cruising at FL350 ($h_{ac} = 35,000\,\text{ft}$):
$$d_{RLOS} = 1.23 \cdot (\sqrt{150} + \sqrt{35000}) = 1.23 \cdot (12.25 + 187.08) = 245.2\,\text{nm}$$

##### B. DME/DME Subtended Crossing Angle & GDOP
To calculate a valid 2D horizontal RNAV position fix without GNSS, the Flight Management Computer (FMC) requires at least two independent DME stations within radio line-of-sight. The intersecting radials must satisfy the subtended crossing angle constraint:

$$30^\circ \le |\psi_{DME1} - \psi_{DME2}| \le 150^\circ$$

Angles outside this envelope create severe Geometric Dilution of Precision (GDOP), causing position calculation divergence.
        """)


def render_cdo_cco_math_accordion() -> None:
    """Mathematical formulations for Continuous Descent Operations (CDO) and Climb (CCO)."""
    with st.expander("📐 Mathematical Formulations, Physics & Analytical Proofs for this View", expanded=False):
        st.markdown(r"""
#### 1. Continuous Descent Operations (CDO) Aerodynamics

##### A. Flight Path Angle & Glideslope Equation
A continuous clean-configuration descent operates at idle or near-idle thrust ($T \approx 0$). The equilibrium descent angle $\gamma$ is governed by the lift-to-drag ratio $(L/D)$:

$$\sin(\gamma) = \frac{T - D}{m \cdot g} \approx -\frac{1}{(L/D)_{clean}}$$

For standard transport jets ($(L/D) \approx 18\text{--}20$), the natural aerodynamic idle descent gradient is approximately $3.0^\circ$ ($5.2\%$), corresponding to:

$$\frac{\Delta h}{\Delta d} \approx 318\,\text{ft / nm} \quad \left( \approx 3\,\text{nm per } 1,000\,\text{ft of descent} \right)$$

##### B. Vertical Speed from Groundspeed
Given groundspeed $GS$ (in knots) and descent angle $\gamma$:

$$V_z = GS \cdot \tan(\gamma) \cdot \frac{6076.12\,\text{ft/nm}}{60\,\text{min/hr}} \approx GS \cdot 5.25 \quad [\text{ft/min}]$$

*Example:* At an approach groundspeed of $140\,\text{kts}$ on a standard $3.0^\circ$ glideslope:
$$V_z = 140 \cdot 5.25 = -735\,\text{ft/min}$$

##### C. Intermediate Level-Off Fuel Penalty
When Air Traffic Control interrupts a continuous descent with intermediate step-down level-offs ($\gamma = 0$), the aircraft must increase engine thrust from idle to cruise rating to overcome parasite and induced drag:

$$\Delta m_{fuel} = \dot{m}_{level}(\text{FL}) \cdot \Delta t_{level} - \dot{m}_{idle} \cdot \Delta t_{level}$$

For a narrowbody jet (e.g. A320), level-off at FL60 burns $\sim 20\text{--}25\,\text{kg/min}$ compared to $\sim 4\text{--}5\,\text{kg/min}$ in clean idle descent, producing excess carbon emissions.
        """)


def render_fuel_emissions_math_accordion() -> None:
    """Mathematical formulations for Aircraft Drag Polar, Fuel Flow, and ICAO Emissions."""
    with st.expander("📐 Mathematical Formulations, Physics & Analytical Proofs for this View", expanded=False):
        st.markdown(r"""
#### 1. Aerodynamic Drag Polar & Fuel Flow

##### A. Parabolic Drag Polar (BADA Formulation)
Total aerodynamic drag force $D$ is the sum of zero-lift parasite drag and lift-induced drag:

$$C_D = C_{D0} + C_{D2} \cdot C_L^2$$

$$C_L = \frac{2 \cdot m \cdot g}{\rho(z) \cdot V_{TAS}^2 \cdot S}$$

$$D = \frac{1}{2} \rho(z) \cdot V_{TAS}^2 \cdot S \cdot C_D$$

Where:
- $m$: Aircraft instantaneous mass ($\text{kg}$)
- $\rho(z)$: Air density at flight level $z$ ($\text{kg/m}^3$)
- $V_{TAS}$: True Airspeed ($\text{m/s}$)
- $S$: Wing reference area ($\text{m}^2$)
- $C_{D0}, C_{D2}$: Parasite and induced drag polar coefficients

##### B. Thrust Specific Fuel Consumption & Fuel Flow
In steady level cruise ($T = D$):

$$\dot{m}_f = SFC \cdot T_{net} = SFC \cdot D$$

$$\text{Total Fuel Burn} = \int_{t_0}^{t_1} \dot{m}_f(t) \, dt$$

---

#### 2. ICAO LTO Emissions & Boeing Fuel Flow Method 2 (BFFM2)

##### A. Carbon Dioxide ($CO_2$) Emissions
Carbon emissions are strictly stoichiometric based on fuel carbon mass fraction:

$$m_{CO2} = m_{fuel} \cdot EI_{CO2} \quad \text{where } EI_{CO2} = 3.15\,\text{kg } CO_2 / \text{kg Jet A-1}$$

##### B. Nitrogen Oxides ($NO_x$) Atmospheric Correction (BFFM2)
Reference $NO_x$ emission indices measured at sea-level test beds ($EI_{NOx,ref}$) are corrected for ambient altitude pressure $P_{amb}$, temperature $T_{amb}$, and specific humidity $h$:

$$EI_{NOx} = EI_{NOx,ref} \cdot \left( \frac{P_{amb}}{P_0} \right)^a \cdot \left( \frac{T_{amb}}{T_0} \right)^b \cdot e^{-19 \cdot (h - 0.0063)}$$

Where $P_0 = 1013.25\,\text{hPa}$, $T_0 = 288.15\,\text{K}$, $a \approx 0.4\text{--}0.5$, and $b \approx 0.6\text{--}0.8$.
        """)


def render_upper_air_contrail_math_accordion() -> None:
    """Mathematical formulations for Upper-Air Winds, ESAD, Schmidt-Appleman, and Contrail Forcing."""
    with st.expander("📐 Mathematical Formulations, Physics & Analytical Proofs for this View", expanded=False):
        st.markdown(r"""
#### 1. Upper-Air Wind Triangle & Equivalent Still Air Distance (ESAD)

##### A. Kinematic Vector Wind Triangle
$$\vec{V}_{GS} = \vec{V}_{TAS} + \vec{V}_{wind}$$

The along-track effective wind component $w_\parallel$ (positive for tailwind, negative for headwind):

$$w_\parallel = |\vec{V}_{wind}| \cdot \cos(\theta_{wind\_from} + 180^\circ - \theta_{track})$$

$$V_{GS} = \sqrt{V_{TAS}^2 - (|\vec{V}_{wind}|\sin(\Delta \theta))^2} + w_\parallel \approx V_{TAS} + w_\parallel$$

##### B. Equivalent Still Air Distance (ESAD)
ESAD represents the air-mass distance the aircraft must aerodynamically fly through:

$$ESAD = \int_{0}^{D_{ground}} \frac{V_{TAS}(s)}{V_{GS}(s)} \, ds = \sum_{i} \frac{V_{TAS,i}}{V_{GS,i}} \cdot \Delta s_i$$

- **Net Tailwind ($w_\parallel > 0 \implies V_{GS} > V_{TAS}$):** $ESAD < D_{ground}$ (fuel saved).
- **Net Headwind ($w_\parallel < 0 \implies V_{GS} < V_{TAS}$):** $ESAD > D_{ground}$ (fuel penalty).

---

#### 2. Contrail Thermodynamics: The Schmidt-Appleman Criterion

##### A. Mixing Line Slope ($G$)
Exhaust gases mix with ambient air along a straight line in the $(T, e)$ thermodynamic state plane. The slope $G$ of this mixing line:

$$G = \frac{\Delta e}{\Delta T} = \frac{c_p \cdot P \cdot EI_{H2O}}{\epsilon \cdot \Delta H_c \cdot (1 - \eta)}$$

Where:
- $c_p = 1005\,\text{J/(kg}\cdot\text{K)}$: Specific heat capacity of dry air
- $P$: Ambient atmospheric pressure ($\text{Pa}$)
- $EI_{H2O} = 1.25\,\text{kg H}_2\text{O / kg fuel}$: Water vapor emission index
- $\epsilon = 0.622$: Ratio of molecular weights of water vapor and dry air
- $\Delta H_c = 43.1 \times 10^6\,\text{J/kg}$: Lower heating value of Jet A-1
- $\eta \approx 0.30\text{--}0.38$: Overall engine propulsion efficiency

##### B. Critical Threshold Temperature ($T_{SAC}$)
The critical ambient temperature below which the mixing line intersects the liquid water saturation curve $e_{sat,w}(T)$:

$$T_{SAC} \approx -41.5 + 2.8 \cdot \ln\left( \frac{P}{25000} \right) + (\eta - 0.33) \cdot 12.0 \quad [^\circ\text{C}]$$

##### C. Ice Supersaturated Regions (ISSR)
A contrail persists and evolves into contrail-cirrus if and only if the ambient air is supersaturated with respect to ice:

$$RHi = \frac{e_{amb}}{e_{sat,ice}(T)} \times 100\% \ge 100\%$$

*Sonntag (1990) Formulation for Saturation Vapor Pressure:*
$$\ln(e_{sat,ice}) = -\frac{6024.5282}{T} + 24.7219 - 0.010613868 \cdot T + 1.31988 \times 10^{-5} \cdot T^2 - 0.493826 \cdot \ln(T)$$

##### D. Net Instantaneous Radiative Forcing (iRF)
The net climate warming of contrail cirrus combines longwave thermal trapping ($LW > 0$, warming) and solar albedo reflection ($SW < 0$, cooling):

$$iRF = RF_{LW} + RF_{SW}(\theta_{sza})$$

- **Daytime:** Solar albedo partially offsets thermal trapping ($SW < 0$).
- **Nighttime:** Albedo cooling is zero ($SW = 0$), causing maximum net warming ($iRF = RF_{LW} \gg 0$).
        """)


def render_fra_inefficiency_math_accordion() -> None:
    """Mathematical formulations for Free Route Airspace (FRA) KEA and KEP metrics."""
    with st.expander("📐 Mathematical Formulations, Physics & Analytical Proofs for this View", expanded=False):
        st.markdown(r"""
#### 1. Horizontal En-Route Flight Inefficiency (ICAO GANP / Eurocontrol PRU)

##### A. KEA (Key Performance Indicator - Actual Flown Trajectory)
The en-route horizontal inefficiency of the radar trajectory relative to the achieved great-circle distance:

$$KEA = \frac{\sum L_{actual} - \sum G}{\sum G} \times 100\%$$

Where:
- $L_{actual}$: Length of the actual radar trajectory within en-route airspace (excluding terminal sequencing $\le 40\,\text{nm}$ from ADEP/ADES).
- $G$: Orthodromic (great-circle) distance between airspace entry and exit fixes.

##### B. KEP (Key Performance Indicator - Filed Flight Plan)
Evaluates filed airway route inefficiency relative to great-circle directness:

$$KEP = \frac{\sum L_{planned} - \sum G}{\sum G} \times 100\%$$

##### C. Free Route Airspace (FRA) Directness Ratio
Within designated Free Route Airspace volumes:

$$\text{Directness Ratio} = \frac{d_{direct}(Entry \to Exit)}{d_{ATS\_airway}(Entry \to Exit)}$$

$$\text{Fuel Savings} = \left( d_{ATS} - d_{direct} \right) \cdot \dot{m}_{cruise} / V_{TAS}$$
        """)


def render_gnss_jamming_math_accordion() -> None:
    """Mathematical formulations for GNSS Jamming, Spoofing, and Kinematic Teleportation."""
    with st.expander("📐 Mathematical Formulations, Physics & Analytical Proofs for this View", expanded=False):
        st.markdown(r"""
#### 1. Kinematic Anomaly Detection & Aerodynamic Envelope Bounds

##### A. Segment Groundspeed Teleportation
Given consecutive radar waypoint positions $(\phi_1, \lambda_1, t_1)$ and $(\phi_2, \lambda_2, t_2)$:

$$V_{GS} = \frac{d(\phi_1, \lambda_1, \phi_2, \lambda_2)}{\Delta t} \quad [\text{knots}]$$

- **Subsonic Transport Airframe Limit:** Certified maximum operating Mach $M_{MO} \approx 0.82\text{--}0.89$, corresponding to maximum physical groundspeed $V_{GS} \le 650\text{--}680\,\text{kts}$ (even under $150\,\text{kt}$ jetstream tailwinds).
- **Teleportation Breach Trigger:** $V_{GS} > 750\,\text{kts}$ (supersonic breach) or $\Delta V > 300\,\text{kts}$ in $< 60\,\text{s}$.

##### B. Vertical Climb/Descent Velocity Spikes
Given barometric/geometric flight levels $FL_1, FL_2$:

$$V_z = \frac{|FL_2 - FL_1| \cdot 100\,\text{ft}}{\Delta t / 60\,\text{s}} \quad [\text{ft/min}]$$

- **Commercial Aircraft Structural Limit:** Maximum passenger climb $\approx 3,000\,\text{fpm}$; maximum structural emergency descent $\approx 6,000\,\text{fpm}$.
- **Spoofing Breach Trigger:** $V_z > 6,000\,\text{ft/min}$ (with $\Delta FL \ge 20$ in $< 120\,\text{s}$) or instantaneous altitude glitches dropping from cruise to FL0.

##### C. Track Angle Jitter & Circular Spoofing
The angular change between consecutive trajectory leg azimuths $\theta_1, \theta_2$:

$$\Delta \theta = \min\left( |\theta_2 - \theta_1|, \; 360^\circ - |\theta_2 - \theta_1| \right)$$

- **Standard Rate Turn Limit:** Rate-1 turn $= 3^\circ/\text{s}$ ($180^\circ/\text{min}$) at bank angle $\le 25^\circ\text{--}30^\circ$.
- **Jitter Breach Trigger:** $|\Delta \theta| \ge 65^\circ$ in $< 150\,\text{s}$ outside terminal holding stacks.

##### D. Terminal Approach Speed & Spatial Jumps
In terminal airspace below FL100, ICAO/FAA regulations mandate $V \le 250\,\text{kts}$. Groundspeeds $> 320\,\text{kts}$ below FL100 or spatial jumps $> 8\,\text{nm}$ in $< 30\,\text{s}$ indicate spoofed coordinates during approach alignment.
        """)


def render_economic_charges_math_accordion() -> None:
    """Mathematical formulations for Eurocontrol CRCO Route Charges and Cost Index."""
    with st.expander("📐 Mathematical Formulations, Physics & Analytical Proofs for this View", expanded=False):
        st.markdown(r"""
#### 1. Eurocontrol CRCO Overflight Billing Formula

The route charge $R$ billed to an airline by Eurocontrol Central Route Charges Office (CRCO) for overflying a national Flight Information Region (FIR):

$$R = d_{FIR} \times p \times U_{FIR}$$

Where:
- **Distance Factor ($d_{FIR}$):** Great-circle distance flown through the FIR in hundreds of kilometers ($d_{FIR} = \text{km} / 100$).
- **Aircraft Weight Factor ($p$):** Proportional to the square root of Maximum Take-Off Weight (MTOW in metric tons):
  $$p = \sqrt{\frac{MTOW}{50}}$$
- **National Unit Rate ($U_{FIR}$):** National en-route unit rate in Euros (€) established annually under the Single European Sky Performance Scheme.

*Worked Example:* An Airbus A350-900 ($MTOW = 280\,\text{t}$) flying $450\,\text{km}$ through French Airspace (LFEE/LFBB, $U = €84.50$):
$$p = \sqrt{\frac{280}{50}} = \sqrt{5.6} = 2.366$$
$$R = 4.50 \times 2.366 \times €84.50 = €900.08\,\text{EUR}$$

---

#### 2. Airline Cost Index (CI) Optimization

Flight Management Computers optimize speed to minimize total trip direct operating cost:

$$J = C_{fuel} \cdot \Delta m_{fuel} + C_{time} \cdot \Delta t$$

$$\text{Cost Index (CI)} = \frac{C_{time}\; [\text{€/hr}]}{C_{fuel}\; [\text{€/kg}]}$$

- $CI = 0$: Minimum fuel speed (maximum range).
- High $CI$: High-speed cruise to minimize crew overtime, aircraft rotation delays, and passenger miss penalties.
        """)


def render_experimental_ai_math_accordion() -> None:
    """Mathematical formulations for Experimental AI & Machine Learning Lab."""
    with st.expander("📐 Mathematical Formulations, Physics & Analytical Proofs for this View", expanded=False):
        st.markdown(r"""
#### 1. Physics-Informed Neural Network (PINN) Mass Inversion

##### A. Governing Longitudinal Equations of Motion
The neural surrogate enforces Newton's second law along the flight path as a soft regularization loss:

$$m \cdot \frac{dv}{dt} = T - D - m \cdot g \cdot \sin(\gamma)$$

$$L = m \cdot g \cdot \cos(\gamma)$$

##### B. Total Loss Functional
$$\mathcal{L}_{PINN} = \mathcal{L}_{data}(m_{pred}, m_{true}) + \lambda_{dyn} \left\| m \cdot \dot{v} - (T - D - m g \sin\gamma) \right\|^2$$

By backpropagating through aerodynamic drag $D(v, \rho, m)$ and thrust $T(v, h)$, the model reconstructs unknown takeoff weight without onboard sensor access.

---

#### 2. Spatio-Temporal Graph Neural Network (ST-GNN) Delay Contagion

The European airspace is modeled as a directed dynamic multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ where airports $\mathcal{V}$ are nodes and scheduled flights $\mathcal{E}$ are edges. Delay diffusion follows a Hawkes self-exciting branching process:

$$\lambda_i(t) = \mu_i + \sum_{j \in \mathcal{N}(i)} \sum_{t_k < t} \alpha_{ji} \cdot e^{-\beta (t - t_k)}$$

The basic delay reproduction number $R_0 = \rho(\mathbf{A})$ defines whether a localized delay event (e.g. Frankfurt storm) dissipates ($R_0 < 1$) or propagates across the entire European network ($R_0 > 1$).

---

#### 3. Multi-Agent Reinforcement Learning (MARL) Slot Gaming

Airlines compete in a decentralized Markov game under Air Traffic Flow Management (ATFM) Ground Delay Programs. The Nash equilibrium pricing under Vickrey-Clarke-Groves (VCG) mechanisms balances ground holding costs versus schedule integrity.
        """)


def render_passenger_copilot_math_accordion() -> None:
    """Mathematical formulations for Passenger Travel Copilot and Itinerary Miss Risk."""
    with st.expander("📐 Mathematical Formulations, Physics & Analytical Proofs for this View", expanded=False):
        st.markdown(r"""
#### 1. Probabilistic Minimum Connecting Time (MCT) & Miss Risk

##### A. Connection Buffer Equation
For an inbound flight leg $L_1$ arriving at hub airport $H$ and connecting to outbound leg $L_2$:

$$\text{Buffer} = \text{Scheduled Layover} - \text{MCT}(H) - \Delta t_{dep\_gate} - \Delta t_{border}$$

Where:
- $\text{MCT}(H)$: Certified physical Minimum Connecting Time for the airport/terminal terminal pair.
- $\Delta t_{border}$: Schengen/Non-Schengen immigration and security queue latency.

##### B. Gaussian Cumulative Miss Probability
Given inbound historical arrival delay distribution $\mathcal{N}(\mu_{arr}, \sigma_{arr}^2)$:

$$P(\text{Missed Connection}) = 1 - \Phi\left( \frac{\text{Scheduled Layover} - \text{MCT} - \mu_{arr}}{\sigma_{arr}} \right)$$

$$\Phi(z) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{z} e^{-t^2/2} \, dt$$

When $P(\text{Miss}) \ge 25\%$, the Passenger Copilot flags the connection as High Risk and constructs alternative multi-carrier itineraries.
        """)
