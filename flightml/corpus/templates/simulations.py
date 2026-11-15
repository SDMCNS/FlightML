from __future__ import annotations

import random
from typing import Any

from flightml.corpus.schemas import SFTConversation, SFTMessage
from flightml.corpus.templates.base import BaseTemplate


class SimulationTemplate(BaseTemplate):
    """Generates Track 4 samples: 4D Airspace Closures, Tail-Swaps & OR Simulators."""

    def __init__(self):
        super().__init__(track_name="simulations")

    def generate(self, rng: random.Random, context: dict[str, Any]) -> SFTConversation | None:
        fir_id = context.get("target_fir", rng.choice(["LSASFIR", "LFFFFIR", "EDUUUIR"]))
        duration_hrs = context.get("duration_hrs", rng.choice([2.0, 3.0, 4.0]))
        flights_impacted = context.get("flights_impacted", rng.randint(350, 850))
        rerouted_count = int(flights_impacted * 0.88)
        extra_nm = flights_impacted * rng.uniform(42.0, 68.0)
        extra_fuel_tonnes = (extra_nm * 3.8) / 1000.0
        airline_loss_eur = extra_fuel_tonnes * 1000.0 * 0.6421 + (flights_impacted * 1850.0)
        recovery_hours = duration_hrs * 0.75 + (flights_impacted / 500.0)

        prompts = [
            f"Simulate a temporary {duration_hrs:.0f}-hour tactical closure of Flight Information Region {fir_id}. What is the macro-level impact on European traffic reroutes, detour nautical miles, additional Jet-A fuel burn, airline operating losses in Euros (€ EUR), and post-reopening recovery queue duration?",
            f"An unexpected airspace disruption forces the immediate closure of {fir_id} for {duration_hrs:.0f} hours. Perform an operations research counterfactual analysis quantifying rerouted flights, fuel waste, airline balance sheet impact, and recovery kinetics.",
            f"How does a {duration_hrs:.0f}-hour shutdown of {fir_id} propagate across the European ATM network? Derive extra nautical miles flown, CO2 emissions, and airline economic losses in Euros.",
        ]
        user_prompt = rng.choice(prompts)

        cot = (
            f"<thought>\n"
            f"1. Scenario Setup:\n"
            f"   - Target Airspace (FIR): {fir_id}\n"
            f"   - Closure Duration: {duration_hrs:.1f} hours\n"
            f"   - Historical Baseline Flight Influx: {flights_impacted:,} flights scheduled to transit sector\n"
            f"2. Tactical Rerouting & Nautical Mile Penalties:\n"
            f"   - Diverted / Rerouted Flights: {rerouted_count:,} flights (88% tactical reroute rate, remainder ground-delayed or cancelled)\n"
            f"   - Total Extra Flight Distance: +{extra_nm:,.0f} NM\n"
            f"   - Average Detour per Rerouted Flight: +{extra_nm / rerouted_count:.1f} NM\n"
            f"3. Fuel Burn & Environmental Emissions:\n"
            f"   - Extra Jet-A Fuel Consumed: +{extra_fuel_tonnes:,.1f} metric tonnes (+{extra_fuel_tonnes * 1000:,.0f} kg)\n"
            f"   - Additional CO2 Emissions: +{extra_fuel_tonnes * 3.16:,.1f} tonnes CO2\n"
            f"4. Airline Economic Impact (Euro Compliance):\n"
            f"   - Fuel Detour Cost: {extra_fuel_tonnes * 1000:,.0f} kg * €0.6421/kg = €{extra_fuel_tonnes * 1000 * 0.6421:,.2f} EUR\n"
            f"   - Total Airline Operating Loss (Fuel + Ground Delays + Crew Duty): €{airline_loss_eur:,.2f} EUR\n"
            f"5. Non-Linear Post-Reopening Queue Recovery Kinetics:\n"
            f"   - Backlog Recovery Time: {recovery_hours:.1f} hours after reopening before sector traffic density normalizes\n"
            f"</thought>\n"
        )

        response_body = (
            f"### Counterfactual Airspace Closure Assessment: {fir_id} ({duration_hrs:.0f}-Hour Window)\n\n"
            f"**Disrupted Airspace:** `{fir_id}` | **Disruption Duration:** {duration_hrs:.0f}.0 hours\n\n"
            f"#### 1. Air Traffic Demand & Tactical Rerouting Kinetics\n"
            f"- **Total Scheduled Flights Disrupted**: **{flights_impacted:,} flights**\n"
            f"- **Tactically Rerouted Flights**: **{rerouted_count:,} flights** ({rerouted_count / flights_impacted * 100:.1f}% reroute compliance rate)\n"
            f"- **Aggregate Network Detour**: **+{extra_nm:,.0f} Nautical Miles** (mean detour of +{extra_nm / rerouted_count:.1f} NM per diverted airframe)\n\n"
            f"#### 2. Fuel Waste & Environmental Externalities\n"
            f"- **Extra Jet-A Fuel Burned**: **+{extra_fuel_tonnes:,.1f} metric tonnes**\n"
            f"- **Additional Atmospheric Carbon ($\text{{CO}}_2$)**: **+{extra_fuel_tonnes * 3.16:,.1f} metric tonnes**\n"
            f"- **En-Route Fuel Cost Penalty**: **€{extra_fuel_tonnes * 1000 * 0.6421:,.2f} EUR** (at benchmark spot price of €0.6421/kg)\n\n"
            f"#### 3. Macroeconomic Airline Balance Sheet Impact\n"
            f"- **Total Estimated Airline Direct Operating Loss**: **€{airline_loss_eur:,.2f} EUR**\n"
            f"  - Incorporates fuel detour penalties, extended crew flight duty period (FDP) overtime, ATFM slot delay ground compensation, and passenger misconnection handling.\n\n"
            f"#### 4. Post-Reopening Queue Drainage Dynamics\n"
            f"- **Estimated Sector Recovery Horizon**: **{recovery_hours:.1f} hours** post-reopening.\n"
            f"- When `{fir_id}` reopens, the immediate influx of airborne waiting holds, release of ground-stopped departures, and normal scheduled traffic creates a secondary arrival wave. "
            f"Sector capacity recovery follows a non-linear drainage curve requiring regulated meter-fix metering to prevent secondary loss of separation alerts."
        )

        assistant_msg = cot + response_body

        return SFTConversation(
            messages=[
                SFTMessage(role="system", content=self.get_system_prompt(rng)),
                SFTMessage(role="user", content=user_prompt),
                SFTMessage(role="assistant", content=assistant_msg),
            ],
            track=self.track_name,
            metadata={
                "target_fir": fir_id,
                "duration_hrs": duration_hrs,
                "flights_impacted": flights_impacted,
                "extra_nm": extra_nm,
                "airline_loss_eur": airline_loss_eur,
                "recovery_hours": recovery_hours,
            },
        )
