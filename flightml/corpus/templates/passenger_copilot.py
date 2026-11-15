from __future__ import annotations

import random
from typing import Any

from flightml.corpus.schemas import SFTConversation, SFTMessage
from flightml.corpus.templates.base import BaseTemplate


class PassengerCopilotTemplate(BaseTemplate):
    """Generates Track 5 samples: Passenger Travel Copilot & Statutory Rights (EU261/UK261)."""

    def __init__(self):
        super().__init__(track_name="passenger_copilot")

    def generate(self, rng: random.Random, context: dict[str, Any]) -> SFTConversation | None:
        origin = context.get("origin", "EINN")
        hub = context.get("hub", "EGLL")
        dest = context.get("dest", "EBBR")
        layover_min = context.get("layover_min", 75)
        day_of_week = context.get("day_of_week", "Tuesday")
        wave = context.get("time_of_day", "morning")
        operator = context.get("operator", "BAW")

        # Geodesic distance calculation
        distance_km = context.get("distance_km", 912.0)
        tier = "Short-haul (≤ 1,500 km)" if distance_km <= 1500 else ("Medium-haul (1,500–3,500 km)" if distance_km <= 3500 else "Long-haul (> 3,500 km)")
        comp_eur = 250.0 if distance_km <= 1500 else (400.0 if distance_km <= 3500 else 600.0)
        duty_hours = 2.0 if distance_km <= 1500 else (3.0 if distance_km <= 3500 else 4.0)

        mct_nominal = 60 if hub == "EGLL" else (50 if hub == "EHAM" else 45)
        weather_transit_penalty = context.get("weather_transit_penalty", 0)
        eff_mct = mct_nominal + weather_transit_penalty
        inbound_p90 = context.get("inbound_p90", 9.0)
        effective_slack = layover_min - eff_mct
        missed_risk_pct = 14.2 if effective_slack >= 15 else 38.5

        prompts = [
            f"I am flying from {origin} to {dest} via {hub} on a {day_of_week} {wave} with {operator}. I have a {layover_min}-minute scheduled layover at {hub}. Assess my risk of missing the connection, evaluate the gate conformance rules at {hub}, and explain my statutory compensation rights under EU261 / UK261 in Euros (€ EUR).",
            f"Evaluate the travel risk of a connecting flight itinerary: {origin} ➔ {hub} (Layover: {layover_min}m) ➔ {dest}. What is the Minimum Connection Time (MCT), what happens if the inbound flight is delayed by 15 minutes, and what cash compensation applies under European regulation?",
            f"A passenger books a {layover_min}-minute transfer at {hub} traveling from {origin} to {dest}. Explain the biometric flight connection cutoff rule, compound weather risks at the hub, luggage transfer feasibility, and statutory airline obligations.",
        ]
        user_prompt = rng.choice(prompts)

        cot = (
            f"<thought>\n"
            f"1. Itinerary Decomposition:\n"
            f"   - Leg 1: {origin} ➔ {hub} ({operator}, Day: {day_of_week}, Wave: {wave})\n"
            f"   - Transfer Hub: {hub} (Scheduled Layover: {layover_min} min)\n"
            f"   - Leg 2: {hub} ➔ {dest}\n"
            f"2. Hub Transfer & Minimum Connection Time (MCT) Analysis:\n"
            f"   - Hub: {hub} -> Nominal MCT = {mct_nominal} min\n"
            f"   - Weather Transit Penalty: +{weather_transit_penalty} min -> Effective MCT = {eff_mct} min\n"
            f"   - Effective Slack Buffer: Layover ({layover_min}m) - Effective MCT ({eff_mct}m) = +{effective_slack:.1f} min\n"
            f"   - Missed Connection Risk: {missed_risk_pct:.1f}%\n"
            f"3. Hub Terminal & Gate Conformance Nuances:\n"
            f"   - Heathrow T5 biometric connections enforce a strict 20-minute gate conformance cutoff.\n"
            f"   - Boarding closes 20 minutes prior to scheduled departure. Outbound departure taxi delays do NOT hold the gate open.\n"
            f"4. Statutory Passenger Rights (Regulation EC 261/2004 & UK261):\n"
            f"   - Great-Circle Distance: {distance_km:.0f} km\n"
            f"   - Classification: {tier}\n"
            f"   - Statutory Cash Compensation: €{comp_eur:,.0f} EUR per passenger if arrival at final destination is delayed >= 3 hours\n"
            f"   - Duty of Care (Right to Care): Meals, refreshments, and communications after {duty_hours:.0f} hours delay\n"
            f"</thought>\n"
        )

        response_body = (
            f"### Passenger Travel Risk Advisory & Statutory Rights: {origin} ➔ {hub} ➔ {dest}\n\n"
            f"**Route:** {origin} ➔ {hub} (Hub) ➔ {dest} | **Flight Window:** {day_of_week} {wave.capitalize()}\n"
            f"**Scheduled Layover:** {layover_min} minutes at {hub} | **Primary Carrier:** {operator}\n\n"
            f"#### 1. Transfer Hub Feasibility & Connection Slack Analysis\n"
            f"- **Nominal Minimum Connection Time (MCT)**: **{mct_nominal} minutes** at {hub}\n"
            f"- **Effective Connection Slack**: **+{effective_slack:.0f} minutes** buffer above MCT ({layover_min}m layover − {eff_mct}m MCT).\n"
            f"- **Missed Connection Probability**: **{missed_risk_pct:.1f}%** ({'TIGHT connection with moderate misconnect risk' if effective_slack < 20 else 'COMFORTABLE buffer under standard operating conditions'}).\n"
            f"- **Checked Luggage Transfer**: "
            f"{'LOW RISK of luggage separation' if effective_slack >= 15 else 'HIGH RISK of baggage misconnection due to compressed ramp sorting times'}.\n\n"
            f"#### 2. Terminal Architecture & Gate Conformance Rules\n"
            f"> [!WARNING]\n"
            f"> **Strict 20-Minute Gate Conformance Cutoff at {hub}**:\n"
            f"> At modern hubs such as London Heathrow Terminal 5, biometric flight connection gates close strictly **20 minutes before scheduled departure off-blocks**. "
            f"Even if the outbound flight incurs a 25-minute ground taxi queue before takeoff, gate agents **will not hold the boarding gate open**. "
            f"Passengers arriving at the gate with under 20 minutes remaining will be offloaded.\n\n"
            f"#### 3. Statutory Passenger Rights & Compensation Charter (EU261 / UK261)\n"
            f"- **Great-Circle Journey Distance**: **{distance_km:.0f} km** ({tier})\n"
            f"- **Statutory Cash Compensation**: **€{comp_eur:,.0f} EUR** per passenger if you arrive at final destination `{dest}` with a delay of **≥ 3 hours** "
            f"(payable unless the airline proves delay was caused by extraordinary circumstances such as unannounced ATC strikes or severe gale-force weather).\n"
            f"- **Right to Care (Duty of Care)**: Airline must provide complimentary food, drink vouchers, and 2 communications after **{duty_hours:.0f} hours** of delay, regardless of the delay's root cause."
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
                "origin": origin,
                "hub": hub,
                "dest": dest,
                "layover_min": layover_min,
                "effective_slack": effective_slack,
                "comp_eur": comp_eur,
                "distance_km": distance_km,
            },
        )
