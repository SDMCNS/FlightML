from __future__ import annotations

import random
from typing import Any

from flightml.corpus.schemas import SFTConversation, SFTMessage
from flightml.corpus.templates.base import BaseTemplate


class DelayContagionTemplate(BaseTemplate):
    """Generates Track 3 samples: Rotational Delay Contagion & Tail Tracking."""

    def __init__(self):
        super().__init__(track_name="delay_contagion")

    def generate(self, rng: random.Random, context: dict[str, Any]) -> SFTConversation | None:
        reg = context.get("ac_reg", "EI-SYNA")
        operator = context.get("operator", "SYN")
        ac_type = context.get("ac_type", "A320")
        hub = context.get("hub", "EGLL")
        outstation = context.get("outstation", "LEMD")

        inbound_delay_min = context.get("inbound_delay_min", rng.randint(45, 95))
        scheduled_turn_min = context.get("scheduled_turn_min", 45)
        minimum_turn_min = context.get("minimum_turn_min", 35)
        slack_buffer_min = scheduled_turn_min - minimum_turn_min  # e.g. 10 min

        reactionary_delay_min = max(0, inbound_delay_min - slack_buffer_min)
        cost_per_delay_min_eur = 88.0  # Eurocontrol standard input benchmark
        reactionary_cost_eur = reactionary_delay_min * cost_per_delay_min_eur

        prompts = [
            f"Explain how rotational delay contagion propagates along airframe {reg} ({operator} {ac_type}) if leg 1 arrives at {hub} with a {inbound_delay_min}-minute delay. Quantify the downstream reactionary departure delay on leg 2 ({hub} to {outstation}) and explain the countervailing slot relief phenomenon.",
            f"An airline operations controller at {hub} observes that aircraft {reg} incurred a {inbound_delay_min}-minute primary arrival delay. The scheduled turnaround buffer is {scheduled_turn_min} minutes (minimum turnaround time is {minimum_turn_min} minutes). Calculate the reactionary delay and economic impact in Euros (€ EUR).",
            f"Trace the tail-rotation delay contagion for {operator} airframe {reg}. What is the downstream impact on subsequent legs if turnaround slack is exhausted, and how do stranded return legs accelerate hub backlog clearance?",
        ]
        user_prompt = rng.choice(prompts)

        cot = (
            f"<thought>\n"
            f"1. Rotational Sequence & Airframe Identification:\n"
            f"   - Airframe: {reg} ({operator} {ac_type})\n"
            f"   - Turnaround Station: {hub}\n"
            f"   - Next Downstream Leg: {hub} ➔ {outstation}\n"
            f"2. Turnaround Buffer Mechanics:\n"
            f"   - Inbound Delay (D_inbound): {inbound_delay_min} min\n"
            f"   - Scheduled Ground Time: {scheduled_turn_min} min\n"
            f"   - Minimum Required Turnaround (T_min): {minimum_turn_min} min\n"
            f"   - Scheduled Turnaround Slack: S = {scheduled_turn_min} - {minimum_turn_min} = {slack_buffer_min} min\n"
            f"3. Reactionary Delay Propagation Formula:\n"
            f"   - Downstream Delay D_react = max(0, D_inbound - S) = max(0, {inbound_delay_min} - {slack_buffer_min}) = {reactionary_delay_min} min\n"
            f"   - Eurocontrol Standard Cost Benchmark: €{cost_per_delay_min_eur:.2f} EUR/min\n"
            f"   - Tactical Economic Cost: {reactionary_delay_min} * €{cost_per_delay_min_eur:.2f} = €{reactionary_cost_eur:,.2f} EUR\n"
            f"4. Countervailing Slot Relief Dynamic:\n"
            f"   - If late airframe causes cancellation of downstream return leg {outstation} ➔ {hub}, those departure and arrival slots are vacated.\n"
            f"   - Vacated slots liberate runway and gate capacity for remaining traffic, dampening system-wide backlog.\n"
            f"</thought>\n"
        )

        response_body = (
            f"### Airframe Tail-Tracking & Rotational Delay Contagion Analysis\n\n"
            f"**Airframe Registration:** `{reg}` ({operator} {ac_type}) | **Turnaround Hub:** {hub}\n"
            f"**Downstream Route:** {hub} ➔ {outstation}\n\n"
            f"#### 1. Rotational Delay Mechanics & Turnaround Slack\n"
            f"Commercial narrowbody aircraft operate high-utilization rotations (4 to 8 legs daily) with compressed ground turnarounds. "
            f"When an inbound leg incurs significant delay, downstream departures inherit reactionary delay once ground slack is depleted:\n\n"
            f"$$\\text{{Turnaround Slack Buffer }} (S) = T_{{\\text{{scheduled}}}} - T_{{\\text{{minimum}}}} = {scheduled_turn_min}\\text{{ min}} - {minimum_turn_min}\\text{{ min}} = {slack_buffer_min}\\text{{ minutes}}$$\n"
            f"$$\\text{{Reactionary Delay }} (D_{{\\text{{react}}}}) = \\max(0,\\; D_{{\\text{{inbound}}}} - S) = \\max(0,\\; {inbound_delay_min} - {slack_buffer_min}) = \\mathbf{{{reactionary_delay_min}\\text{{ minutes}}}}$$\n\n"
            f"- **Inbound Primary Delay**: +{inbound_delay_min} minutes\n"
            f"- **Ground Slack Absorbed**: {slack_buffer_min} minutes\n"
            f"- **Downstream Departure Delay on Leg 2**: **+{reactionary_delay_min} minutes**\n"
            f"- **Estimated Tactical Reactionary Cost**: **€{reactionary_cost_eur:,.2f} EUR** "
            f"(computed using the Eurocontrol Standard Inputs benchmark of €{cost_per_delay_min_eur:.2f}/min for crew, maintenance, passenger misconnections, and handling).\n\n"
            f"#### 2. Downstream Outstation Impact & Countervailing Slot Relief\n"
            f"Rotational delay propagation creates two contrasting operational effects across the European network:\n"
            f"1. **Outstation Contagion Spread**: The reactionary delay on leg 2 arrives late into outstation `{outstation}`. Because ground staff and gate windows at outstations are constrained, outbound return passengers face compounding delays or flight cancellations.\n"
            f"2. **Countervailing Slot Relief (Backlog Clearance)**: If `{reg}` is delayed so severely that subsequent legs in the chain must be cancelled or consolidated, those unserved flights **vacate their scheduled runway takeoff, landing, and gate slots**. This paradoxically frees up critical bottleneck capacity at hub `{hub}`, accelerating queue clearance for other arriving flights."
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
                "ac_reg": reg,
                "hub": hub,
                "outstation": outstation,
                "inbound_delay_min": inbound_delay_min,
                "reactionary_delay_min": reactionary_delay_min,
                "reactionary_cost_eur": reactionary_cost_eur,
            },
        )
