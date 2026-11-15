from __future__ import annotations

import random
from flightml.corpus.schemas import DPOPair, SFTConversation


COMMON_DOMAIN_HALLUCINATIONS = {
    "crco_fuel": [
        (
            "The Eurocontrol en-route charge is calculated by multiplying the flight distance by a flat fee of $120.00 USD. "
            "For this flight, the estimated route fee is approximately $1,800.00 USD. Fuel burn is calculated using typical fuel burn charts."
        ),
        (
            "The navigation fee is £1,450.00 GBP based on standard British airspace pricing. "
            "Weight factor does not apply to narrowbody aircraft under 100 tonnes."
        ),
    ],
    "pru_physics": [
        (
            "KPI04 (KEP) measures the actual flown radar distance, while KPI05 (KEA) measures the filed flight plan. "
            "Because KEA was lower than KEP, the airline filed a longer route than permitted."
        ),
        (
            "En-route efficiency is simply the straight line distance divided by flight time. "
            "There were no tactical ATC interventions observed because the pilots flew on autopilot."
        ),
    ],
    "delay_contagion": [
        (
            "Rotational delay does not affect subsequent legs because airlines always have spare aircraft waiting at every airport. "
            "The aircraft will make up the 45-minute delay by flying faster in the air."
        ),
        (
            "When a flight is delayed, the subsequent flight departs immediately on time regardless of where the physical airframe is located."
        ),
    ],
    "passenger_copilot": [
        (
            "Don't worry about the 20-minute connection cutoff. If the outbound flight is held in a departure taxi queue, "
            "the gate agents will hold the boarding gate open until you arrive at the gate."
        ),
        (
            "Under EU261, you are automatically entitled to £520 GBP for any delay over 1 hour. "
            "There are no distance tiers under European passenger regulations."
        ),
        (
            "Heathrow has a standard 20-minute connection time across all terminals, so a 45-minute layover provides plenty of extra time."
        ),
    ],
    "simulations": [
        (
            "When an airspace region closes, all aircraft immediately land at the nearest airport. "
            "There are no en-route rerouting procedures in European ATM."
        ),
        (
            "Airspace closure incurs zero extra fuel burn because planes simply pause their journey."
        ),
    ],
}


class NegativeSampler:
    """Generates realistic domain-specific negative / hallucinated responses for DPO pairs."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_dpo_pair(self, sft_sample: SFTConversation) -> DPOPair:
        """Converts an SFT conversation into a high-contrast DPO Pair."""
        system_content = ""
        user_prompt = ""
        chosen_resp = ""

        for msg in sft_sample.messages:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "user":
                user_prompt = msg.content
            elif msg.role == "assistant":
                chosen_resp = msg.content

        track = sft_sample.track
        hallucinations = COMMON_DOMAIN_HALLUCINATIONS.get(
            track,
            [
                "The European aviation network uses approximate standard pricing and schedules are generally fixed without reactionary delay propagation."
            ],
        )

        rejected_resp = self.rng.choice(hallucinations)

        return DPOPair(
            system=system_content,
            prompt=user_prompt,
            chosen=chosen_resp,
            rejected=rejected_resp,
            track=track,
            metadata=sft_sample.metadata,
        )
