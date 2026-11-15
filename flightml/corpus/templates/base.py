from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from flightml.corpus.schemas import SFTConversation, SFTMessage


EUROPEAN_AVIATION_SYSTEM_PROMPTS = [
    (
        "You are EuroAvia-7B, an authoritative expert on European air traffic management, "
        "ICAO GANP performance indicators, Eurocontrol PRU metrics, 4D flight trajectory physics, "
        "and airline economics. Provide rigorous, step-by-step mathematical reasoning and cite exact figures in Euros (€ EUR)."
    ),
    (
        "You are an expert aviation intelligence assistant specializing in the European Air Traffic Management (ATM) "
        "network. When analyzing flight data, derive operational metrics systematically, show intermediate calculations, "
        "and explain downstream knock-on effects on airport capacity and airline operations."
    ),
    (
        "You are an AI advisor for European airline operations control (AOC) and air traffic flow management. "
        "Analyze flight efficiency, fuel burn, Eurocontrol CRCO navigation charges (€ EUR), and regulatory passenger "
        "rights with uncompromising analytical precision."
    ),
]


class BaseTemplate(ABC):
    """Abstract base class for all synthetic prompt-response generators."""

    def __init__(self, track_name: str):
        self.track_name = track_name

    def get_system_prompt(self, rng: random.Random) -> str:
        """Select a varied system prompt."""
        return rng.choice(EUROPEAN_AVIATION_SYSTEM_PROMPTS)

    @abstractmethod
    def generate(self, rng: random.Random, context: dict[str, Any]) -> SFTConversation | None:
        """Synthesize a complete SFT conversation from context data."""
        pass
