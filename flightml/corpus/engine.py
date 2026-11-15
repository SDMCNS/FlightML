from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from flightml.corpus.exporter import CorpusExporter
from flightml.corpus.negative_sampler import NegativeSampler
from flightml.corpus.schemas import DPOPair, SFTConversation, ToolCallSample
from flightml.corpus.templates.crco_fuel import CRCOFuelTemplate
from flightml.corpus.templates.delay_contagion import DelayContagionTemplate
from flightml.corpus.templates.passenger_copilot import PassengerCopilotTemplate
from flightml.corpus.templates.pru_physics import PRUPhysicsTemplate
from flightml.corpus.templates.simulations import SimulationTemplate
from flightml.corpus.templates.tool_calling import ToolCallingTemplate
from flightml.data.dataset import FlightDataset
from flightml.eval.crco_economics import compute_flight_crco_charges
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
from flightml.eval.pru_kpis import compute_flight_pru_kpis


EUROPEAN_CITY_PAIRS = [
    ("EGLL", "LIRF"),
    ("EIDW", "EHAM"),
    ("LEMD", "EBBR"),
    ("LFPG", "EDDF"),
    ("EGLL", "LEBL"),
    ("EHAM", "LOWW"),
    ("EDDF", "LEMD"),
    ("EINN", "EGLL"),
    ("EGLL", "EBBR"),
    ("LFPO", "LFMN"),
    ("EDDM", "LIRF"),
    ("EGKK", "LEPA"),
]


class CorpusGenerator:
    """Orchestrates multi-track European aviation synthetic training corpus generation."""

    def __init__(
        self,
        dataset: FlightDataset | None = None,
        period: str = "auto",
        seed: int = 42,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.period = period
        self.dataset = dataset

        # Templates
        self.template_pru = PRUPhysicsTemplate()
        self.template_crco = CRCOFuelTemplate()
        self.template_contagion = DelayContagionTemplate()
        self.template_sims = SimulationTemplate()
        self.template_passenger = PassengerCopilotTemplate()
        self.template_tools = ToolCallingTemplate()

        self.negative_sampler = NegativeSampler(seed=seed)

    def _ensure_dataset(self) -> FlightDataset | None:
        """Lazily initialize dataset if not already loaded."""
        if self.dataset is None:
            try:
                self.dataset = FlightDataset(period=self.period)
            except Exception:
                self.dataset = None
        return self.dataset

    def generate_corpus(
        self,
        total_samples: int = 1000,
        tracks: list[str] | str = "all",
        output_dir: str | Path = "data/llm_corpus",
        formats: list[str] | None = None,
        train_split: float = 0.9,
    ) -> dict[str, Any]:
        """Synthesize and export structured LLM training datasets."""
        if formats is None:
            formats = ["chatml", "dpo", "tools"]

        valid_tracks = ["pru_physics", "crco_fuel", "delay_contagion", "simulations", "passenger_copilot"]
        if tracks == "all":
            selected_tracks = valid_tracks
        elif isinstance(tracks, str):
            selected_tracks = [t.strip() for t in tracks.split(",") if t.strip() in valid_tracks]
        else:
            selected_tracks = [t for t in tracks if t in valid_tracks]

        if not selected_tracks:
            selected_tracks = valid_tracks

        samples_per_track = max(1, total_samples // len(selected_tracks))
        ds = self._ensure_dataset()

        sft_samples: list[SFTConversation] = []
        dpo_pairs: list[DPOPair] = []
        tool_samples: list[ToolCallSample] = []

        # 1. Trajectory Physics & CRCO Generation (Data-backed where possible)
        historical_flights: list[dict[str, Any]] = []
        if ds is not None:
            try:
                df = ds.search_flights(limit=min(samples_per_track * 2, 200))
                if not df.empty:
                    historical_flights = df.to_dict(orient="records")
            except Exception:
                historical_flights = []

        for track in selected_tracks:
            count = 0
            while count < samples_per_track:
                pair = self.rng.choice(EUROPEAN_CITY_PAIRS)
                context: dict[str, Any] = {
                    "origin": pair[0],
                    "dest": pair[1],
                }

                if track in ("pru_physics", "crco_fuel") and historical_flights:
                    fl_row = self.rng.choice(historical_flights)
                    ectrl_id = int(fl_row.get("ectrl_id", 900104215))
                    context["ectrl_id"] = ectrl_id
                    try:
                        if track == "pru_physics":
                            context["pru"] = compute_flight_pru_kpis(ds, ectrl_id=ectrl_id)
                        else:
                            context["fuel_impact"] = compute_flight_fuel_and_emissions(ds, ectrl_id=ectrl_id)
                            context["crco"] = compute_flight_crco_charges(ds, ectrl_id=ectrl_id)
                    except Exception:
                        pass

                # Track-specific dispatch
                conv: SFTConversation | None = None
                if track == "pru_physics":
                    conv = self.template_pru.generate(self.rng, context)
                elif track == "crco_fuel":
                    conv = self.template_crco.generate(self.rng, context)
                elif track == "delay_contagion":
                    context["hub"] = pair[0]
                    context["outstation"] = pair[1]
                    conv = self.template_contagion.generate(self.rng, context)
                elif track == "simulations":
                    conv = self.template_sims.generate(self.rng, context)
                elif track == "passenger_copilot":
                    context["hub"] = "EGLL" if pair[0] != "EGLL" else "EDDF"
                    conv = self.template_passenger.generate(self.rng, context)

                if conv:
                    sft_samples.append(conv)
                    dpo_pairs.append(self.negative_sampler.generate_dpo_pair(conv))
                    count += 1

        # 2. Function-Calling Agent Trajectories
        if "tools" in formats:
            tools_count = max(10, total_samples // 10)
            for _ in range(tools_count):
                pair = self.rng.choice(EUROPEAN_CITY_PAIRS)
                tool_ctx = {"origin": pair[0], "dest": pair[1], "hub": "EGLL"}
                sample = self.template_tools.generate(self.rng, tool_ctx)
                tool_samples.append(sample)

        # 3. Export to Files
        exporter = CorpusExporter(output_dir=output_dir, train_split=train_split)
        export_summary: dict[str, Any] = {}

        if "chatml" in formats:
            export_summary["chatml"] = exporter.export_sft(sft_samples, format_type="chatml")
        if "alpaca" in formats:
            export_summary["alpaca"] = exporter.export_sft(sft_samples, format_type="alpaca")
        if "dpo" in formats:
            export_summary["dpo"] = exporter.export_dpo(dpo_pairs)
        if "tools" in formats:
            export_summary["tools"] = exporter.export_tools(tool_samples)

        return {
            "total_sft_generated": len(sft_samples),
            "total_dpo_generated": len(dpo_pairs),
            "total_tool_trajectories": len(tool_samples),
            "tracks": selected_tracks,
            "export_summary": export_summary,
        }
