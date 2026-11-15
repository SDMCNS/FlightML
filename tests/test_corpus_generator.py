from __future__ import annotations

import json
from pathlib import Path
import pytest

from flightml.corpus.engine import CorpusGenerator
from flightml.corpus.exporter import CorpusExporter
from flightml.corpus.negative_sampler import NegativeSampler
from flightml.corpus.schemas import DPOPair, SFTConversation, SFTMessage, ToolCallSample
from flightml.corpus.templates.crco_fuel import CRCOFuelTemplate
from flightml.corpus.templates.delay_contagion import DelayContagionTemplate
from flightml.corpus.templates.passenger_copilot import PassengerCopilotTemplate
from flightml.corpus.templates.pru_physics import PRUPhysicsTemplate
from flightml.corpus.templates.simulations import SimulationTemplate
from flightml.corpus.templates.tool_calling import ToolCallingTemplate
from flightml.corpus.training_config import HardwareProfile, QLoRAHyperparameters


def test_sft_schemas_and_conversion():
    conv = SFTConversation(
        messages=[
            SFTMessage(role="system", content="You are EuroAvia-7B."),
            SFTMessage(role="user", content="Calculate KEP for flight 900104215."),
            SFTMessage(role="assistant", content="<thought>Computing...</thought>KEP is 4.18%."),
        ],
        track="pru_physics",
        metadata={"ectrl_id": 900104215},
    )

    chatml = conv.to_chatml()
    assert len(chatml["messages"]) == 3
    assert chatml["messages"][0]["role"] == "system"
    assert chatml["track"] == "pru_physics"

    alpaca = conv.to_alpaca()
    assert alpaca["instruction"] == "You are EuroAvia-7B."
    assert alpaca["input"] == "Calculate KEP for flight 900104215."
    assert "KEP is 4.18%" in alpaca["output"]


def test_dpo_schemas():
    dpo = DPOPair(
        system="You are EuroAvia-7B.",
        prompt="What is the CRCO fee?",
        chosen="The fee is €1,584.20 EUR.",
        rejected="The fee is $1,500 USD.",
        track="crco_fuel",
    )
    dpo_dict = dpo.to_dict()
    assert "<|user|>\nWhat is the CRCO fee?" in dpo_dict["prompt"]
    assert "€1,584.20 EUR" in dpo_dict["chosen"]
    assert "$1,500 USD" in dpo_dict["rejected"]


def test_qlora_hardware_budget_16gb():
    cfg = QLoRAHyperparameters()
    mem = cfg.estimate_vram_footprint_gb()

    assert mem["model_weights_gb"] > 4.0
    assert mem["total_estimated_vram_gb"] < 14.5  # Must comfortably fit within 16GB
    assert mem["available_headroom_gb"] >= 1.5   # Must have safety buffer

    script = cfg.generate_training_script_snippet()
    assert "Qwen/Qwen2.5-7B-Instruct" in script
    assert "FastLanguageModel" in script
    assert "paged_adamw_8bit" in script


def test_templates_generate_valid_content():
    import random
    rng = random.Random(42)

    # 1. PRU Physics
    pru_tpl = PRUPhysicsTemplate()
    sft_pru = pru_tpl.generate(rng, {"ectrl_id": 900104215, "adep": "EGLL", "ades": "LIRF"})
    assert sft_pru is not None
    assert "<thought>" in sft_pru.messages[-1].content
    assert "KPI04 KEP" in sft_pru.messages[-1].content

    # 2. CRCO Fuel
    crco_tpl = CRCOFuelTemplate()
    sft_crco = crco_tpl.generate(rng, {"ectrl_id": 900104215, "adep": "EGLL", "ades": "LIRF"})
    assert sft_crco is not None
    assert "€" in sft_crco.messages[-1].content
    assert "Service Units" in sft_crco.messages[-1].content

    # 3. Delay Contagion
    contagion_tpl = DelayContagionTemplate()
    sft_cont = contagion_tpl.generate(rng, {"ac_reg": "EI-SYNA", "hub": "EGLL", "outstation": "LEMD"})
    assert sft_cont is not None
    assert "Reactionary Delay" in sft_cont.messages[-1].content
    assert "Slot Relief" in sft_cont.messages[-1].content

    # 4. Airspace Simulation
    sim_tpl = SimulationTemplate()
    sft_sim = sim_tpl.generate(rng, {"target_fir": "LSASFIR", "duration_hrs": 3.0})
    assert sft_sim is not None
    assert "LSASFIR" in sft_sim.messages[-1].content
    assert "€" in sft_sim.messages[-1].content

    # 5. Passenger Copilot
    pass_tpl = PassengerCopilotTemplate()
    sft_pass = pass_tpl.generate(rng, {"origin": "EINN", "hub": "EGLL", "dest": "EBBR"})
    assert sft_pass is not None
    assert "Gate Conformance Cutoff" in sft_pass.messages[-1].content
    assert "EU261" in sft_pass.messages[-1].content

    # 6. Tool Calling
    tool_tpl = ToolCallingTemplate()
    tool_sample = tool_tpl.generate(rng, {"ectrl_id": 900104215})
    assert tool_sample is not None
    assert len(tool_sample.tools) >= 2
    assert any(m.get("tool_calls") for m in tool_sample.messages)


def test_negative_sampler_contrast():
    import random
    rng = random.Random(42)
    crco_tpl = CRCOFuelTemplate()
    sft = crco_tpl.generate(rng, {})

    sampler = NegativeSampler(seed=42)
    dpo = sampler.generate_dpo_pair(sft)

    assert "€" in dpo.chosen
    assert ("$" in dpo.rejected or "£" in dpo.rejected or "flat fee" in dpo.rejected)


def test_corpus_generator_e2e(tmp_path: Path):
    generator = CorpusGenerator(seed=42)
    res = generator.generate_corpus(
        total_samples=15,
        tracks=["pru_physics", "crco_fuel", "passenger_copilot"],
        output_dir=tmp_path,
        formats=["chatml", "dpo", "tools"],
    )

    assert res["total_sft_generated"] >= 15
    assert res["total_dpo_generated"] >= 15
    assert res["total_tool_trajectories"] >= 10

    # Verify output files exist and are valid JSONL
    chatml_train = tmp_path / "sft_chatml_train.jsonl"
    dpo_train = tmp_path / "dpo_train.jsonl"
    tools_file = tmp_path / "tool_calling_samples.jsonl"

    assert chatml_train.exists()
    assert dpo_train.exists()
    assert tools_file.exists()

    with open(chatml_train, "r", encoding="utf-8") as f:
        line = json.loads(f.readline())
        assert "messages" in line
        assert len(line["messages"]) == 3

    with open(dpo_train, "r", encoding="utf-8") as f:
        line = json.loads(f.readline())
        assert "prompt" in line
        assert "chosen" in line
        assert "rejected" in line
