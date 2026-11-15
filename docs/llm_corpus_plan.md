# FlightML-Instruct: Synthetic LLM Training Corpus Generator Plan ✈️🧠

## Executive Summary

The objective of this initiative is to leverage FlightML's high-performance DuckDB columnar stores, physics models (OpenAP, ICAO EDB v32B), Eurocontrol PRU KPI algorithms, CRCO en-route cost accounting, operations research simulators, and Passenger Travel Copilot to automatically generate **zero-hallucination, mathematically verified, structured training corpora**.

This dataset enables fine-tuning open **4B–8B parameter models** (such as **Qwen 2.5 7B**, **Llama 3.1 8B**, or **Mistral 7B**) into authoritative, production-grade **European Aviation Domain Intelligence Systems**.

---

## 🎯 Hardware Target & Resource Balancing

The user workstation profile:
- **CPU**: Intel Core i7
- **System Memory**: 32 GB RAM
- **GPU**: NVIDIA RTX 4060 Ti / 5060 Ti with **16 GB VRAM**

### VRAM & Training Resource Allocation Strategy (16 GB VRAM)
In 16 GB VRAM, 4-bit quantized base models (via bitsandbytes NF4 or Unsloth) occupy approximately:
- **Base Model (4-bit Qwen 2.5 7B / Llama 3.1 8B)**: ~5.5 GB to 6.2 GB VRAM.
- **LoRA Adapter Weights ($r=64, \alpha=128$)**: ~0.4 GB VRAM.
- **Available Working VRAM for Activations & KV**: ~9.4 GB to 10.1 GB VRAM.

Because time is not a constraining factor, we optimize for **sample depth and stability**:
- **Sequence Length**: 2,048 to 4,096 tokens (ample room for detailed Chain-of-Thought mathematical derivations).
- **Per-Device Batch Size**: `micro_batch_size = 2`.
- **Gradient Accumulation**: `gradient_accumulation_steps = 8` (effective batch size of 16).
- **Optimizer**: `paged_adamw_8bit` (automatically pages states to system RAM if momentary spikes occur, preventing out-of-memory crashes).
- **Gradient Checkpointing**: Enabled.

---

## 📚 1. Six Targeted Curriculum Tracks

The corpus generator synthesizes prompt-response pairs across six core tracks:

### Track 1: 4D Trajectory & ICAO GANP / Eurocontrol PRU Scorecards
- **Focus**: Deriving `KPI04 KEP` (filed extension), `KPI05 KEA` (actual extension), tactical ATC shortcut attribution, continuous descent (`CDO`), continuous climb (`CCO`), `KPI08 ASMA` (40 nm terminal holding), and `KPI19` cruise level-capping.
- **Reasoning Format**: Multi-step mathematical derivation showing great-circle distance $H$, filed distance $L$, actual distance $A$, and tactical ATC delta ($KEA - KEP$).

### Track 2: Fuel Burn, ICAO EDB v32B Emissions & Eurocontrol CRCO Economics (`€ EUR`)
- **Focus**: Step-by-step derivation of fuel consumption, excess fuel burn vs. optimal 3-way cruise altitude, ICAO Engine Emissions Databank (EDB v32B) LTO pollutants ($\text{NO}_x$, $\text{CO}$, $\text{HC}$, $\text{nvPM}$ soot), and Eurocontrol CRCO en-route fees:
  $$\text{Charge}_{\text{FIR}} = d_{\text{FIR}} \times \sqrt{\frac{\text{MTOW}}{50}} \times \text{UnitRate}_{\text{FIR}}$$
- **Monetary Compliance**: Strictly in Euros (`€`, `EUR`).

### Track 3: Delay Contagion, Network Bottlenecks & Aircraft Rotations (`ac_reg`)
- **Focus**: Tracing multi-leg rotational delay propagation along physical airframe registrations (`ac_reg`), calculating inbound delay $D_{\text{arr}}$, scheduled ground turnaround slack $S$, downstream departure delay:
  $$D_{\text{react}} = \max(0,\; D_{\text{arr}} - S)$$
  and countervailing slot relief (unoperated return legs vacating runway capacity).

### Track 4: Prescriptive Operations Research & Airspace What-If Simulators
- **Focus**: Strategic decision-making under network disruptions:
  - Temporary 4D FIR closures (e.g. Swiss `LSASFIR`, French `LFFFFIR`): tactical reroutes, nautical mile penalties, extra Jet-A burn, airline operating losses (`€ EUR`), and queue recovery times.
  - Hub tail-swap optimization (`Tail A ⇄ Tail B`) to eliminate primary and reactionary delays.
  - Ground Delay Programs (GDP / Ration-by-Schedule slot rationing).
  - Multi-Agent RL ATFM slot gaming (cooperative vs. competitive Nash equilibria).

### Track 5: Public Passenger Travel Copilot & Statutory Rights (EU261 / UK261)
- **Focus**: Consumer journey intelligence:
  - Direct and multi-leg connections ($Leg_1 \to Hub \to Leg_2$).
  - Airport-specific Minimum Connection Times (MCT) and effective slack buffers.
  - Heathrow Terminal 5 biometric flight connections and the strict 20-minute gate closure rule.
  - Compound hub weather impacts (inbound holding + slippery ramp transit bus delays).
  - Geodesic great-circle distance tiers and statutory cash compensation (€250, €400, €600).
  - Duty of care obligations (meals, refreshments, hotel accommodation).

### Track 6: FastMCP Tool-Calling & Autonomous Agent Trajectories
- **Focus**: Training the model to act as an autonomous agent that invokes FlightML FastMCP tools:
  - Emitting structured JSON tool calls: `flightml_assess_passenger_trip`, `flightml_get_pru_kpis`, `flightml_diagnose_flight`.
  - Parsing the tool output and synthesizing authoritative, formatted answers with cited flight IDs and euro figures.

---

## 🏗️ 2. Module Architecture

```text
flightml/
└── corpus/
    ├── __init__.py
    ├── engine.py                 # Core CorpusGenerator engine orchestrating generation
    ├── schemas.py                # Pydantic schemas (SFTMessage, SFTConversation, DPOPair, ToolCallSample)
    ├── templates/
    │   ├── __init__.py
    │   ├── base.py               # Abstract template class with persona & prompt perturbators
    │   ├── pru_physics.py        # Track 1: Trajectory & PRU KPI derivation templates
    │   ├── crco_fuel.py          # Track 2: Fuel burn, emissions & CRCO charges templates
    │   ├── delay_contagion.py    # Track 3: Rotational delay contagion templates
    │   ├── simulations.py        # Track 4: Airspace closure & OR scenario templates
    │   ├── passenger_copilot.py  # Track 5: Passenger journey & EU261 templates
    │   └── tool_calling.py       # Track 6: FastMCP function-calling agent templates
    ├── negative_sampler.py       # Plausible hallucination generator for DPO rejected pairs
    ├── exporter.py               # Streaming JSONL exporter with train/test splits & token count
    └── training_config.py        # Hardware-tuned QLoRA hyperparameter configuration for 16 GB VRAM
```
