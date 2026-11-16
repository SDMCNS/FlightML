# EuroAvia LLM: Aeronautical Fine-Tuning, MCP Tools & 5-Stage Benchmark Evaluation

## 1. Overview

**EuroAvia** is FlightML's specialized Large Language Model (LLM) fine-tuning and evaluation framework tailored for domain-specific aeronautical network intelligence, trajectory physics, environmental emissions, and operations research.

General-purpose open-source LLMs (such as base Qwen 2.5 7B, Llama 3, or Mistral) frequently hallucinate exact radar waypoint coordinates, miscalculate Continuous Descent Operation (CDO) descent scores, confuse ICAO GANP KPIs, and misattribute Eurocontrol CRCO FIR navigation charges.

EuroAvia solves this by combining:
1. **Aeronautical Instruction Corpus Generation** (`flightml/corpus/engine.py`): Synthesizes thousands of high-fidelity, verified multi-turn aeronautical dialogue records derived deterministically from historical radar telemetry and physics models.
2. **QLoRA Parameter-Efficient Fine-Tuning** (`scripts/train_qlora.py`): 4-bit NF4 quantized LoRA fine-tuning for consumer and workstation GPUs (16 GB – 24 GB VRAM).
3. **Model Context Protocol (MCP) Integration** (`flightml/mcp/server.py`): Native FastMCP server providing end-user CLI tools with `--json` output so models can query actual radar databases deterministically.
4. **5-Stage Comparative Evaluation Benchmark** (`scripts/run_5stage_eval.py`): Rigorous empirical framework testing deterministic math against base LLMs, fine-tuned LLMs, and MCP tool-calling agents.

---

## 2. Multi-Turn Aeronautical Training Corpus

The corpus engine (`flightml/corpus/engine.py`) programmatically extracts trajectories, sector handoffs, and performance models from DuckDB archives to synthesize structured conversations in ShareGPT and Alpaca formats:

### Supported Domain Modules:
* **Single Flight 4D Telemetry**: Off-block times, Great Circle Distance (GCD), actual distance, fuel burn, cruise altitude profile.
* **CDO / CCO Profile Analysis**: Top of Descent (TOD), intermediate level-off steps, vertical efficiency scores.
* **Delay & Runway Safety**: Departure/arrival delays, runway surface condition, stopping distances.
* **Fuel & Emissions**: 3-way optimal vs actual cruise flight levels, excess fuel waste, $\text{CO}_2$ and $\text{NO}_x$ emissions.
* **Eurocontrol CRCO Charges**: FIR traversals, entry/exit points, Service Units (SU), national unit rates, and fees in Euros (`€`).
* **Surface Movements (TAXI_OUT)**: Unimpeded taxi time (UTOT p20), ICAO KPI02 Additional Taxi-Out Time (ATOT).
* **Free Route Airspace (FRA)**: FIR boundary directness, airway funneling, detour deviations, multi-horizon benefits.
* **Operations Research**: Ground Delay Program (GDP) slot allocation, hub tail swaps.

### Data Privacy & Synthetic Anonymization:
In strict compliance with Eurocontrol OneSky Online Data Dissemination Policies, the generator automatically masks commercial identifiers:
* Callsigns and registrations use synthetic prefixes (e.g. `SYN101`, `G-SYNA`, `EI-SYNB`).
* Real radar trajectory points and physics calculations are preserved with zero distortion.

### CLI Generation Commands:
```bash
# Generate 1,000 multi-turn training conversations from period 202510
flightml corpus generate --period 202510 --count 1000 --out data/euroavia_train.jsonl

# Generate sample audit records for human inspection
flightml corpus sample --count 20 --out data/euroavia_sample.json
```

---

## 3. QLoRA Fine-Tuning Pipeline

The training script (`scripts/train_qlora.py`) fine-tunes base foundation models (default: `Qwen/Qwen2.5-7B-Instruct`) using **QLoRA (Dettmers et al.)**:

### Configuration:
* **Base Model**: `Qwen/Qwen2.5-7B-Instruct` (or `meta-llama/Llama-3.1-8B-Instruct`)
* **Quantization**: 4-bit NormalFloat (`NF4`) with double quantization and bfloat16 compute
* **LoRA Rank ($r$)**: `16` | **Alpha ($\alpha$)**: `32` | **Dropout**: `0.05`
* **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
* **Context Length**: `2048` tokens
* **Optimizer**: `paged_adamw_8bit` | **Batch Size**: 4 per device with gradient accumulation 4 (effective batch 16)
* **Learning Rate**: `2e-4` with cosine annealing schedule

### Launching Training:
```bash
python scripts/train_qlora.py \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset_path "data/euroavia_train.jsonl" \
  --output_dir "models/euroavia-7b-qlora" \
  --epochs 3 \
  --batch_size 4
```

---

## 4. Model Context Protocol (MCP) Server

FlightML includes a native FastMCP server (`flightml-mcp`) enabling local LLMs (via LM Studio, Claude Desktop, Cursor, or LangChain) to query the underlying DuckDB engine autonomously via structured JSON-RPC tool calls.

### Exposed MCP Tools:
1. `get_flight_telemetry`: Retrieve departure, arrival, aircraft type, operator, off-block time, and trajectory points.
2. `analyze_cdo_descent`: Continuous Descent Operations compliance, level-off steps, and descent fuel waste.
3. `analyze_cco_climb`: Continuous Climb Operations compliance and climb fuel penalties.
4. `compute_fuel_and_emissions`: 3-way optimal vs flown flight levels, fuel burn, and carbon emissions.
5. `compute_crco_charges`: Eurocontrol CRCO route charges, FIR breakdown, and direct operating cost in Euros (`€`).
6. `analyze_taxi_out`: Surface taxi delay, unimpeded UTOT, and ICAO KPI02 ATOT metrics.
7. `compute_pru_kpis`: ICAO GANP / Eurocontrol PRU KPIs (KEP, KEA, ASMA, level-capping).
8. `analyze_fra_boundaries`: Free Route Airspace FIR boundary inefficiencies, directness index, and detour metrics.
9. `calculate_fra_forecast`: Multi-horizon 1, 5, and 20-year FRA benefits across energy scenarios and SAF mandates.
10. `get_fra_worst_flights`: Retrieve the most detoured flights across a bottleneck FIR boundary.
11. `get_network_summary`: Global European network flight counts, delays, and airport statistics.
12. `optimize_tail_swaps`: Prescriptive hub tail-swap recovery schedule to minimize reactionary passenger delays.

---

## 5. The 5-Stage Comparative Evaluation Benchmark

To critically evaluate model capabilities and verify tool integration, FlightML implements a 5-stage evaluation benchmark (`scripts/run_5stage_eval.py`):

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        5-STAGE BENCHMARK DESIGN                        │
├─────────┬─────────────────────────┬────────────────────────────────────┤
│ Stage   │ Architecture            │ Operational Characteristic         │
├─────────┼─────────────────────────┼────────────────────────────────────┤
│ Stage 1 │ Deterministic CLI       │ Ground Truth: Pure math & DuckDB   │
│ Stage 2 │ Base Qwen 2.5 7B        │ Unassisted general LLM reasoning   │
│ Stage 3 │ EuroAvia 7B Fine-Tuned  │ Domain-adapted aeronautical LLM    │
│ Stage 4 │ Base Qwen 2.5 7B + MCP  │ Tool-augmented general LLM         │
│ Stage 5 │ EuroAvia 7B + MCP       │ Fully autonomous specialist agent  │
└─────────┴─────────────────────────┴────────────────────────────────────┘
```

### Benchmark Tasks Evaluated:
1. **Task 1: Flight Trajectory Telemetry & Great Circle Directness** (Flight ID verification, Great Circle vs Actual distance, detour NM).
2. **Task 2: Continuous Descent Operation (CDO) Quality** (Top of descent detection, intermediate level-off count, descent fuel waste).
3. **Task 3: Eurocontrol CRCO Route Charges in Euros (€)** (FIR traversal distances, FIR unit rates, exact CRCO fee in EUR).
4. **Task 4: Free Route Airspace (FRA) Bottleneck Identification** (Choke point ranking, detour % deviation, 20-year savings).
5. **Task 5: Hub Reactionary Delay Propagation** (Hub tail-swap recovery, rotational delay mitigation, passenger delay savings).

### Running the Evaluation Suite:
```bash
python scripts/run_5stage_eval.py \
  --period 202510 \
  --output_dir llm_tests \
  --compare
```

Detailed results and critical analysis reports are saved to [`llm_tests/eval_report.md`](file:///c:/Projects/data/FlightML/llm_tests/eval_report.md).
