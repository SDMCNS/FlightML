"""
Interactive Inference & Benchmark Evaluation Script for EuroAvia-7B QLoRA
Tuned for: Intel Core i7, 32 GB RAM, 16 GB VRAM GPU

Usage:
    # Run the automated 5-track benchmark suite comparing model outputs against ground truth:
    python scripts/test_inference.py --adapter models/euroavia-7b-qlora --benchmark

    # Launch interactive streaming chat in terminal:
    python scripts/test_inference.py --adapter models/euroavia-7b-qlora --interactive

    # Compare un-finetuned base model vs. QLoRA adapter:
    python scripts/test_inference.py --adapter models/euroavia-7b-qlora --compare-base
"""

from __future__ import annotations

import argparse
import sys
import torch
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextStreamer,
    )
    from peft import PeftModel
except ImportError:
    print("❌ Required packages missing. Please run:")
    print("   pip install transformers peft bitsandbytes accelerate")
    sys.exit(1)

console = Console()

SYSTEM_PROMPT = (
    "You are EuroAvia-7B, an authoritative expert on European air traffic management, "
    "ICAO GANP performance indicators, Eurocontrol PRU metrics, 4D flight trajectory physics, "
    "and airline economics. Provide rigorous, step-by-step mathematical reasoning and cite exact figures in Euros (€ EUR)."
)

BENCHMARK_SCENARIOS = [
    {
        "track": "Track 1: Trajectory Physics & PRU Scorecards",
        "prompt": (
            "Analyze the en-route horizontal and vertical flight efficiency for synthetic flight 900104215 "
            "(SYN A320, EGLL to LIRF, Great-Circle Distance: 780.4 NM). Filed en-route distance >40nm is 731.2 NM, "
            "and actual radar en-route distance >40nm is 720.4 NM. Compute Eurocontrol PRU KPI04 (KEP) and KPI05 (KEA), "
            "explain tactical ATC intervention, and evaluate terminal holding."
        ),
        "expected_ground_truth": (
            "• Reference en-route distance (H): 780.4 - 80 = 700.4 NM\n"
            "• KPI04 KEP (Filed extension): ((731.2 - 700.4) / 700.4) * 100 = 4.40% (or ~4.18% depending on terminal radius)\n"
            "• KPI05 KEA (Actual extension): ((720.4 - 700.4) / 700.4) * 100 = 2.86% (or ~2.64%)\n"
            "• Tactical Delta (KEA - KEP): Negative value (~-1.54%), classified as ATC_TACTICAL_SHORTCUT\n"
            "• Terminal Transit (KPI08 ASMA): Displays additional terminal holding delay above unimpeded reference"
        ),
    },
    {
        "track": "Track 2: Fuel Burn & Eurocontrol CRCO Navigation Fees (€ EUR)",
        "prompt": (
            "Calculate the Eurocontrol CRCO en-route navigation charges and direct fuel operating costs in Euros (€ EUR) "
            "for flight 900104215 (A320, MTOW 78.0 tonnes) operating from EGLL to LIRF across 1,445 km. "
            "Show the exact Service Unit formula and derive fuel cost at €0.6421/kg for 4,620 kg of Jet-A."
        ),
        "expected_ground_truth": (
            "• Aircraft Weight Factor: sqrt(MTOW / 50) = sqrt(78.0 / 50) = 1.2490\n"
            "• Total Service Units: (1,445 km / 100) * 1.2490 = 18.05 SU\n"
            "• En-Route Fee: ~€1,584.20 EUR (based on blended bilateral unit rates)\n"
            "• Jet-A Fuel Cost: 4,620 kg * €0.6421/kg = €2,966.50 EUR\n"
            "• Total Direct Operating Cost (DOC): €2,966.50 + €1,584.20 = €4,550.70 EUR (Currency strictly in Euros € EUR)"
        ),
    },
    {
        "track": "Track 3: Rotational Delay Contagion & Tail Tracking",
        "prompt": (
            "Explain how rotational delay contagion propagates along airframe EI-SYNA (SYN A320) if leg 1 arrives at EGLL "
            "with a 65-minute delay. The scheduled turnaround buffer is 45 minutes, and minimum required turn time is 35 minutes. "
            "Quantify downstream reactionary departure delay on leg 2 (EGLL to LEMD) and explain the countervailing slot relief dynamic."
        ),
        "expected_ground_truth": (
            "• Turnaround Slack (S): 45 min scheduled - 35 min minimum = 10 minutes buffer\n"
            "• Reactionary Delay: max(0, 65 min - 10 min) = 55 minutes departure delay on leg 2\n"
            "• Tactical Reactionary Cost: 55 min * €88.00/min = €4,840.00 EUR\n"
            "• Countervailing Slot Relief: Cancelled or delayed downstream return legs vacate departure and arrival slots, "
            "accelerating hub backlog clearance for remaining traffic"
        ),
    },
    {
        "track": "Track 4: 4D Airspace Closure Counterfactual",
        "prompt": (
            "Simulate a temporary 3-hour tactical closure of Swiss airspace (LSASFIR). "
            "What is the macro-level impact on European traffic reroutes, detour nautical miles, additional Jet-A fuel burn, "
            "airline operating losses in Euros (€ EUR), and post-reopening recovery queue duration?"
        ),
        "expected_ground_truth": (
            "• Disrupted Flights: ~500 to 800 flights scheduled through Swiss airspace\n"
            "• Tactical Reroute Compliance: ~88% of flights rerouted laterally around the Alps via France or Italy\n"
            "• Extra Detour: +40 to +65 NM average per diverted flight\n"
            "• Extra Fuel Burn: +100 to +160 metric tonnes of Jet-A (+300 to +500 tonnes CO2)\n"
            "• Airline Economic Losses: In millions of Euros (€ EUR)\n"
            "• Post-Reopening Queue: Non-linear recovery backlog requiring 2.5 to 4.0 hours to normalize"
        ),
    },
    {
        "track": "Track 5: Passenger Travel Copilot & Statutory Rights (EU261 / UK261)",
        "prompt": (
            "I am flying from Shannon (EINN) to Brussels (EBBR) via London Heathrow (EGLL) on a Tuesday morning with British Airways. "
            "I have a 75-minute scheduled layover at Heathrow T5. Assess my risk of missing the connection, evaluate the gate conformance "
            "rules at Heathrow, and explain my statutory compensation rights under EU261 / UK261 in Euros (€ EUR)."
        ),
        "expected_ground_truth": (
            "• Nominal MCT at Heathrow: 60 minutes\n"
            "• Effective Connection Slack: 75 min - 60 min = +15 minutes (Tight connection, ~14% missed flight risk)\n"
            "• Heathrow T5 Biometric Conformance Rule: Gate closes STRICTLY 20 minutes before scheduled off-blocks. "
            "Outbound departure taxi queues will NOT hold the gate open!\n"
            "• EU261 / UK261 Distance Tier: Short-haul (912 km <= 1,500 km)\n"
            "• Statutory Compensation: €250.00 EUR per passenger if delayed >= 3 hours at destination\n"
            "• Duty of Care: Complimentary meals, refreshments, and communications after 2 hours delay"
        ),
    },
]


def load_model_and_adapter(base_model_id: str, adapter_path: str | None, load_base_only: bool = False):
    """Loads the base model in 4-bit NF4 and attaches the QLoRA adapter."""
    console.print(f"[bold cyan]📦 Loading base model in 4-bit NF4:[/bold cyan] [yellow]{base_model_id}[/yellow]")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if load_base_only or adapter_path is None:
        console.print("[green]✓ Base model loaded without adapters.[/green]")
        return base_model, tokenizer

    console.print(f"[bold cyan]🔧 Attaching QLoRA adapter:[/bold cyan] [yellow]{adapter_path}[/yellow]")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    console.print("[bold green]✓ QLoRA adapter successfully attached and active![/bold green]\n")
    return model, tokenizer


def generate_response(model, tokenizer, user_prompt: str, stream: bool = True) -> str:
    """Generates an answer from the model using ChatML formatting."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,  # Low temperature for analytical precision
            top_p=0.9,
            repetition_penalty=1.05,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return full_output


def run_benchmarks(model, tokenizer):
    """Executes the automated 5-track European aviation benchmark evaluation."""
    console.print(Panel(
        "[bold cyan]🏆 European Aviation 7B QLoRA Benchmark Evaluation[/bold cyan]\n"
        "Evaluating model across 5 core ATM physics, economics, delay, and consumer rights tracks.",
        expand=False,
    ))

    for idx, test in enumerate(BENCHMARK_SCENARIOS, 1):
        console.print(f"\n[bold magenta]════════════════════════════════════════════════════════════════════[/bold magenta]")
        console.print(f"[bold yellow]Test #{idx}: {test['track']}[/bold yellow]")
        console.print(f"[bold cyan]User Prompt:[/bold cyan] {test['prompt']}\n")

        console.print("[bold green]Expected Ground Truth (FlightML Deterministic Engine):[/bold green]")
        console.print(Panel(test["expected_ground_truth"], style="green"))

        console.print("[bold cyan]Model Generation (EuroAvia-7B QLoRA):[/bold cyan]")
        _ = generate_response(model, tokenizer, test["prompt"], stream=True)


def run_interactive(model, tokenizer):
    """Runs an interactive chat loop in the terminal."""
    console.print(Panel(
        "[bold green]💬 EuroAvia-7B Interactive Aviation Copilot[/bold green]\n"
        "Ask questions about European flight plans, PRU KPIs, fuel burn, CRCO fees (€ EUR), "
        "delay contagion, or passenger rights. Type [bold red]'exit'[/bold red] to quit.",
        expand=False,
    ))

    while True:
        try:
            user_input = console.input("\n[bold yellow]User ✈️ > [/bold yellow]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[cyan]Exiting EuroAvia-7B session. Goodbye![/cyan]")
                break

            console.print("\n[bold cyan]EuroAvia-7B 🧠 >[/bold cyan]")
            _ = generate_response(model, tokenizer, user_input, stream=True)
        except KeyboardInterrupt:
            console.print("\n[cyan]Session interrupted. Goodbye![/cyan]")
            break


def run_comparison(base_model_id: str, adapter_path: str):
    """Side-by-side comparison between un-finetuned base model and QLoRA adapter."""
    console.print(Panel(
        "[bold yellow]⚖️ Side-by-Side Model Comparison[/bold yellow]\n"
        "Comparing Base Model (un-finetuned) vs. EuroAvia-7B (QLoRA fine-tuned).",
        expand=False,
    ))

    test_prompt = BENCHMARK_SCENARIOS[1]["prompt"]  # CRCO & Fuel Operating Cost prompt

    console.print(f"[bold cyan]Test Prompt:[/bold cyan]\n{test_prompt}\n")

    console.print("[bold red]1. Un-Finetuned Base Model (Expect Hallucinations & Non-EUR currency):[/bold red]")
    base_model, tokenizer = load_model_and_adapter(base_model_id, adapter_path=None, load_base_only=True)
    _ = generate_response(base_model, tokenizer, test_prompt, stream=True)

    del base_model
    torch.cuda.empty_cache()

    console.print("\n[bold green]2. EuroAvia-7B QLoRA Adapted Model (Expect Exact Math & € EUR Compliance):[/bold green]")
    fine_tuned_model, _ = load_model_and_adapter(base_model_id, adapter_path=adapter_path)
    _ = generate_response(fine_tuned_model, tokenizer, test_prompt, stream=True)


def main():
    parser = argparse.ArgumentParser(description="EuroAvia-7B QLoRA Inference & Evaluation Engine")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base HuggingFace model ID")
    parser.add_argument("--adapter", default="models/euroavia-7b-qlora", help="Path to trained LoRA adapter directory")
    parser.add_argument("--benchmark", action="store_true", help="Run automated 5-track benchmark suite against ground truth")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat session in terminal")
    parser.add_argument("--compare-base", action="store_true", help="Compare un-adapted base model vs. QLoRA adapter")

    args = parser.parse_args()

    if args.compare_base:
        run_comparison(args.model, args.adapter)
        return

    model, tokenizer = load_model_and_adapter(args.model, args.adapter)

    if args.interactive:
        run_interactive(model, tokenizer)
    else:
        # Default to benchmark evaluation
        run_benchmarks(model, tokenizer)


if __name__ == "__main__":
    main()
