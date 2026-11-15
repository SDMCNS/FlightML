"""
European Aviation 7B/8B QLoRA Fine-Tuning Pipeline
Tuned for: Intel Core i7, 32 GB RAM, 16 GB VRAM (RTX 4060 Ti / 5060 Ti)

Usage:
    python scripts/train_qlora.py --data data/llm_corpus/sft_chatml_train.jsonl --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import os
import sys
import torch

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False


def check_vram_and_resources():
    """Verify GPU capabilities and available VRAM headroom."""
    if not torch.cuda.is_available():
        print("⚠️ CUDA is not available. GPU is required for QLoRA training.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🖥️ Detected GPU: {device_name} ({total_vram_gb:.2f} GB VRAM)")

    if total_vram_gb < 15.0:
        print(f"⚠️ Warning: Detected {total_vram_gb:.2f} GB VRAM. 16 GB VRAM recommended for 7B/8B QLoRA.")
    else:
        print("✓ VRAM profile matches 16 GB target. Safe memory headroom confirmed.")


def train_with_unsloth(
    dataset_path: str,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "models/euroavia-7b-qlora",
    max_seq_length: int = 2048,
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    epochs: int = 3,
):
    """Fine-tune 7B/8B model using Unsloth's fast kernels and 4-bit quantization."""
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"🚀 Initializing base model: {base_model} (4-bit NF4)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    print("🔧 Attaching LoRA adapters (r=64, alpha=128 on all linear projection layers)")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print(f"📖 Loading synthetic dataset: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.03,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",  # Pages memory to 32GB system RAM if momentary spikes occur
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        output_dir=output_dir,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="messages",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    print("⏳ Starting QLoRA fine-tuning run...")
    trainer.train()

    print(f"💾 Saving trained LoRA weights to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"📦 Exporting merged 16-bit standalone model to {output_dir}_merged")
    model.save_pretrained_merged(f"{output_dir}_merged", tokenizer, save_method="merged_16bit")

    print(f"⚡ Exporting GGUF (Q4_K_M) for local LM Studio / FastMCP execution to {output_dir}_gguf")
    try:
        model.save_pretrained_gguf(f"{output_dir}_gguf", tokenizer, quantization_method="q4_k_m")
        print("✓ GGUF export complete! Can be loaded directly into LM Studio, Ollama, or FlightML.")
    except Exception as e:
        print(f"Note: GGUF export requires llama.cpp installed. Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="FlightML 7B/8B QLoRA Fine-Tuning Pipeline (16GB VRAM)")
    parser.add_argument("--data", default="data/llm_corpus/sft_chatml_train.jsonl", help="Path to SFT JSONL file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base HuggingFace model ID")
    parser.add_argument("--output", default="models/euroavia-7b-qlora", help="Output directory for weights")
    parser.add_argument("--seq-len", type=int, default=2048, help="Maximum sequence length (2048 or 4096)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")

    args = parser.parse_args()

    check_vram_and_resources()

    if not os.path.exists(args.data):
        print(f"❌ Dataset file not found at: {args.data}")
        print("💡 Generate one first by running: flightml generate-corpus --output-dir data/llm_corpus --samples 5000")
        sys.exit(1)

    if UNSLOTH_AVAILABLE:
        train_with_unsloth(
            dataset_path=args.data,
            base_model=args.model,
            output_dir=args.output,
            max_seq_length=args.seq_len,
            epochs=args.epochs,
        )
    else:
        print("Unsloth is recommended for 2x faster training and 70% less memory.")
        print("Install via: pip install unsloth")


if __name__ == "__main__":
    main()
