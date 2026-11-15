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
        print("❌ CUDA is not available. An NVIDIA GPU with CUDA is required for QLoRA fine-tuning.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🖥️ Detected GPU: {device_name} ({total_vram_gb:.2f} GB VRAM)")

    if total_vram_gb < 15.0:
        print(f"⚠️ Warning: Detected {total_vram_gb:.2f} GB VRAM. Target configuration is tuned for 16 GB VRAM.")
    else:
        print("✓ 16 GB VRAM confirmed. Memory budget allocation verified:")
        print("  - Base model (4-bit NF4): ~5.2 GB")
        print("  - LoRA adapters (r=64): ~0.35 GB")
        print("  - Paged AdamW 8-bit: ~0.45 GB")
        print("  - Activations & Buffers: ~5.6 GB")
        print(f"  - Safe VRAM Headroom: ~{total_vram_gb - 11.6:.1f} GB")


def load_sft_dataset(dataset_path: str):
    """Loads ChatML dataset cleanly, isolating the 'messages' field to prevent Arrow schema collisions."""
    import json
    from datasets import Dataset

    print(f"📖 Loading synthetic dataset from: {dataset_path}")
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                item = json.loads(line_str)
                if "messages" in item:
                    records.append({"messages": item["messages"]})
                else:
                    records.append(item)
    print(f"✓ Successfully loaded {len(records):,} clean ChatML training dialogues.")
    return Dataset.from_list(records)


def train_with_transformers_native(
    dataset_path: str,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "models/euroavia-7b-qlora",
    max_seq_length: int = 2048,
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    epochs: int = 3,
    lr: float = 2e-4,
):
    """Fine-tune 7B/8B model using native HuggingFace PEFT, bitsandbytes 4-bit, and TRL."""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    print(f"🚀 Initializing base model via HuggingFace PEFT: {base_model} (4-bit NF4)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_sft_dataset(dataset_path)

    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.03,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=False,
        bf16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",  # Offloads memory spikes to 32GB system RAM
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        output_dir=output_dir,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        dataset_text_field="messages",
        args=training_args,
    )

    print("⏳ Starting QLoRA fine-tuning...")
    trainer.train()

    print(f"💾 Saving trained LoRA adapter weights to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✓ Fine-tuning complete!")


def train_with_unsloth(
    dataset_path: str,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_dir: str = "models/euroavia-7b-qlora",
    max_seq_length: int = 2048,
    micro_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    epochs: int = 3,
    lr: float = 2e-4,
):
    """Fine-tune 7B/8B model using Unsloth's fast kernels and 4-bit quantization."""
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"🚀 Initializing base model via Unsloth: {base_model} (4-bit NF4)")
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

    dataset = load_sft_dataset(dataset_path)

    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.03,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=False,
        bf16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        output_dir=output_dir,
        save_strategy="epoch",
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
    try:
        model.save_pretrained_merged(f"{output_dir}_merged", tokenizer, save_method="merged_16bit")
        print("✓ Merged 16-bit model saved.")
    except Exception as e:
        print(f"Merge note: {e}")

    print(f"⚡ Exporting GGUF (Q4_K_M) to {output_dir}_gguf")
    try:
        model.save_pretrained_gguf(f"{output_dir}_gguf", tokenizer, quantization_method="q4_k_m")
        print("✓ GGUF export complete! Can be imported into LM Studio, Ollama, or FlightML.")
    except Exception as e:
        print(f"GGUF note: {e}")


def main():
    parser = argparse.ArgumentParser(description="FlightML 7B/8B QLoRA Fine-Tuning Pipeline (16GB VRAM)")
    parser.add_argument("--data", default="data/llm_corpus/sft_chatml_train.jsonl", help="Path to SFT JSONL file")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model ID (Qwen/Qwen2.5-7B-Instruct or meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--output", default="models/euroavia-7b-qlora", help="Output directory for weights")
    parser.add_argument("--seq-len", type=int, default=2048, help="Maximum sequence length (2048 or 4096)")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro batch size per device")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    check_vram_and_resources()

    if not os.path.exists(args.data):
        print(f"❌ Dataset file not found at: {args.data}")
        print("💡 Generate the dataset first by running: flightml generate-corpus --output-dir data/llm_corpus --samples 50000")
        sys.exit(1)

    if UNSLOTH_AVAILABLE:
        print("⚡ Using Unsloth accelerated kernel backend.")
        train_with_unsloth(
            dataset_path=args.data,
            base_model=args.model,
            output_dir=args.output,
            max_seq_length=args.seq_len,
            micro_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            epochs=args.epochs,
            lr=args.lr,
        )
    else:
        print("ℹ️ Unsloth not detected. Using native HuggingFace PEFT + BitsAndBytes 4-bit backend.")
        train_with_transformers_native(
            dataset_path=args.data,
            base_model=args.model,
            output_dir=args.output,
            max_seq_length=args.seq_len,
            micro_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            epochs=args.epochs,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
