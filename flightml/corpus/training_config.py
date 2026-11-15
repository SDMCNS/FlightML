from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class HardwareProfile:
    """Hardware specifications for resource balancing."""
    device_name: str = "NVIDIA GeForce RTX 5060 Ti / 4060 Ti (16GB)"
    vram_gb: float = 16.0
    system_ram_gb: float = 32.0
    cpu_cores: int = 8
    target_utilization_pct: float = 85.0  # Safe headroom to prevent CUDA OOM


@dataclass
class QLoRAHyperparameters:
    """Hardware-balanced QLoRA hyperparameter configuration for 16 GB VRAM."""
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"  # or "meta-llama/Llama-3.1-8B-Instruct"
    quantization_bits: int = 4
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA Adapter Configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Batching & Memory Trade-offs for 16GB VRAM
    max_seq_length: int = 2048  # Expandable to 4096 with gradient checkpointing
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch size = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01

    # Stability & Memory Optimizations
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    optimizer: str = "paged_adamw_8bit"  # Pages memory to 32GB system RAM during spikes
    fp16: bool = False
    bf16: bool = True  # Native bfloat16 on Ampere, Ada Lovelace, and Blackwell

    # Logging & Checkpoints
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    output_dir: str = "models/euroavia-7b-qlora"

    def estimate_vram_footprint_gb(self) -> dict[str, float]:
        """Calculates expected VRAM allocation across model components."""
        # Base 7B model in 4-bit: ~7.0B params * 0.5 bytes = ~3.5 GB + ~1.5 GB overhead = ~5.0 GB
        model_weights_gb = 5.2
        # LoRA adapters (r=64 on all linear layers): ~0.35 GB
        lora_weights_gb = 0.35
        # 8-bit AdamW optimizer states for LoRA weights: ~0.45 GB
        optimizer_states_gb = 0.45
        # Activation memory per token under gradient checkpointing (batch=2, seq=2048): ~3.8 GB
        activations_gb = (self.max_seq_length / 2048.0) * (self.micro_batch_size / 2.0) * 3.8
        # KV Cache & temporary CUDA buffers: ~1.8 GB
        cuda_overhead_gb = 1.8

        total_estimated = (
            model_weights_gb
            + lora_weights_gb
            + optimizer_states_gb
            + activations_gb
            + cuda_overhead_gb
        )

        return {
            "model_weights_gb": model_weights_gb,
            "lora_weights_gb": lora_weights_gb,
            "optimizer_states_gb": optimizer_states_gb,
            "activations_gb": activations_gb,
            "cuda_overhead_gb": cuda_overhead_gb,
            "total_estimated_vram_gb": total_estimated,
            "available_headroom_gb": 16.0 - total_estimated,
        }

    def generate_training_script_snippet(self, dataset_path: str = "data/llm_corpus/sft_train.jsonl") -> str:
        """Generates a standalone Python fine-tuning script snippet using Unsloth or TRL."""
        return f'''# Auto-generated European Aviation 7B QLoRA Trainer
# Optimized for 16GB VRAM (RTX 4060 Ti / 5060 Ti) + 32GB System RAM

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Load 4-bit base model with fast kernels
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{self.base_model}",
    max_seq_length={self.max_seq_length},
    dtype=torch.bfloat16,
    load_in_4bit={self.quantization_bits == 4},
)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r={self.lora_r},
    target_modules={self.target_modules},
    lora_alpha={self.lora_alpha},
    lora_dropout={self.lora_dropout},
    bias="none",
    use_gradient_checkpointing="{self.gradient_checkpointing}",
)

# 3. Load FlightML Synthetic Corpus
dataset = load_dataset("json", data_files="{dataset_path}", split="train")

# 4. Configure Training Arguments balanced for 16GB VRAM
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="messages",
    max_seq_length={self.max_seq_length},
    args=TrainingArguments(
        per_device_train_batch_size={self.micro_batch_size},
        gradient_accumulation_steps={self.gradient_accumulation_steps},
        warmup_ratio={self.warmup_ratio},
        learning_rate={self.learning_rate},
        fp16={self.fp16},
        bf16={self.bf16},
        logging_steps={self.logging_steps},
        optim="{self.optimizer}",
        weight_decay={self.weight_decay},
        lr_scheduler_type="{self.lr_scheduler_type}",
        output_dir="{self.output_dir}",
    ),
)

# 5. Train and Export
trainer.train()
model.save_pretrained_merged("{self.output_dir}_merged", tokenizer, save_method="merged_16bit")
model.save_pretrained_gguf("{self.output_dir}_gguf", tokenizer, quantization_method="q4_k_m")
'''
