from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flightml.corpus.schemas import DPOPair, SFTConversation, ToolCallSample


class CorpusExporter:
    """Streams structured training samples into train/validation JSONL datasets."""

    def __init__(self, output_dir: str | Path, train_split: float = 0.9):
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_sft(
        self,
        samples: list[SFTConversation],
        format_type: str = "chatml",
    ) -> dict[str, Any]:
        """Exports SFT samples to train/val JSONL files in ChatML or Alpaca format."""
        train_path = self.output_dir / f"sft_{format_type}_train.jsonl"
        val_path = self.output_dir / f"sft_{format_type}_val.jsonl"

        train_count = int(len(samples) * self.train_split)
        train_samples = samples[:train_count]
        val_samples = samples[train_count:]

        total_tokens_est = 0
        track_counts: dict[str, int] = {}

        with open(train_path, "w", encoding="utf-8") as f_train:
            for s in train_samples:
                record = s.to_chatml() if format_type == "chatml" else s.to_alpaca()
                raw_json = json.dumps(record, ensure_ascii=False)
                total_tokens_est += len(raw_json) // 4
                track_counts[s.track] = track_counts.get(s.track, 0) + 1
                f_train.write(raw_json + "\n")

        with open(val_path, "w", encoding="utf-8") as f_val:
            for s in val_samples:
                record = s.to_chatml() if format_type == "chatml" else s.to_alpaca()
                raw_json = json.dumps(record, ensure_ascii=False)
                total_tokens_est += len(raw_json) // 4
                f_val.write(raw_json + "\n")

        return {
            "format": format_type,
            "total_samples": len(samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "train_path": str(train_path),
            "val_path": str(val_path),
            "estimated_tokens": total_tokens_est,
            "track_distribution": track_counts,
        }

    def export_dpo(self, dpo_pairs: list[DPOPair]) -> dict[str, Any]:
        """Exports DPO preference pairs to train/val JSONL files."""
        train_path = self.output_dir / "dpo_train.jsonl"
        val_path = self.output_dir / "dpo_val.jsonl"

        train_count = int(len(dpo_pairs) * self.train_split)
        train_pairs = dpo_pairs[:train_count]
        val_pairs = dpo_pairs[train_count:]

        with open(train_path, "w", encoding="utf-8") as f_train:
            for p in train_pairs:
                f_train.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

        with open(val_path, "w", encoding="utf-8") as f_val:
            for p in val_pairs:
                f_val.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")

        return {
            "total_pairs": len(dpo_pairs),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "train_path": str(train_path),
            "val_path": str(val_path),
        }

    def export_tools(self, tool_samples: list[ToolCallSample]) -> dict[str, Any]:
        """Exports FastMCP function-calling execution trajectories."""
        path = self.output_dir / "tool_calling_samples.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in tool_samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

        return {
            "total_samples": len(tool_samples),
            "output_path": str(path),
        }
