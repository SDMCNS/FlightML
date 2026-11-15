from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class SFTMessage(BaseModel):
    """Single turn in a multi-turn conversational dataset."""
    role: Literal["system", "user", "assistant"]
    content: str


class SFTConversation(BaseModel):
    """Complete conversational episode for Supervised Fine-Tuning (SFT)."""
    messages: list[SFTMessage] = Field(default_factory=list)
    track: str = "general"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_chatml(self) -> dict[str, Any]:
        """Export to standard OpenAI / ChatML dictionary format."""
        return {
            "messages": [msg.model_dump() for msg in self.messages],
            "track": self.track,
        }

    def to_alpaca(self) -> dict[str, Any]:
        """Export to Alpaca instruction/input/output format."""
        system_prompt = ""
        user_prompt = ""
        assistant_resp = ""

        for msg in self.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_prompt = msg.content
            elif msg.role == "assistant":
                assistant_resp = msg.content

        return {
            "instruction": system_prompt,
            "input": user_prompt,
            "output": assistant_resp,
            "track": self.track,
        }


class DPOPair(BaseModel):
    """Pairwise comparison sample for Direct Preference Optimization (DPO)."""
    system: str = ""
    prompt: str
    chosen: str
    rejected: str
    track: str = "general"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Export to standard HuggingFace TRL / DPOTrainer dictionary."""
        return {
            "prompt": f"<|system|>\n{self.system}\n<|user|>\n{self.prompt}",
            "chosen": self.chosen,
            "rejected": self.rejected,
            "track": self.track,
            "metadata": self.metadata,
        }


class ToolCallSample(BaseModel):
    """Function-calling agent training sample matching FastMCP / OpenAI tool schemas."""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    track: str = "tool_calling"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "tools": self.tools,
            "track": self.track,
            "metadata": self.metadata,
        }
