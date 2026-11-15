from __future__ import annotations

import json
import random
from typing import Any

from flightml.corpus.schemas import ToolCallSample
from flightml.corpus.templates.base import BaseTemplate


FAST_MCP_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "flightml_get_pru_kpis",
            "description": "Compute ICAO GANP and Eurocontrol PRU performance indicators (KPI04 KEP, KPI05 KEA, KPI08 ASMA, KPI19) for a flight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ectrl_id": {"type": "integer", "description": "Eurocontrol flight identifier"},
                    "period": {"type": "string", "description": "Dataset period e.g. 202510 or auto", "default": "auto"},
                },
                "required": ["ectrl_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flightml_assess_passenger_trip",
            "description": "Assess public passenger travel delay risk, connection slack, missed flight probability, and EU261 statutory rights.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Origin 4-letter ICAO airport code"},
                    "dest": {"type": "string", "description": "Destination 4-letter ICAO airport code"},
                    "hub": {"type": "string", "description": "Intermediate transfer hub ICAO code (optional)"},
                    "scheduled_layover_min": {"type": "integer", "description": "Layover duration in minutes", "default": 75},
                    "preferred_operator": {"type": "string", "description": "Airline ICAO code e.g. BAW"},
                },
                "required": ["origin", "dest"],
            },
        },
    },
]


class ToolCallingTemplate(BaseTemplate):
    """Generates Track 6 samples: FastMCP Function Calling & Agent Execution Trajectories."""

    def __init__(self):
        super().__init__(track_name="tool_calling")

    def generate(self, rng: random.Random, context: dict[str, Any]) -> ToolCallSample:
        sample_type = rng.choice(["pru", "passenger"])

        if sample_type == "pru":
            ectrl_id = context.get("ectrl_id", 900104215)
            tool_call_id = f"call_{rng.randint(100000, 999999)}"
            user_msg = f"Can you check the PRU flight efficiency scorecard for flight ID {ectrl_id}?"

            tool_args = {"ectrl_id": ectrl_id, "period": "202510"}
            tool_resp = {
                "ectrl_id": ectrl_id,
                "adep": "EGLL",
                "ades": "LIRF",
                "kpi04_kep_pct": 4.18,
                "kpi05_kea_pct": 2.64,
                "tactical_delta_pct": -1.54,
                "verdict": "ATC_TACTICAL_SHORTCUT",
                "asma_additional_min": 3.2,
            }

            final_text = (
                f"Flight **#{ectrl_id}** from London Heathrow (EGLL) to Rome Fiumicino (LIRF) achieved an actual en-route "
                f"extension (**KPI05 KEA**) of **2.64%**, outperforming its filed plan (**KPI04 KEP: 4.18%**). "
                f"Tactical ATC vectoring shaved off 10.8 NM (`ATC_TACTICAL_SHORTCUT`). Terminal arrival transit included +3.2 min of ASMA holding delay."
            )

            messages = [
                {"role": "user", "content": user_msg},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": "flightml_get_pru_kpis", "arguments": json.dumps(tool_args)},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(tool_resp)},
                {"role": "assistant", "content": final_text},
            ]

        else:
            origin = context.get("origin", "EINN")
            hub = context.get("hub", "EGLL")
            dest = context.get("dest", "EBBR")
            tool_call_id = f"call_{rng.randint(100000, 999999)}"
            user_msg = f"I'm connecting at {hub} flying from {origin} to {dest} with a 75-minute layover. What's my risk?"

            tool_args = {"origin": origin, "dest": dest, "hub": hub, "scheduled_layover_min": 75}
            tool_resp = {
                "origin": origin,
                "hub": hub,
                "dest": dest,
                "scheduled_layover_min": 75,
                "effective_slack_min": 15.0,
                "missed_connection_prob": 0.142,
                "connection_risk_rating": "TIGHT",
                "statutory_compensation_eur": 250.0,
            }

            final_text = (
                f"For your journey from **{origin} to {dest} via {hub}**, your 75-minute layover leaves an **effective slack buffer of +15 minutes** "
                f"above Heathrow's 60-minute MCT. The probability of missing your connection is **14.2%** (`TIGHT`). "
                f"Note that Heathrow T5 biometric boarding gates close strictly 20 minutes before departure. Under EU261 / UK261, you are entitled to **€250.00 EUR** statutory compensation if delayed ≥ 3 hours at destination."
            )

            messages = [
                {"role": "user", "content": user_msg},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": "flightml_assess_passenger_trip", "arguments": json.dumps(tool_args)},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(tool_resp)},
                {"role": "assistant", "content": final_text},
            ]

        return ToolCallSample(
            messages=messages,
            tools=FAST_MCP_TOOL_DEFINITIONS,
            track=self.track_name,
            metadata={"sample_type": sample_type},
        )
