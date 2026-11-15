from __future__ import annotations

import random
from typing import Any

from flightml.corpus.schemas import SFTConversation, SFTMessage
from flightml.corpus.templates.base import BaseTemplate


class PRUPhysicsTemplate(BaseTemplate):
    """Generates Track 1 samples: 4D Trajectory & ICAO GANP / Eurocontrol PRU KPIs."""

    def __init__(self):
        super().__init__(track_name="pru_physics")

    def generate(self, rng: random.Random, context: dict[str, Any]) -> SFTConversation | None:
        pru = context.get("pru")

        ectrl_id = getattr(pru, "ectrl_id", context.get("ectrl_id", 900104215))
        adep = getattr(pru, "adep", context.get("adep", "EGLL"))
        ades = getattr(pru, "ades", context.get("ades", "LIRF"))
        ac_type = getattr(pru, "aircraft_type", context.get("ac_type", "A320"))
        ac_operator = getattr(pru, "ac_operator", context.get("ac_operator", "SYN"))
        ac_reg = getattr(pru, "ac_reg", context.get("ac_reg", "G-SYNA"))

        gc_dist = getattr(pru, "great_circle_distance_nm", context.get("gc_dist", 780.4))
        enroute_gc = getattr(pru, "enroute_gcd_nm", max(gc_dist - 80.0, 10.0))
        filed_enroute = getattr(pru, "enroute_filed_distance_nm", enroute_gc * 1.042)
        actual_enroute = getattr(pru, "enroute_actual_distance_nm", enroute_gc * 1.026)

        kep_pct = getattr(pru, "kpi04_kep_pct", ((filed_enroute - enroute_gc) / enroute_gc) * 100.0)
        kea_pct = getattr(pru, "kpi05_kea_pct", ((actual_enroute - enroute_gc) / enroute_gc) * 100.0)
        tactical_delta = getattr(pru, "tactical_delta_pct", kea_pct - kep_pct)
        verdict = getattr(pru, "enroute_routing_verdict", "ATC_TACTICAL_SHORTCUT" if tactical_delta < 0 else "ATC_TACTICAL_EXTENSION")

        asma_time = getattr(pru, "asma_40nm_actual_min", 18.5)
        asma_add = getattr(pru, "kpi08_asma_40nm_additional_min", 3.2)
        req_fl = getattr(pru, "requested_fl", 370.0)
        act_fl = getattr(pru, "max_actual_fl", 350.0)
        is_capped = getattr(pru, "is_level_capped", act_fl < req_fl - 5.0)

        # Question variations
        prompts = [
            f"Analyze the en-route horizontal and vertical flight efficiency for flight {ectrl_id} ({ac_operator} {ac_type}, {adep} to {ades}). Compute Eurocontrol PRU KPI04 (KEP) and KPI05 (KEA), explain tactical ATC intervention, and evaluate whether level capping occurred.",
            f"Given radar trajectory records for {ac_operator} flight {ectrl_id} ({adep} ➔ {ades}, {ac_type}), perform a detailed ICAO GANP and Eurocontrol PRU assessment. Show step-by-step calculations for lateral extension and vertical holding.",
            f"Evaluate the flight efficiency scorecard of {adep}-{ades} flight {ectrl_id}. What was the difference between filed en-route distance and actual flown distance, and did tactical air traffic control optimize or extend the route?",
        ]
        user_prompt = rng.choice(prompts)

        # Chain of Thought reasoning block
        cot = (
            f"<thought>\n"
            f"1. Flight Identification:\n"
            f"   - ECTRL ID: {ectrl_id} ({ac_operator} {ac_type}, registration {ac_reg})\n"
            f"   - City Pair: {adep} ➔ {ades} (Great-Circle Distance H = {gc_dist:.1f} NM)\n"
            f"   - En-Route Terminal Boundary (40 NM buffer deducted from each aerodrome): H_enroute = {enroute_gc:.1f} NM\n"
            f"2. Horizontal En-Route Flight Efficiency Calculations:\n"
            f"   - Filed En-Route Distance (L): {filed_enroute:.1f} NM\n"
            f"   - KPI04 (KEP): ((L - H_enroute) / H_enroute) * 100 = (({filed_enroute:.1f} - {enroute_gc:.1f}) / {enroute_gc:.1f}) * 100 = {kep_pct:.2f}%\n"
            f"   - Actual Flown Radar En-Route Distance (A): {actual_enroute:.1f} NM\n"
            f"   - KPI05 (KEA): ((A - H_enroute) / H_enroute) * 100 = (({actual_enroute:.1f} - {enroute_gc:.1f}) / {enroute_gc:.1f}) * 100 = {kea_pct:.2f}%\n"
            f"3. Tactical ATC Attribution:\n"
            f"   - Delta = KEA - KEP = {kea_pct:.2f}% - {kep_pct:.2f}% = {tactical_delta:+.2f}%\n"
            f"   - Distance Delta: {actual_enroute - filed_enroute:+.1f} NM\n"
            f"   - Operational Verdict: {verdict}\n"
            f"4. Terminal & Vertical Efficiency:\n"
            f"   - KPI08 ASMA (40 NM Terminal Inbound): {asma_time:.1f} min (+{asma_add:.1f} min additional delay holding vs unimpeded baseline)\n"
            f"   - KPI19 Cruise Level: Requested FL{req_fl:.0f} vs Max Actual FL{act_fl:.0f} (Capped: {is_capped})\n"
            f"</thought>\n"
        )

        response_body = (
            f"### Flight Efficiency & Eurocontrol PRU Scorecard: Flight #{ectrl_id}\n\n"
            f"**Route Overview:** {adep} ➔ {ades} | **Operator:** {ac_operator} | **Aircraft:** {ac_type} (`{ac_reg}`)\n"
            f"**Great-Circle Distance:** {gc_dist:.1f} NM (En-Route Reference >40 NM: {enroute_gc:.1f} NM)\n\n"
            f"#### 1. Horizontal En-Route Efficiency (ICAO GANP)\n"
            f"- **KPI04 KEP (Filed Flight Plan Extension)**: **{kep_pct:.2f}%**\n"
            f"  - Filed En-Route Distance: {filed_enroute:.1f} NM (+{filed_enroute - enroute_gc:.1f} NM above reference)\n"
            f"- **KPI05 KEA (Actual Radar Flown Extension)**: **{kea_pct:.2f}%**\n"
            f"  - Actual En-Route Distance: {actual_enroute:.1f} NM (+{actual_enroute - enroute_gc:.1f} NM above reference)\n"
            f"- **Tactical ATC Attribution (KEA − KEP)**: **{tactical_delta:+.2f}%** ({actual_enroute - filed_enroute:+.1f} NM)\n"
            f"  - **Verdict**: `{verdict}` — tactical ATC radar vectoring {'provided direct shortcuts shaving off mileage' if tactical_delta < 0 else 'extended the route due to sector congestion or weather circumnavigation'}.\n\n"
            f"#### 2. Terminal Inbound Transit & Vertical Profile\n"
            f"- **KPI08 ASMA (40 NM Terminal Arrival Transit)**: **{asma_time:.1f} minutes**\n"
            f"  - Additional Holding / Sequencing Delay: **+{asma_add:.1f} minutes** beyond unimpeded reference transit.\n"
            f"- **KPI19 Cruise Level-Capping**: "
            f"{f'**Level-Capped at FL{act_fl:.0f}** (filed for FL{req_fl:.0f}, incurring cruise aerodynamic drag penalty)' if is_capped else f'**Optimal cruise level FL{act_fl:.0f} achieved** without en-route level capping.'}"
        )

        assistant_msg = cot + response_body

        return SFTConversation(
            messages=[
                SFTMessage(role="system", content=self.get_system_prompt(rng)),
                SFTMessage(role="user", content=user_prompt),
                SFTMessage(role="assistant", content=assistant_msg),
            ],
            track=self.track_name,
            metadata={
                "ectrl_id": ectrl_id,
                "adep": adep,
                "ades": ades,
                "ac_type": ac_type,
                "kep_pct": kep_pct,
                "kea_pct": kea_pct,
            },
        )
