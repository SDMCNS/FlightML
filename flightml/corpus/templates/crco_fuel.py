from __future__ import annotations

import math
import random
from typing import Any

from flightml.corpus.schemas import SFTConversation, SFTMessage
from flightml.corpus.templates.base import BaseTemplate


class CRCOFuelTemplate(BaseTemplate):
    """Generates Track 2 samples: Fuel Burn, Emissions & Eurocontrol CRCO Navigation Charges (€ EUR)."""

    def __init__(self):
        super().__init__(track_name="crco_fuel")

    def generate(self, rng: random.Random, context: dict[str, Any]) -> SFTConversation | None:
        impact = context.get("fuel_impact")
        crco = context.get("crco")

        ectrl_id = getattr(impact, "ectrl_id", getattr(crco, "ectrl_id", 900104215))
        adep = getattr(impact, "adep", getattr(crco, "adep", "EGLL"))
        ades = getattr(impact, "ades", getattr(crco, "ades", "LIRF"))
        ac_type = getattr(impact, "aircraft_type", getattr(crco, "aircraft_type", "A320"))
        mtow_t = getattr(crco, "mtow_tonnes", 78.0)
        weight_factor = math.sqrt(mtow_t / 50.0)

        total_fuel_kg = getattr(impact, "estimated_total_fuel_kg", 4620.0)
        excess_fuel_kg = getattr(impact, "total_excess_fuel_kg", 222.4)
        jet_a_eur_per_kg = getattr(impact, "jet_a_price_eur_per_kg", 0.6421)
        total_fuel_cost_eur = total_fuel_kg * jet_a_eur_per_kg
        excess_fuel_cost_eur = excess_fuel_kg * jet_a_eur_per_kg

        co2_emissions_kg = getattr(impact, "estimated_co2_kg", total_fuel_kg * 3.16)
        nox_emissions_kg = getattr(impact, "estimated_nox_kg", total_fuel_kg * 0.0135)

        total_distance_km = getattr(crco, "actual_distance_km", 1445.0)
        total_su = getattr(crco, "actual_total_service_units", (total_distance_km / 100.0) * weight_factor)
        actual_crco_eur = getattr(crco, "actual_crco_charge_eur", total_su * 87.80)
        filed_crco_eur = getattr(crco, "filed_crco_charge_eur", actual_crco_eur * 1.018)
        crco_delta_eur = actual_crco_eur - filed_crco_eur

        total_direct_cost_eur = total_fuel_cost_eur + actual_crco_eur
        crco_cost_share_pct = (actual_crco_eur / total_direct_cost_eur) * 100.0

        prompts = [
            f"Calculate the Eurocontrol CRCO en-route navigation charges and direct fuel operating costs in Euros (€ EUR) for flight {ectrl_id} ({ac_type}, MTOW {mtow_t:.1f} tonnes) operating from {adep} to {ades}. Show the exact Service Unit formula.",
            f"Derive the flight operating economics for {ac_type} flight {ectrl_id} ({adep} ➔ {ades}). What are the fuel burn, ICAO engine emissions, and Eurocontrol CRCO navigation fees in Euros, and how does the aircraft's weight factor influence the route charge?",
            f"Perform an airline direct operating cost (DOC) breakdown for {adep}-{ades} flight {ectrl_id}. Compute fuel expenditure at €{jet_a_eur_per_kg:.4f}/kg, excess fuel penalty, and CRCO FIR charging units.",
        ]
        user_prompt = rng.choice(prompts)

        cot = (
            f"<thought>\n"
            f"1. Flight & Airframe Parameters:\n"
            f"   - Aircraft: {ac_type} (MTOW = {mtow_t:.1f} metric tonnes)\n"
            f"   - Route: {adep} ➔ {ades} (Actual Distance Flown = {total_distance_km:,.1f} km)\n"
            f"   - Jet-A Spot Price: €{jet_a_eur_per_kg:.4f} EUR/kg\n"
            f"2. Eurocontrol CRCO Service Unit (SU) Derivation:\n"
            f"   - Aircraft Weight Factor w = sqrt(MTOW / 50) = sqrt({mtow_t:.1f} / 50) = {weight_factor:.4f}\n"
            f"   - Distance Factor d = {total_distance_km:,.1f} km / 100 = {total_distance_km / 100.0:.2f}\n"
            f"   - Total Service Units: SU = d * w = {total_su:.2f} SU\n"
            f"3. Navigation Charges:\n"
            f"   - Actual CRCO Fee: €{actual_crco_eur:,.2f} EUR (Filed: €{filed_crco_eur:,.2f} EUR, Delta: €{crco_delta_eur:+,.2f} EUR)\n"
            f"4. Fuel Burn & Environmental Emissions (OpenAP & ICAO EDB v32B):\n"
            f"   - Total Fuel Burn: {total_fuel_kg:,.1f} kg -> €{total_fuel_cost_eur:,.2f} EUR\n"
            f"   - Detour & Holding Excess Fuel: {excess_fuel_kg:,.1f} kg -> €{excess_fuel_cost_eur:,.2f} EUR\n"
            f"   - CO2 Factor (3.16 kg CO2/kg fuel): {co2_emissions_kg:,.1f} kg CO2\n"
            f"   - LTO NOx Emissions: {nox_emissions_kg:,.1f} kg NOx\n"
            f"5. Total Direct Operating Cost (DOC) Allocation:\n"
            f"   - Total Route Cost = Fuel (€{total_fuel_cost_eur:,.2f}) + CRCO (€{actual_crco_eur:,.2f}) = €{total_direct_cost_eur:,.2f} EUR\n"
            f"   - CRCO Navigation Share: {crco_cost_share_pct:.1f}%\n"
            f"</thought>\n"
        )

        response_body = (
            f"### Flight Operating Economics & CRCO Cost Derivation: Flight #{ectrl_id}\n\n"
            f"**City Pair:** {adep} ➔ {ades} | **Aircraft:** {ac_type} (MTOW: {mtow_t:.1f} tonnes)\n"
            f"**Distance Traversed:** {total_distance_km:,.1f} km | **Jet-A Fuel Price:** €{jet_a_eur_per_kg:.4f} EUR/kg\n\n"
            f"#### 1. Eurocontrol CRCO Navigation Charges Derivation\n"
            f"Eurocontrol route charges are governed by the bilateral multilateral charging formula based on distance and aircraft weight factor:\n"
            f"$$\\text{{Weight Factor}} = \\sqrt{{\\frac{{\\text{{MTOW}}}}{{50}}}} = \\sqrt{{\\frac{{{mtow_t:.1f}}}{{50}}}} = {weight_factor:.4f}$$\n"
            f"$$\\text{{Total Service Units (SU)}} = \\left(\\frac{{{total_distance_km:,.1f}\\text{{ km}}}}{{100}}\\right) \\times {weight_factor:.4f} = {total_su:.2f}\\text{{ SU}}$$\n\n"
            f"- **Actual CRCO Navigation Charge**: **€{actual_crco_eur:,.2f} EUR**\n"
            f"- **Filed Plan CRCO Charge**: €{filed_crco_eur:,.2f} EUR (Savings via tactical en-route shortcuts: **€{abs(crco_delta_eur):,.2f} EUR**)\n\n"
            f"#### 2. OpenAP Fuel Consumption & Emissions\n"
            f"- **Total Jet-A Fuel Burned**: **{total_fuel_kg:,.1f} kg** (Cost: **€{total_fuel_cost_eur:,.2f} EUR**)\n"
            f"- **Excess Fuel (Detours & Holding)**: **+{excess_fuel_kg:,.1f} kg** (Cost penalty: **€{excess_fuel_cost_eur:,.2f} EUR**)\n"
            f"- **Greenhouse Gas Emissions**: **{co2_emissions_kg:,.1f} kg $\\text{{CO}}_2$** | **{nox_emissions_kg:,.1f} kg $\\text{{NO}}_x$**\n\n"
            f"#### 3. Direct Operating Cost (DOC) Synthesis\n"
            f"- **Total Direct Route Operating Cost**: **€{total_direct_cost_eur:,.2f} EUR**\n"
            f"  - Jet-A Fuel Share: **{100.0 - crco_cost_share_pct:.1f}%** (€{total_fuel_cost_eur:,.2f} EUR)\n"
            f"  - Eurocontrol Airspace Navigation Share: **{crco_cost_share_pct:.1f}%** (€{actual_crco_eur:,.2f} EUR)"
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
                "total_fuel_kg": total_fuel_kg,
                "actual_crco_eur": actual_crco_eur,
                "total_direct_cost_eur": total_direct_cost_eur,
            },
        )
