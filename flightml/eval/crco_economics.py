"""
Eurocontrol Central Route Charges Office (CRCO) Airspace Economics & Fuel Trade-Off Engine (Sprint 4).

Implements the official Eurocontrol CRCO En-Route Charge formula:
    Weight Factor (WF) = sqrt(MTOW_metric_tons / 50.0)
    Distance Factor (DF_i) = d_i_km / 100.0
    Service Units (SU_i) = DF_i * WF
    Route Charge (EUR_i) = Unit_Rate_EUR_i * SU_i

Combines CRCO navigation charges with Jet-A fuel burn cost (via weekly EIA spot prices)
to evaluate Total Route Direct Operating Cost (Fuel USD + Navigation Charges USD) and detect
Route-Charge Avoidance detours where airlines fly longer distances to bypass high-unit-rate FIRs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.eval.fuel_carbon import compute_flight_fuel_and_emissions
from flightml.geo.geodesy import haversine_nm

NM_TO_KM = 1.852


@dataclass
class FIRChargeBreakdown:
    fir_id: str
    zone_code: str
    country_name: str
    dwell_min: float
    estimated_distance_nm: float
    estimated_distance_km: float
    distance_factor: float
    weight_factor: float
    service_units: float
    unit_rate_eur: float
    charge_eur: float


@dataclass
class CRCOChargeReport:
    ectrl_id: int
    period: str
    adep: str
    ades: str
    ac_operator: str
    ac_type: str
    mtow_tons: float
    crco_weight_factor: float
    actual_distance_nm: float
    actual_distance_km: float
    total_service_units: float
    actual_crco_charge_eur: float
    filed_crco_charge_eur: float
    crco_charge_delta_eur: float
    estimated_total_fuel_kg: float
    estimated_total_fuel_cost_eur: float
    excess_fuel_cost_eur: float
    total_direct_route_cost_eur: float
    crco_share_of_direct_cost_pct: float
    most_expensive_fir: str | None
    fir_breakdown: list[FIRChargeBreakdown]
    # Backwards-compatible aliases (mapped to EUR)
    actual_crco_charge_usd: float = 0.0
    filed_crco_charge_usd: float = 0.0
    crco_charge_delta_usd: float = 0.0
    estimated_total_fuel_cost_usd: float = 0.0
    excess_fuel_cost_usd: float = 0.0
    total_direct_route_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in (
            "actual_crco_charge_usd",
            "filed_crco_charge_usd",
            "crco_charge_delta_usd",
            "estimated_total_fuel_cost_usd",
            "excess_fuel_cost_usd",
            "total_direct_route_cost_usd",
        ):
            d.pop(k, None)
        return d


def compute_flight_crco_charges(
    dataset: FlightDataset,
    ectrl_id: int,
) -> CRCOChargeReport | None:
    """
    Compute Eurocontrol CRCO En-Route Navigation Charges (in EUR by FIR Charging Zone) and combine
    with Jet-A fuel cost (in EUR) to quantify Total Direct Route Cost (Fuel EUR + CRCO Charges EUR).
    """
    flight = dataset.get_flight(ectrl_id)
    if not flight:
        return None

    adep = str(flight.get("adep") or "")
    ades = str(flight.get("ades") or "")
    ac_op = str(flight.get("ac_operator") or "")
    ac_type = str(flight.get("ac_type") or "B738")
    act_dist_nm = float(flight.get("actual_distance_nm") or 0.0)

    trajs = dataset.get_trajectory(ectrl_id)
    actual_df = trajs["actual"]
    if act_dist_nm <= 1.0 and len(actual_df) >= 2:
        act_dist_nm = haversine_nm(
            float(actual_df.iloc[0]["latitude"]),
            float(actual_df.iloc[0]["longitude"]),
            float(actual_df.iloc[-1]["latitude"]),
            float(actual_df.iloc[-1]["longitude"]),
        )
    act_dist_nm = max(20.0, act_dist_nm)
    act_dist_km = act_dist_nm * NM_TO_KM

    # Resolve aircraft MTOW and Eurocontrol Weight Factor sqrt(MTOW_tons / 50)
    ac_perf = dataset.get_aircraft_performance(ac_type)
    if ac_perf and ac_perf.get("mtow_kg"):
        mtow_tons = float(ac_perf["mtow_kg"]) / 1000.0
        wf = float(ac_perf.get("crco_weight_factor") or math.sqrt(mtow_tons / 50.0))
    else:
        mtow_tons = 79.0
        wf = math.sqrt(mtow_tons / 50.0)

    # Compute sector breakdown from `flight_firs_actual` if available, else infer from origin/destination ICAO prefixes
    fir_rows: list[dict[str, Any]] = []
    filed_fir_rows: list[dict[str, Any]] = []

    if dataset.has_fir_traversals():
        f_df = dataset.get_flight_fir_traversals(ectrl_id)
        if not f_df.empty:
            # Aggregate by FIR charging zone (merging lower FIR and upper UIR of same center in sequence or summing dwell)
            for _, r in f_df.iterrows():
                fid = str(r["fir_id"])
                if fid == "TAXI_OUT":
                    continue
                act_d = max(1.0, float(r["actual_dwell_min"] or 0.0))
                fld_d = float(r["filed_dwell_min"]) if pd.notna(r["filed_dwell_min"]) else act_d
                fir_rows.append({"fir_id": fid, "dwell_min": act_d})
                filed_fir_rows.append({"fir_id": fid, "dwell_min": max(1.0, fld_d)})

    if not fir_rows:
        # Fallback for 201611 or flights without explicit flight_firs_actual:
        total_air = max(15.0, float(flight.get("actual_air_time_min") or (act_dist_nm / 420.0 * 60.0)))
        f1 = f"{adep[:2]}XXFIR" if len(adep) >= 2 else "EGTTFIR"
        f2 = f"{ades[:2]}XXFIR" if len(ades) >= 2 else "LFFFFIR"
        if f1 == f2:
            fir_rows = [{"fir_id": f1, "dwell_min": total_air}]
        else:
            fir_rows = [
                {"fir_id": f1, "dwell_min": total_air * 0.5},
                {"fir_id": f2, "dwell_min": total_air * 0.5},
            ]
        filed_fir_rows = list(fir_rows)

    total_dwell = max(1.0, sum(x["dwell_min"] for x in fir_rows))
    breakdown: list[FIRChargeBreakdown] = []
    total_su = 0.0
    total_eur = 0.0

    for item in fir_rows:
        fid = item["fir_id"]
        d_min = item["dwell_min"]
        share = d_min / total_dwell
        seg_nm = act_dist_nm * share
        seg_km = seg_nm * NM_TO_KM
        df_factor = seg_km / 100.0
        su = df_factor * wf

        rate_info = dataset.get_crco_unit_rate(fid) or {
            "zone_code": fid[:2],
            "country_name": f"Non-CRCO / External ({fid[:2]})",
            "unit_rate_eur": 45.0,
        }
        u_eur = float(rate_info.get("unit_rate_eur") or rate_info.get("total_unit_rate_eur") or 45.0)
        c_eur = su * u_eur

        total_su += su
        total_eur += c_eur

        breakdown.append(
            FIRChargeBreakdown(
                fir_id=fid,
                zone_code=str(rate_info.get("zone_code") or fid[:2]),
                country_name=str(rate_info.get("country_name") or rate_info.get("zone_name") or fid),
                dwell_min=round(d_min, 1),
                estimated_distance_nm=round(seg_nm, 1),
                estimated_distance_km=round(seg_km, 1),
                distance_factor=round(df_factor, 3),
                weight_factor=round(wf, 3),
                service_units=round(su, 3),
                unit_rate_eur=round(u_eur, 2),
                charge_eur=round(c_eur, 2),
            )
        )

    # Estimate Filed CRCO charge
    total_fld_dwell = max(1.0, sum(x["dwell_min"] for x in filed_fir_rows))
    filed_eur = 0.0
    for item in filed_fir_rows:
        fid = item["fir_id"]
        share = item["dwell_min"] / total_fld_dwell
        # Scale distance by filed vs actual dwell ratio
        fld_dist_km = act_dist_km * (total_fld_dwell / total_dwell) * share
        su_f = (fld_dist_km / 100.0) * wf
        r_info = dataset.get_crco_unit_rate(fid)
        u_eur = float(r_info.get("unit_rate_eur") or 45.0) if r_info else 45.0
        filed_eur += su_f * u_eur

    # Combine with flight fuel & emissions model (in EUR)
    fuel_impact = compute_flight_fuel_and_emissions(dataset, ectrl_id)
    air_hours = max(0.25, float(flight.get("actual_air_time_min") or (act_dist_nm / 430.0 * 60.0)) / 60.0)
    wake = str(ac_perf.get("wake_category") or "M") if ac_perf else "M"
    nom_ff_kg_hr = 6800.0 if wake == "H" else 11500.0 if wake == "J" else 1200.0 if wake == "L" else 2500.0
    est_total_fuel_kg = air_hours * nom_ff_kg_hr + fuel_impact.total_excess_fuel_kg
    est_total_fuel_eur = est_total_fuel_kg * fuel_impact.fuel_price_eur_per_kg

    total_doc_eur = est_total_fuel_eur + total_eur
    crco_share_pct = (total_eur / max(1.0, total_doc_eur)) * 100.0

    most_exp = max(breakdown, key=lambda b: b.charge_eur).fir_id if breakdown else None
    delta_eur = round(total_eur - filed_eur, 2)

    return CRCOChargeReport(
        ectrl_id=int(ectrl_id),
        period=dataset.period,
        adep=adep,
        ades=ades,
        ac_operator=ac_op,
        ac_type=ac_type,
        mtow_tons=round(mtow_tons, 1),
        crco_weight_factor=round(wf, 3),
        actual_distance_nm=round(act_dist_nm, 1),
        actual_distance_km=round(act_dist_km, 1),
        total_service_units=round(total_su, 2),
        actual_crco_charge_eur=round(total_eur, 2),
        filed_crco_charge_eur=round(filed_eur, 2),
        crco_charge_delta_eur=delta_eur,
        estimated_total_fuel_kg=round(est_total_fuel_kg, 1),
        estimated_total_fuel_cost_eur=round(est_total_fuel_eur, 2),
        excess_fuel_cost_eur=round(fuel_impact.estimated_excess_fuel_cost_eur, 2),
        total_direct_route_cost_eur=round(total_doc_eur, 2),
        crco_share_of_direct_cost_pct=round(crco_share_pct, 1),
        most_expensive_fir=most_exp,
        fir_breakdown=breakdown,
        actual_crco_charge_usd=round(total_eur, 2),
        filed_crco_charge_usd=round(filed_eur, 2),
        crco_charge_delta_usd=delta_eur,
        estimated_total_fuel_cost_usd=round(est_total_fuel_eur, 2),
        excess_fuel_cost_usd=round(fuel_impact.estimated_excess_fuel_cost_eur, 2),
        total_direct_route_cost_usd=round(total_doc_eur, 2),
    )


def detect_route_charge_avoidance(
    dataset: FlightDataset,
    adep: str,
    ades: str,
    ac_type: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """
    Compare flights on a city pair (`adep` -> `ades`) to evaluate the trade-off between
    Eurocontrol CRCO route charges and Jet-A fuel burn, flagging flights that flew longer
    distances through lower-unit-rate FIRs to reduce navigation fees at the expense of higher CO2.
    """
    where_clauses = ["adep = ?", "ades = ?"]
    params: list[Any] = [adep.strip().upper(), ades.strip().upper()]
    if ac_type and ac_type.strip():
        where_clauses.append("ac_type = ?")
        params.append(ac_type.strip().upper())

    sql = f"""
        SELECT ectrl_id, ac_operator, ac_type, actual_distance_nm
        FROM flights
        WHERE {" AND ".join(where_clauses)}
        ORDER BY actual_distance_nm ASC
        LIMIT {int(limit)};
    """
    df = dataset.cursor.execute(sql, params).df()
    if df.empty:
        return {
            "period": dataset.period,
            "corridor": f"{adep.upper()} ➔ {ades.upper()}",
            "analyzed_flights": 0,
            "avoidance_candidates": [],
            "flights": [],
        }

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        fid = int(row["ectrl_id"])
        rep = compute_flight_crco_charges(dataset, fid)
        if rep:
            records.append(
                {
                    "ectrl_id": rep.ectrl_id,
                    "ac_operator": rep.ac_operator,
                    "ac_type": rep.ac_type,
                    "actual_distance_nm": rep.actual_distance_nm,
                    "crco_charge_eur": rep.actual_crco_charge_eur,
                    "fuel_cost_eur": rep.estimated_total_fuel_cost_eur,
                    "total_direct_cost_eur": rep.total_direct_route_cost_eur,
                    "estimated_fuel_kg": rep.estimated_total_fuel_kg,
                    "most_expensive_fir": rep.most_expensive_fir,
                }
            )

    if not records:
        return {
            "period": dataset.period,
            "corridor": f"{adep.upper()} ➔ {ades.upper()}",
            "analyzed_flights": 0,
            "avoidance_candidates": [],
            "flights": [],
        }

    shortest = min(records, key=lambda x: x["actual_distance_nm"])
    avoidance_cases: list[dict[str, Any]] = []
    for rec in records:
        dist_extra_nm = rec["actual_distance_nm"] - shortest["actual_distance_nm"]
        crco_saved_eur = shortest["crco_charge_eur"] - rec["crco_charge_eur"]
        extra_fuel_kg = max(0.0, rec["estimated_fuel_kg"] - shortest["estimated_fuel_kg"])
        if dist_extra_nm >= 10.0 and crco_saved_eur > 5.0:
            avoidance_cases.append(
                {
                    **rec,
                    "extra_distance_nm": round(dist_extra_nm, 1),
                    "crco_fee_saved_eur": round(crco_saved_eur, 2),
                    "extra_fuel_burned_kg": round(extra_fuel_kg, 1),
                    "extra_co2_emitted_kg": round(extra_fuel_kg * 3.16, 1),
                }
            )

    return {
        "period": dataset.period,
        "corridor": f"{adep.upper()} ➔ {ades.upper()}",
        "analyzed_flights": len(records),
        "shortest_route_distance_nm": shortest["actual_distance_nm"],
        "shortest_route_crco_eur": shortest["crco_charge_eur"],
        "avoidance_candidates_count": len(avoidance_cases),
        "avoidance_candidates": avoidance_cases,
        "flights": records,
    }

