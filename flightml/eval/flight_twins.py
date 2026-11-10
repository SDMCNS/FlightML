from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import folium
import pandas as pd

from flightml.data.dataset import FlightDataset
from flightml.geo.trajectory import analyze_trajectory
from flightml.geo.geodesy import haversine_distance_nm
from flightml.eval.fuel_carbon import DEFAULT_FUEL_FLOW_KG_HR, CO2_FACTOR
from flightml.eval.jet_fuel_prices import get_weekly_fuel_price


@dataclass
class TwinFlightComparison:
    adep: str
    ades: str
    ac_type: str
    fast_id: int
    fast_operator: str
    fast_air_time_min: float
    fast_actual_distance_nm: float
    fast_extension_ratio: float
    fast_fuel_kg: float
    slow_id: int
    slow_operator: str
    slow_air_time_min: float
    slow_actual_distance_nm: float
    slow_extension_ratio: float
    slow_fuel_kg: float
    distance_penalty_nm: float
    air_time_penalty_min: float
    excess_fuel_burned_kg: float
    excess_co2_kg: float
    excess_fuel_cost_usd: float
    divergence_lat: float | None
    divergence_lon: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": f"{self.adep} ➔ {self.ades}",
            "ac_type": self.ac_type,
            "efficient_flight": {
                "ectrl_id": self.fast_id,
                "operator": self.fast_operator,
                "air_time_min": round(self.fast_air_time_min, 1),
                "distance_nm": round(self.fast_actual_distance_nm, 1),
                "extension_ratio": round(self.fast_extension_ratio, 3),
                "fuel_burned_kg": round(self.fast_fuel_kg, 1),
            },
            "inefficient_flight": {
                "ectrl_id": self.slow_id,
                "operator": self.slow_operator,
                "air_time_min": round(self.slow_air_time_min, 1),
                "distance_nm": round(self.slow_actual_distance_nm, 1),
                "extension_ratio": round(self.slow_extension_ratio, 3),
                "fuel_burned_kg": round(self.slow_fuel_kg, 1),
            },
            "disparity_deltas": {
                "extra_distance_nm": round(self.distance_penalty_nm, 1),
                "extra_air_time_min": round(self.air_time_penalty_min, 1),
                "excess_fuel_burned_kg": round(self.excess_fuel_burned_kg, 1),
                "excess_co2_kg": round(self.excess_co2_kg, 1),
                "excess_co2_metric_tons": round(self.excess_co2_kg / 1000.0, 2),
                "excess_cost_usd": round(self.excess_fuel_cost_usd, 2),
            },
            "divergence_point": {
                "lat": round(self.divergence_lat, 3) if self.divergence_lat else None,
                "lon": round(self.divergence_lon, 3) if self.divergence_lon else None,
            },
        }


def compare_twin_flights(
    dataset: FlightDataset,
    adep: str = "RKSI",
    ades: str = "EGLL",
    ac_type: str = "B77W",
    fuel_price_per_kg: float | None = None,
) -> TwinFlightComparison | None:
    """
    Compare and contrast the most and least efficient flights of identical complexity
    (same origin, destination, and aircraft type).
    Automatically resolves the EIA weekly jet fuel spot price for the dataset period.
    """
    if fuel_price_per_kg is None:
        fuel_price_per_kg = get_weekly_fuel_price(None, period_hint=dataset.period).price_usd_per_kg
    adep = adep.upper()
    ades = ades.upper()
    ac_type = ac_type.upper()

    query = """
        SELECT
            ectrl_id,
            ac_operator,
            actual_distance_nm,
            actual_air_time_min
        FROM flights
        WHERE adep = ? AND ades = ? AND ac_type = ?
          AND actual_distance_nm IS NOT NULL
          AND actual_air_time_min > 30
        ORDER BY actual_distance_nm ASC;
    """
    df = dataset.query(query, [adep, ades, ac_type])
    if len(df) < 2:
        return None

    fast_row = df.iloc[0]
    slow_row = df.iloc[-1]

    fast_id = int(fast_row["ectrl_id"])
    slow_id = int(slow_row["ectrl_id"])

    t_fast = dataset.get_trajectory(fast_id)
    t_slow = dataset.get_trajectory(slow_id)

    m_fast = analyze_trajectory(t_fast["actual"], t_fast["planned"])
    m_slow = analyze_trajectory(t_slow["actual"], t_slow["planned"])

    fuel_flow = DEFAULT_FUEL_FLOW_KG_HR.get(ac_type, DEFAULT_FUEL_FLOW_KG_HR["DEFAULT_JET"])
    fuel_fast = (m_fast.actual_distance_nm / 450.0) * fuel_flow
    fuel_slow = (m_slow.actual_distance_nm / 450.0) * fuel_flow

    excess_fuel = max(0.0, fuel_slow - fuel_fast)
    excess_co2 = excess_fuel * CO2_FACTOR
    excess_cost = excess_fuel * fuel_price_per_kg

    # Find geographic point of trajectory divergence
    div_lat, div_lon = None, None
    if not t_fast["actual"].empty and not t_slow["actual"].empty:
        pts_fast = t_fast["actual"][["latitude", "longitude"]].values
        pts_slow = t_slow["actual"][["latitude", "longitude"]].values
        min_pts = min(len(pts_fast), len(pts_slow))

        for k in range(min_pts):
            d = haversine_distance_nm(
                pts_fast[k][0], pts_fast[k][1], pts_slow[k][0], pts_slow[k][1]
            )
            if d > 100.0:  # Trajectories diverged by more than 100 nm
                div_lat = float(pts_fast[k][0])
                div_lon = float(pts_fast[k][1])
                break

    return TwinFlightComparison(
        adep=adep,
        ades=ades,
        ac_type=ac_type,
        fast_id=fast_id,
        fast_operator=str(fast_row["ac_operator"]),
        fast_air_time_min=float(fast_row["actual_air_time_min"]),
        fast_actual_distance_nm=m_fast.actual_distance_nm,
        fast_extension_ratio=m_fast.horizontal_extension_ratio,
        fast_fuel_kg=fuel_fast,
        slow_id=slow_id,
        slow_operator=str(slow_row["ac_operator"]),
        slow_air_time_min=float(slow_row["actual_air_time_min"]),
        slow_actual_distance_nm=m_slow.actual_distance_nm,
        slow_extension_ratio=m_slow.horizontal_extension_ratio,
        slow_fuel_kg=fuel_slow,
        distance_penalty_nm=m_slow.actual_distance_nm - m_fast.actual_distance_nm,
        air_time_penalty_min=float(slow_row["actual_air_time_min"]) - float(fast_row["actual_air_time_min"]),
        excess_fuel_burned_kg=excess_fuel,
        excess_co2_kg=excess_co2,
        excess_fuel_cost_usd=excess_cost,
        divergence_lat=div_lat,
        divergence_lon=div_lon,
    )


def render_twin_comparison_map(
    dataset: FlightDataset,
    comparison: TwinFlightComparison,
    output_html_path: Path | str | None = None,
) -> folium.Map:
    """
    Render an interactive dual-trajectory comparison map on unblocked open basemaps:
    - High-Efficiency Flight: Emerald Green
    - Low-Efficiency Flight: Crimson Red
    - Divergence Marker: Orange pin
    """
    t_fast = dataset.get_trajectory(comparison.fast_id)
    t_slow = dataset.get_trajectory(comparison.slow_id)

    actual_fast = t_fast["actual"]
    actual_slow = t_slow["actual"]

    mid_lat = float(actual_fast["latitude"].mean()) if not actual_fast.empty else 50.0
    mid_lon = float(actual_fast["longitude"].mean()) if not actual_fast.empty else 10.0

    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=3, tiles=None)

    # Add unblocked Humanitarian OSM France tile layer
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors, HOT style hosted by OSM France",
        name="Humanitarian (Unblocked)",
        subdomains="abc",
        show=True,
    ).add_to(m)

    # 1. High-Efficiency Flight Line (Emerald Green)
    if not actual_fast.empty:
        fast_coords = list(zip(actual_fast["latitude"], actual_fast["longitude"]))
        folium.PolyLine(
            fast_coords,
            color="#27ae60",
            weight=4,
            opacity=0.9,
            tooltip=f"High Efficiency: Flight {comparison.fast_id} ({comparison.fast_actual_distance_nm:.0f} nm)",
        ).add_to(m)

    # 2. Low-Efficiency Flight Line (Crimson Red)
    if not actual_slow.empty:
        slow_coords = list(zip(actual_slow["latitude"], actual_slow["longitude"]))
        folium.PolyLine(
            slow_coords,
            color="#e74c3c",
            weight=4,
            opacity=0.9,
            tooltip=f"Low Efficiency: Flight {comparison.slow_id} ({comparison.slow_actual_distance_nm:.0f} nm, +{comparison.distance_penalty_nm:.0f} nm)",
        ).add_to(m)

    # 3. Departure & Destination Markers
    if not actual_fast.empty:
        folium.Marker(
            [actual_fast.iloc[0]["latitude"], actual_fast.iloc[0]["longitude"]],
            tooltip=f"Origin: {comparison.adep}",
            icon=folium.Icon(color="green", icon="plane-departure", prefix="fa"),
        ).add_to(m)
        folium.Marker(
            [actual_fast.iloc[-1]["latitude"], actual_fast.iloc[-1]["longitude"]],
            tooltip=f"Destination: {comparison.ades}",
            icon=folium.Icon(color="red", icon="plane-arrival", prefix="fa"),
        ).add_to(m)

    # 4. Divergence Point Marker
    if comparison.divergence_lat and comparison.divergence_lon:
        folium.Marker(
            [comparison.divergence_lat, comparison.divergence_lon],
            tooltip="Major Route Divergence Point (>100 nm separation)",
            icon=folium.Icon(color="orange", icon="code-branch", prefix="fa"),
        ).add_to(m)

    # Add Info Box
    box_html = f"""
    <div style="position: fixed; bottom: 25px; left: 15px; width: 360px; height: 140px;
                background-color: white; border: 2px solid #bdc3c7; border-radius: 8px;
                z-index: 9999; font-size: 13px; font-family: sans-serif;
                padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.2);">
        <b style="font-size: 15px; color: #2c3e50;">✈️ Twin-Flight Efficiency Comparison</b><br>
        <b>Route:</b> {comparison.adep} &rarr; {comparison.ades} ({comparison.ac_type})<br>
        <span style="color: #27ae60;">■ Flight {comparison.fast_id}: {comparison.fast_actual_distance_nm:.0f} nm ({comparison.fast_air_time_min:.0f} min)</span><br>
        <span style="color: #e74c3c;">■ Flight {comparison.slow_id}: {comparison.slow_actual_distance_nm:.0f} nm ({comparison.slow_air_time_min:.0f} min)</span><br>
        <b>Disparity:</b> +{comparison.distance_penalty_nm:.0f} nm detour | +{comparison.air_time_penalty_min:.0f} min<br>
        <b>Excess Fuel & CO₂:</b> +{comparison.excess_fuel_burned_kg:,.0f} kg fuel (+{comparison.excess_co2_kg/1000.0:.1f}t CO₂)
    </div>
    """
    m.get_root().html.add_child(folium.Element(box_html))

    if output_html_path:
        out_p = Path(output_html_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(out_p))

    return m

