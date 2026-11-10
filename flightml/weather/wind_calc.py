import math
from dataclasses import dataclass
from typing import Any


@dataclass
class RunwayWindComponent:
    runway_ident: str
    runway_heading_deg: float
    wind_dir_deg: float
    wind_speed_kts: float
    gust_kts: float | None
    crosswind_kts: float
    crosswind_side: str  # 'LEFT' or 'RIGHT'
    headwind_kts: float
    is_tailwind: bool
    tailwind_kts: float
    crosswind_limit_exceeded: bool
    tailwind_limit_exceeded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "runway_ident": self.runway_ident,
            "runway_heading_deg": self.runway_heading_deg,
            "wind_dir_deg": self.wind_dir_deg,
            "wind_speed_kts": self.wind_speed_kts,
            "gust_kts": self.gust_kts,
            "crosswind_kts": round(self.crosswind_kts, 1),
            "crosswind_side": self.crosswind_side,
            "headwind_kts": round(self.headwind_kts, 1),
            "is_tailwind": self.is_tailwind,
            "tailwind_kts": round(self.tailwind_kts, 1),
            "crosswind_limit_exceeded": self.crosswind_limit_exceeded,
            "tailwind_limit_exceeded": self.tailwind_limit_exceeded,
        }


def compute_runway_wind(
    runway_ident: str,
    runway_heading_deg: float,
    wind_dir_deg: float,
    wind_speed_kts: float,
    gust_kts: float | None = None,
    max_crosswind_kts: float = 30.0,
    max_tailwind_kts: float = 10.0,
) -> RunwayWindComponent:
    """
    Decompose wind vector relative to a specific runway heading.
    """
    effective_speed = gust_kts if gust_kts and gust_kts > wind_speed_kts else wind_speed_kts

    # Relative wind angle
    angle_diff = (wind_dir_deg - runway_heading_deg) % 360
    angle_rad = math.radians(angle_diff)

    crosswind_raw = effective_speed * math.sin(angle_rad)
    crosswind = abs(crosswind_raw)
    side = "RIGHT" if crosswind_raw > 0 else "LEFT"

    headwind_raw = effective_speed * math.cos(angle_rad)
    is_tailwind = headwind_raw < 0
    tailwind = abs(headwind_raw) if is_tailwind else 0.0
    headwind = headwind_raw if not is_tailwind else 0.0

    return RunwayWindComponent(
        runway_ident=runway_ident,
        runway_heading_deg=runway_heading_deg,
        wind_dir_deg=wind_dir_deg,
        wind_speed_kts=wind_speed_kts,
        gust_kts=gust_kts,
        crosswind_kts=crosswind,
        crosswind_side=side,
        headwind_kts=headwind,
        is_tailwind=is_tailwind,
        tailwind_kts=tailwind,
        crosswind_limit_exceeded=crosswind > max_crosswind_kts,
        tailwind_limit_exceeded=tailwind > max_tailwind_kts,
    )


def find_optimal_runways(
    runways: list[dict[str, Any]],
    wind_dir_deg: float,
    wind_speed_kts: float,
    gust_kts: float | None = None,
    max_crosswind_kts: float = 30.0,
    max_tailwind_kts: float = 10.0,
) -> list[RunwayWindComponent]:
    """
    Evaluate all runway ends (le_ident and he_ident) and rank them by minimum crosswind and headwind preference.
    """
    candidates: list[RunwayWindComponent] = []

    for r in runways:
        # Low-end runway (e.g. 09L)
        if r.get("le_ident") and r.get("le_heading_degT") is not None:
            c = compute_runway_wind(
                runway_ident=str(r["le_ident"]),
                runway_heading_deg=float(r["le_heading_degT"]),
                wind_dir_deg=wind_dir_deg,
                wind_speed_kts=wind_speed_kts,
                gust_kts=gust_kts,
                max_crosswind_kts=max_crosswind_kts,
                max_tailwind_kts=max_tailwind_kts,
            )
            candidates.append(c)

        # High-end runway reciprocal (e.g. 27R)
        if r.get("he_ident") and r.get("he_heading_degT") is not None:
            c = compute_runway_wind(
                runway_ident=str(r["he_ident"]),
                runway_heading_deg=float(r["he_heading_degT"]),
                wind_dir_deg=wind_dir_deg,
                wind_speed_kts=wind_speed_kts,
                gust_kts=gust_kts,
                max_crosswind_kts=max_crosswind_kts,
                max_tailwind_kts=max_tailwind_kts,
            )
            candidates.append(c)

    # Sort: prefer valid winds, prefer headwinds over tailwinds, then minimize crosswind
    candidates.sort(
        key=lambda c: (
            c.tailwind_limit_exceeded or c.crosswind_limit_exceeded,
            c.is_tailwind,
            c.tailwind_kts,
            c.crosswind_kts,
            -c.headwind_kts,
        )
    )
    return candidates

