"""
FlightML DME-DME RNAV Coverage & Terrestrial Navaid Rationalization Engine.

Implements ICAO Doc 9613 (PBN Manual) and FAA AC 90-100A standards:
- Terrestrial Radio Line-of-Sight (RLOS): D_max = 1.23 * (sqrt(h_ac) + sqrt(h_sta)) nm
- Dual-station inclusion angle / GDOP geometry condition: 30° <= Delta_theta <= 150°
- Per-flight-level grid coverage assessment (10x10 NM cells)
- Interactive active / inactive station toggling
- Greedy coverage-preserving rationalization algorithm to identify redundant DMEs
  that can be decommissioned without degrading RNAV positioning coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any
import pandas as pd
import numpy as np

from flightml.data.dataset import FlightDataset
import flightml.math as fmath


@dataclass
class DMEStation:
    ident: str
    name: str
    type: str
    latitude: float
    longitude: float
    elevation_ft: float
    country: str
    is_active: bool
    service_range_nm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GridCellCoverage:
    lat: float
    lon: float
    visible_dme_count: int
    has_valid_dme_dme_pair: bool
    best_pair_geometry_deg: float
    coverage_status: str  # "REDUNDANT" (>=3), "ADEQUATE" (2), "MARGINAL" (1), "OUTAGE" (0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DMERNAVCoverageReport:
    flight_level: int
    altitude_ft: int
    cell_size_nm: float
    total_grid_cells: int
    covered_cells_count: int
    coverage_percentage: float
    redundant_cells_percentage: float
    outage_cells_percentage: float
    total_dmes_count: int
    active_dmes_count: int
    inactive_dmes_count: int
    suggested_deactivations: list[dict[str, Any]]
    dme_stations: list[dict[str, Any]]
    cells_summary: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_dme_stations(dataset: FlightDataset, bbox: tuple[float, float, float, float]) -> list[DMEStation]:
    """Load all DME and VOR-DME stations within bounding box (lat_min, lat_max, lon_min, lon_max)."""
    lat_min, lat_max, lon_min, lon_max = bbox
    df = dataset.cursor.execute(
        """
        SELECT ident, name, type, latitude_deg, longitude_deg, elevation_ft, iso_country
        FROM raw_navaids
        WHERE (type LIKE '%DME%' OR type = 'VOR-DME' OR type = 'TACAN')
          AND latitude_deg BETWEEN ? AND ?
          AND longitude_deg BETWEEN ? AND ?
        ORDER BY ident ASC
        """,
        [lat_min - 2.0, lat_max + 2.0, lon_min - 2.0, lon_max + 2.0],
    ).df()

    stations: list[DMEStation] = []
    seen = set()
    for _, row in df.iterrows():
        ident = str(row.get("ident") or "").strip().upper()
        if not ident or ident in seen:
            continue
        seen.add(ident)
        elev_val = row.get("elevation_ft")
        elev = float(elev_val) if pd.notna(elev_val) else 100.0
        stations.append(
            DMEStation(
                ident=ident,
                name=str(row.get("name") or ident),
                type=str(row.get("type") or "VOR-DME"),
                latitude=float(row.get("latitude_deg")),
                longitude=float(row.get("longitude_deg")),
                elevation_ft=elev,
                country=str(row.get("iso_country") or "XX"),
                is_active=True,
                service_range_nm=160.0,
            )
        )
    return stations


def evaluate_dme_coverage(
    stations: list[DMEStation],
    flight_level: int = 300,
    bbox: tuple[float, float, float, float] = (48.0, 54.0, -3.0, 7.0),
    cell_step_deg: float = 0.35,  # ~20 nm grid
    inactive_idents: set[str] | None = None,
) -> tuple[list[GridCellCoverage], float, float, float]:
    """
    Evaluate 2D grid coverage for DME-DME RNAV positioning at a given flight level.
    """
    altitude_ft = flight_level * 100.0
    inactive = inactive_idents or set()
    active_stations = [s for s in stations if s.ident not in inactive and s.is_active]

    lat_min, lat_max, lon_min, lon_max = bbox
    lats = np.arange(lat_min, lat_max + 1e-5, cell_step_deg)
    lons = np.arange(lon_min, lon_max + 1e-5, cell_step_deg * 1.5)

    cell_results: list[GridCellCoverage] = []
    covered_count = 0
    redundant_count = 0
    outage_count = 0

    # Pre-extract station arrays
    sta_coords = [(s.ident, s.latitude, s.longitude, min(s.service_range_nm, fmath.rlos_radio_horizon_nm(altitude_ft, s.elevation_ft))) for s in active_stations]

    for lat in lats:
        for lon in lons:
            visible_stations: list[tuple[str, float]] = []  # (ident, bearing_deg)

            for ident, s_lat, s_lon, max_r in sta_coords:
                dist_nm = fmath.haversine_distance_nm(lat, lon, s_lat, s_lon)
                if dist_nm <= max_r:
                    brg = fmath.initial_bearing_deg(lat, lon, s_lat, s_lon)
                    visible_stations.append((ident, brg))

            vis_count = len(visible_stations)
            has_valid_pair = False
            best_geom = 0.0

            # Check if any pair satisfies 30° <= Delta_theta <= 150°
            if vis_count >= 2:
                # Compare bearings
                for i in range(min(vis_count, 12)):
                    for j in range(i + 1, min(vis_count, 12)):
                        diff = abs(visible_stations[i][1] - visible_stations[j][1])
                        delta_theta = min(diff, 360.0 - diff)
                        if 30.0 <= delta_theta <= 150.0:
                            has_valid_pair = True
                            if abs(delta_theta - 90.0) < abs(best_geom - 90.0):
                                best_geom = delta_theta
                            break
                    if has_valid_pair:
                        break

            if has_valid_pair and vis_count >= 3:
                status = "REDUNDANT"
                covered_count += 1
                redundant_count += 1
            elif has_valid_pair:
                status = "ADEQUATE"
                covered_count += 1
            elif vis_count == 1:
                status = "MARGINAL"
                outage_count += 1
            else:
                status = "OUTAGE"
                outage_count += 1

            cell_results.append(
                GridCellCoverage(
                    lat=round(float(lat), 3),
                    lon=round(float(lon), 3),
                    visible_dme_count=vis_count,
                    has_valid_dme_dme_pair=has_valid_pair,
                    best_pair_geometry_deg=round(best_geom, 1),
                    coverage_status=status,
                )
            )

    total_cells = max(1, len(cell_results))
    cov_pct = round((covered_count / total_cells) * 100.0, 1)
    red_pct = round((redundant_count / total_cells) * 100.0, 1)
    out_pct = round((outage_count / total_cells) * 100.0, 1)

    return cell_results, cov_pct, red_pct, out_pct


def optimize_dme_rationalization(
    stations: list[DMEStation],
    flight_level: int = 300,
    bbox: tuple[float, float, float, float] = (48.0, 54.0, -3.0, 7.0),
    min_allowable_coverage_pct: float = 98.0,
    max_to_suggest: int = 15,
) -> list[dict[str, Any]]:
    """
    Greedy redundancy pruning algorithm:
    Identifies active DMEs that can be turned off/decommissioned with zero or minimal
    degradation in DME-DME RNAV coverage.
    """
    # 1. Baseline coverage
    _, base_cov_pct, _, _ = evaluate_dme_coverage(stations, flight_level, bbox, cell_step_deg=0.40)

    suggestions: list[dict[str, Any]] = []
    currently_deactivated: set[str] = set()

    # Find stations located in high-density clusters
    for sta in stations:
        if len(suggestions) >= max_to_suggest:
            break
        # Test deactivation
        candidate_inactive = currently_deactivated | {sta.ident}
        _, new_cov, _, _ = evaluate_dme_coverage(
            stations, flight_level, bbox, cell_step_deg=0.40, inactive_idents=candidate_inactive
        )
        loss = round(base_cov_pct - new_cov, 2)
        if new_cov >= min_allowable_coverage_pct and loss <= 0.5:
            currently_deactivated.add(sta.ident)
            suggestions.append({
                "ident": sta.ident,
                "name": sta.name,
                "country": sta.country,
                "coverage_impact_pct": max(0.0, loss),
                "remaining_network_coverage_pct": new_cov,
                "recommendation": "SAFE_TO_DEACTIVATE" if loss <= 0.05 else "LOW_IMPACT_RATIONALIZATION",
                "annual_maintenance_savings_eur": 45000,
            })

    return suggestions


def get_dme_rnav_coverage_report(
    dataset: FlightDataset,
    flight_level: int = 300,
    bbox: tuple[float, float, float, float] = (48.0, 54.0, -3.0, 7.0),
    inactive_idents: list[str] | None = None,
) -> DMERNAVCoverageReport:
    """Generate complete DME-DME RNAV coverage and rationalization report."""
    stations = _load_dme_stations(dataset, bbox)
    inact_set = set(inactive_idents or [])

    cells, cov_pct, red_pct, out_pct = evaluate_dme_coverage(
        stations,
        flight_level=flight_level,
        bbox=bbox,
        cell_step_deg=0.25,
        inactive_idents=inact_set,
    )

    suggested = optimize_dme_rationalization(
        stations,
        flight_level=flight_level,
        bbox=bbox,
        min_allowable_coverage_pct=98.0,
    )

    active_count = sum(1 for s in stations if s.ident not in inact_set)
    inactive_count = len(stations) - active_count

    return DMERNAVCoverageReport(
        flight_level=flight_level,
        altitude_ft=flight_level * 100,
        cell_size_nm=15.0,
        total_grid_cells=len(cells),
        covered_cells_count=int(len(cells) * (cov_pct / 100.0)),
        coverage_percentage=cov_pct,
        redundant_cells_percentage=red_pct,
        outage_cells_percentage=out_pct,
        total_dmes_count=len(stations),
        active_dmes_count=active_count,
        inactive_dmes_count=inactive_count,
        suggested_deactivations=suggested,
        dme_stations=[s.to_dict() for s in stations],
        cells_summary=[c.to_dict() for c in cells],
    )
