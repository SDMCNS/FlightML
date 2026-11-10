from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import duckdb
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from flightml.config import PARQUET_DIR, DATA_DIR


@dataclass
class AirspaceSector:
    airspace_id: str
    min_fl: float
    max_fl: float
    polygon: Polygon


class FIREngine:
    """
    Spacial indexing and boundary intersection engine for Flight Information Regions (FIRs).
    Uses Shapely STRtree for sub-millisecond point-in-polygon queries.
    """

    def __init__(self, fir_parquet_path: Path | str | None = None) -> None:
        self.parquet_path = Path(fir_parquet_path or PARQUET_DIR / "fir.parquet")
        self.sectors: list[AirspaceSector] = []
        self.geometries: list[Polygon] = []
        self.tree: STRtree | None = None
        self._load_fir_boundaries()

    def _load_fir_boundaries(self) -> None:
        con = duckdb.connect()
        src = str(self.parquet_path).replace("\\", "/")
        query = f"""
            SELECT
                "Airspace ID" AS airspace_id,
                "Min Flight Level" AS min_fl,
                "Max Flight Level" AS max_fl,
                "Sequence Number" AS seq_num,
                Latitude AS lat,
                Longitude AS lon
            FROM '{src}'
            ORDER BY airspace_id, seq_num ASC;
        """
        df = con.execute(query).df()

        grouped = df.groupby(["airspace_id", "min_fl", "max_fl"])
        for (airspace_id, min_fl, max_fl), group in grouped:
            coords = list(zip(group["lon"], group["lat"]))
            if len(coords) < 3:
                continue

            # Ensure polygon is closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)

            sector = AirspaceSector(
                airspace_id=str(airspace_id),
                min_fl=float(min_fl),
                max_fl=float(max_fl),
                polygon=poly,
            )
            self.sectors.append(sector)
            self.geometries.append(poly)

        if self.geometries:
            self.tree = STRtree(self.geometries)

    def locate_fir(self, lat: float, lon: float, flight_level: float = 350.0) -> list[str]:
        """
        Return the Airspace IDs of the FIR sectors containing the given coordinate and flight level.
        """
        if not self.tree:
            return []

        pt = Point(lon, lat)
        candidate_indices = self.tree.query(pt)

        matching_firs = []
        for idx in candidate_indices:
            sector = self.sectors[idx]
            if sector.min_fl <= flight_level <= sector.max_fl:
                if sector.polygon.contains(pt):
                    matching_firs.append(sector.airspace_id)

        return sorted(list(set(matching_firs)))

    def tag_trajectory(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tag every waypoint in a trajectory with its corresponding FIR.
        """
        df = trajectory_df.copy()
        firs = []
        for _, row in df.iterrows():
            lat = row["latitude"]
            lon = row["longitude"]
            fl = row.get("flight_level", 350.0)
            res = self.locate_fir(lat, lon, fl)
            firs.append(res[0] if res else "UNKNOWN")

        df["fir"] = firs
        return df

