from __future__ import annotations

from typing import Any
import pandas as pd

from flightml.data.dataset import FlightDataset


def analyze_fir_bottlenecks(dataset: FlightDataset, limit: int = 15) -> pd.DataFrame:
    """
    Identify the most congested Flight Information Region (FIR) airspaces
    and quantify traffic demand and peak movement density.
    """
    # Spatially map flights to FIR airspaces by joining waypoint coordinates
    query = """
        WITH fir_sampled AS (
            SELECT
                COALESCE(fir."Airspace ID", 'EN-ROUTE') AS airspace_id,
                min(fir."Min Flight Level") AS min_fl,
                max(fir."Max Flight Level") AS max_fl,
                count(distinct t.ectrl_id) AS unique_flights,
                count(*) AS waypoint_traversals
            FROM trajectory_actual t
            JOIN raw_fir fir
                ON round(t.latitude, 0) = round(fir.Latitude, 0)
                AND round(t.longitude, 0) = round(fir.Longitude, 0)
            GROUP BY COALESCE(fir."Airspace ID", 'EN-ROUTE')
        )
        SELECT
            airspace_id,
            unique_flights,
            waypoint_traversals,
            min_fl,
            max_fl
        FROM fir_sampled
        ORDER BY unique_flights DESC
        LIMIT ?;
    """
    return dataset.query(query, [limit])

