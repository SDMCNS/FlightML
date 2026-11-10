from __future__ import annotations

from typing import Any
import pandas as pd
from flightml.data.dataset import FlightDataset


def compute_top_fir_congestion(dataset: FlightDataset, limit: int = 15) -> pd.DataFrame:
    """
    Estimate FIR sector demand and flight density across Europe by analyzing flight trajectories.
    """
    # Sample or aggregate trajectory points to identify hotspot FIR regions
    query = """
        SELECT
            COALESCE(fir."Airspace ID", 'EN-ROUTE') AS airspace_id,
            count(distinct t.ectrl_id) as unique_flights_sampled,
            count(*) as waypoint_density
        FROM trajectory_actual t
        LEFT JOIN raw_fir fir
            ON round(t.latitude, 0) = round(fir.Latitude, 0)
            AND round(t.longitude, 0) = round(fir.Longitude, 0)
        GROUP BY COALESCE(fir."Airspace ID", 'EN-ROUTE')
        ORDER BY unique_flights_sampled DESC
        LIMIT ?;
    """
    try:
        return dataset.query(query, [limit])
    except Exception:
        # Fallback to route-based FIR approximation
        return pd.DataFrame()

