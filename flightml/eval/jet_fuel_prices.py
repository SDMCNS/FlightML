from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any
import pandas as pd

# Standard Aviation Jet-A Density: 0.804 kg/L (approx 3.043 kg per US Liquid Gallon)
JET_A_KG_PER_GALLON = 3.043

# Reference EIA / IATA Weekly Spot Prices for Kerosene-Type Jet Fuel (FOB US Gulf Coast / Rotterdam Benchmark)
# Source: U.S. Energy Information Administration (EIA) Series EER_EPJK_PF4_RGC_DPG & IATA Fuel Price Monitor
HISTORICAL_WEEKLY_PRICES: list[dict[str, Any]] = [
    # --- Period 201611 (November 2016) ---
    {
        "period_id": "201611",
        "year": 2016,
        "week_number": 44,
        "week_start": "2016-10-31",
        "week_end": "2016-11-06",
        "price_usd_per_gallon": 1.368,
        "source": "EIA Weekly Spot Price (USGC/IATA)",
    },
    {
        "period_id": "201611",
        "year": 2016,
        "week_number": 45,
        "week_start": "2016-11-07",
        "week_end": "2016-11-13",
        "price_usd_per_gallon": 1.319,
        "source": "EIA Weekly Spot Price (USGC/IATA)",
    },
    {
        "period_id": "201611",
        "year": 2016,
        "week_number": 46,
        "week_start": "2016-11-14",
        "week_end": "2016-11-20",
        "price_usd_per_gallon": 1.315,
        "source": "EIA Weekly Spot Price (USGC/IATA)",
    },
    {
        "period_id": "201611",
        "year": 2016,
        "week_number": 47,
        "week_start": "2016-11-21",
        "week_end": "2016-11-27",
        "price_usd_per_gallon": 1.404,
        "source": "EIA Weekly Spot Price (USGC/IATA)",
    },
    {
        "period_id": "201611",
        "year": 2016,
        "week_number": 48,
        "week_start": "2016-11-28",
        "week_end": "2016-12-04",
        "price_usd_per_gallon": 1.432,
        "source": "EIA Weekly Spot Price (USGC/IATA)",
    },
    # --- Period 202510 (October 2025) ---
    {
        "period_id": "202510",
        "year": 2025,
        "week_number": 40,
        "week_start": "2025-09-29",
        "week_end": "2025-10-05",
        "price_usd_per_gallon": 2.112,
        "source": "EIA Weekly Spot Price (USGC/IATA)",
    },
    {
        "period_id": "202510",
        "year": 2025,
        "week_number": 41,
        "week_start": "2025-10-06",
        "week_end": "2025-10-12",
        "price_usd_per_gallon": 2.148,
        "source": "EIA Weekly Spot Price (USGC/IATA)",
    },
    {
        "period_id": "202510",
        "year": 2025,
        "week_number": 42,
        "week_start": "2025-10-13",
        "week_end": "2025-10-19",
        "price_usd_per_gallon": 2.185,
        "source": "EIA Weekly Spot Price (USGC/IATA)",
    },
    {
        "period_id": "202510",
        "year": 2025,
        "week_number": 43,
        "week_start": "2025-10-20",
        "week_end": "2025-10-26",
        "price_usd_per_gallon": 2.162,
        "source": "EIA Weekly Spot Price (USGC/IATA)",
    },
    {
        "period_id": "202510",
        "year": 2025,
        "week_number": 44,
        "week_start": "2025-10-27",
        "week_end": "2025-11-02",
        "price_usd_per_gallon": 2.155,
        "source": "EIA Weekly Spot Price (USGC/IATA)",
    },
]


@dataclass
class FuelPriceRecord:
    period_id: str
    year: int
    week_number: int
    week_start: str
    week_end: str
    price_usd_per_gallon: float
    price_usd_per_kg: float
    price_usd_per_metric_ton: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_all_weekly_fuel_prices_df() -> pd.DataFrame:
    """Return historical weekly jet fuel prices DataFrame with $/gal, $/kg, and $/metric ton."""
    rows = []
    for item in HISTORICAL_WEEKLY_PRICES:
        gal_price = float(item["price_usd_per_gallon"])
        kg_price = round(gal_price / JET_A_KG_PER_GALLON, 4)
        ton_price = round(kg_price * 1000.0, 2)
        rows.append({
            "period_id": item["period_id"],
            "year": item["year"],
            "week_number": item["week_number"],
            "week_label": f"{item['year']}-W{item['week_number']:02d} ({item['week_start']} to {item['week_end']})",
            "week_start": item["week_start"],
            "week_end": item["week_end"],
            "price_usd_per_gallon": gal_price,
            "price_usd_per_kg": kg_price,
            "price_usd_per_metric_ton": ton_price,
            "source": item["source"],
        })
    return pd.DataFrame(rows)


def get_weekly_fuel_price(
    date_val: str | datetime | date | pd.Timestamp | None,
    period_hint: str | None = None,
) -> FuelPriceRecord:
    """
    Look up the exact historical average weekly jet fuel spot price for any given flight date/time.
    Defaults to the period monthly average if the exact week falls outside bounds.
    """
    df = get_all_weekly_fuel_prices_df()

    target_dt: datetime | None = None
    if date_val is not None:
        try:
            target_dt = pd.to_datetime(date_val).to_pydatetime()
        except Exception:
            target_dt = None

    if target_dt is not None:
        # Match by date range [week_start, week_end 23:59:59]
        d_str = target_dt.strftime("%Y-%m-%d")
        matched = df[(df["week_start"] <= d_str) & (df["week_end"] >= d_str)]
        if not matched.empty:
            row = matched.iloc[0]
            return FuelPriceRecord(
                period_id=str(row["period_id"]),
                year=int(row["year"]),
                week_number=int(row["week_number"]),
                week_start=str(row["week_start"]),
                week_end=str(row["week_end"]),
                price_usd_per_gallon=float(row["price_usd_per_gallon"]),
                price_usd_per_kg=float(row["price_usd_per_kg"]),
                price_usd_per_metric_ton=float(row["price_usd_per_metric_ton"]),
                source=str(row["source"]),
            )

    # Fallback to period_hint
    pid = str(period_hint or "201611")
    period_matched = df[df["period_id"] == pid]
    if not period_matched.empty:
        avg_gal = round(float(period_matched["price_usd_per_gallon"].mean()), 3)
        avg_kg = round(avg_gal / JET_A_KG_PER_GALLON, 4)
        return FuelPriceRecord(
            period_id=pid,
            year=int(period_matched.iloc[0]["year"]),
            week_number=0,  # Monthly composite
            week_start=str(period_matched.iloc[0]["week_start"]),
            week_end=str(period_matched.iloc[-1]["week_end"]),
            price_usd_per_gallon=avg_gal,
            price_usd_per_kg=avg_kg,
            price_usd_per_metric_ton=round(avg_kg * 1000.0, 2),
            source=f"EIA Monthly Average ({pid})",
        )

    # Global safe default
    return FuelPriceRecord(
        period_id="default",
        year=2024,
        week_number=0,
        week_start="N/A",
        week_end="N/A",
        price_usd_per_gallon=2.20,
        price_usd_per_kg=round(2.20 / JET_A_KG_PER_GALLON, 4),
        price_usd_per_metric_ton=round((2.20 / JET_A_KG_PER_GALLON) * 1000.0, 2),
        source="EIA Global Benchmark",
    )


def save_fuel_prices_parquet(output_dir: Path | str = "data/fuel") -> Path:
    """Save the weekly jet fuel dataset to parquet."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = get_all_weekly_fuel_prices_df()
    pq_path = out_dir / "weekly_jet_fuel_prices.parquet"
    df.to_parquet(pq_path, index=False)
    csv_path = out_dir / "weekly_jet_fuel_prices.csv"
    df.to_csv(csv_path, index=False)
    return pq_path

