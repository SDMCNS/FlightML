from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from flightml.config import CACHE_DIR
from flightml.data.dataset import FlightDataset

MODEL_DIR = CACHE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class DelayPredictor:
    """
    Gradient-boosted machine learning model to predict flight arrival/departure delays
    using flight plan parameters, schedule congestion, and target-encoded airport/operator priors.
    """

    def __init__(self, target_col: str = "arr_delay_min") -> None:
        self.target_col = target_col
        self.model: HistGradientBoostingRegressor | None = None
        self.target_encodings: dict[str, dict[str, float]] = {}
        self.global_mean: float = 0.0
        self.feature_names: list[str] = []

    def prepare_data(
        self, dataset: FlightDataset, sample_size: int = 50000
    ) -> pd.DataFrame:
        """
        Extract clean training tabular data from DuckDB flights.
        """
        query = f"""
            SELECT
                ectrl_id,
                coalesce(adep, 'UNKNOWN') AS adep,
                coalesce(ades, 'UNKNOWN') AS ades,
                coalesce(ac_operator, 'UNKNOWN') AS ac_operator,
                coalesce(market_segment, 'UNKNOWN') AS market_segment,
                coalesce(requested_fl, 300) AS requested_fl,
                coalesce(actual_distance_nm, 500) AS actual_distance_nm,
                dep_delay_min,
                arr_delay_min,
                coalesce(extract(hour from filed_off_block_time), 12) AS dep_hour,
                coalesce(extract(dayofweek from filed_off_block_time), 1) AS dep_dayofweek,
                coalesce(filed_air_time_min, 60) AS filed_air_time_min
            FROM flights
            WHERE filed_off_block_time IS NOT NULL
              AND actual_arrival_time IS NOT NULL
              AND arr_delay_min BETWEEN -60 AND 360
            USING SAMPLE {sample_size};
        """
        df = dataset.query(query)
        return df

    def train(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> dict[str, float]:
        """
        Train the model using smoothed target encoding for high-cardinality categorical features.
        """
        cat_cols = ["adep", "ades", "ac_operator", "market_segment"]
        num_cols = ["requested_fl", "actual_distance_nm", "dep_hour", "dep_dayofweek", "filed_air_time_min"]

        df_clean = df.copy()
        y_all = pd.to_numeric(df_clean[self.target_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self.global_mean = float(np.mean(y_all))

        # Split into train and test before computing target encodings to prevent data leakage
        train_idx, test_idx = train_test_split(
            np.arange(len(df_clean)), test_size=test_size, random_state=42
        )
        train_df = df_clean.iloc[train_idx].copy()
        test_df = df_clean.iloc[test_idx].copy()

        # Compute smoothed target encodings on training partition
        self.target_encodings = {}
        smooth_weight = 10.0

        for col in cat_cols:
            grouped = train_df.groupby(col)[self.target_col].agg(["count", "mean"])
            enc_map = {}
            for val, row in grouped.iterrows():
                count = row["count"]
                mean_val = row["mean"]
                smoothed = (count * mean_val + smooth_weight * self.global_mean) / (count + smooth_weight)
                enc_map[str(val)] = float(smoothed)
            self.target_encodings[col] = enc_map

        # Function to map encoded features
        def apply_encoding(source_df: pd.DataFrame) -> np.ndarray:
            features = []
            for col in cat_cols:
                mapping = self.target_encodings[col]
                enc_series = source_df[col].astype(str).map(lambda x: mapping.get(x, self.global_mean))
                features.append(enc_series.to_numpy(dtype=np.float64))

            for col in num_cols:
                num_series = pd.to_numeric(source_df[col], errors="coerce").fillna(0.0)
                features.append(num_series.to_numpy(dtype=np.float64))

            return np.column_stack(features)

        self.feature_names = [f"{c}_target_enc" for c in cat_cols] + num_cols
        X_train = apply_encoding(train_df)
        y_train = y_all[train_idx]

        X_test = apply_encoding(test_df)
        y_test = y_all[test_idx]

        self.model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}

    def predict(self, flight_info: dict[str, Any]) -> float:
        """
        Predict delay (minutes) for a single flight dictionary.
        """
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        cat_cols = ["adep", "ades", "ac_operator", "market_segment"]
        num_cols = ["requested_fl", "actual_distance_nm", "dep_hour", "dep_dayofweek", "filed_air_time_min"]

        row = []
        for col in cat_cols:
            val = str(flight_info.get(col, "UNKNOWN"))
            mapping = self.target_encodings.get(col, {})
            row.append(mapping.get(val, self.global_mean))

        for col in num_cols:
            try:
                row.append(float(flight_info.get(col, 0.0) or 0.0))
            except Exception:
                row.append(0.0)

        pred = self.model.predict(np.array([row], dtype=np.float64))[0]
        return round(float(pred), 1)

    def save(self, filepath: Path | str | None = None) -> Path:
        path = Path(filepath or MODEL_DIR / f"{self.target_col}_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "target_encodings": self.target_encodings,
                    "global_mean": self.global_mean,
                    "features": self.feature_names,
                },
                f,
            )
        return path

    @classmethod
    def load(cls, filepath: Path | str) -> DelayPredictor:
        inst = cls()
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            inst.model = data["model"]
            inst.target_encodings = data["target_encodings"]
            inst.global_mean = data["global_mean"]
            inst.feature_names = data["features"]
        return inst
