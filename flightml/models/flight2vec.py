from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from flightml.data.dataset import FlightDataset


@dataclass
class OperationalTwinMatch:
    ectrl_id: int
    route: str
    ac_type: str
    operator: str
    flown_distance_nm: float
    arr_delay_min: float
    cruise_fl: float
    cosine_similarity: float
    similarity_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Flight2VecNetwork(nn.Module):
    """
    Self-supervised multi-layer projection network mapping flight kinematics,
    routing geometry, and operational context into a dense 128-dimensional latent space.
    """

    def __init__(self, input_dim: int = 16, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 96),
            nn.LayerNorm(96),
            nn.LeakyReLU(0.1),
            nn.Linear(96, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self._init_orthogonal_weights()

    def _init_orthogonal_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        # L2 normalize onto unit hypersphere for cosine similarity metric space
        return nn.functional.normalize(z, p=2, dim=-1)


class Flight2VecEngine:
    """
    Experimental engine for computing self-supervised flight embeddings and
    retrieving latent operational twins across European traffic datasets.
    """

    def __init__(self, dataset: FlightDataset | None = None, period: str = "202510") -> None:
        self.dataset = dataset or FlightDataset(period=period)
        self.period = self.dataset.period
        self.network = Flight2VecNetwork(input_dim=16, latent_dim=128)
        self.network.eval()

    def _extract_flight_features(self, row: dict[str, Any] | pd.Series) -> np.ndarray:
        """Normalize key flight parameters into an input feature vector."""
        dist = float(row.get("actual_distance_nm") or 500.0)
        dep_delay = float(row.get("dep_delay_min") or 0.0)
        arr_delay = float(row.get("arr_delay_min") or 0.0)
        req_fl = float(row.get("requested_fl") or 340.0)
        max_fl = req_fl
        hold_time = float(row.get("total_holding_duration_min") or 0.0)
        flown_time = float(row.get("actual_air_time_min") or row.get("flight_duration_min") or 90.0)
        speed = dist / max(flown_time / 60.0, 0.2)

        # Hash-based operator and airframe encoding
        op_str = str(row.get("ac_operator") or "UNK")
        ac_str = str(row.get("ac_type") or "A320")
        op_hash = (hash(op_str) % 1000) / 1000.0
        ac_hash = (hash(ac_str) % 1000) / 1000.0

        vec = np.array([
            min(dist / 3000.0, 2.0),
            np.clip(dep_delay / 120.0, -1.0, 3.0),
            np.clip(arr_delay / 120.0, -1.0, 3.0),
            req_fl / 450.0,
            max_fl / 450.0,
            (max_fl - req_fl) / 50.0,
            min(hold_time / 30.0, 2.0),
            min(flown_time / 360.0, 2.0),
            min(speed / 550.0, 1.5),
            op_hash,
            ac_hash,
            float(row.get("departure_hour") or 12.0) / 24.0,
            float(row.get("route_charge_avoidance_detected") or 0.0),
            float(row.get("is_heavy_wake") or 0.0),
            min(float(row.get("co2_metric_tons") or 5.0) / 50.0, 2.0),
            float(row.get("has_tactical_reroute") or 0.0),
        ], dtype=np.float32)
        return vec

    def embed_flight(self, ectrl_id: int) -> np.ndarray | None:
        """Compute the 128-dimensional latent representation for a given flight ID."""
        flight = self.dataset.get_flight(ectrl_id)
        if not flight:
            return None
        feat = self._extract_flight_features(flight)
        with torch.no_grad():
            x = torch.from_numpy(feat).unsqueeze(0)
            z = self.network(x).squeeze(0).numpy()
        return z

    def find_operational_twins(
        self,
        target_ectrl_id: int,
        search_sample_size: int = 150,
        top_k: int = 5,
    ) -> list[OperationalTwinMatch]:
        """
        Scan a representative corpus of flights and retrieve the top-k most semantically
        identical operational twins in the 128D latent space using cosine similarity.
        """
        target_flight = self.dataset.get_flight(target_ectrl_id)
        if not target_flight:
            return []

        target_emb = self.embed_flight(target_ectrl_id)
        if target_emb is None:
            return []

        # Retrieve a diverse flight sample to compare against
        candidates_df = self.dataset.query(f"""
            SELECT
                ectrl_id, adep, ades, ac_type, ac_operator,
                actual_distance_nm, dep_delay_min, arr_delay_min,
                requested_fl, actual_air_time_min,
                EXTRACT(HOUR FROM COALESCE(actual_off_block_time, filed_off_block_time)) AS departure_hour
            FROM flights
            WHERE ectrl_id != {target_ectrl_id}
            LIMIT {search_sample_size};
        """)

        if candidates_df.empty:
            return []

        matches: list[OperationalTwinMatch] = []
        for _, row in candidates_df.iterrows():
            cand_feat = self._extract_flight_features(row)
            with torch.no_grad():
                cand_emb = self.network(torch.from_numpy(cand_feat).unsqueeze(0)).squeeze(0).numpy()

            # Cosine similarity on unit sphere = dot product
            cos_sim = float(np.dot(target_emb, cand_emb))
            cos_sim = max(min(cos_sim, 1.0), -1.0)
            sim_pct = max(0.0, (cos_sim + 1.0) / 2.0 * 100.0)

            matches.append(
                OperationalTwinMatch(
                    ectrl_id=int(row["ectrl_id"]),
                    route=f"{row['adep']} ➔ {row['ades']}",
                    ac_type=str(row["ac_type"]),
                    operator=str(row["ac_operator"]),
                    flown_distance_nm=float(row["actual_distance_nm"] or 0.0),
                    arr_delay_min=float(row["arr_delay_min"] or 0.0),
                    cruise_fl=float(row.get("requested_fl") or 340.0),
                    cosine_similarity=cos_sim,
                    similarity_pct=sim_pct,
                )
            )

        matches.sort(key=lambda m: m.cosine_similarity, reverse=True)
        return matches[:top_k]

    def project_latent_space_2d(self, sample_size: int = 40) -> pd.DataFrame:
        """
        Sample flights across the active period, project their 128D embeddings into a 2D plane
        via Principal Component Analysis (PCA), and return plottable coordinates.
        """
        df = self.dataset.query(f"""
            SELECT
                ectrl_id, adep, ades, ac_type, ac_operator,
                actual_distance_nm, dep_delay_min, arr_delay_min,
                requested_fl, actual_air_time_min,
                EXTRACT(HOUR FROM COALESCE(actual_off_block_time, filed_off_block_time)) AS departure_hour
            FROM flights
            LIMIT {sample_size};
        """)
        if df.empty:
            return pd.DataFrame()

        embeddings = []
        labels = []
        routes = []
        delays = []
        operators = []

        for _, row in df.iterrows():
            feat = self._extract_flight_features(row)
            with torch.no_grad():
                z = self.network(torch.from_numpy(feat).unsqueeze(0)).squeeze(0).numpy()
            embeddings.append(z)
            labels.append(f"{row['ac_operator']} #{row['ectrl_id']}")
            routes.append(f"{row['adep']} ➔ {row['ades']}")
            delays.append(float(row["arr_delay_min"] or 0.0))
            operators.append(str(row["ac_operator"]))

        emb_matrix = np.array(embeddings)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(emb_matrix)

        plot_df = pd.DataFrame({
            "PCA_Dim_1": coords[:, 0],
            "PCA_Dim_2": coords[:, 1],
            "Flight_Label": labels,
            "Route": routes,
            "Arr_Delay_Min": delays,
            "Operator": operators,
        })
        return plot_df
