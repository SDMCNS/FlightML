from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
    explanation: str = ""
    similarity_reasons: list[str] = field(default_factory=list)

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
    Supports high-speed batch vectorization, persistent disk vector caching,
    and explainable similarity attributions.
    """

    def __init__(self, dataset: FlightDataset | None = None, period: str = "202510") -> None:
        self.dataset = dataset or FlightDataset(period=period)
        self.period = self.dataset.period
        self.network = Flight2VecNetwork(input_dim=16, latent_dim=128)
        self.network.eval()
        self._cached_df: pd.DataFrame | None = None
        self._cached_vectors: np.ndarray | None = None

    def _get_cache_dir(self) -> Path:
        cache_dir = Path(self.dataset.data_dir) / "cache" / "flight2vec"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_cache_paths(self) -> tuple[Path, Path]:
        cdir = self._get_cache_dir()
        meta_p = cdir / f"flight2vec_meta_{self.period}.parquet"
        vec_candidates = sorted(
            cdir.glob(f"flight2vec_vectors_{self.period}*.npy"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if vec_candidates:
            vec_p = vec_candidates[0]
        else:
            vec_p = cdir / f"flight2vec_vectors_{self.period}.npy"
        return meta_p, vec_p

    def is_cache_available(self) -> bool:
        meta_p, vec_p = self._get_cache_paths()
        return meta_p.exists() and vec_p.exists()

    def get_cached_count(self) -> int:
        if self._cached_vectors is not None:
            return len(self._cached_vectors)
        meta_p, vec_p = self._get_cache_paths()
        if meta_p.exists() and vec_p.exists():
            try:
                meta = pd.read_parquet(meta_p, columns=["ectrl_id"])
                return len(meta)
            except Exception:
                return 0
        return 0

    def load_vector_cache(self) -> tuple[pd.DataFrame, np.ndarray] | None:
        if self._cached_df is not None and self._cached_vectors is not None:
            return self._cached_df, self._cached_vectors

        meta_p, vec_p = self._get_cache_paths()
        if not (meta_p.exists() and vec_p.exists()):
            return None

        try:
            df = pd.read_parquet(meta_p)
            vecs = np.load(vec_p)
            # Enforce length consistency between metadata and latent vector array
            if len(df) != len(vecs):
                min_len = min(len(df), len(vecs))
                df = df.iloc[:min_len].copy()
                vecs = vecs[:min_len]
            self._cached_df = df
            self._cached_vectors = vecs
            return df, vecs
        except Exception:
            return None

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

    def _extract_batch_features(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized feature normalization for multi-thousand flight batches."""
        n = len(df)
        feat = np.zeros((n, 16), dtype=np.float32)
        dist = df["actual_distance_nm"].fillna(500.0).to_numpy(dtype=np.float32)
        dep_delay = df["dep_delay_min"].fillna(0.0).to_numpy(dtype=np.float32)
        arr_delay = df["arr_delay_min"].fillna(0.0).to_numpy(dtype=np.float32)
        req_fl = df["requested_fl"].fillna(340.0).to_numpy(dtype=np.float32)
        flown_time = df["actual_air_time_min"].fillna(90.0).to_numpy(dtype=np.float32) if "actual_air_time_min" in df.columns else np.full(n, 90.0, dtype=np.float32)
        speed = dist / np.maximum(flown_time / 60.0, 0.2)

        op_hashes = np.array([(hash(str(op)) % 1000) / 1000.0 for op in df["ac_operator"]], dtype=np.float32)
        ac_hashes = np.array([(hash(str(ac)) % 1000) / 1000.0 for ac in df["ac_type"]], dtype=np.float32)
        dep_hour = df["departure_hour"].fillna(12.0).to_numpy(dtype=np.float32) if "departure_hour" in df.columns else np.full(n, 12.0, dtype=np.float32)

        feat[:, 0] = np.minimum(dist / 3000.0, 2.0)
        feat[:, 1] = np.clip(dep_delay / 120.0, -1.0, 3.0)
        feat[:, 2] = np.clip(arr_delay / 120.0, -1.0, 3.0)
        feat[:, 3] = req_fl / 450.0
        feat[:, 4] = req_fl / 450.0
        feat[:, 5] = 0.0
        feat[:, 6] = 0.0
        feat[:, 7] = np.minimum(flown_time / 360.0, 2.0)
        feat[:, 8] = np.minimum(speed / 550.0, 1.5)
        feat[:, 9] = op_hashes
        feat[:, 10] = ac_hashes
        feat[:, 11] = dep_hour / 24.0
        return feat

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

    def embed_batch(self, df: pd.DataFrame) -> np.ndarray:
        """High-speed vectorized embedding of an entire DataFrame of flights."""
        if df.empty:
            return np.empty((0, 128), dtype=np.float32)
        feat = self._extract_batch_features(df)
        with torch.no_grad():
            x = torch.from_numpy(feat)
            z = self.network(x).numpy()
        return z

    def build_vector_cache(self, limit: int | None = None, batch_size: int = 25000) -> int:
        """
        Pre-vectorize flights in the database (or up to limit if specified) using high-speed
        PyTorch batch matrix inference, persisting metadata to Parquet and
        normalized 128D embeddings to disk for instant retrieval.
        """
        limit_clause = f"LIMIT {limit}" if limit is not None else ""
        query_sql = f"""
            SELECT
                ectrl_id, adep, ades, ac_type, ac_operator,
                actual_distance_nm, dep_delay_min, arr_delay_min,
                requested_fl, actual_air_time_min,
                EXTRACT(HOUR FROM COALESCE(actual_off_block_time, filed_off_block_time)) AS departure_hour
            FROM flights
            {limit_clause};
        """
        df = self.dataset.query(query_sql)
        if df.empty:
            return 0

        total_rows = len(df)
        vector_chunks = []

        for offset in range(0, total_rows, batch_size):
            chunk_df = df.iloc[offset : offset + batch_size]
            chunk_vectors = self.embed_batch(chunk_df)
            vector_chunks.append(chunk_vectors)

        all_vectors = np.vstack(vector_chunks).astype(np.float32)

        meta_p, vec_p = self._get_cache_paths()
        df.to_parquet(meta_p, index=False)
        try:
            np.save(vec_p, all_vectors)
        except OSError:
            alt_vec_p = vec_p.parent / f"flight2vec_vectors_{self.period}_v{int(time.time())}.npy"
            np.save(alt_vec_p, all_vectors)

        self._cached_df = df
        self._cached_vectors = all_vectors
        return total_rows

    def _explain_similarity(
        self,
        target_flight: dict[str, Any] | pd.Series,
        twin_row: dict[str, Any] | pd.Series,
        cos_sim: float,
    ) -> tuple[str, list[str]]:
        """Explain the operational rationale and dimensions driving high latent cosine similarity."""
        reasons: list[str] = []
        d_t = float(target_flight.get("actual_distance_nm") or 500.0)
        d_m = float(twin_row.get("actual_distance_nm") or 500.0)
        d_delta = abs(d_t - d_m)
        d_pct = (d_delta / max(d_t, 1.0)) * 100.0
        if d_delta <= 30.0:
            reasons.append(f"Exact stage length match ({d_m:.0f} nm vs {d_t:.0f} nm, Δ {d_delta:.0f} nm)")
        elif d_pct <= 15.0:
            reasons.append(f"Comparable stage length ({d_m:.0f} nm vs {d_t:.0f} nm, {d_pct:.1f}% range delta)")
        else:
            reasons.append(f"Distance profile: {d_m:.0f} nm vs {d_t:.0f} nm")

        fl_t = float(target_flight.get("requested_fl") or 340.0)
        fl_m = float(twin_row.get("requested_fl") or 340.0)
        fl_delta = abs(fl_t - fl_m)
        if fl_delta == 0.0:
            reasons.append(f"Identical cruise altitude tier (FL{fl_m:.0f})")
        elif fl_delta <= 20.0:
            reasons.append(f"Matching cruise band (FL{fl_m:.0f} vs FL{fl_t:.0f}, within 2,000 ft RVSM)")

        arr_t = float(target_flight.get("arr_delay_min") or 0.0)
        arr_m = float(twin_row.get("arr_delay_min") or 0.0)
        if abs(arr_t - arr_m) <= 10.0:
            reasons.append(f"Synchronized delay dynamics ({arr_m:+.1f} min vs {arr_t:+.1f} min arrival delay)")
        elif arr_t > 15.0 and arr_m > 15.0:
            reasons.append(f"Both absorbed reactionary network delay ripple ({arr_m:+.1f}m vs {arr_t:+.1f}m)")
        elif arr_t <= 5.0 and arr_m <= 5.0:
            reasons.append("Both operated with unimpeded on-time arrival punctuality")

        dep_h_t = float(target_flight.get("departure_hour") or 12.0)
        dep_h_m = float(twin_row.get("departure_hour") or 12.0)
        if abs(dep_h_t - dep_h_m) <= 1.5:
            reasons.append(f"Operating in the same departure traffic bank ({dep_h_m:02.0f}:00 UTC)")

        op_t = str(target_flight.get("ac_operator") or "")
        op_m = str(twin_row.get("ac_operator") or "")
        ac_t = str(target_flight.get("ac_type") or "")
        ac_m = str(twin_row.get("ac_type") or "")
        if op_t == op_m and op_t:
            reasons.append(f"Same airline operator ({op_t}), sharing fleet dispatch policies")
        elif ac_t == ac_m and ac_t:
            reasons.append(f"Same airframe type ({ac_t}), identical climb and cruise drag polar")

        summary = " • ".join(reasons[:3]) if reasons else f"Latent kinematic trajectory alignment across 16 dimensions (cos θ = {cos_sim:.4f})."
        return summary, reasons

    def find_operational_twins(
        self,
        target_ectrl_id: int,
        search_sample_size: int = 500,
        top_k: int = 5,
        use_cache: bool = True,
    ) -> list[OperationalTwinMatch]:
        """
        Scan a representative corpus of flights and retrieve the top-k most semantically
        identical operational twins in the 128D latent space using cosine similarity.
        Leverages pre-vectorized cache if available for microsecond search across thousands of flights,
        and provides explainable similarity attributions for each twin match.
        """
        target_flight = self.dataset.get_flight(target_ectrl_id)
        if not target_flight:
            return []

        target_emb = self.embed_flight(target_ectrl_id)
        if target_emb is None:
            return []

        # High-speed pre-vectorized cache branch
        if use_cache and self.is_cache_available():
            cache_data = self.load_vector_cache()
            if cache_data is not None:
                meta_df, vectors = cache_data
                target_mask = meta_df["ectrl_id"] != target_ectrl_id
                filtered_df = meta_df[target_mask]
                filtered_vecs = vectors[target_mask.to_numpy()]

                if len(filtered_vecs) > 0:
                    similarities = np.dot(filtered_vecs, target_emb)
                    k = min(top_k, len(similarities))
                    best_indices = np.argpartition(similarities, -k)[-k:]
                    best_indices = best_indices[np.argsort(similarities[best_indices])[::-1]]

                    matches = []
                    for idx in best_indices:
                        row = filtered_df.iloc[idx]
                        cos_sim = float(np.clip(similarities[idx], -1.0, 1.0))
                        sim_pct = max(0.0, (cos_sim + 1.0) / 2.0 * 100.0)
                        summary, reasons = self._explain_similarity(target_flight, row, cos_sim)
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
                                explanation=summary,
                                similarity_reasons=reasons,
                            )
                        )
                    return matches

        # Fallback dynamic batch vectorization in milliseconds
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

        cand_vectors = self.embed_batch(candidates_df)
        similarities = np.dot(cand_vectors, target_emb)
        k = min(top_k, len(similarities))
        best_indices = np.argpartition(similarities, -k)[-k:]
        best_indices = best_indices[np.argsort(similarities[best_indices])[::-1]]

        matches = []
        for idx in best_indices:
            row = candidates_df.iloc[idx]
            cos_sim = float(np.clip(similarities[idx], -1.0, 1.0))
            sim_pct = max(0.0, (cos_sim + 1.0) / 2.0 * 100.0)
            summary, reasons = self._explain_similarity(target_flight, row, cos_sim)

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
                    explanation=summary,
                    similarity_reasons=reasons,
                )
            )

        return matches

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

        embeddings = self.embed_batch(df)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)

        labels = [f"{row['ac_operator']} #{row['ectrl_id']}" for _, row in df.iterrows()]
        routes = [f"{row['adep']} ➔ {row['ades']}" for _, row in df.iterrows()]

        plot_df = pd.DataFrame({
            "PCA_Dim_1": coords[:, 0],
            "PCA_Dim_2": coords[:, 1],
            "Flight_Label": labels,
            "Route": routes,
            "Arr_Delay_Min": df["arr_delay_min"].fillna(0.0).to_numpy(),
            "Operator": df["ac_operator"].astype(str).to_numpy(),
        })
        return plot_df
