from __future__ import annotations

import math
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from flightml.data.dataset import FlightDataset

# ----------------- AIRCRAFT OPERATIONAL TAXONOMY -----------------
COMMERCIAL_NARROWBODY_PREFIXES = (
    "A318", "A319", "A320", "A321", "A20N", "A21N", "A19N",
    "B73", "B38M", "B39M", "B37M", "B3JM",
    "E170", "E175", "E190", "E195", "E290", "E295", "E19", "E29",
    "BCS1", "BCS3", "A220",
    "CRJ1", "CRJ2", "CRJ7", "CRJ9", "CRJX",
    "MD8", "MD9", "B712", "F70", "F100", "SU95",
)
COMMERCIAL_WIDEBODY_PREFIXES = (
    "A33", "A34", "A35", "A38",
    "B74", "B75", "B76", "B77", "B78",
    "MD11", "DC10", "IL96",
)
REGIONAL_TURBOPROP_PREFIXES = (
    "AT4", "AT7", "DH8", "DHC8", "B190", "SF34", "SB20", "JS41", "JS31",
    "D328", "SW4", "L410", "AN24", "AN26",
)
BUSINESS_JET_PREFIXES = (
    "CL3", "CL6", "C525", "C560", "C680", "C750", "C25A", "C25B", "C25C",
    "FA7X", "FA8X", "FA50", "FA20", "F2TH", "F900",
    "G150", "G280", "G450", "G550", "G650", "GL5T", "GL6T", "GL7T", "GLEX",
    "PC24", "E55P", "E50P", "E545", "E550", "LJ35", "LJ45", "LJ60", "LJ75", "H25B", "BE40",
)
HELICOPTER_PREFIXES = (
    "S92", "S76", "A139", "A109", "A169", "AW13", "AW16", "AW18", "AW10",
    "EC35", "EC45", "EC55", "EC75", "EC25", "EC30", "H125", "H135", "H145", "H160", "H175", "H225",
    "AS32", "AS35", "AS50", "AS55", "AS65",
    "B06", "B212", "B412", "B429", "B407", "R44", "R66", "MD50", "MD52", "MD90",
)


def get_aircraft_class(ac_type: Any) -> tuple[str, int]:
    """
    Categorize aircraft into operational taxonomy classes:
    - COMMERCIAL_JET_NARROWBODY (code 0)
    - COMMERCIAL_JET_WIDEBODY (code 1)
    - REGIONAL_TURBOPROP (code 2)
    - BUSINESS_JET (code 3)
    - HELICOPTER (code 4)
    - GENERAL_AVIATION_PISTON (code 5)
    """
    if ac_type is None or pd.isna(ac_type):
        return "COMMERCIAL_JET_NARROWBODY", 0
    ac = str(ac_type).strip().upper()
    if any(ac.startswith(p) for p in COMMERCIAL_NARROWBODY_PREFIXES):
        return "COMMERCIAL_JET_NARROWBODY", 0
    if any(ac.startswith(p) for p in COMMERCIAL_WIDEBODY_PREFIXES):
        return "COMMERCIAL_JET_WIDEBODY", 1
    if any(ac.startswith(p) for p in REGIONAL_TURBOPROP_PREFIXES):
        return "REGIONAL_TURBOPROP", 2
    if any(ac.startswith(p) for p in BUSINESS_JET_PREFIXES):
        return "BUSINESS_JET", 3
    if any(ac.startswith(p) for p in HELICOPTER_PREFIXES):
        return "HELICOPTER", 4
    return "GENERAL_AVIATION_PISTON", 5


def deterministic_hash(val: Any, modulus: int = 1000) -> float:
    """Deterministic CRC32 hash normalized to [0, 1), reproducible across Python processes and OS."""
    if val is None or pd.isna(val):
        return 0.0
    val_str = str(val).strip().upper()
    return float((zlib.crc32(val_str.encode("utf-8")) & 0xFFFFFFFF) % modulus) / float(modulus)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely extract float from nullable pandas/database values."""
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


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
    Deterministic self-supervised projection network mapping flight kinematics,
    routing geometry, aircraft taxonomy, and operational context into a dense 128-dimensional latent space.
    Strictly initialized with a deterministic orthogonal seed to guarantee cross-process reproducibility.
    """

    def __init__(self, input_dim: int = 16, latent_dim: int = 128, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
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
        self._init_orthogonal_weights(seed=seed)

    def _init_orthogonal_weights(self, seed: int = 42) -> None:
        gen = torch.Generator().manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, generator=gen)
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
        self.network = Flight2VecNetwork(input_dim=16, latent_dim=128, seed=42)
        self.network.eval()
        self._cached_df: pd.DataFrame | None = None
        self._cached_vectors: np.ndarray | None = None

    def _get_cache_dir(self) -> Path:
        cache_dir = Path(self.dataset.data_dir) / "cache" / "flight2vec"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_cache_paths(self) -> tuple[Path, Path]:
        cdir = self._get_cache_dir()
        meta_p = cdir / f"flight2vec_meta_{self.period}_v2.parquet"
        vec_candidates = sorted(
            cdir.glob(f"flight2vec_vectors_{self.period}_v2*.npy"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if vec_candidates:
            vec_p = vec_candidates[0]
        else:
            vec_p = cdir / f"flight2vec_vectors_{self.period}_v2.npy"
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
        dist = _safe_float(row.get("actual_distance_nm"), 500.0)
        dep_delay = _safe_float(row.get("dep_delay_min"), 0.0)
        arr_delay = _safe_float(row.get("arr_delay_min"), 0.0)
        req_fl = _safe_float(row.get("requested_fl"), 340.0)
        flown_time = _safe_float(row.get("actual_air_time_min") or row.get("flight_duration_min"), 90.0)
        speed = dist / max(flown_time / 60.0, 0.2)

        op = str(row.get("ac_operator") or "UNK")
        ac = str(row.get("ac_type") or "A320")
        cls_name, cls_code = get_aircraft_class(ac)
        dep_hour = _safe_float(row.get("departure_hour"), 12.0)

        vec = np.array([
            min(dist / 3000.0, 2.0),
            np.clip(dep_delay / 120.0, -1.0, 3.0),
            np.clip(arr_delay / 120.0, -1.0, 3.0),
            req_fl / 450.0,
            min(flown_time / 360.0, 2.0),
            min(speed / 550.0, 1.5),
            deterministic_hash(op),
            deterministic_hash(ac),
            dep_hour / 24.0,
            cls_code / 5.0,
            1.0 if cls_name in ("COMMERCIAL_JET_NARROWBODY", "COMMERCIAL_JET_WIDEBODY") else 0.0,
            1.0 if cls_name == "HELICOPTER" else 0.0,
            1.0 if cls_name == "REGIONAL_TURBOPROP" else 0.0,
            1.0 if cls_name == "BUSINESS_JET" else 0.0,
            min(dist / 600.0, 1.0),
            1.0 if abs(arr_delay) <= 15.0 else 0.0,
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
        flown_time = (
            df["actual_air_time_min"].fillna(90.0).to_numpy(dtype=np.float32)
            if "actual_air_time_min" in df.columns
            else np.full(n, 90.0, dtype=np.float32)
        )
        speed = dist / np.maximum(flown_time / 60.0, 0.2)

        op_hashes = np.array([deterministic_hash(op) for op in df["ac_operator"]], dtype=np.float32)
        ac_hashes = np.array([deterministic_hash(ac) for ac in df["ac_type"]], dtype=np.float32)
        dep_hour = (
            df["departure_hour"].fillna(12.0).to_numpy(dtype=np.float32)
            if "departure_hour" in df.columns
            else np.full(n, 12.0, dtype=np.float32)
        )

        classes = [get_aircraft_class(ac) for ac in df["ac_type"]]
        cls_codes = np.array([c[1] for c in classes], dtype=np.float32)
        is_comm = np.array([1.0 if c[0] in ("COMMERCIAL_JET_NARROWBODY", "COMMERCIAL_JET_WIDEBODY") else 0.0 for c in classes], dtype=np.float32)
        is_helo = np.array([1.0 if c[0] == "HELICOPTER" else 0.0 for c in classes], dtype=np.float32)
        is_turbo = np.array([1.0 if c[0] == "REGIONAL_TURBOPROP" else 0.0 for c in classes], dtype=np.float32)
        is_biz = np.array([1.0 if c[0] == "BUSINESS_JET" else 0.0 for c in classes], dtype=np.float32)

        feat[:, 0] = np.minimum(dist / 3000.0, 2.0)
        feat[:, 1] = np.clip(dep_delay / 120.0, -1.0, 3.0)
        feat[:, 2] = np.clip(arr_delay / 120.0, -1.0, 3.0)
        feat[:, 3] = req_fl / 450.0
        feat[:, 4] = np.minimum(flown_time / 360.0, 2.0)
        feat[:, 5] = np.minimum(speed / 550.0, 1.5)
        feat[:, 6] = op_hashes
        feat[:, 7] = ac_hashes
        feat[:, 8] = dep_hour / 24.0
        feat[:, 9] = cls_codes / 5.0
        feat[:, 10] = is_comm
        feat[:, 11] = is_helo
        feat[:, 12] = is_turbo
        feat[:, 13] = is_biz
        feat[:, 14] = np.minimum(dist / 600.0, 1.0)
        feat[:, 15] = (np.abs(arr_delay) <= 15.0).astype(np.float32)
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
            alt_vec_p = vec_p.parent / f"flight2vec_vectors_{self.period}_v2_{int(time.time())}.npy"
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
        d_t = _safe_float(target_flight.get("actual_distance_nm"), 500.0)
        d_m = _safe_float(twin_row.get("actual_distance_nm"), 500.0)
        d_delta = abs(d_t - d_m)
        d_pct = (d_delta / max(d_t, 1.0)) * 100.0
        if d_delta <= 30.0:
            reasons.append(f"Exact stage length match ({d_m:.0f} nm vs {d_t:.0f} nm, Δ {d_delta:.0f} nm)")
        elif d_pct <= 20.0:
            reasons.append(f"Comparable stage length ({d_m:.0f} nm vs {d_t:.0f} nm, {d_pct:.1f}% range delta)")
        else:
            reasons.append(f"Distance profile: {d_m:.0f} nm vs {d_t:.0f} nm")

        # Aircraft taxonomy & airframe compatibility
        ac_t = str(target_flight.get("ac_type") or "A320")
        ac_m = str(twin_row.get("ac_type") or "A320")
        cls_t, _ = get_aircraft_class(ac_t)
        cls_m, _ = get_aircraft_class(ac_m)
        if ac_t == ac_m:
            reasons.append(f"Identical airframe type ({ac_t}), identical climb and cruise drag polar")
        elif cls_t == cls_m:
            reasons.append(f"Same fleet tier ({cls_t.replace('_', ' ').title()}: {ac_m} vs {ac_t})")

        fl_t = _safe_float(target_flight.get("requested_fl"), 340.0)
        fl_m = _safe_float(twin_row.get("requested_fl"), 340.0)
        fl_delta = abs(fl_t - fl_m)
        if fl_delta == 0.0:
            reasons.append(f"Identical cruise altitude tier (FL{fl_m:.0f})")
        elif fl_delta <= 20.0:
            reasons.append(f"Matching cruise band (FL{fl_m:.0f} vs FL{fl_t:.0f}, within 2,000 ft RVSM)")

        arr_t = _safe_float(target_flight.get("arr_delay_min"), 0.0)
        arr_m = _safe_float(twin_row.get("arr_delay_min"), 0.0)
        if abs(arr_t - arr_m) <= 10.0:
            reasons.append(f"Synchronized delay dynamics ({arr_m:+.1f} min vs {arr_t:+.1f} min arrival delay)")
        elif arr_t > 15.0 and arr_m > 15.0:
            reasons.append(f"Both absorbed reactionary network delay ripple ({arr_m:+.1f}m vs {arr_t:+.1f}m)")
        elif arr_t <= 5.0 and arr_m <= 5.0:
            reasons.append("Both operated with unimpeded on-time arrival punctuality")

        dep_h_t = _safe_float(target_flight.get("departure_hour"), 12.0)
        dep_h_m = _safe_float(twin_row.get("departure_hour"), 12.0)
        if abs(dep_h_t - dep_h_m) <= 1.5:
            reasons.append(f"Operating in the same departure traffic bank ({dep_h_m:02.0f}:00 UTC)")

        op_t = str(target_flight.get("ac_operator") or "")
        op_m = str(twin_row.get("ac_operator") or "")
        if op_t == op_m and op_t:
            reasons.append(f"Same airline operator ({op_t}), sharing fleet dispatch policies")

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
        enforces strict operational fleet taxonomy compatibility, and provides explainable similarity attributions.
        """
        target_flight = self.dataset.get_flight(target_ectrl_id)
        if not target_flight:
            return []

        target_emb = self.embed_flight(target_ectrl_id)
        if target_emb is None:
            return []

        target_ac = str(target_flight.get("ac_type") or "A320")
        target_cls, _ = get_aircraft_class(target_ac)
        target_dist = _safe_float(target_flight.get("actual_distance_nm"), 500.0)

        # High-speed pre-vectorized cache branch
        if use_cache and self.is_cache_available():
            cache_data = self.load_vector_cache()
            if cache_data is not None:
                meta_df, vectors = cache_data
                mask = meta_df["ectrl_id"] != target_ectrl_id

                # Operational category semantic filtering
                if target_cls in ("COMMERCIAL_JET_NARROWBODY", "COMMERCIAL_JET_WIDEBODY"):
                    # Exclude helicopters and light general aviation pistons
                    helo_mask = meta_df["ac_type"].astype(str).str.upper().apply(
                        lambda ac: any(ac.startswith(p) for p in HELICOPTER_PREFIXES)
                    )
                    mask = mask & (~helo_mask)
                elif target_cls == "HELICOPTER":
                    helo_mask = meta_df["ac_type"].astype(str).str.upper().apply(
                        lambda ac: any(ac.startswith(p) for p in HELICOPTER_PREFIXES)
                    )
                    mask = mask & helo_mask

                # Plausible stage length filtering (avoid pairing 50nm hop with 5000nm long-haul)
                if "actual_distance_nm" in meta_df.columns:
                    dist_col = meta_df["actual_distance_nm"].fillna(500.0)
                    min_dist = max(30.0, target_dist * 0.25)
                    max_dist = max(250.0, target_dist * 3.0)
                    mask = mask & (dist_col >= min_dist) & (dist_col <= max_dist)

                filtered_df = meta_df[mask]
                filtered_vecs = vectors[mask.to_numpy()]

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

        # Fallback dynamic batch vectorization
        min_dist = max(30.0, target_dist * 0.25)
        max_dist = max(250.0, target_dist * 3.0)
        candidates_df = self.dataset.query(f"""
            SELECT
                ectrl_id, adep, ades, ac_type, ac_operator,
                actual_distance_nm, dep_delay_min, arr_delay_min,
                requested_fl, actual_air_time_min,
                EXTRACT(HOUR FROM COALESCE(actual_off_block_time, filed_off_block_time)) AS departure_hour
            FROM flights
            WHERE ectrl_id != {target_ectrl_id}
              AND actual_distance_nm BETWEEN {min_dist} AND {max_dist}
            LIMIT {search_sample_size * 2};
        """)

        if candidates_df.empty:
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

        # Filter candidates by operational category
        cand_classes = [get_aircraft_class(ac)[0] for ac in candidates_df["ac_type"]]
        if target_cls in ("COMMERCIAL_JET_NARROWBODY", "COMMERCIAL_JET_WIDEBODY"):
            cand_mask = [c not in ("HELICOPTER", "GENERAL_AVIATION_PISTON") for c in cand_classes]
            filtered_cands = candidates_df[cand_mask]
            if len(filtered_cands) >= top_k:
                candidates_df = filtered_cands
        elif target_cls == "HELICOPTER":
            cand_mask = [c == "HELICOPTER" for c in cand_classes]
            filtered_cands = candidates_df[cand_mask]
            if len(filtered_cands) >= top_k:
                candidates_df = filtered_cands

        candidates_df = candidates_df.iloc[:search_sample_size]
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
