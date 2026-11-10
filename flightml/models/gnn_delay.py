from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from flightml.config import CACHE_DIR
from flightml.data.dataset import FlightDataset

GNN_MODEL_DIR = CACHE_DIR / "models"
GNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class GraphConvolution(nn.Module):
    """
    Standard Kipf & Welling Graph Convolutional Layer.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # H = A_norm * X * W + b
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj_norm, support) + self.bias
        return output


class AirportSpatioTemporalGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for network-wide delay propagation forecasting.
    """

    def __init__(self, num_nodes: int, in_features: int = 5, hidden_dim: int = 32) -> None:
        super().__init__()
        self.gcn1 = GraphConvolution(in_features, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # Predicts arrival delay for each node
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.gcn1(x, adj_norm))
        h1 = self.dropout(h1)
        h2 = F.relu(self.gcn2(h1, adj_norm))
        out = self.fc(h2).squeeze(-1)
        return out


class AirportGraphDelayModel:
    """
    High-level manager for constructing the airport network graph,
    training the PyTorch GNN, and forecasting network-wide delays.
    """

    def __init__(self, top_n_airports: int = 20) -> None:
        self.top_n = top_n_airports
        self.airports: list[str] = []
        self.adj_norm: torch.Tensor | None = None
        self.model: AirportSpatioTemporalGNN | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_graph(self, dataset: FlightDataset) -> tuple[torch.Tensor, list[str]]:
        """
        Build normalized adjacency matrix from flight network flight counts.
        """
        top_df = dataset.get_top_airports(limit=self.top_n)
        self.airports = top_df["ident"].tolist()
        apt_set = set(self.airports)

        # Query route connection weights among top hubs
        route_df = dataset.query(f"""
            SELECT adep, ades, count(*) AS flight_count
            FROM flights
            WHERE adep IN ({", ".join(f"'{a}'" for a in self.airports)})
              AND ades IN ({", ".join(f"'{a}'" for a in self.airports)})
            GROUP BY adep, ades;
        """)

        n = len(self.airports)
        adj = np.zeros((n, n), dtype=np.float32)
        idx_map = {apt: i for i, apt in enumerate(self.airports)}

        for _, row in route_df.iterrows():
            u = idx_map.get(row["adep"])
            v = idx_map.get(row["ades"])
            if u is not None and v is not None:
                adj[u, v] = float(row["flight_count"])

        # Symmetrize and add self loops
        adj = adj + adj.T + np.eye(n, dtype=np.float32)

        # Normalize: D^(-1/2) * A * D^(-1/2)
        row_sum = np.array(adj.sum(1), dtype=np.float32)
        d_inv_sqrt = np.zeros_like(row_sum)
        mask = row_sum > 0
        d_inv_sqrt[mask] = np.power(row_sum[mask], -0.5)
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        adj_normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

        self.adj_norm = torch.tensor(adj_normalized, dtype=torch.float32, device=self.device)
        return self.adj_norm, self.airports

    def train_baseline(self, dataset: FlightDataset, epochs: int = 50) -> dict[str, float]:
        """
        Train the GNN on daily airport network snapshots.
        """
        if self.adj_norm is None:
            self.build_graph(dataset)

        n_nodes = len(self.airports)
        self.model = AirportSpatioTemporalGNN(num_nodes=n_nodes, in_features=4, hidden_dim=32).to(self.device)

        # Generate daily/hourly aggregate feature tensors from DuckDB
        data_df = dataset.query(f"""
            SELECT
                adep,
                extract(day from actual_off_block_time) AS day,
                avg(dep_delay_min) AS avg_dep_delay,
                avg(arr_delay_min) AS avg_arr_delay,
                count(*) AS volume
            FROM flights
            WHERE adep IN ({", ".join(f"'{a}'" for a in self.airports)})
            GROUP BY adep, extract(day from actual_off_block_time)
            ORDER BY day, adep;
        """)

        # Build feature snapshots by day
        idx_map = {apt: i for i, apt in enumerate(self.airports)}
        days = sorted(data_df["day"].dropna().unique().tolist())

        snapshots = []
        targets = []

        for d in days:
            sub = data_df[data_df["day"] == d]
            feat = np.zeros((n_nodes, 4), dtype=np.float32)
            target = np.zeros(n_nodes, dtype=np.float32)

            for _, row in sub.iterrows():
                idx = idx_map.get(row["adep"])
                if idx is not None:
                    feat[idx, 0] = float(row["avg_dep_delay"] or 0.0)
                    feat[idx, 1] = float(row["volume"] or 0.0) / 100.0
                    feat[idx, 2] = float(d) / 30.0
                    feat[idx, 3] = 1.0  # active indicator
                    target[idx] = float(row["avg_arr_delay"] or 0.0)

            snapshots.append(feat)
            targets.append(target)

        if len(snapshots) < 5:
            return {"loss": 0.0}

        X_tensors = torch.tensor(np.array(snapshots[:-1]), dtype=torch.float32, device=self.device)
        Y_tensors = torch.tensor(np.array(targets[1:]), dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.MSELoss()

        self.model.train()
        losses = []
        for _ in range(epochs):
            total_loss = 0.0
            for i in range(len(X_tensors)):
                optimizer.zero_grad()
                pred = self.model(X_tensors[i], self.adj_norm)
                loss = criterion(pred, Y_tensors[i])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            losses.append(total_loss / len(X_tensors))

        final_loss = losses[-1] if losses else 0.0
        return {"final_loss": round(float(final_loss), 4), "epochs": epochs, "device": str(self.device)}

    def forecast_network(self, current_delays: dict[str, float] | None = None) -> dict[str, float]:
        """
        Forecast next-step arrival delays across all network nodes.
        """
        if self.model is None or self.adj_norm is None:
            raise ValueError("GNN model is not trained.")

        self.model.eval()
        n_nodes = len(self.airports)
        idx_map = {apt: i for i, apt in enumerate(self.airports)}

        feat = np.zeros((n_nodes, 4), dtype=np.float32)
        if current_delays:
            for apt, d in current_delays.items():
                idx = idx_map.get(apt)
                if idx is not None:
                    feat[idx, 0] = d

        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32, device=self.device)
            preds = self.model(x, self.adj_norm).cpu().numpy()

        return {apt: round(float(preds[i]), 1) for i, apt in enumerate(self.airports)}
