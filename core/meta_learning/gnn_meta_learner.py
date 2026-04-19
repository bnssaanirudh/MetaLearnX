"""
MetaLearnX — Graph Neural Network Meta-Learner
2-layer GCN that predicts best pipeline from a dataset knowledge graph.
Uses PyTorch Geometric (graceful fallback if unavailable).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "svm",
    "mlp",
]

WEIGHTS_PATH = Path(__file__).parents[3] / "models" / "neural_meta" / "gnn_meta.pt"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


if PYG_AVAILABLE:
    class GNNMetaLearner(nn.Module):
        """
        2-layer Graph Convolutional Network for pipeline recommendation.

        Architecture:
          Node features (meta_feature_vec) → GCNConv → ReLU → Dropout
          → GCNConv → ReLU → Linear head → model family probabilities

        Training:
          Semi-supervised: labeled nodes = datasets with completed experiments
          Loss: CrossEntropyLoss on labeled nodes
          Update: called after each experiment batch
        """

        def __init__(
            self,
            in_channels: int = 33,
            hidden_channels: int = 64,
            out_channels: int = len(MODEL_NAMES),
            dropout: float = 0.3,
        ) -> None:
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.classifier = nn.Linear(hidden_channels, out_channels)
            self.dropout = dropout

        def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            return self.classifier(x)

        def predict_proba(self, x, edge_index) -> np.ndarray:
            self.eval()
            with torch.no_grad():
                logits = self.forward(x, edge_index)
                return F.softmax(logits, dim=-1).cpu().numpy()

else:
    class GNNMetaLearner:
        """Stub when PyTorch Geometric is not installed."""
        def __init__(self, *args, **kwargs):
            pass


def train_gnn(
    feature_matrix: np.ndarray,      # (N, meta_feature_dim)
    labels: np.ndarray,               # (N,) model family index, -1 = unlabeled
    edge_src: List[int],
    edge_tgt: List[int],
    epochs: int = 100,
    lr: float = 5e-3,
    device: str = "cpu",
) -> Optional["GNNMetaLearner"]:
    """
    Train GNN meta-learner on the knowledge graph.
    Semi-supervised: trains only on labeled nodes (completed experiments).
    """
    if not PYG_AVAILABLE:
        logger.warning("PyTorch Geometric not available — skipping GNN training.")
        return None

    N, D = feature_matrix.shape
    model = GNNMetaLearner(in_channels=D, hidden_channels=64, out_channels=len(MODEL_NAMES))
    model = model.to(device)

    x = torch.tensor(feature_matrix, dtype=torch.float).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)

    if edge_src:
        edge_index = torch.tensor([edge_src + edge_tgt, edge_tgt + edge_src], dtype=torch.long).to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)

    labeled_mask = y >= 0
    if labeled_mask.sum() < 2:
        logger.warning("Not enough labeled nodes to train GNN (need ≥ 2).")
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[labeled_mask], y[labeled_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                preds = out[labeled_mask].argmax(dim=1)
                acc = (preds == y[labeled_mask]).float().mean().item()
            logger.info(f"GNN epoch {epoch+1}/{epochs} — loss={loss.item():.4f}, acc={acc:.3f}")

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(WEIGHTS_PATH))
    logger.info(f"GNN meta-learner saved to {WEIGHTS_PATH}")
    return model


def build_graph_from_meta_db() -> Optional["GNNMetaLearner"]:
    """
    Pull all datasets + experiments from meta-db, construct graph, and train GNN.
    Called periodically to update the GNN as new experiments complete.
    """
    from core.tracking.meta_db import get_all_meta_features, list_datasets, list_experiments
    from sklearn.metrics.pairwise import cosine_similarity

    all_ds = list_datasets()
    if len(all_ds) < 3:
        logger.info("Not enough datasets to train GNN (need ≥ 3).")
        return None

    id_list = [d["id"] for d in all_ds]
    mf_data = dict(get_all_meta_features())   # {id: np.array}

    # Filter to datasets with embeddings
    valid_ids = [did for did in id_list if did in mf_data]
    if len(valid_ids) < 3:
        return None

    X = np.vstack([mf_data[did] for did in valid_ids])

    # Build labels from experiments
    exp_map: dict = {}
    for exp in list_experiments():
        did = exp.get("dataset_id")
        model = exp.get("pipeline_config", {}).get("model_name", "")
        if did and model in MODEL_NAMES and exp.get("status") == "done":
            score = exp.get("metrics", {}).get("primary_metric", 0)
            if did not in exp_map or score > exp_map[did][1]:
                exp_map[did] = (model, score)

    labels = np.full(len(valid_ids), -1, dtype=int)
    for i, did in enumerate(valid_ids):
        if did in exp_map:
            model, _ = exp_map[did]
            if model in MODEL_NAMES:
                labels[i] = MODEL_NAMES.index(model)

    # Build edges via cosine similarity threshold
    sim_matrix = cosine_similarity(X)
    threshold = 0.5
    edge_src, edge_tgt = [], []
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            if sim_matrix[i, j] >= threshold:
                edge_src.append(i)
                edge_tgt.append(j)

    logger.info(
        f"GNN graph: {len(valid_ids)} nodes, {len(edge_src)} edges, "
        f"{(labels >= 0).sum()} labeled"
    )
    return train_gnn(X, labels, edge_src, edge_tgt)
