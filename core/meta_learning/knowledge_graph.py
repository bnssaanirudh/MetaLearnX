"""
MetaLearnX — Dataset Knowledge Graph
Constructs and maintains a dynamic graph: datasets as nodes, similarity as edges.
Uses PyTorch Geometric (GCN) if available, falls back to NetworkX.

Graph structure:
  Nodes: datasets (node features = meta-feature vector)
  Edges: weighted by cosine similarity (threshold > 0.6)
  Node attributes: best_model, best_metric
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# ── Optional PyTorch Geometric import ─────────────────────────────────────
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    logger.info("PyTorch Geometric not installed — Knowledge Graph uses NetworkX fallback.")

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

GRAPH_PATH = Path(__file__).parents[3] / "data" / "meta_db" / "knowledge_graph.pkl"
GRAPH_WEIGHTS_PATH = Path(__file__).parents[3] / "models" / "neural_meta" / "gnn_meta.pt"


# ─────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────

class DatasetKnowledgeGraph:
    """
    Dynamic dataset knowledge graph.

    Nodes: datasets with meta-feature vectors and performance attributes.
    Edges: cosine similarity between dataset meta-feature vectors (threshold-filtered).
    """

    def __init__(self, similarity_threshold: float = 0.5):
        self.threshold = similarity_threshold
        self.nodes: Dict[str, Dict[str, Any]] = {}   # dataset_id → attrs
        self.edges: List[Tuple[str, str, float]] = []
        self._nx_graph = None

    # ── Public API ─────────────────────────────────────────────────────

    def add_or_update_node(
        self,
        dataset_id: str,
        name: str,
        meta_feature_vec: np.ndarray,
        best_model: Optional[str] = None,
        best_metric: Optional[float] = None,
        task_type: str = "classification",
    ) -> None:
        """Add a dataset node with its meta-feature vector and performance info."""
        self.nodes[dataset_id] = {
            "name": name,
            "meta_features": meta_feature_vec.astype(np.float32),
            "best_model": best_model or "unknown",
            "best_metric": best_metric or 0.0,
            "task_type": task_type,
        }
        self._recompute_edges()
        self._nx_graph = None  # invalidate

    def get_neighbors(self, dataset_id: str, top_k: int = 5) -> List[Dict]:
        """Return top-k most similar datasets to the query node."""
        if dataset_id not in self.nodes:
            return []

        q_vec = self.nodes[dataset_id]["meta_features"]
        similarities = []
        for nid, attrs in self.nodes.items():
            if nid == dataset_id:
                continue
            sim = float(_cosine_sim(q_vec, attrs["meta_features"]))
            similarities.append((nid, sim, attrs))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "dataset_id": nid,
                "name": attrs["name"],
                "similarity": round(sim, 4),
                "best_model": attrs["best_model"],
                "best_metric": round(attrs["best_metric"], 4),
                "task_type": attrs["task_type"],
            }
            for nid, sim, attrs in similarities[:top_k]
        ]

    def predict_best_model(self, query_vec: np.ndarray, task_type: str = "classification") -> str:
        """
        Predict best model for a new dataset using GNN (if available) or kNN fallback.
        """
        if PYG_AVAILABLE and GRAPH_WEIGHTS_PATH.exists():
            try:
                return self._gnn_predict(query_vec)
            except Exception as e:
                logger.warning(f"GNN predict failed: {e} — using kNN fallback.")

        return self._knn_predict(query_vec, task_type)

    def get_networkx_graph(self):
        """Return the NetworkX representation for visualization."""
        if not NX_AVAILABLE:
            return None
        if self._nx_graph is None:
            self._build_nx_graph()
        return self._nx_graph

    def to_json(self) -> Dict[str, Any]:
        """Serialize graph to JSON-serializable format for dashboard."""
        nodes_json = []
        for nid, attrs in self.nodes.items():
            nodes_json.append({
                "id": nid,
                "name": attrs["name"],
                "best_model": attrs["best_model"],
                "best_metric": attrs["best_metric"],
                "task_type": attrs["task_type"],
            })

        edges_json = [
            {"source": src, "target": tgt, "weight": round(w, 3)}
            for src, tgt, w in self.edges
            if w >= self.threshold
        ]

        return {
            "nodes": nodes_json,
            "edges": edges_json,
            "n_nodes": len(nodes_json),
            "n_edges": len(edges_json),
            "pyg_available": PYG_AVAILABLE,
        }

    def save(self) -> None:
        GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Knowledge graph saved ({len(self.nodes)} nodes, {len(self.edges)} edges)")

    @staticmethod
    def load() -> "DatasetKnowledgeGraph":
        if GRAPH_PATH.exists():
            with open(GRAPH_PATH, "rb") as f:
                g = pickle.load(f)
            logger.info(f"Knowledge graph loaded ({len(g.nodes)} nodes)")
            return g
        return DatasetKnowledgeGraph()

    # ── Private helpers ────────────────────────────────────────────────

    def _recompute_edges(self) -> None:
        """Recompute all pairwise edges based on current node set."""
        ids = list(self.nodes.keys())
        self.edges = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                vi = self.nodes[ids[i]]["meta_features"]
                vj = self.nodes[ids[j]]["meta_features"]
                sim = float(_cosine_sim(vi, vj))
                self.edges.append((ids[i], ids[j], sim))

    def _build_nx_graph(self) -> None:
        G = nx.Graph()
        for nid, attrs in self.nodes.items():
            G.add_node(nid, **{k: v for k, v in attrs.items() if k != "meta_features"})
        for src, tgt, w in self.edges:
            if w >= self.threshold:
                G.add_edge(src, tgt, weight=w)
        self._nx_graph = G

    def _knn_predict(self, query_vec: np.ndarray, task_type: str) -> str:
        """Majority vote on best model from top-k similar datasets."""
        from collections import Counter
        similarities = [
            (nid, float(_cosine_sim(query_vec, attrs["meta_features"])), attrs["best_model"])
            for nid, attrs in self.nodes.items()
            if attrs.get("task_type") == task_type
        ]
        if not similarities:
            return "random_forest"
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:5]
        votes = Counter(m for _, _, m in top_k)
        return votes.most_common(1)[0][0]

    def _gnn_predict(self, query_vec: np.ndarray) -> str:
        """Use trained GCN to predict best model for the query vector."""
        from core.meta_learning.gnn_meta_learner import GNNMetaLearner, MODEL_NAMES

        model = GNNMetaLearner(
            in_channels=len(query_vec),
            hidden_channels=64,
            out_channels=len(MODEL_NAMES),
        )
        state = torch.load(str(GRAPH_WEIGHTS_PATH), map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        # Build mini-graph: existing nodes + query as isolated node
        all_ids = list(self.nodes.keys())
        all_vecs = [self.nodes[nid]["meta_features"] for nid in all_ids] + [query_vec]
        x = torch.tensor(np.vstack(all_vecs), dtype=torch.float)

        edge_list = []
        src_edges, tgt_edges = [], []
        for i, (s, t, w) in enumerate(self.edges):
            if w >= self.threshold:
                si = all_ids.index(s)
                ti = all_ids.index(t)
                src_edges += [si, ti]
                tgt_edges += [ti, si]

        if src_edges:
            edge_index = torch.tensor([src_edges, tgt_edges], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        query_idx = len(all_ids)
        pred = out[query_idx].argmax().item()
        return MODEL_NAMES[pred]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────
# Singleton + helpers
# ─────────────────────────────────────────────

_graph_instance: Optional[DatasetKnowledgeGraph] = None


def get_knowledge_graph() -> DatasetKnowledgeGraph:
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = DatasetKnowledgeGraph.load()
    return _graph_instance


def update_graph_after_experiment(
    dataset_id: str,
    name: str,
    meta_feature_vec: np.ndarray,
    best_model: str,
    best_metric: float,
    task_type: str,
) -> None:
    """Called by orchestrator after each successful experiment."""
    g = get_knowledge_graph()
    g.add_or_update_node(
        dataset_id=dataset_id,
        name=name,
        meta_feature_vec=meta_feature_vec,
        best_model=best_model,
        best_metric=best_metric,
        task_type=task_type,
    )
    g.save()
    logger.info(f"Knowledge graph updated: {len(g.nodes)} nodes, {len(g.edges)} edges")
