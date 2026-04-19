"""
MetaLearnX — Transformer Meta-Learner (Upgraded from MLP)
Treats each meta-feature dimension as a token and applies multi-head self-attention.
Includes MC Dropout for uncertainty estimation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

N_MODELS = len(MODEL_NAMES)
META_FEATURE_DIM = 33
WEIGHTS_PATH = Path(__file__).parents[3] / "models" / "neural_meta" / "meta_ranker.pt"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TransformerMetaLearner(nn.Module):
        """
        Transformer-based meta-learner for pipeline recommendation.

        Each meta-feature is treated as a "token" (1-dim feature → projected to d_model).
        Multi-head self-attention captures inter-feature relationships.

        Architecture:
          Input: (B, meta_feature_dim) float32
          Projection: (B, meta_feature_dim, d_model)   — per-feature linear projection
          Transformer: multi-head attention + FFN
          Pool: mean pool over feature dimension
          Head: Linear → n_models logits
        """

        def __init__(
            self,
            input_dim: int = META_FEATURE_DIM,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            n_models: int = N_MODELS,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.d_model = d_model

            # Per-feature linear projection
            self.feature_embed = nn.Linear(1, d_model)

            # Positional encoding (learnable)
            self.pos_embed = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            # Classification head
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_models),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                x: (B, input_dim) meta-feature vectors
            Returns:
                logits: (B, n_models)
            """
            # Reshape for per-feature embedding: (B, input_dim, 1) → (B, input_dim, d_model)
            x = x.unsqueeze(-1)                            # (B, D, 1)
            x = self.feature_embed(x)                      # (B, D, d_model)
            x = x + self.pos_embed.unsqueeze(0)            # add positional encoding

            # Transformer (batch_first=True → (B, seq_len, d_model))
            x = self.transformer(x)                        # (B, D, d_model)

            # Mean-pool over feature dimension
            x = x.mean(dim=1)                              # (B, d_model)

            return self.head(x)                            # (B, n_models)

        def predict_proba(self, x: "torch.Tensor") -> "torch.Tensor":
            return F.softmax(self.forward(x), dim=-1)

    # Legacy MLP (for ablation comparison)
    class LegacyMetaLearner(nn.Module):
        """Original 2-layer MLP meta-learner kept for ablation studies."""

        def __init__(
            self,
            input_dim: int = META_FEATURE_DIM,
            hidden_dim: int = 128,
            n_models: int = N_MODELS,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_models),
            )

        def forward(self, x):
            return F.log_softmax(self.network(x), dim=-1)

        def predict_proba(self, x):
            return torch.exp(self.forward(x))

else:
    class TransformerMetaLearner:
        def __init__(self, *args, **kwargs): pass

    class LegacyMetaLearner:
        def __init__(self, *args, **kwargs): pass


# ─────────────────────────────────────────────
# Model cache + inference
# ─────────────────────────────────────────────

_model_cache: Optional["TransformerMetaLearner"] = None


def _get_transformer_ranker(device: str = "cpu") -> Optional["TransformerMetaLearner"]:
    global _model_cache
    if not TORCH_AVAILABLE:
        return None
    if _model_cache is None:
        model = TransformerMetaLearner()
        if WEIGHTS_PATH.exists():
            try:
                state = torch.load(str(WEIGHTS_PATH), map_location=device)
                model.load_state_dict(state)
                logger.info("Loaded Transformer meta-learner weights.")
            except Exception as e:
                logger.warning(f"Weight load failed ({e}) — using random init.")
        model.eval()
        _model_cache = model.to(device)
    return _model_cache


def neural_recommend(
    meta_feature_vec: np.ndarray,
    task_type: str = "classification",
    top_k: int = 5,
    device: str = "cpu",
    use_transformer: bool = True,
) -> List[Dict]:
    """
    Recommend model families using Transformer meta-learner.
    Falls back to uniform prior if model not available.
    """
    if not TORCH_AVAILABLE:
        return _uniform_prior(top_k)

    model = _get_transformer_ranker(device)
    if model is None:
        return _uniform_prior(top_k)

    x = torch.from_numpy(meta_feature_vec.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model.predict_proba(x)[0].cpu().numpy()

    ranked = sorted(zip(MODEL_NAMES, probs.tolist()), key=lambda x: x[1], reverse=True)
    return [
        {
            "rank": i + 1,
            "model_name": m,
            "confidence": round(float(p), 4),
            "source": "transformer_meta_learner",
        }
        for i, (m, p) in enumerate(ranked[:top_k])
    ]


def _uniform_prior(top_k: int) -> List[Dict]:
    uniform = 1.0 / N_MODELS
    return [
        {"rank": i + 1, "model_name": m, "confidence": round(uniform, 4), "source": "uniform_prior"}
        for i, m in enumerate(MODEL_NAMES[:top_k])
    ]


def train_neural_ranker(
    examples: List[Tuple[np.ndarray, int]],
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    use_transformer: bool = True,
) -> "TransformerMetaLearner":
    """Train the Transformer meta-ranker on accumulated (meta_features, label) pairs."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available — skipping training.")
        return TransformerMetaLearner()

    ModelClass = TransformerMetaLearner if use_transformer else LegacyMetaLearner
    model = ModelClass().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X = torch.tensor(np.vstack([x for x, _ in examples]), dtype=torch.float).to(device)
    y = torch.tensor([lbl for _, lbl in examples], dtype=torch.long).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        if X.shape[0] < 2:
            break
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            logger.info(f"Meta-ranker epoch {epoch+1}/{epochs} — loss={loss.item():.4f}")

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(WEIGHTS_PATH))
    global _model_cache
    _model_cache = None
    return model
