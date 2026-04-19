"""
MetaLearnX — Uncertainty-Aware AutoML with MC Dropout
Provides epistemic uncertainty estimates for model recommendations.

Method: Monte Carlo Dropout
  - Keep dropout layers active at inference time
  - Run T=50 stochastic forward passes
  - Compute mean + variance of predictions
  - Output: confidence interval and uncertainty score per model family

Reference:
  Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "svm",
    "mlp",
]


if TORCH_AVAILABLE:
    class MCDropoutMetaLearner(nn.Module):
        """
        MC Dropout Meta-Learner for uncertainty-aware recommendations.
        Identical architecture to TransformerMetaLearner but with always-on dropout.

        During MC Dropout inference:
          - model.train() is called (keeps dropout stochastic)
          - T forward passes collect a distribution of predictions
          - std across passes = epistemic uncertainty
        """

        def __init__(
            self,
            input_dim: int = 33,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            n_models: int = len(MODEL_NAMES),
            dropout: float = 0.3,   # slightly higher for better uncertainty spread
        ) -> None:
            super().__init__()
            self.feature_embed = nn.Linear(1, d_model)
            self.pos_embed = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_models),
            )
            self.dropout_rate = dropout

        def forward(self, x):
            x = x.unsqueeze(-1)
            x = self.feature_embed(x) + self.pos_embed.unsqueeze(0)
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.head(x)

        def mc_predict(
            self,
            x: "torch.Tensor",
            n_passes: int = 50,
        ) -> Dict:
            """
            Run T stochastic forward passes with dropout active.
            Returns mean, std, and confidence intervals.
            """
            self.train()  # keeps dropout stochastic
            predictions = []
            with torch.no_grad():
                for _ in range(n_passes):
                    probs = F.softmax(self.forward(x), dim=-1)
                    predictions.append(probs.cpu().numpy())

            preds_arr = np.stack(predictions, axis=0)   # (T, B, n_models)
            mean = preds_arr.mean(axis=0)               # (B, n_models)
            std = preds_arr.std(axis=0)                 # (B, n_models)
            ci_95_low = np.percentile(preds_arr, 2.5, axis=0)
            ci_95_high = np.percentile(preds_arr, 97.5, axis=0)

            # Predictive entropy as global uncertainty score
            eps = 1e-9
            entropy = -(mean * np.log(mean + eps)).sum(axis=-1)   # (B,)

            return {
                "mean": mean,
                "std": std,
                "ci_95_low": ci_95_low,
                "ci_95_high": ci_95_high,
                "entropy": entropy,
                "n_passes": n_passes,
            }

else:
    class MCDropoutMetaLearner:
        def __init__(self, *a, **k): pass


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

from pathlib import Path
WEIGHTS_PATH = Path(__file__).parents[3] / "models" / "neural_meta" / "meta_ranker.pt"
_mc_model_cache: Optional["MCDropoutMetaLearner"] = None


def _get_mc_model() -> Optional["MCDropoutMetaLearner"]:
    global _mc_model_cache
    if not TORCH_AVAILABLE:
        return None
    if _mc_model_cache is None:
        m = MCDropoutMetaLearner()
        if WEIGHTS_PATH.exists():
            try:
                state = torch.load(str(WEIGHTS_PATH), map_location="cpu")
                m.load_state_dict(state, strict=False)
            except Exception:
                pass
        _mc_model_cache = m
    return _mc_model_cache


def uncertain_recommend(
    meta_feature_vec: np.ndarray,
    n_passes: int = 50,
    top_k: int = 5,
    task_type: str = "classification",
) -> List[Dict]:
    """
    Recommend models with epistemic uncertainty scores via MC Dropout.

    Returns list of:
      {model_name, mean_confidence, std, ci_95_low, ci_95_high, uncertainty_label}
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch unavailable — returning dummy uncertainty.")
        return _dummy_uncertain(top_k)

    model = _get_mc_model()
    if model is None:
        return _dummy_uncertain(top_k)

    x = torch.from_numpy(meta_feature_vec.astype(np.float32)).unsqueeze(0)
    result = model.mc_predict(x, n_passes=n_passes)

    mean = result["mean"][0]       # (n_models,)
    std = result["std"][0]
    ci_low = result["ci_95_low"][0]
    ci_high = result["ci_95_high"][0]
    entropy = float(result["entropy"][0])

    ranked = sorted(
        zip(MODEL_NAMES, mean.tolist(), std.tolist(), ci_low.tolist(), ci_high.tolist()),
        key=lambda row: row[1], reverse=True
    )

    recommendations = []
    for rank_i, (m, mn, sd, cl, ch) in enumerate(ranked[:top_k]):
        uncertainty_label = _uncertainty_label(sd)
        recommendations.append({
            "rank": rank_i + 1,
            "model_name": m,
            "mean_confidence": round(float(mn), 4),
            "std": round(float(sd), 4),
            "ci_95_low": round(float(cl), 4),
            "ci_95_high": round(float(ch), 4),
            "uncertainty_label": uncertainty_label,
            "source": "mc_dropout",
        })

    return {
        "recommendations": recommendations,
        "predictive_entropy": round(entropy, 4),
        "n_passes": n_passes,
        "global_uncertainty": _uncertainty_label(entropy / np.log(len(MODEL_NAMES))),
    }


def _uncertainty_label(normalized_std: float) -> str:
    if normalized_std < 0.05:
        return "very_confident"
    elif normalized_std < 0.12:
        return "confident"
    elif normalized_std < 0.20:
        return "uncertain"
    else:
        return "very_uncertain"


def _dummy_uncertain(top_k: int) -> List[Dict]:
    p = 1.0 / len(MODEL_NAMES)
    return {
        "recommendations": [
            {
                "rank": i + 1,
                "model_name": m,
                "mean_confidence": round(p, 4),
                "std": 0.0,
                "ci_95_low": round(p, 4),
                "ci_95_high": round(p, 4),
                "uncertainty_label": "very_uncertain",
                "source": "torch_unavailable",
            }
            for i, m in enumerate(MODEL_NAMES[:top_k])
        ],
        "predictive_entropy": round(np.log(len(MODEL_NAMES)), 4),
        "n_passes": 0,
        "global_uncertainty": "very_uncertain",
    }
