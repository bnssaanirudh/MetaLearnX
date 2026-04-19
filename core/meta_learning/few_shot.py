"""
MetaLearnX — Few-Shot AutoML with MAML-Inspired Fast Adaptation
Rapidly adapts the meta-learner to new datasets with very few samples.

Approach:
  MAML-lite: gradient-based inner-loop adaptation on the neural meta-learner.
  Unlike full MAML, we adapt only the Transformer meta-learner weights
  (not the full sklearn pipeline), which is computationally tractable.

Research connection:
  Finn et al. (2017) MAML: "Model-Agnostic Meta-Learning for Fast Adaptation
  of Deep Networks"

Use case:
  - Dataset with < 50 samples
  - New domain where meta-db has no similar datasets
  - "Cold start" adaptation with minimal labels
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

# Models that work well with few samples (< 50)
FEW_SHOT_SAFE_MODELS = ["logistic_regression", "svm", "random_forest"]
TABPFN_THRESHOLD = 1_000  # TabPFN optimal for < 1000 samples


def detect_few_shot_regime(n_samples: int, n_features: int) -> Dict:
    """Determine if a dataset is in the few-shot regime and what to do."""
    is_few_shot = n_samples < 200
    is_very_few = n_samples < 50
    complexity_ratio = n_features / max(n_samples, 1)

    regime = "standard"
    if is_very_few:
        regime = "very_few_shot"
    elif is_few_shot:
        regime = "few_shot"

    recommendations = []
    if is_very_few:
        recommendations.append("Use TabPFN (prior-fitted network, zero-shot capable)")
        recommendations.append("Logistic Regression with strong regularization (C=0.01)")
        recommendations.append("SVM with RBF kernel and cross-validated C")
    elif is_few_shot:
        recommendations.append("Random Forest with max_depth=3 (avoid overfitting)")
        recommendations.append("Logistic Regression (interpretable + safe)")
        if n_samples >= 100:
            recommendations.append("LightGBM with heavy regularization (min_child_samples=20)")

    return {
        "regime": regime,
        "is_few_shot": is_few_shot,
        "n_samples": n_samples,
        "complexity_ratio": round(complexity_ratio, 3),
        "recommendations": recommendations,
        "use_tabpfn": n_samples < TABPFN_THRESHOLD,
        "safe_models": FEW_SHOT_SAFE_MODELS if is_few_shot else MODEL_NAMES,
    }


class MAMLFewShotAdapter:
    """
    MAML-inspired fast adaptation for the neural meta-learner.

    Inner loop: fine-tune meta-learner on few support examples from new dataset.
    This adapts the recommendation model to the specific characteristics
    of a new, data-scarce dataset without full retraining.

    Note: This adapts only the meta-learner (Transformer), not sklearn pipelines.
    """

    def __init__(self, inner_lr: float = 0.01, n_inner_steps: int = 5):
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps

    def adapt(
        self,
        meta_feature_vec: np.ndarray,
        support_model_labels: Optional[List[int]] = None,
        support_meta_features: Optional[List[np.ndarray]] = None,
    ) -> Dict:
        """
        Adapt the meta-learner to a few-shot task.

        Args:
            meta_feature_vec: Meta-feature vector for the query dataset.
            support_model_labels: Known best-model labels for support datasets.
            support_meta_features: Meta-feature vectors for support datasets.

        Returns:
            Adapted model recommendations with fast-adaptation scores.
        """
        if not TORCH_AVAILABLE:
            return self._fallback_recommend(meta_feature_vec)

        from core.meta_learning.neural_meta_learner import TransformerMetaLearner
        from pathlib import Path

        weights_path = Path(__file__).parents[3] / "models" / "neural_meta" / "meta_ranker.pt"
        base_model = TransformerMetaLearner()
        if weights_path.exists():
            try:
                state = torch.load(str(weights_path), map_location="cpu")
                base_model.load_state_dict(state, strict=False)
            except Exception:
                pass

        # Clone model for adaptation (don't modify base weights)
        adapted_model = self._clone_model(base_model)

        if support_model_labels and support_meta_features:
            # Inner loop: gradient steps on support set
            adapted_model = self._inner_loop(
                adapted_model,
                support_meta_features,
                support_model_labels,
            )
            logger.info(
                f"MAML adaptation: {self.n_inner_steps} inner steps on "
                f"{len(support_model_labels)} support examples."
            )

        # Predict on query
        x = torch.from_numpy(meta_feature_vec.astype(np.float32)).unsqueeze(0)
        adapted_model.eval()
        with torch.no_grad():
            probs = F.softmax(adapted_model(x), dim=-1)[0].cpu().numpy()

        ranked = sorted(zip(MODEL_NAMES, probs.tolist()), key=lambda r: r[1], reverse=True)
        return {
            "adapted_recommendations": [
                {
                    "rank": i + 1,
                    "model_name": m,
                    "adapted_confidence": round(float(p), 4),
                    "source": "maml_adapted" if support_model_labels else "base_model",
                }
                for i, (m, p) in enumerate(ranked[:5])
            ],
            "n_support": len(support_model_labels) if support_model_labels else 0,
            "n_inner_steps": self.n_inner_steps,
            "inner_lr": self.inner_lr,
        }

    def _inner_loop(
        self,
        model: "TransformerMetaLearner",
        support_mf: List[np.ndarray],
        support_labels: List[int],
    ) -> "TransformerMetaLearner":
        criterion = nn.CrossEntropyLoss()
        X = torch.tensor(np.vstack(support_mf), dtype=torch.float)
        y = torch.tensor(support_labels, dtype=torch.long)

        for step in range(self.n_inner_steps):
            loss = criterion(model(X), y)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            # Manual gradient descent (MAML inner loop)
            with torch.no_grad():
                for param, grad in zip(model.parameters(), grads):
                    if grad is not None:
                        param -= self.inner_lr * grad

        return model

    def _clone_model(self, model: "TransformerMetaLearner") -> "TransformerMetaLearner":
        """Deep copy of model parameters for independent adaptation."""
        import copy
        cloned = copy.deepcopy(model)
        return cloned

    def _fallback_recommend(self, meta_feature_vec: np.ndarray) -> Dict:
        n_features = int(meta_feature_vec.get(2, 10) if hasattr(meta_feature_vec, 'get') else 10)
        return {
            "adapted_recommendations": [
                {"rank": i+1, "model_name": m, "adapted_confidence": 0.33, "source": "heuristic"}
                for i, m in enumerate(FEW_SHOT_SAFE_MODELS[:3])
            ],
            "n_support": 0,
            "n_inner_steps": 0,
            "inner_lr": 0.0,
        }


def few_shot_recommend(
    meta_feature_vec: np.ndarray,
    n_samples: int,
    n_features: int,
    dataset_id: Optional[str] = None,
    top_k: int = 5,
) -> Dict:
    """
    Full few-shot recommendation pipeline.
    1. Detect regime
    2. If few-shot: retrieve support set from meta-db
    3. Run MAML adaptation
    4. Return adapted recommendations + safety flags
    """
    regime_info = detect_few_shot_regime(n_samples, n_features)
    adapter = MAMLFewShotAdapter()

    # Retrieve support examples from meta-db
    support_mf, support_labels = [], []
    try:
        from core.tracking.meta_db import list_experiments, get_all_meta_features
        mf_dict = dict(get_all_meta_features())
        for exp in list_experiments():
            if exp.get("status") != "done":
                continue
            did = exp.get("dataset_id")
            model = exp.get("pipeline_config", {}).get("model_name")
            if did and model in MODEL_NAMES and did in mf_dict:
                support_mf.append(mf_dict[did])
                support_labels.append(MODEL_NAMES.index(model))
                if len(support_mf) >= 10:
                    break
    except Exception as e:
        logger.warning(f"Could not retrieve support set: {e}")

    adapted = adapter.adapt(
        meta_feature_vec,
        support_model_labels=support_labels if support_labels else None,
        support_meta_features=support_mf if support_mf else None,
    )

    return {
        **regime_info,
        **adapted,
    }
