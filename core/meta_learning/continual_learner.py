"""
MetaLearnX — Continual Meta-Learning with EWC + Replay Buffer
Prevents catastrophic forgetting as the meta-knowledge base grows.

Techniques:
  1. Experience Replay Buffer (circular, capacity 500)
  2. Elastic Weight Consolidation (EWC) - penalizes changes to important weights
  3. Incremental training after each experiment

References:
  - Kirkpatrick et al. (2017): "Overcoming Catastrophic Forgetting in Neural Networks"
  - Ring (1998): "CHILD: A First Step Towards Continual Machine Learning"
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
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

BUFFER_PATH = Path(__file__).parents[3] / "models" / "neural_meta" / "replay_buffer.pkl"
EWC_PATH = Path(__file__).parents[3] / "models" / "neural_meta" / "ewc_state.pt"


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular experience replay buffer for meta-learning examples.
    Stores (meta_feature_vector, best_model_index) pairs.
    Prevents catastrophic forgetting by replaying past experiences during training.
    """

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.n_total_added = 0

    def add(self, meta_features: np.ndarray, model_label: int) -> None:
        """Add a new experience. Oldest entries evicted when full."""
        self.buffer.append((meta_features.astype(np.float32).copy(), int(model_label)))
        self.n_total_added += 1

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int]]:
        """Sample a random batch from the replay buffer."""
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)

    def sample_as_tensors(self, batch_size: int):
        """Sample batch and return as PyTorch tensors."""
        if not TORCH_AVAILABLE:
            return None, None
        samples = self.sample(batch_size)
        if not samples:
            return None, None
        X = torch.tensor(np.vstack([s[0] for s in samples]), dtype=torch.float)
        y = torch.tensor([s[1] for s in samples], dtype=torch.long)
        return X, y

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def utilization(self) -> float:
        return len(self.buffer) / self.capacity

    def stats(self) -> Dict:
        labels = [s[1] for s in self.buffer]
        from collections import Counter
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization_pct": round(self.utilization * 100, 1),
            "total_added": self.n_total_added,
            "model_distribution": dict(Counter(labels)),
        }

    def save(self) -> None:
        import pickle
        BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BUFFER_PATH, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load() -> "ReplayBuffer":
        import pickle
        if BUFFER_PATH.exists():
            with open(BUFFER_PATH, "rb") as f:
                return pickle.load(f)
        return ReplayBuffer()


# ─────────────────────────────────────────────
# EWC (Elastic Weight Consolidation)
# ─────────────────────────────────────────────

class EWC:
    """
    Elastic Weight Consolidation.
    Computes Fisher information for each parameter after initial training,
    then adds a regularization term during subsequent training:

      L_total = L_new_task + λ/2 * Σ_i F_i(θ_i - θ*_i)^2

    where F_i = Fisher information (weight importance),
          θ*_i = optimal parameters for previous task.
    """

    def __init__(self, model: nn.Module, dataloader_samples: List[Tuple], lam: float = 1000.0):
        self.lam = lam
        self._params_star: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}

        if TORCH_AVAILABLE and model is not None:
            self._compute_fisher(model, dataloader_samples)

    def _compute_fisher(self, model: nn.Module, samples: List[Tuple]) -> None:
        """Compute empirical Fisher information matrix (diagonal approximation)."""
        model.eval()
        criterion = nn.CrossEntropyLoss()

        # Save current optimal parameters
        for name, param in model.named_parameters():
            self._params_star[name] = param.data.clone()
            self._fisher[name] = torch.zeros_like(param.data)

        for x_np, y_int in samples[:min(100, len(samples))]:
            try:
                x = torch.tensor(x_np, dtype=torch.float).unsqueeze(0)
                y = torch.tensor([y_int], dtype=torch.long)

                model.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        self._fisher[name] += param.grad.data.pow(2)
            except Exception:
                continue

        # Normalize
        n = max(len(samples), 1)
        for name in self._fisher:
            self._fisher[name] /= n

        logger.info(f"EWC Fisher computed over {n} samples")

    def penalty(self, model: nn.Module) -> "torch.Tensor":
        """Compute EWC penalty for current model parameters."""
        if not TORCH_AVAILABLE or not self._fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if name in self._fisher:
                diff = (param - self._params_star[name]).pow(2)
                loss += (self._fisher[name] * diff).sum()
        return (self.lam / 2) * loss

    def save(self) -> None:
        if TORCH_AVAILABLE:
            torch.save({"fisher": self._fisher, "params_star": self._params_star, "lam": self.lam},
                       str(EWC_PATH))

    @staticmethod
    def load(model: nn.Module, lam: float = 1000.0) -> "EWC":
        ewc = EWC(model=None, dataloader_samples=[], lam=lam)
        if TORCH_AVAILABLE and EWC_PATH.exists():
            state = torch.load(str(EWC_PATH), map_location="cpu")
            ewc._fisher = state["fisher"]
            ewc._params_star = state["params_star"]
            ewc.lam = state["lam"]
        return ewc


# ─────────────────────────────────────────────
# Continual Meta-Learner
# ─────────────────────────────────────────────

class ContinualMetaLearner:
    """
    Wraps the neural meta-learner with:
      - Replay buffer (anti-forgetting)
      - EWC regularization (weight protection)
      - Incremental update API
    """

    def __init__(self, meta_feature_dim: int = 33, n_models: int = 7):
        self.dim = meta_feature_dim
        self.n_models = n_models
        self.replay_buffer = ReplayBuffer.load()
        self._model: Optional[nn.Module] = None
        self._ewc: Optional[EWC] = None
        self._n_updates = 0

    def _get_model(self):
        if self._model is None:
            from core.meta_learning.neural_meta_learner import TransformerMetaLearner
            self._model = TransformerMetaLearner(
                input_dim=self.dim, n_models=self.n_models
            )
            # Load existing weights if available
            from pathlib import Path
            weights = Path(__file__).parents[3] / "models" / "neural_meta" / "meta_ranker.pt"
            if weights.exists() and TORCH_AVAILABLE:
                import torch
                self._model.load_state_dict(torch.load(str(weights), map_location="cpu"))
        return self._model

    def update(
        self,
        new_meta_features: np.ndarray,
        best_model_idx: int,
        lr: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 5,
    ) -> Dict:
        """
        Incremental update after a new experiment completes.
        1. Add new experience to replay buffer
        2. Sample mixed batch (new + replay)
        3. Train with EWC regularization
        """
        # 1. Add to replay buffer
        self.replay_buffer.add(new_meta_features, best_model_idx)

        if not TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable — skipping continual update.")
            self.replay_buffer.save()
            return {"status": "pytorch_unavailable"}

        import torch

        model = self._get_model()

        # Compute EWC penalty (if we have prior Fisher info)
        if self._ewc is None:
            self._ewc = EWC.load(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        losses = []
        model.train()
        for epoch in range(n_epochs):
            # Sample from replay buffer
            X_buf, y_buf = self.replay_buffer.sample_as_tensors(batch_size)
            if X_buf is None:
                break

            # New sample
            x_new = torch.tensor(new_meta_features, dtype=torch.float).unsqueeze(0)
            y_new = torch.tensor([best_model_idx], dtype=torch.long)

            X = torch.cat([X_buf, x_new], dim=0)
            y = torch.cat([y_buf, y_new], dim=0)

            optimizer.zero_grad()
            if X.shape[0] < 2:
                continue
            out = model(X)
            ce_loss = criterion(out, y)
            ewc_loss = self._ewc.penalty(model)
            total_loss = ce_loss + ewc_loss
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        # Update EWC after sufficient data
        if len(self.replay_buffer) >= 10:
            samples = self.replay_buffer.sample(50)
            self._ewc = EWC(model, samples)
            self._ewc.save()

        # Save updated weights
        from pathlib import Path
        weights_path = Path(__file__).parents[3] / "models" / "neural_meta" / "meta_ranker.pt"
        torch.save(model.state_dict(), str(weights_path))
        self.replay_buffer.save()
        self._n_updates += 1

        result = {
            "status": "updated",
            "n_updates": self._n_updates,
            "replay_buffer_size": len(self.replay_buffer),
            "avg_loss": round(float(np.mean(losses)) if losses else 0, 5),
            "buffer_stats": self.replay_buffer.stats(),
        }
        logger.info(f"Continual update #{self._n_updates}: loss={result['avg_loss']:.5f}, "
                    f"buffer={result['replay_buffer_size']}")
        return result

    def get_stats(self) -> Dict:
        return {
            "n_updates": self._n_updates,
            "replay_buffer": self.replay_buffer.stats(),
            "ewc_active": self._ewc is not None and bool(self._ewc._fisher),
        }


# Singleton
_continual_learner: Optional[ContinualMetaLearner] = None


def get_continual_learner() -> ContinualMetaLearner:
    global _continual_learner
    if _continual_learner is None:
        _continual_learner = ContinualMetaLearner()
    return _continual_learner
