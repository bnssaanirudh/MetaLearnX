"""
MetaLearnX — Learned Dataset Embedding (Dataset2Vec-inspired)
A lightweight DeepSets/SetTransformer that encodes a dataset sample
(subset of rows × features) into a fixed-size embedding vector.

Architecture:
  Input: (B, N_rows, N_features)  -- random sample of dataset rows
  φ:     per-row MLP              -- maps each row → h_dim
  ρ:     mean-pooling             -- aggregates row representations
  ψ:     output MLP               -- maps to embedding_dim

This is inspired by Dataset2Vec (Jomaa et al., 2021) but implemented as a
lightweight self-contained module that can be fine-tuned on the meta-db.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

EMBEDDING_DIM = 64
HIDDEN_DIM = 128
N_SAMPLE_ROWS = 100     # rows to sample per dataset
N_SAMPLE_COLS = 50      # max features to sample per dataset

WEIGHTS_PATH = Path(__file__).parents[3] / "models" / "neural_meta" / "dataset_embedder.pt"


class DatasetEmbedder(nn.Module):
    """
    DeepSets-style dataset encoder.
    Maps a variable-size set of rows to a fixed-size embedding.
    """

    def __init__(
        self,
        input_dim: int = N_SAMPLE_COLS,
        hidden_dim: int = HIDDEN_DIM,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # φ: per-row encoder
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # ψ: post-aggregation output MLP
        self.psi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_rows, input_dim)
        Returns:
            embedding: (batch, embedding_dim) — L2-normalized
        """
        h = self.phi(x)               # (B, N, hidden_dim)
        h = h.mean(dim=1)             # (B, hidden_dim)  — DeepSets aggregation
        emb = self.psi(h)             # (B, embedding_dim)
        return F.normalize(emb, dim=-1)


def _prepare_dataset_tensor(
    X: np.ndarray,
    n_rows: int = N_SAMPLE_ROWS,
    n_cols: int = N_SAMPLE_COLS,
    seed: int = 42,
) -> torch.Tensor:
    """
    Sample and pad/truncate a numpy matrix to (1, n_rows, n_cols).
    Handles datasets with fewer rows/cols gracefully.
    """
    rng = np.random.default_rng(seed)

    nrows, ncols = X.shape
    # Column sampling / padding
    if ncols >= n_cols:
        col_idx = rng.choice(ncols, size=n_cols, replace=False)
    else:
        # Pad to n_cols with zeros
        pad = np.zeros((nrows, n_cols - ncols), dtype=np.float32)
        X = np.hstack([X, pad])
        col_idx = np.arange(n_cols)

    X_sub = X[:, col_idx]

    # Row sampling / padding
    if nrows >= n_rows:
        row_idx = rng.choice(nrows, size=n_rows, replace=False)
        X_sub = X_sub[row_idx]
    else:
        pad_r = np.zeros((n_rows - nrows, n_cols), dtype=np.float32)
        X_sub = np.vstack([X_sub, pad_r])

    # Normalize per-column (robust to outliers)
    X_sub = X_sub.astype(np.float32)
    col_std = X_sub.std(axis=0) + 1e-8
    col_mean = X_sub.mean(axis=0)
    X_sub = (X_sub - col_mean) / col_std

    # Replace NaN/Inf
    X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=3.0, neginf=-3.0)

    return torch.from_numpy(X_sub).unsqueeze(0)  # (1, n_rows, n_cols)


_embedder_cache: Optional[DatasetEmbedder] = None


def _get_embedder(device: str = "cpu") -> DatasetEmbedder:
    global _embedder_cache
    if _embedder_cache is None:
        model = DatasetEmbedder()
        if WEIGHTS_PATH.exists():
            state = torch.load(str(WEIGHTS_PATH), map_location=device)
            model.load_state_dict(state)
            logger.info(f"Loaded dataset embedder weights from {WEIGHTS_PATH}")
        else:
            logger.info(
                "Dataset embedder weights not found — using random initialization. "
                "Embeddings are structurally valid but not yet trained on meta-db."
            )
        model.eval()
        _embedder_cache = model.to(device)
    return _embedder_cache


def embed_dataset(
    X: np.ndarray,
    n_rows: int = N_SAMPLE_ROWS,
    n_cols: int = N_SAMPLE_COLS,
    device: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """
    Compute a learned embedding for a dataset matrix X.

    Args:
        X: (n_samples, n_features) numeric numpy array (already encoded)
        n_rows: number of rows to sample
        n_cols: number of columns to use (padded/truncated)
        device: 'cpu' or 'cuda'
        seed: random seed for reproducibility

    Returns:
        embedding: (EMBEDDING_DIM,) float32 numpy array
    """
    model = _get_embedder(device)
    tensor = _prepare_dataset_tensor(X, n_rows=n_rows, n_cols=n_cols, seed=seed)
    tensor = tensor.to(device)

    with torch.no_grad():
        emb = model(tensor)  # (1, EMBEDDING_DIM)

    return emb.squeeze(0).cpu().numpy().astype(np.float32)


def embed_dataset_from_df(
    df_X: "pd.DataFrame",  # type: ignore[name-defined]
    device: str = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """
    Convenience wrapper: accepts a pandas DataFrame (numeric only),
    returns (EMBEDDING_DIM,) embedding vector.
    """
    import pandas as pd

    # Encode categoricals numerically
    df_enc = df_X.copy()
    for col in df_enc.select_dtypes(exclude=[np.number]).columns:
        df_enc[col] = pd.factorize(df_enc[col])[0].astype(np.float32)

    X = df_enc.fillna(0).values.astype(np.float32)
    return embed_dataset(X, device=device, seed=seed)


def save_embedder_weights(model: DatasetEmbedder) -> None:
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(WEIGHTS_PATH))
    logger.info(f"Saved embedder weights to {WEIGHTS_PATH}")
    # Invalidate cache
    global _embedder_cache
    _embedder_cache = None


def fine_tune_embedder(
    dataset_tensors: List[torch.Tensor],
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> DatasetEmbedder:
    """
    Self-supervised fine-tuning via contrastive learning.
    Creates positive pairs by augmenting (two random samples from the same
    dataset) and trains using NT-Xent loss.
    This is called by the meta-knowledge update step after each experiment.
    """
    model = DatasetEmbedder().to(device)
    if WEIGHTS_PATH.exists():
        state = torch.load(str(WEIGHTS_PATH), map_location=device)
        model.load_state_dict(state)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for t in dataset_tensors:
            t = t.to(device)
            # Augment: two different row-samples from same dataset
            aug1 = _random_row_drop(t)
            aug2 = _random_row_drop(t)

            z1 = model(aug1)
            z2 = model(aug2)

            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Embedder fine-tune epoch {epoch+1}/{epochs} — loss={total_loss:.4f}")

    save_embedder_weights(model)
    return model


def _random_row_drop(t: torch.Tensor, drop_ratio: float = 0.2) -> torch.Tensor:
    """Augmentation: randomly zero out drop_ratio of rows."""
    t = t.clone()
    n_rows = t.shape[1]
    drop_n = max(1, int(n_rows * drop_ratio))
    drop_idx = torch.randperm(n_rows)[:drop_n]
    t[:, drop_idx, :] = 0.0
    return t


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Normalized temperature-scaled cross entropy loss (NT-Xent).
    Used for contrastive self-supervised embedding training.
    """
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.mm(z, z.T) / temperature  # (2B, 2B)
    B = z1.shape[0]
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])  # positive pair indices
    sim.fill_diagonal_(float("-inf"))
    return F.cross_entropy(sim, labels)
