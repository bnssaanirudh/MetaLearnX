"""
MetaLearnX — Handcrafted Meta-Feature Extractor
Converts a DatasetProfile into a normalized numeric vector for meta-learning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.dataset_intelligence.profiler import DatasetProfile, profile_dataset

# Ordered list of meta-feature keys used for the fixed-length vector.
# Order MUST stay stable across versions (add new ones at the end only).
META_FEATURE_KEYS: List[str] = [
    "n_samples",
    "n_features",
    "n_numeric",
    "n_categorical",
    "n_binary",
    "numeric_ratio",
    "categorical_ratio",
    "missing_rate_mean",
    "missing_rate_max",
    "n_cols_with_missing",
    "has_missing",
    "n_classes",
    "class_imbalance_ratio",
    "target_entropy",
    "target_mean",
    "target_std",
    "skewness_mean",
    "skewness_std",
    "kurtosis_mean",
    "kurtosis_std",
    "corr_mean",
    "corr_max",
    "corr_std",
    "cardinality_mean",
    "cardinality_max",
    "high_cardinality_ratio",
    "sparsity",
    "feature_scale_range",
    "duplicate_row_rate",
    "constant_feature_count",
    "outlier_rate",
    "near_zero_var_count",
    "complexity_score",
]

META_FEATURE_DIM = len(META_FEATURE_KEYS)


def extract_meta_features(profile: DatasetProfile) -> np.ndarray:
    """
    Convert a DatasetProfile into a fixed-length numeric vector.
    Returns shape (META_FEATURE_DIM,) float32 array.
    Missing keys default to 0.
    """
    mf_dict = profile.to_meta_feature_dict()
    vec = np.array(
        [float(mf_dict.get(k, 0.0)) for k in META_FEATURE_KEYS],
        dtype=np.float32,
    )
    # Replace NaNs / Infs with 0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def meta_features_to_dict(vec: np.ndarray) -> Dict[str, float]:
    """Reconstruct a human-readable dict from a meta-feature vector."""
    return {k: float(v) for k, v in zip(META_FEATURE_KEYS, vec)}


def log_transform_features(vec: np.ndarray) -> np.ndarray:
    """
    Apply log1p to high-range features (n_samples, n_features, etc.)
    to improve meta-learning model performance.
    Log-transformed indices: n_samples=0, n_features=1, n_numeric=2,
    n_categorical=3, cardinality_max=24, feature_scale_range=26.
    """
    log_indices = [0, 1, 2, 3, 24, 27]
    result = vec.copy()
    for idx in log_indices:
        if idx < len(result):
            result[idx] = np.log1p(max(result[idx], 0))
    return result


def extract_and_vectorize(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
    name: str = "dataset",
    apply_log_transform: bool = True,
) -> Tuple[DatasetProfile, np.ndarray]:
    """
    End-to-end convenience: profile → meta-feature vector.
    Returns (profile, vector).
    """
    profile = profile_dataset(df, target_col, task_type=task_type, name=name)
    vec = extract_meta_features(profile)
    if apply_log_transform:
        vec = log_transform_features(vec)
    return profile, vec
