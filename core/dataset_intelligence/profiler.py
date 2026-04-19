"""
MetaLearnX — Dataset Intelligence Profiler
Extracts 30+ meta-features plus data quality alerts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


@dataclass
class DatasetProfile:
    # ── Identity ──────────────────────────────────────────────
    name: str = ""
    task_type: str = "classification"  # 'classification' | 'regression'
    n_samples: int = 0
    n_features: int = 0

    # ── Feature types ─────────────────────────────────────────
    n_numeric: int = 0
    n_categorical: int = 0
    n_binary: int = 0
    numeric_ratio: float = 0.0
    categorical_ratio: float = 0.0

    # ── Missingness ───────────────────────────────────────────
    missing_rate_mean: float = 0.0
    missing_rate_max: float = 0.0
    n_cols_with_missing: int = 0
    has_missing: bool = False

    # ── Target ────────────────────────────────────────────────
    n_classes: int = 0
    class_imbalance_ratio: float = 0.0   # minority/majority
    target_entropy: float = 0.0
    target_mean: float = 0.0             # regression only
    target_std: float = 0.0              # regression only

    # ── Statistical moments ───────────────────────────────────
    skewness_mean: float = 0.0
    skewness_std: float = 0.0
    kurtosis_mean: float = 0.0
    kurtosis_std: float = 0.0

    # ── Correlation ───────────────────────────────────────────
    corr_mean: float = 0.0
    corr_max: float = 0.0
    corr_std: float = 0.0

    # ── Cardinality ───────────────────────────────────────────
    cardinality_mean: float = 0.0
    cardinality_max: int = 0
    high_cardinality_ratio: float = 0.0  # cols with cardinality > 50

    # ── Advanced Metrics (TDA & Info Theory) ──────────────────
    mutual_information_max: float = 0.0
    mutual_information_mean: float = 0.0
    tda_betti_zero_proxy: int = 0        # Proxy for connected components

    # ── Sparsity / Scale ──────────────────────────────────────
    sparsity: float = 0.0                # fraction of zeros
    feature_scale_range: float = 0.0    # mean(max-min) of numeric cols

    # ── Quality indicators ────────────────────────────────────
    duplicate_row_rate: float = 0.0
    constant_feature_count: int = 0
    outlier_rate: float = 0.0           # IQR-based
    near_zero_var_count: int = 0

    # ── Complexity score ──────────────────────────────────────
    complexity_score: float = 0.0       # 0-1, drives optimization budget

    # ── Quality alerts ────────────────────────────────────────
    alerts: List[str] = field(default_factory=list)
    leakage_suspects: List[str] = field(default_factory=list)

    def to_meta_feature_dict(self) -> Dict[str, float]:
        """Return numeric-only dict for meta-learning vectors."""
        exclude = {"name", "task_type", "alerts", "leakage_suspects"}
        return {
            k: float(v)
            for k, v in vars(self).items()
            if k not in exclude and isinstance(v, (int, float, bool))
        }

    def to_dict(self) -> Dict[str, Any]:
        d = vars(self).copy()
        return d


def profile_dataset(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
    name: str = "dataset",
) -> DatasetProfile:
    """
    Full dataset profiling. Returns a DatasetProfile with 30+ meta-features
    and annotated quality alerts.
    """
    t0 = time.time()
    p = DatasetProfile(name=name, task_type=task_type)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ── Identity ──────────────────────────────────────────────
    p.n_samples, p.n_features = X.shape

    # ── Feature types ─────────────────────────────────────────
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    bin_cols = [c for c in num_cols if X[c].nunique() == 2]
    p.n_numeric = len(num_cols)
    p.n_categorical = len(cat_cols)
    p.n_binary = len(bin_cols)
    p.numeric_ratio = len(num_cols) / max(p.n_features, 1)
    p.categorical_ratio = len(cat_cols) / max(p.n_features, 1)

    # ── Missingness ───────────────────────────────────────────
    miss_rates = X.isnull().mean()
    p.missing_rate_mean = float(miss_rates.mean())
    p.missing_rate_max = float(miss_rates.max())
    p.n_cols_with_missing = int((miss_rates > 0).sum())
    p.has_missing = p.missing_rate_mean > 0

    # ── Target analysis ───────────────────────────────────────
    if task_type == "classification":
        vc = y.value_counts()
        p.n_classes = int(y.nunique())
        p.class_imbalance_ratio = float(vc.min() / vc.max()) if len(vc) > 1 else 1.0
        probs = vc / len(y)
        p.target_entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        if p.class_imbalance_ratio < 0.3:
            p.alerts.append(
                f"⚠️ Class imbalance detected: minority/majority ratio = {p.class_imbalance_ratio:.2f}"
            )
    else:
        p.n_classes = 0
        p.target_mean = float(y.mean())
        p.target_std = float(y.std())

    # ── Statistical moments (numeric cols only) ───────────────
    if num_cols:
        X_num = X[num_cols].fillna(X[num_cols].median())
        sk = X_num.apply(lambda c: stats.skew(c.dropna()))
        ku = X_num.apply(lambda c: stats.kurtosis(c.dropna()))
        p.skewness_mean = float(sk.mean())
        p.skewness_std = float(sk.std())
        p.kurtosis_mean = float(ku.mean())
        p.kurtosis_std = float(ku.std())
        if abs(p.skewness_mean) > 2.0:
            p.alerts.append(
                f"⚠️ High average feature skewness ({p.skewness_mean:.2f}) — consider log transform."
            )

    # ── Correlation ───────────────────────────────────────────
    if len(num_cols) >= 2:
        corr = X[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        vals = upper.stack().values
        p.corr_mean = float(vals.mean()) if len(vals) else 0.0
        p.corr_max = float(vals.max()) if len(vals) else 0.0
        p.corr_std = float(vals.std()) if len(vals) else 0.0

    # ── Cardinality ───────────────────────────────────────────
    cardinalities = X.nunique()
    p.cardinality_mean = float(cardinalities.mean())
    p.cardinality_max = int(cardinalities.max())
    high_card = (cardinalities > 50).sum()
    p.high_cardinality_ratio = float(high_card / max(p.n_features, 1))
    if p.high_cardinality_ratio > 0.2:
        p.alerts.append(
            f"⚠️ {high_card} high-cardinality categorical features (>50 unique). "
            "Consider target encoding or hashing."
        )

    # ── Sparsity / Scale ──────────────────────────────────────
    p.sparsity = float((X == 0).sum().sum() / max(X.size, 1))
    if num_cols:
        ranges = X[num_cols].max() - X[num_cols].min()
        p.feature_scale_range = float(ranges.mean())

    # ── Advanced Metrics (Info Theory & TDA) ──────────────────
    try:
        sample_X = X.select_dtypes(include=[np.number]).fillna(0)
        if not sample_X.empty and len(sample_X) > 10:
            sub_X = sample_X.sample(n=min(len(sample_X), 1000), random_state=42)
            sub_y = y.loc[sub_X.index]
            if task_type == 'classification':
                mi_scores = mutual_info_classif(sub_X, pd.factorize(sub_y)[0])
            else:
                mi_scores = mutual_info_regression(sub_X, sub_y)
            p.mutual_information_max = float(mi_scores.max())
            p.mutual_information_mean = float(mi_scores.mean())
            
            # Simple TDA proxy: count dense regions (Betti-0 proxy via thresholding)
            dist_mat = np.linalg.norm(sub_X.values[:, None, :] - sub_X.values[None, :, :], axis=2)
            np.fill_diagonal(dist_mat, np.inf)
            eps = np.percentile(dist_mat, 5)
            p.tda_betti_zero_proxy = int(np.sum(np.min(dist_mat, axis=1) > eps))
    except Exception as e:
        logger.debug(f"Advanced metrics failed: {e}")

    # ── Quality indicators ────────────────────────────────────
    p.duplicate_row_rate = float(X.duplicated().sum() / max(p.n_samples, 1))
    if p.duplicate_row_rate > 0.05:
        p.alerts.append(
            f"⚠️ {p.duplicate_row_rate*100:.1f}% duplicate rows detected."
        )

    const_cols = [c for c in X.columns if X[c].nunique() <= 1]
    p.constant_feature_count = len(const_cols)
    if const_cols:
        p.alerts.append(f"⚠️ {len(const_cols)} constant features: {const_cols[:3]}...")

    near_zero = [
        c for c in num_cols
        if X[c].std() < 1e-4 and X[c].std() != 0
    ]
    p.near_zero_var_count = len(near_zero)

    # ── Outlier rate (IQR) ────────────────────────────────────
    if num_cols:
        X_num = X[num_cols]
        Q1 = X_num.quantile(0.25)
        Q3 = X_num.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((X_num < Q1 - 1.5 * IQR) | (X_num > Q3 + 1.5 * IQR))
        p.outlier_rate = float(outlier_mask.any(axis=1).mean())
        if p.outlier_rate > 0.15:
            p.alerts.append(
                f"⚠️ {p.outlier_rate*100:.1f}% of rows contain outliers (IQR rule)."
            )

    # ── Leakage hints ─────────────────────────────────────────
    if task_type == "classification" and num_cols:
        y_enc = pd.factorize(y)[0]
        for c in num_cols:
            try:
                corr_with_y = abs(np.corrcoef(X[c].fillna(0), y_enc)[0, 1])
                if corr_with_y > 0.95:
                    p.leakage_suspects.append(c)
                    p.alerts.append(
                        f"🚨 Possible leakage: '{c}' has {corr_with_y:.3f} correlation with target."
                    )
            except Exception:
                pass

    # ── Complexity score ──────────────────────────────────────
    # Combines several difficulty signals into a single 0-1 score
    dim_score = min(p.n_features / 500, 1.0) * 0.2
    size_score = min(p.n_samples / 100_000, 1.0) * 0.2
    miss_score = min(p.missing_rate_mean * 5, 1.0) * 0.15
    imb_score = (1 - p.class_imbalance_ratio) * 0.15 if task_type == "classification" else 0.0
    card_score = p.high_cardinality_ratio * 0.15
    alert_score = min(len(p.alerts) / 5, 1.0) * 0.15
    p.complexity_score = round(
        dim_score + size_score + miss_score + imb_score + card_score + alert_score, 4
    )

    elapsed = round(time.time() - t0, 3)
    logger.info(
        f"Profiled '{name}': {p.n_samples}×{p.n_features}, "
        f"complexity={p.complexity_score:.2f}, {len(p.alerts)} alerts, {elapsed}s"
    )
    return p
