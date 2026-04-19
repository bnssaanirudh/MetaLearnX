"""
MetaLearnX — Dynamic Pipeline Builder
Constructs sklearn-compatible Pipeline objects from a config dict.
Handles imputation, scaling, encoding, feature selection, and the estimator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from models.classical.registry import get_model


def build_pipeline(
    config: Dict[str, Any],
    numeric_cols: List[str],
    categorical_cols: List[str],
    task_type: str = "classification",
) -> Pipeline:
    """
    Build a full sklearn Pipeline from a config dictionary.

    Config schema:
        imputer:         "mean" | "median" | "most_frequent" | "constant"
        scaler:          "standard" | "robust" | "minmax" | "none"
        encoder:         "ordinal" | "onehot" | "target" | "none"
        feature_selector: "kbest" | "from_model" | "none"
        selector_k:      int (for kbest)
        model_name:      str (key from model registry)
        model_params:    dict
    """
    # ── Numeric transformer ───────────────────────────────────
    imputer = _build_imputer(config.get("imputer", "median"))
    scaler = _build_scaler(config.get("scaler", "standard"))
    numeric_steps = [("imputer", imputer), ("scaler", scaler)]

    selector = _build_selector(
        config.get("feature_selector", "none"),
        k=config.get("selector_k", 20),
        task_type=task_type,
    )
    if selector is not None:
        numeric_steps.append(("selector", selector))

    numeric_transformer = Pipeline(steps=numeric_steps)

    # ── Categorical transformer ───────────────────────────────
    cat_imputer = SimpleImputer(strategy="most_frequent")
    encoder = _build_encoder(config.get("encoder", "ordinal"))
    categorical_transformer = Pipeline(
        steps=[("cat_imputer", cat_imputer), ("encoder", encoder)]
    )

    # ── Column transformer ────────────────────────────────────
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError("No numeric or categorical columns found.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # ── Estimator ─────────────────────────────────────────────
    model_name = config.get("model_name", "random_forest")
    model_params = config.get("model_params", {})
    estimator = get_model(model_name, task_type=task_type, **model_params)

    # ── Full pipeline ─────────────────────────────────────────
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    return pipeline


def get_column_types(df: "pd.DataFrame", target_col: str) -> Tuple[List[str], List[str]]:  # type: ignore
    """Return (numeric_cols, categorical_cols) for a dataframe (excluding target)."""
    import pandas as pd

    X = df.drop(columns=[target_col])
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def config_to_description(config: Dict[str, Any]) -> str:
    """Human-readable pipeline description."""
    parts = [
        f"Imputer: {config.get('imputer', 'median')}",
        f"Scaler: {config.get('scaler', 'standard')}",
        f"Encoder: {config.get('encoder', 'ordinal')}",
        f"Selector: {config.get('feature_selector', 'none')}",
        f"Model: {config.get('model_name', 'random_forest')}",
    ]
    return " | ".join(parts)


# ─────────────────────────────────────────────
# Builder helpers
# ─────────────────────────────────────────────

def _build_imputer(strategy: str) -> SimpleImputer:
    valid = {"mean", "median", "most_frequent", "constant"}
    if strategy not in valid:
        strategy = "median"
    return SimpleImputer(strategy=strategy)


def _build_scaler(name: str):
    return {
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "minmax": MinMaxScaler(),
        "none": "passthrough",
    }.get(name, StandardScaler())


def _build_encoder(name: str):
    return {
        "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        "onehot": "passthrough",   # handled manually for high-cardinality
        "target": "passthrough",   # TargetEncoder requires y; simplified for v1
        "none": "passthrough",
    }.get(name, OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))


def _build_selector(name: str, k: int = 20, task_type: str = "classification"):
    if name == "none":
        return None
    if name == "kbest":
        score_func = f_classif if task_type == "classification" else f_regression
        return SelectKBest(score_func=score_func, k=min(k, 50))
    if name == "from_model":
        estimator = LogisticRegression(max_iter=200, solver="liblinear")
        return SelectFromModel(estimator=estimator, threshold="mean")
    return None
