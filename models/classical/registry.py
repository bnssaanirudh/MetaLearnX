"""
MetaLearnX — Model Registry
Centralized registry of all model families with sklearn-compatible interfaces.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "svm",
    "mlp",
    "tabpfn",  # optional foundation model
]


def get_model(name: str, task_type: str = "classification", **kwargs):
    """
    Instantiate a model by name.
    Raises ValueError for unknown models.
    """
    name = name.lower().strip()

    if name == "logistic_regression":
        if task_type == "classification":
            return LogisticRegression(
                max_iter=kwargs.get("max_iter", 500),
                C=kwargs.get("C", 1.0),
                solver=kwargs.get("solver", "lbfgs"),
                multi_class="auto",
            )
        return Ridge(alpha=kwargs.get("alpha", 1.0))

    if name == "random_forest":
        cls = RandomForestClassifier if task_type == "classification" else RandomForestRegressor
        return cls(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1,
        )

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor

            cls = XGBClassifier if task_type == "classification" else XGBRegressor
            return cls(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.1),
                subsample=kwargs.get("subsample", 0.8),
                colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                eval_metric="logloss" if task_type == "classification" else "rmse",
                random_state=kwargs.get("random_state", 42),
                verbosity=0,
            )
        except ImportError:
            return _fallback_rf(task_type, **kwargs)

    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor

            cls = LGBMClassifier if task_type == "classification" else LGBMRegressor
            return cls(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", -1),
                learning_rate=kwargs.get("learning_rate", 0.05),
                num_leaves=kwargs.get("num_leaves", 31),
                subsample=kwargs.get("subsample", 0.8),
                random_state=kwargs.get("random_state", 42),
                verbose=-1,
            )
        except ImportError:
            return _fallback_rf(task_type, **kwargs)

    if name == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            cls = CatBoostClassifier if task_type == "classification" else CatBoostRegressor
            return cls(
                iterations=kwargs.get("iterations", 200),
                depth=kwargs.get("depth", 6),
                learning_rate=kwargs.get("learning_rate", 0.05),
                random_seed=kwargs.get("random_state", 42),
                verbose=0,
            )
        except ImportError:
            return _fallback_rf(task_type, **kwargs)

    if name == "svm":
        if task_type == "classification":
            return SVC(
                C=kwargs.get("C", 1.0),
                kernel=kwargs.get("kernel", "rbf"),
                probability=True,
            )
        return SVR(
            C=kwargs.get("C", 1.0),
            kernel=kwargs.get("kernel", "rbf"),
        )

    if name == "mlp":
        layer_size = kwargs.get("hidden_layer_sizes", (100, 50))
        if task_type == "classification":
            return MLPClassifier(
                hidden_layer_sizes=layer_size,
                max_iter=kwargs.get("max_iter", 200),
                random_state=kwargs.get("random_state", 42),
            )
        return MLPRegressor(
            hidden_layer_sizes=layer_size,
            max_iter=kwargs.get("max_iter", 200),
            random_state=kwargs.get("random_state", 42),
        )

    if name == "tabpfn":
        try:
            from models.foundation_tabular.tabpfn_wrapper import TabPFNWrapper
            return TabPFNWrapper(task_type=task_type)
        except Exception:
            # Graceful degradation
            return _fallback_rf(task_type, **kwargs)

    raise ValueError(f"Unknown model name: '{name}'. Available: {MODEL_NAMES}")


def _fallback_rf(task_type: str, **kwargs):
    cls = RandomForestClassifier if task_type == "classification" else RandomForestRegressor
    return cls(n_estimators=100, random_state=42, n_jobs=-1)


def get_model_display_name(name: str) -> str:
    return {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "svm": "Support Vector Machine",
        "mlp": "Neural Network (MLP)",
        "tabpfn": "TabPFN (Foundation Model)",
    }.get(name, name)
