"""
MetaLearnX — TabPFN Foundation Model Wrapper
Wraps TabPFN with sklearn-compatible interface and graceful fallback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class TabPFNWrapper:
    """
    Scikit-learn compatible wrapper for TabPFN.
    Gracefully falls back to RandomForest if TabPFN is not installed.
    """

    def __init__(self, task_type: str = "classification", device: str = "cpu") -> None:
        self.task_type = task_type
        self.device = device
        self._model = None
        self._available = False
        self._classes_ = None

        try:
            if task_type == "classification":
                from tabpfn import TabPFNClassifier
                self._model = TabPFNClassifier(device=device)
            else:
                # TabPFN v2 supports regression
                try:
                    from tabpfn import TabPFNRegressor
                    self._model = TabPFNRegressor(device=device)
                except ImportError:
                    raise ImportError("TabPFN regressor not available")
            self._available = True
        except (ImportError, Exception):
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if task_type == "classification":
                self._model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                self._model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self._available = False

    @property
    def is_foundation_model(self) -> bool:
        return self._available

    @property
    def model_description(self) -> str:
        if self._available:
            return "TabPFN (Prior-Fitted Network — tabular foundation model)"
        return "Random Forest (TabPFN fallback)"

    def fit(self, X, y):
        if self.task_type == "classification":
            self._classes_ = np.unique(y)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        # Regression fallback — return dummy proba
        preds = self.predict(X)
        return np.column_stack([1 - preds, preds])

    def get_params(self, deep: bool = True):
        return {"task_type": self.task_type, "device": self.device}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
