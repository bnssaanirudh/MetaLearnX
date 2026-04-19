"""
MetaLearnX — Synthetic Drift Injection Module
Simulates real-world IoT sensor failure and covariate shift by injecting
Gaussian noise and random missingness into validation data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


def inject_drift(
    df: pd.DataFrame,
    noise_frac: float = 0.10,
    missing_frac: float = 0.05,
    columns: Optional[list] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Inject synthetic distribution drift into a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The clean input dataframe (typically a validation split).
    noise_frac : float
        Multiplicative factor for Gaussian noise σ = column_std × noise_frac.
    missing_frac : float
        Fraction of cells (per numeric column) to replace with NaN.
    columns : list, optional
        Restrict injection to these columns. If None, all numeric columns.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A copy of the input with noise and missingness injected.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()
    num_cols = columns or list(out.select_dtypes(include=[np.number]).columns)

    if not num_cols:
        logger.warning("drift_injector: No numeric columns found — skipping.")
        return out

    n_rows = len(out)
    total_noise_cells = 0
    total_missing_cells = 0

    for col in num_cols:
        col_std = out[col].std()
        if pd.isna(col_std) or col_std == 0:
            continue

        # --- Gaussian noise ---
        sigma = col_std * noise_frac
        noise = rng.normal(loc=0, scale=sigma, size=n_rows)
        out[col] = out[col] + noise
        total_noise_cells += n_rows

        # --- Random missingness ---
        n_missing = max(1, int(n_rows * missing_frac))
        missing_idx = rng.choice(n_rows, size=n_missing, replace=False)
        out.iloc[missing_idx, out.columns.get_loc(col)] = np.nan
        total_missing_cells += n_missing

    logger.info(
        f"drift_injector: Injected noise into {total_noise_cells} cells "
        f"and {total_missing_cells} NaN cells across {len(num_cols)} columns."
    )
    return out
