"""
MetaLearnX — Pareto Frontier Scatter Plotter
Generates publication-quality 2D scatter plots of the Pareto front
(Accuracy vs. CPU Training Time) for each optimization run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

PLOT_DIR = Path(__file__).parents[2] / "data" / "plots"


def plot_pareto_front(
    pareto_front: List[Dict],
    all_trials: List[Dict],
    dataset_name: str = "dataset",
    experiment_id: str = "exp",
    save_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Generate a 2D Pareto frontier scatter plot (Accuracy vs. Training Time).

    Parameters
    ----------
    pareto_front : list of dict
        Each dict must contain 'primary_metric' and 'train_time'.
    all_trials : list of dict
        All completed trial dicts for background plotting.
    dataset_name : str
        Used in the plot title and filename.
    experiment_id : str
        Used in the filename.
    save_dir : Path, optional
        Override default save directory.

    Returns
    -------
    str or None
        Absolute path to the saved PNG, or None on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping Pareto plot.")
        return None

    if not pareto_front:
        logger.info("No Pareto front data to plot.")
        return None

    out_dir = save_dir or PLOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Extract data ---
    all_acc = [t.get("primary_metric", 0) for t in all_trials if t.get("primary_metric", 0) > 0]
    all_time = [t.get("train_time", 0) for t in all_trials if t.get("primary_metric", 0) > 0]

    pareto_acc = [p.get("primary_metric", 0) for p in pareto_front]
    pareto_time = [p.get("train_time", 0) for p in pareto_front]

    # --- Build the figure ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Background: all trials
    if all_acc:
        ax.scatter(all_time, all_acc, c="#cbd5e1", s=30, alpha=0.5, label="All Trials", zorder=1)

    # Pareto front: highlighted
    ax.scatter(pareto_time, pareto_acc, c="#FF0080", s=70, edgecolors="#0f172a",
               linewidths=0.8, label="Pareto Front", zorder=2)

    # Connect Pareto points with a line (sorted by time)
    if len(pareto_time) > 1:
        sorted_idx = np.argsort(pareto_time)
        ax.plot(
            np.array(pareto_time)[sorted_idx],
            np.array(pareto_acc)[sorted_idx],
            color="#FF0080", linestyle="--", linewidth=1.2, alpha=0.6, zorder=1,
        )

    ax.set_xlabel("CPU Training Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Primary Metric (F1 / R²)", fontsize=11, fontweight="bold")
    ax.set_title(f"Pareto Frontier — {dataset_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.15)

    filename = f"pareto_{dataset_name}_{experiment_id}.png"
    filepath = out_dir / filename
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

    logger.info(f"Pareto scatter saved to {filepath}")
    return str(filepath)
