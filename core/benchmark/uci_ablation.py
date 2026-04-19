"""
MetaLearnX — UCI Repository Ablation Harness
Downloads lightweight UCI / OpenML datasets and benchmarks MetaLearnX
against a pure Random-Search baseline to prove LLM warm-start lift.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# 15 lightweight UCI / OpenML classification datasets
UCI_DATASETS = [
    {"openml_id": 31,   "name": "credit-g"},
    {"openml_id": 37,   "name": "diabetes"},
    {"openml_id": 44,   "name": "spambase"},
    {"openml_id": 50,   "name": "tic-tac-toe"},
    {"openml_id": 54,   "name": "vehicle"},
    {"openml_id": 151,  "name": "electricity"},
    {"openml_id": 1462, "name": "banknote-auth"},
    {"openml_id": 1464, "name": "blood-transfusion"},
    {"openml_id": 1471, "name": "eeg-eye-state"},
    {"openml_id": 1489, "name": "phoneme"},
    {"openml_id": 1494, "name": "qsar-biodeg"},
    {"openml_id": 1510, "name": "wdbc"},
    {"openml_id": 6,    "name": "letter"},
    {"openml_id": 23,   "name": "cmc"},
    {"openml_id": 28,   "name": "optdigits"},
]

RESULTS_DIR = Path(__file__).parents[2] / "data" / "benchmark_results"


def _random_search_baseline(X: pd.DataFrame, y: pd.Series, n_iter: int = 20, seed: int = 42) -> float:
    """Simple random-search RandomForest baseline."""
    rng = np.random.default_rng(seed)
    best_score = 0.0
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    for _ in range(n_iter):
        clf = RandomForestClassifier(
            n_estimators=int(rng.integers(50, 500)),
            max_depth=int(rng.integers(3, 20)),
            min_samples_split=int(rng.integers(2, 20)),
            random_state=seed,
            n_jobs=-1,
        )
        scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score

    return round(best_score, 4)


def run_ablation(max_datasets: int = 15) -> List[Dict]:
    """
    Run the full UCI ablation benchmark.

    Returns a list of result dicts and writes them to a CSV file.
    """
    results = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for entry in UCI_DATASETS[:max_datasets]:
        ds_name = entry["name"]
        openml_id = entry["openml_id"]
        logger.info(f"[UCI Ablation] Loading {ds_name} (OpenML #{openml_id})...")

        try:
            data = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")
            df = data.data.copy()
            target = data.target.copy()

            # Basic cleanup
            df = df.select_dtypes(include=[np.number]).fillna(0)
            if df.empty or len(df) < 30:
                logger.warning(f"  Skipping {ds_name}: too few numeric features/samples.")
                continue

            y = pd.factorize(target)[0]

            # --- Random Search Baseline ---
            t0 = time.time()
            baseline_f1 = _random_search_baseline(df, y)
            baseline_time = round(time.time() - t0, 2)

            # --- MetaLearnX Pipeline ---
            metax_f1 = 0.0
            metax_time = 0.0
            try:
                from core.orchestrator import run_full_pipeline
                full_df = df.copy()
                full_df["__target__"] = target

                t1 = time.time()
                res = run_full_pipeline(
                    df=full_df,
                    target_col="__target__",
                    task_type="classification",
                    dataset_name=ds_name,
                    n_trials=15,
                    max_agent_iterations=1,
                )
                metax_time = round(time.time() - t1, 2)
                metax_f1 = round(res.get("best_metrics", {}).get("primary_metric", 0), 4)
            except Exception as e:
                logger.error(f"  MetaLearnX pipeline failed on {ds_name}: {e}")

            lift = round(metax_f1 - baseline_f1, 4)

            result = {
                "dataset": ds_name,
                "openml_id": openml_id,
                "n_samples": len(df),
                "n_features": df.shape[1],
                "baseline_f1": baseline_f1,
                "baseline_time_s": baseline_time,
                "metalearnx_f1": metax_f1,
                "metalearnx_time_s": metax_time,
                "lift": lift,
            }
            results.append(result)
            logger.info(f"  {ds_name}: Baseline={baseline_f1}, MetaLearnX={metax_f1}, Lift={lift}")

        except Exception as e:
            logger.error(f"  Failed to load {ds_name}: {e}")

    # Write CSV
    if results:
        csv_path = RESULTS_DIR / "uci_ablation_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Ablation results saved to {csv_path}")

    return results


if __name__ == "__main__":
    run_ablation()
