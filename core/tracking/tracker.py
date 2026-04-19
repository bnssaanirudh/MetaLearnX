"""
MetaLearnX — MLflow Experiment Tracking Wrapper
Logs all MetaLearnX artifacts, metrics, configs, and embeddings.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
from loguru import logger

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///mlruns")
EXPERIMENT_NAME = "MetaLearnX"


def _setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            mlflow.create_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as e:
        logger.warning(f"MLflow setup warning: {e}")


def log_experiment(
    experiment_id: str,
    dataset_name: str,
    task_type: str,
    meta_features: Dict[str, float],
    embedding: Optional[List[float]],
    recommendations: List[Dict],
    best_config: Dict,
    best_metrics: Dict,
    pareto_front: List[Dict],
    all_trials: List[Dict],
    n_trials: int,
    runtime_seconds: float,
    explainability: Optional[Dict] = None,
) -> str:
    """
    Log a complete MetaLearnX experiment to MLflow.
    Returns the MLflow run_id.
    """
    _setup_mlflow()

    with mlflow.start_run(run_name=f"metax_{dataset_name}_{experiment_id[:8]}") as run:
        run_id = run.info.run_id

        # ── Tags ──────────────────────────────────────────────
        mlflow.set_tags({
            "experiment_id": experiment_id,
            "dataset_name": dataset_name,
            "task_type": task_type,
            "best_model": best_config.get("model_name", "unknown"),
            "framework": "MetaLearnX",
        })

        # ── Params ────────────────────────────────────────────
        flat_config = {
            f"pipeline_{k}": str(v)
            for k, v in best_config.items()
            if k != "model_params"
        }
        flat_params = {
            f"hp_{k}": str(v)
            for k, v in best_config.get("model_params", {}).items()
        }
        mlflow.log_params({**flat_config, **flat_params, "n_trials": n_trials})

        # ── Metrics ───────────────────────────────────────────
        mlflow.log_metrics({
            "primary_metric": round(best_metrics.get("primary_metric", 0), 6),
            "training_time_s": round(best_metrics.get("train_time", 0), 3),
            "model_size_bytes": best_metrics.get("model_size_bytes", 0),
            "total_runtime_s": runtime_seconds,
            "n_pareto_solutions": len(pareto_front),
            "n_similar_datasets": len(recommendations) if recommendations else 0,
        })

        # Log individual trial metrics
        for i, trial in enumerate(all_trials[:20]):
            mlflow.log_metric(
                "trial_primary_metric",
                trial.get("primary_metric", 0),
                step=trial.get("trial_number", i),
            )

        # ── Artifacts ─────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp:
            # Meta-features JSON
            mf_path = Path(tmp) / "meta_features.json"
            mf_path.write_text(json.dumps(meta_features, indent=2))
            mlflow.log_artifact(str(mf_path), "dataset_profile")

            # Embedding
            if embedding:
                emb_path = Path(tmp) / "embedding.json"
                emb_path.write_text(json.dumps(embedding))
                mlflow.log_artifact(str(emb_path), "dataset_profile")

            # Best config
            cfg_path = Path(tmp) / "best_config.json"
            cfg_path.write_text(json.dumps(best_config, indent=2))
            mlflow.log_artifact(str(cfg_path), "pipeline")

            # Pareto front
            if pareto_front:
                pareto_path = Path(tmp) / "pareto_front.json"
                pareto_path.write_text(json.dumps(pareto_front, indent=2))
                mlflow.log_artifact(str(pareto_path), "optimization")

            # All trials
            trials_path = Path(tmp) / "all_trials.json"
            trials_path.write_text(json.dumps(all_trials, indent=2))
            mlflow.log_artifact(str(trials_path), "optimization")

            # Recommendations
            if recommendations:
                rec_path = Path(tmp) / "recommendations.json"
                rec_path.write_text(json.dumps(recommendations, indent=2))
                mlflow.log_artifact(str(rec_path), "meta_learning")

            # Explainability
            if explainability:
                exp_path = Path(tmp) / "explainability.json"
                exp_path.write_text(json.dumps(explainability, indent=2))
                mlflow.log_artifact(str(exp_path), "explainability")

        logger.info(f"MLflow run logged: {run_id}")
        return run_id
