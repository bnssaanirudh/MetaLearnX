"""
MetaLearnX — Multi-Objective Optimization Engine with Optuna
Features: Pareto optimization, warm-start from similar datasets, budget-aware search.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score
)

from core.pipeline_builder.builder import build_pipeline, get_column_types

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────
# Search space definitions
# ─────────────────────────────────────────────

def _suggest_config(
    trial: optuna.Trial, 
    task_type: str, 
    llm_bounds: Optional[Dict] = None,
    forced_model: Optional[str] = None,
    forced_pipeline_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Define the full hyperparameter search space."""
    
    # Model Selection
    if forced_model:
        model_name = trial.suggest_categorical("model_name", [forced_model])
    else:
        model_name = trial.suggest_categorical(
            "model_name",
            ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost", "mlp"],
        )
    
    # Architectural Selection (Agent-Control vs Search)
    if forced_pipeline_params:
        p = forced_pipeline_params
        imputer = trial.suggest_categorical("imputer", [p.get("imputer", "median")])
        scaler = trial.suggest_categorical("scaler", [p.get("scaler", "standard")])
        encoder = trial.suggest_categorical("encoder", [p.get("encoder", "ordinal")])
        feature_selector = trial.suggest_categorical("feature_selector", [p.get("feature_selector", "none")])
    else:
        imputer = trial.suggest_categorical("imputer", ["mean", "median", "most_frequent"])
        scaler = trial.suggest_categorical("scaler", ["standard", "robust", "minmax", "none"])
        encoder = trial.suggest_categorical("encoder", ["ordinal", "none"])
        feature_selector = trial.suggest_categorical("feature_selector", ["none", "kbest"])

    config: Dict[str, Any] = {
        "model_name": model_name,
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
        "feature_selector": feature_selector,
        "model_params": {},
    }

    if feature_selector == "kbest":
        config["selector_k"] = trial.suggest_int("selector_k", 5, 50)

    if model_name == "logistic_regression":
        if task_type == "classification":
            config["model_params"] = {
                "C": trial.suggest_float("lr_C", 1e-3, 100, log=True),
                "max_iter": 500,
            }
        else:
            config["model_params"] = {
                "alpha": trial.suggest_float("ridge_alpha", 1e-3, 100, log=True),
            }

    elif model_name == "random_forest":
        if llm_bounds and "random_forest" in llm_bounds:
            bounds = llm_bounds["random_forest"]
            n_min, n_max = int(bounds["n_estimators"][0]), int(bounds["n_estimators"][1])
            d_min, d_max = int(bounds["max_depth"][0]), int(bounds["max_depth"][1])
            config["model_params"] = {
                "n_estimators": trial.suggest_int("rf_n_estimators", n_min, n_max),
                "max_depth": trial.suggest_int("rf_max_depth", d_min, d_max),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
            }
        else:
            config["model_params"] = {
                "n_estimators": trial.suggest_int("rf_n_estimators", 50, 500),
                "max_depth": trial.suggest_int("rf_max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 20),
            }

    elif model_name == "xgboost":
        if llm_bounds and "xgboost" in llm_bounds:
            bounds = llm_bounds["xgboost"]
            n_min, n_max = int(bounds["n_estimators"][0]), int(bounds["n_estimators"][1])
            d_min, d_max = int(bounds["max_depth"][0]), int(bounds["max_depth"][1])
            lr_min, lr_max = float(bounds["learning_rate"][0]), float(bounds["learning_rate"][1])
            config["model_params"] = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", n_min, n_max),
                "max_depth": trial.suggest_int("xgb_max_depth", d_min, d_max),
                "learning_rate": trial.suggest_float("xgb_lr", lr_min, lr_max, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            }
        else:
            config["model_params"] = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 500),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
                "learning_rate": trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            }

    elif model_name == "lightgbm":
        config["model_params"] = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 50, 500),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 150),
            "learning_rate": trial.suggest_float("lgb_lr", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
        }

    elif model_name == "catboost":
        config["model_params"] = {
            "iterations": trial.suggest_int("cb_iterations", 50, 400),
            "depth": trial.suggest_int("cb_depth", 3, 10),
            "learning_rate": trial.suggest_float("cb_lr", 1e-3, 0.3, log=True),
        }

    elif model_name == "mlp":
        n_layers = trial.suggest_int("mlp_n_layers", 1, 3)
        layer_size = trial.suggest_int("mlp_layer_size", 32, 256)
        config["model_params"] = {
            "hidden_layer_sizes": tuple([layer_size] * n_layers),
            "max_iter": 200,
        }

    return config


# ─────────────────────────────────────────────
# Budget policy
# ─────────────────────────────────────────────

def compute_budget(complexity_score: float, base_trials: int = 30) -> int:
    """
    Adaptive trial budget based on dataset complexity.
    complexity_score: 0.0 (simple) → 1.0 (hard)
    """
    multiplier = 1.0 + complexity_score * 2.0   # 1x to 3x
    n_trials = int(base_trials * multiplier)
    n_trials = max(10, min(n_trials, 150))
    logger.info(f"Adaptive budget: complexity={complexity_score:.2f} → {n_trials} trials")
    return n_trials


def _compute_search_volume(llm_bounds: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Quantify the hyperparameter search-space volume reduction
    achieved by the LLM warm-start relative to the unconstrained defaults.

    Returns dict with V_original, V_reduced, and the ratio.
    """
    # Default unconstrained ranges (from _suggest_config)
    defaults = {
        "random_forest": {"n_estimators": (50, 500), "max_depth": (3, 20)},
        "xgboost": {"n_estimators": (50, 500), "max_depth": (3, 10), "learning_rate": (1e-3, 0.3)},
        "lightgbm": {"n_estimators": (50, 500), "num_leaves": (15, 150), "learning_rate": (1e-3, 0.3)},
    }

    v_original = 1.0
    v_reduced = 1.0
    matched_models = []

    for model, params in defaults.items():
        for param, (lo, hi) in params.items():
            v_original *= (hi - lo)
            if llm_bounds and model in llm_bounds and param in llm_bounds[model]:
                blo, bhi = float(llm_bounds[model][param][0]), float(llm_bounds[model][param][1])
                v_reduced *= max(bhi - blo, 1e-9)
                if model not in matched_models:
                    matched_models.append(model)
            else:
                v_reduced *= (hi - lo)

    ratio = round(v_reduced / max(v_original, 1e-12), 6)
    logger.info(f"Search-space volume: V_original={v_original:.0f}, V_reduced={v_reduced:.0f}, ratio={ratio}")
    return {
        "v_original": v_original,
        "v_reduced": v_reduced,
        "reduction_ratio": ratio,
        "models_constrained": matched_models,
    }


# ─────────────────────────────────────────────
# Main optimization engine
# ─────────────────────────────────────────────

def run_optimization(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
    complexity_score: float = 0.3,
    warm_start_configs: Optional[List[Dict]] = None,
    multi_objective: bool = True,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, Dict], None]] = None,
    sampler: str = "tpe",
    meta_feature_vec: Optional[np.ndarray] = None,
    use_search_space_pruning: bool = True,
    dataset_profile_dict: Optional[Dict] = None,
    dataset_description: Optional[str] = None,
    forced_model: Optional[str] = None,
    forced_pipeline_params: Optional[Dict] = None,
    max_inference_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization.

    Returns:
        {
            "best_config": dict,
            "best_metrics": dict,
            "pareto_front": list[dict],  # multi-obj only
            "all_trials": list[dict],
            "n_trials_run": int,
            "runtime_seconds": float,
        }
    """
    t0 = time.time()
    numeric_cols, categorical_cols = get_column_types(df, target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    if task_type == "classification":
        y = pd.factorize(y)[0]

    actual_n_trials = n_trials or compute_budget(complexity_score)

    # ── Search space pruning ──────────────────────────────────
    pruner_info = {}
    if use_search_space_pruning and meta_feature_vec is not None:
        try:
            from core.optimization.search_space_learner import get_pruner
            pruner = get_pruner()
            pruner_info = pruner.get_reduction_stats()
            logger.info(f"Search space pruning active for {len(pruner_info)} model families")
        except Exception as e:
            logger.warning(f"Search space pruner unavailable: {e}")

    # ── LLM Search Space ──────────────────────────────────────
    llm_bounds = None
    if dataset_profile_dict:
        from core.meta_learning.llm_provider import get_llm_provider
        provider = get_llm_provider()
        llm_bounds = provider.synthesize_search_space(dataset_profile_dict, description=dataset_description)
        if llm_bounds:
            logger.info(f"LLM Search bounds applied: {llm_bounds.get('llm_reasoning', '')[:100]}...")

    # ── Objective ─────────────────────────────────────────────
    def objective(trial: optuna.Trial):
        if warm_start_configs and trial.number < len(warm_start_configs):
            base_config = warm_start_configs[trial.number]
            # Give trial hint of what we are doing
            trial.suggest_categorical("model_name", [base_config["model_name"]])
            config = dict(base_config)
        else:
            config = _suggest_config(
                trial, 
                task_type, 
                llm_bounds=llm_bounds, 
                forced_model=forced_model,
                forced_pipeline_params=forced_pipeline_params
            )
        try:
            pipeline = build_pipeline(config, numeric_cols, categorical_cols, task_type)
        except Exception as e:
            logger.warning(f"Build failed: {e}")
            if multi_objective:
                return float("nan"), float("nan"), float("nan"), float("nan")
            return float("nan")

        try:
            t_train = time.time()
            if task_type == "classification":
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
                primary = float(np.mean(scores))
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
                scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", n_jobs=-1)
                primary = float(np.mean(scores))
            train_time = time.time() - t_train

            pipeline.fit(X, y)
            model_size = _estimate_model_size(pipeline)

            # 4th objective: inference latency (predict on 100 samples)
            sample_X = X.iloc[:min(100, len(X))]
            t_infer = time.time()
            pipeline.predict(sample_X)
            inference_latency = (time.time() - t_infer) * 1000  # ms

            # Edge-device latency hard-cap
            if max_inference_ms is not None and inference_latency > max_inference_ms:
                logger.debug(f"Trial disqualified: inference {inference_latency:.1f}ms > cap {max_inference_ms}ms")
                if multi_objective:
                    return float("nan"), float("nan"), float("nan"), float("nan")
                return float("nan")

        except Exception as e:
            logger.debug(f"Trial failed: {e}")
            if multi_objective:
                return float("nan"), float("nan"), float("nan"), float("nan")
            return float("nan")

        trial.set_user_attr("config", config)
        trial.set_user_attr("primary_metric", primary)
        trial.set_user_attr("train_time", train_time)
        trial.set_user_attr("model_size", model_size)
        trial.set_user_attr("inference_latency_ms", inference_latency)

        if progress_callback:
            progress_callback(trial.number, {
                "trial": trial.number,
                "model": config["model_name"],
                "primary_metric": round(primary, 4),
                "train_time": round(train_time, 3),
                "inference_latency_ms": round(inference_latency, 2),
            })

        if multi_objective:
            return primary, train_time, model_size, inference_latency
        return primary

    # ── Study creation ────────────────────────────────────────
    np.random.seed(random_state)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=actual_n_trials, reduction_factor=3)
    
    if multi_objective:
        directions = ["maximize", "minimize", "minimize", "minimize"]  # score, time, size, latency
        study = optuna.create_study(
            directions=directions,
            sampler=optuna.samplers.NSGAIISampler(seed=random_state),
        )
    else:
        _sampler_map = {
            "tpe": optuna.samplers.TPESampler(seed=random_state),
            "random": optuna.samplers.RandomSampler(seed=random_state),
            "cmaes": optuna.samplers.CmaEsSampler(seed=random_state),
        }
        direction = "maximize"
        study = optuna.create_study(
            direction=direction,
            sampler=_sampler_map.get(sampler, optuna.samplers.TPESampler(seed=random_state)),
            pruner=pruner,
        )

    # ── Warm start from similar datasets ─────────────────────
    if warm_start_configs:
        logger.info(f"Warm-starting with {len(warm_start_configs)} configs from similar datasets")
        for wc in warm_start_configs[:5]:
            try:
                study.enqueue_trial(wc)
            except Exception:
                pass

    study.optimize(
        objective,
        n_trials=actual_n_trials,
        timeout=timeout,
        catch=(Exception,),
        show_progress_bar=False,
    )

    runtime = time.time() - t0

    # ── Extract results ───────────────────────────────────────
    all_trials = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            cfg = t.user_attrs.get("config", {})
            all_trials.append({
                "trial_number": t.number,
                "model_name": cfg.get("model_name", "?"),
                "config": cfg,
                "primary_metric": t.user_attrs.get("primary_metric", 0),
                "train_time": t.user_attrs.get("train_time", 0),
                "model_size": t.user_attrs.get("model_size", 0),
            })

    # Sort by primary metric
    all_trials.sort(key=lambda x: x["primary_metric"], reverse=True)

    if multi_objective:
        pareto_trials = study.best_trials
        pareto_front = [
            {
                "trial_number": t.number,
                "values": list(t.values) if t.values else [],
                "config": t.user_attrs.get("config", {}),
                "primary_metric": t.user_attrs.get("primary_metric", 0),
                "train_time": t.user_attrs.get("train_time", 0),
                "model_size": t.user_attrs.get("model_size", 0),
                "inference_latency_ms": t.user_attrs.get("inference_latency_ms", 0),
            }
            for t in pareto_trials
        ]
        hypervolume = _compute_hypervolume(pareto_front)
        balanced_solution = _find_balanced_pareto_solution(pareto_front)
    else:
        pareto_front = []
        hypervolume = 0.0
        balanced_solution = all_trials[0] if all_trials else {}

    best_config = all_trials[0]["config"] if all_trials else {}
    best_metrics = {
        "primary_metric": all_trials[0]["primary_metric"] if all_trials else 0.0,
        "metric_name": "f1_weighted" if task_type == "classification" else "r2",
        "train_time": all_trials[0].get("train_time", 0),
        "model_size_bytes": all_trials[0].get("model_size", 0),
        "inference_latency_ms": all_trials[0].get("inference_latency_ms", 0),
        "all_trials": all_trials[:50],
        "pareto_front": pareto_front,
    }

    logger.info(
        f"Optimization done: {actual_n_trials} trials, "
        f"best={best_metrics['primary_metric']:.4f}, "
        f"runtime={runtime:.1f}s, pareto_size={len(pareto_front)}"
    )

    # ── Search-space volume quantification ─────────────────────
    search_volume = _compute_search_volume(llm_bounds)

    # ── Pareto scatter plot ───────────────────────────────────
    try:
        from core.visualization.pareto_plotter import plot_pareto_front
        plot_pareto_front(pareto_front, all_trials[:50])
    except Exception as e:
        logger.debug(f"Pareto plot skipped: {e}")

    return {
        "best_config": best_config,
        "best_metrics": best_metrics,
        "pareto_front": pareto_front,
        "pareto_hypervolume": hypervolume,
        "balanced_solution": balanced_solution,
        "all_trials": all_trials[:50],
        "n_trials_run": len(all_trials),
        "runtime_seconds": round(runtime, 2),
        "search_space_pruning": pruner_info,
        "search_space_volume": search_volume,
    }


def _estimate_model_size(pipeline) -> int:
    """Rough model size proxy in bytes."""
    import pickle
    try:
        return len(pickle.dumps(pipeline))
    except Exception:
        return 0


def _compute_hypervolume(pareto_front: List[Dict]) -> float:
    """
    Compute dominated hypervolume indicator for the Pareto front.
    Normalizes objectives and computes volume dominated by the Pareto set.
    Reference point: [0, max_time, max_size, max_latency]
    """
    if len(pareto_front) < 2:
        return 0.0
    try:
        scores = np.array([p["primary_metric"] for p in pareto_front])
        times  = np.array([p["train_time"] for p in pareto_front])
        sizes  = np.array([p["model_size"] for p in pareto_front])
        lats   = np.array([p.get("inference_latency_ms", 0) for p in pareto_front])

        # Normalize to [0, 1]
        def _norm(a):
            mn, mx = a.min(), a.max()
            return (a - mn) / (mx - mn + 1e-9)

        s_n = _norm(scores)          # higher is better → keep as is
        t_n = 1 - _norm(times)       # lower time is better
        sz_n = 1 - _norm(sizes)
        l_n  = 1 - _norm(lats)

        # Simple proxy: mean of per-solution volumes
        hv = float(np.mean(s_n * t_n * sz_n * l_n))
        return round(hv, 4)
    except Exception:
        return 0.0


def _find_balanced_pareto_solution(pareto_front: List[Dict]) -> Dict:
    """
    Find the Pareto solution closest to the utopia point
    (maximize score, minimize time/size/latency simultaneously).
    Uses normalized Euclidean distance from the ideal vector.
    """
    if not pareto_front:
        return {}
    if len(pareto_front) == 1:
        return pareto_front[0]
    try:
        scores = np.array([p["primary_metric"] for p in pareto_front])
        times  = np.array([p["train_time"] for p in pareto_front])
        sizes  = np.array([p["model_size"] for p in pareto_front])
        lats   = np.array([p.get("inference_latency_ms", 0) for p in pareto_front])

        # Glass-Box Metric (Penalize deep/opaque ensembles if simpler models exist)
        opaque_scores = np.array([
            1.0 if p["config"].get("model_name", "") in ["mlp", "catboost", "xgboost", "lightgbm"] else 0.0 
            for p in pareto_front
        ])

        def _n(a): 
            span = a.max() - a.min()
            return (a - a.min()) / (span + 1e-9) if span > 0 else np.zeros_like(a)

        # Utopia direction: max score → 1, min rest → 0
        dist = np.sqrt(
            (1 - _n(scores))**2 +
            _n(times)**2 +
            _n(sizes)**2 +
            _n(lats)**2 +
            (_n(opaque_scores) * 0.5)**2  # Mathematical Glass-Box Prioritization Bias
        )
        idx = int(np.argmin(dist))
        best = dict(pareto_front[idx])
        best["is_balanced"] = True
        return best
    except Exception:
        return pareto_front[0]
