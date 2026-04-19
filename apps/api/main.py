"""
MetaLearnX — FastAPI Backend
Production-grade API with async background job execution.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from core.tracking.meta_db import (
    get_dataset,
    get_experiment,
    get_meta_db_stats,
    get_neighbors,
    initialize_db,
    list_datasets,
    list_experiments,
    upsert_dataset,
    create_experiment,
    update_experiment,
)

# ── App init ─────────────────────────────────────────────────
app = FastAPI(
    title="MetaLearnX API",
    description="Self-Adaptive AutoML with Meta-Learning, Dataset Intelligence, and Foundation Models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    initialize_db()
    _seed_benchmark_datasets()
    logger.info("MetaLearnX API started.")


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "MetaLearnX", "version": "1.0.0"}


@app.get("/api/meta-db/stats", tags=["System"])
def meta_db_stats():
    return get_meta_db_stats()


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

@app.post("/api/datasets/upload", tags=["Datasets"])
async def upload_dataset(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    task_type: str = Form("classification"),
    dataset_description: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
):
    """Upload a CSV dataset and run the profiler."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found. Available: {list(df.columns)}",
        )

    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Dataset too small (minimum 10 rows).")

    # Run profiling synchronously (fast), embedding + meta-learning async
    from core.dataset_intelligence.profiler import profile_dataset
    from core.meta_features.handcrafted import extract_meta_features
    from core.meta_features.embedding import embed_dataset_from_df

    name = Path(file.filename).stem
    profile = profile_dataset(df, target_col, task_type=task_type, name=name)
    mf_vec = extract_meta_features(profile)

    # 1. Local Semantic Embedding of Description
    from core.dataset_intelligence.semantic_embedding import get_semantic_embedder
    embedder = get_semantic_embedder()
    sem_emb = embedder.embed_text(dataset_description).tolist()

    try:
        X_only = df.drop(columns=[target_col])
        emb_vec = embed_dataset_from_df(X_only)
        emb_list = emb_vec.tolist()
    except Exception:
        emb_list = None

    dataset_id = upsert_dataset(
        name=name,
        task_type=task_type,
        n_samples=profile.n_samples,
        n_features=profile.n_features,
        meta_features=profile.to_meta_feature_dict(),
        embedding=emb_list,
        source="upload",
        description=dataset_description,
        semantic_embedding=sem_emb,
    )

    # Store CSV for later use
    _save_dataset_csv(dataset_id, contents)

    return {
        "dataset_id": dataset_id,
        "name": name,
        "n_samples": profile.n_samples,
        "n_features": profile.n_features,
        "task_type": task_type,
        "target_col": target_col,
        "profile": profile.to_dict(),
    }


@app.get("/api/datasets", tags=["Datasets"])
def list_all_datasets():
    datasets = list_datasets()
    return {"datasets": datasets, "total": len(datasets)}


@app.get("/api/datasets/{dataset_id}", tags=["Datasets"])
def get_dataset_details(dataset_id: str):
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return ds


@app.get("/api/datasets/{dataset_id}/similar", tags=["Datasets"])
def get_similar_datasets(dataset_id: str, top_k: int = 5):
    neighbors = get_neighbors(dataset_id, top_k=top_k)
    # Enrich with past best experiments
    enriched = []
    for n in neighbors:
        nid = n["neighbor_id"]
        exps = list_experiments(dataset_id=nid)
        best_exp = next((e for e in exps if e["status"] == "done"), None)
        n["best_experiment"] = {
            "model": best_exp["pipeline_config"].get("model_name") if best_exp else None,
            "metric": best_exp["metrics"].get("primary_metric") if best_exp else None,
        }
        enriched.append(n)
    return {"similar_datasets": enriched}


# ─────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────

@app.get("/api/recommend/{dataset_id}", tags=["Meta-Learning"])
def recommend_models(dataset_id: str, task_type: str = "classification"):
    """Zero-shot model recommendation using meta-learning."""
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    from core.meta_learning.knn_recommender import zero_shot_recommend
    from core.meta_learning.neural_meta_learner import neural_recommend
    from core.meta_features.handcrafted import extract_meta_features, META_FEATURE_KEYS
    import numpy as np

    mf_dict = ds.get("meta_features", {})
    mf_vec = np.array([float(mf_dict.get(k, 0.0)) for k in META_FEATURE_KEYS], dtype=np.float32)

    knn = zero_shot_recommend(dataset_id, task_type=task_type, top_k=5)
    neural = neural_recommend(mf_vec, task_type=task_type, top_k=5)

    return {
        "dataset_id": dataset_id,
        "knn_recommendations": knn,
        "neural_recommendations": neural,
    }


# ─────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────

class RunExperimentRequest(BaseModel):
    dataset_id: str
    target_col: str
    task_type: str = "classification"
    multi_objective: bool = True
    n_trials: Optional[int] = None
    random_state: int = 42


@app.post("/api/experiments/run", tags=["Experiments"])
async def run_experiment(
    request: RunExperimentRequest,
    background_tasks: BackgroundTasks,
):
    """Launch a full MetaLearnX optimization in the background."""
    ds = get_dataset(request.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    csv_path = _get_dataset_csv_path(request.dataset_id)
    if not csv_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Dataset CSV not found. Please re-upload the dataset.",
        )

    experiment_id = create_experiment(request.dataset_id)
    background_tasks.add_task(
        _run_experiment_task,
        experiment_id=experiment_id,
        dataset_id=request.dataset_id,
        csv_path=csv_path,
        target_col=request.target_col,
        task_type=request.task_type,
        multi_objective=request.multi_objective,
        n_trials=request.n_trials,
        random_state=request.random_state,
    )

    return {
        "experiment_id": experiment_id,
        "status": "pending",
        "message": "Experiment queued. Poll /api/experiments/{id}/status for updates.",
    }


async def _run_experiment_task(
    experiment_id: str,
    dataset_id: str,
    csv_path: Path,
    target_col: str,
    task_type: str,
    multi_objective: bool,
    n_trials: Optional[int],
    random_state: int,
):
    """Background task: loads CSV and runs the full pipeline."""
    try:
        df = pd.read_csv(csv_path)
        ds = get_dataset(dataset_id)
        dataset_name = ds["name"] if ds else "dataset"

        from core.orchestrator import run_full_pipeline

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_full_pipeline(
                df=df,
                target_col=target_col,
                task_type=task_type,
                dataset_name=dataset_name,
                experiment_id=experiment_id,
                multi_objective=multi_objective,
                n_trials=n_trials,
                random_state=random_state,
            ),
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Background experiment {experiment_id} failed: {e}\n{tb}")
        update_experiment(experiment_id, status="failed", error_msg=str(e))


@app.get("/api/experiments/{experiment_id}/status", tags=["Experiments"])
def experiment_status(experiment_id: str):
    exp = get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return {
        "experiment_id": experiment_id,
        "status": exp["status"],
        "created_at": exp["created_at"],
        "completed_at": exp.get("completed_at"),
        "error": exp.get("error_msg"),
    }


@app.get("/api/experiments/{experiment_id}/results", tags=["Experiments"])
def experiment_results(experiment_id: str):
    exp = get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    if exp["status"] != "done":
        return {"status": exp["status"], "message": "Experiment not yet completed."}
    return exp


@app.get("/api/experiments/{experiment_id}/explain", tags=["Experiments"])
def experiment_explain(experiment_id: str):
    """Return explainability artifacts for an experiment."""
    exp = get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    mlflow_run_id = exp.get("mlflow_run_id")
    # Return stored explainability from MLflow if available
    explain_path = None
    if mlflow_run_id:
        mlflow_path = Path("mlruns") / "1" / mlflow_run_id / "artifacts" / "explainability" / "explainability.json"
        if mlflow_path.exists():
            explain_path = mlflow_path

    if explain_path and explain_path.exists():
        return json.loads(explain_path.read_text())

    return {
        "message": "Explainability data not found. Run experiment first.",
        "experiment_id": experiment_id,
    }


@app.get("/api/experiments/history", tags=["Experiments"])
def experiments_history(dataset_id: Optional[str] = None):
    exps = list_experiments(dataset_id=dataset_id)
    return {"experiments": exps, "total": len(exps)}


# ─────────────────────────────────────────────
# Advanced: Uncertainty-Aware Recommendations
# ─────────────────────────────────────────────

@app.get("/api/recommend/{dataset_id}/uncertain", tags=["Advanced: Uncertainty"])
def uncertain_recommendations(
    dataset_id: str,
    task_type: str = "classification",
    n_passes: int = 50,
):
    """MC Dropout uncertainty-aware model recommendations."""
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    from core.meta_learning.uncertainty import uncertain_recommend
    from core.meta_features.handcrafted import META_FEATURE_KEYS
    import numpy as np

    mf_dict = ds.get("meta_features", {})
    mf_vec = np.array([float(mf_dict.get(k, 0.0)) for k in META_FEATURE_KEYS], dtype=np.float32)
    return uncertain_recommend(mf_vec, n_passes=n_passes, task_type=task_type)


# ─────────────────────────────────────────────
# Advanced: Few-Shot detection + recommendation
# ─────────────────────────────────────────────

@app.get("/api/datasets/{dataset_id}/few-shot", tags=["Advanced: Few-Shot"])
def few_shot_analysis(dataset_id: str):
    """Detect few-shot regime and return MAML-adapted recommendations."""
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    from core.meta_learning.few_shot import few_shot_recommend
    from core.meta_features.handcrafted import META_FEATURE_KEYS
    import numpy as np

    mf_dict = ds.get("meta_features", {})
    mf_vec = np.array([float(mf_dict.get(k, 0.0)) for k in META_FEATURE_KEYS], dtype=np.float32)
    return few_shot_recommend(
        meta_feature_vec=mf_vec,
        n_samples=ds.get("n_samples", 1000),
        n_features=ds.get("n_features", 10),
        dataset_id=dataset_id,
    )


# ─────────────────────────────────────────────
# Advanced: Knowledge Graph
# ─────────────────────────────────────────────

@app.get("/api/knowledge-graph", tags=["Advanced: Knowledge Graph"])
def get_knowledge_graph_json():
    """Return the full dataset knowledge graph as JSON for visualization."""
    from core.meta_learning.knowledge_graph import get_knowledge_graph
    g = get_knowledge_graph()
    return g.to_json()


@app.post("/api/knowledge-graph/rebuild", tags=["Advanced: Knowledge Graph"])
async def rebuild_knowledge_graph(background_tasks: BackgroundTasks):
    """Rebuild knowledge graph from meta-db and optionally retrain GNN."""
    background_tasks.add_task(_rebuild_graph_task)
    return {"message": "Knowledge graph rebuild started.", "status": "running"}


async def _rebuild_graph_task():
    try:
        from core.meta_learning.gnn_meta_learner import build_graph_from_meta_db
        from core.meta_learning.knowledge_graph import get_knowledge_graph
        from core.tracking.meta_db import list_datasets, get_all_meta_features, list_experiments
        from core.meta_features.handcrafted import META_FEATURE_KEYS
        import numpy as np

        g = get_knowledge_graph()
        mf_dict = dict(get_all_meta_features())
        all_ds = list_datasets()
        # Get best model per dataset from experiments
        exp_map = {}
        for exp in list_experiments():
            did = exp.get("dataset_id")
            model = exp.get("pipeline_config", {}).get("model_name", "")
            score = exp.get("metrics", {}).get("primary_metric", 0)
            if did and model and exp.get("status") == "done":
                if did not in exp_map or score > exp_map[did][1]:
                    exp_map[did] = (model, score)

        for ds in all_ds:
            did = ds["id"]
            if did in mf_dict:
                best_model, best_metric = exp_map.get(did, (None, None))
                g.add_or_update_node(
                    dataset_id=did,
                    name=ds["name"],
                    meta_feature_vec=mf_dict[did],
                    best_model=best_model,
                    best_metric=best_metric,
                    task_type=ds.get("task_type", "classification"),
                )
        g.save()
        await asyncio.get_event_loop().run_in_executor(None, build_graph_from_meta_db)
        logger.info("Knowledge graph rebuilt + GNN retrained.")
    except Exception as e:
        logger.error(f"Graph rebuild failed: {e}")


# ─────────────────────────────────────────────
# Advanced: Continual Meta-Learning
# ─────────────────────────────────────────────

@app.get("/api/continual-learning/stats", tags=["Advanced: Continual Learning"])
def continual_learning_stats():
    """Return replay buffer stats and EWC status."""
    from core.meta_learning.continual_learner import get_continual_learner
    cl = get_continual_learner()
    return cl.get_stats()


# ─────────────────────────────────────────────
# Advanced: Search Space Learning
# ─────────────────────────────────────────────

@app.get("/api/search-space/stats", tags=["Advanced: Search Space"])
def search_space_stats():
    """Return search space reduction statistics per model family."""
    from core.optimization.search_space_learner import get_pruner
    pruner = get_pruner()
    return {
        "reduction_stats": pruner.get_reduction_stats(),
        "history_sizes": {
            k: {"good": len(v.get("good", [])), "bad": len(v.get("bad", []))}
            for k, v in pruner.pruning_history.items()
        },
    }


# ─────────────────────────────────────────────
# Advanced: Federated Meta-Learning
# ─────────────────────────────────────────────

@app.post("/api/federated/run", tags=["Advanced: Federated"])
async def run_federated_learning(background_tasks: BackgroundTasks, n_rounds: int = 5):
    """Launch federated meta-learning simulation across 3 organizations."""
    background_tasks.add_task(_run_federated_task, n_rounds)
    return {
        "message": f"Federated learning started ({n_rounds} rounds, 3 organizations).",
        "status": "running",
        "privacy": "Only model weights are shared — no raw data transmitted.",
    }


async def _run_federated_task(n_rounds: int):
    try:
        from core.meta_learning.federated import FederatedMetaLearner
        fl = FederatedMetaLearner(n_rounds=n_rounds, n_orgs=3)
        result = await asyncio.get_event_loop().run_in_executor(None, fl.run)
        logger.info(f"Federated learning complete: {result}")
    except Exception as e:
        logger.error(f"Federated learning failed: {e}")


# ─────────────────────────────────────────────
# Advanced: Ablation Study
# ─────────────────────────────────────────────

@app.post("/api/benchmark/ablation", tags=["Benchmark"])
async def run_ablation_study(background_tasks: BackgroundTasks):
    """Run the full ablation study across all conditions."""
    background_tasks.add_task(_run_ablation_task)
    return {"message": "Ablation study started.", "status": "running"}


async def _run_ablation_task():
    try:
        from core.evaluation.harness import run_benchmark
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: run_benchmark(include_openml=False, run_ablation_study=True)
        )
        logger.info("Ablation study complete.")
    except Exception as e:
        logger.error(f"Ablation failed: {e}")


@app.get("/api/benchmark/results", tags=["Benchmark"])
def get_benchmark_results():
    """Return cached benchmark results."""
    from pathlib import Path
    results_path = Path(__file__).parents[2] / "experiments" / "reports" / "benchmark_results.json"
    if results_path.exists():
        return json.loads(results_path.read_text())
    return {"message": "No benchmark results yet. Run /api/benchmark/run first."}


# ─────────────────────────────────────────────
# Benchmark (original)
# ─────────────────────────────────────────────

@app.post("/api/benchmark/run", tags=["Benchmark"])
async def run_benchmark(background_tasks: BackgroundTasks):
    """Launch the multi-dataset evaluation harness."""
    background_tasks.add_task(_run_benchmark_task)
    return {"message": "Benchmark started in background.", "status": "running"}


async def _run_benchmark_task():
    try:
        from core.evaluation.harness import run_benchmark
        await asyncio.get_event_loop().run_in_executor(None, run_benchmark)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

_DATA_DIR = Path(__file__).parents[2] / "data" / "raw"


def _save_dataset_csv(dataset_id: str, contents: bytes) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    (_DATA_DIR / f"{dataset_id}.csv").write_bytes(contents)


def _get_dataset_csv_path(dataset_id: str) -> Path:
    return _DATA_DIR / f"{dataset_id}.csv"


def _seed_benchmark_datasets():
    """Auto-seed meta-db with sklearn built-in datasets for cold-start."""
    from sklearn.datasets import (
        load_breast_cancer, load_iris, load_wine,
        load_diabetes,
    )
    import numpy as np

    SEEDS = [
        ("iris", load_iris, "classification", "target"),
        ("wine", load_wine, "classification", "target"),
        ("breast_cancer", load_breast_cancer, "classification", "target"),
        ("diabetes", load_diabetes, "regression", "target"),
    ]

    existing = {ds["name"] for ds in list_datasets()}

    for name, loader_fn, task_type, tgt in SEEDS:
        if name in existing:
            continue
        try:
            data = loader_fn()
            X = pd.DataFrame(data["data"], columns=data["feature_names"])
            y = pd.Series(data["target"], name=tgt)
            df = pd.concat([X, y], axis=1)

            from core.dataset_intelligence.profiler import profile_dataset
            from core.meta_features.handcrafted import extract_meta_features
            from core.meta_features.embedding import embed_dataset_from_df

            profile = profile_dataset(df, tgt, task_type=task_type, name=name)
            mf_vec = extract_meta_features(profile)
            try:
                emb_vec = embed_dataset_from_df(X)
                emb_list = emb_vec.tolist()
            except Exception:
                emb_list = None

            dataset_id = upsert_dataset(
                name=name,
                task_type=task_type,
                n_samples=profile.n_samples,
                n_features=profile.n_features,
                meta_features=profile.to_meta_feature_dict(),
                embedding=emb_list,
                source="benchmark",
            )
            # Save CSV for experiments
            df.to_csv(_DATA_DIR / f"{dataset_id}.csv", index=False)
            logger.info(f"Seeded benchmark dataset: {name} ({dataset_id})")
        except Exception as e:
            logger.warning(f"Failed to seed {name}: {e}")


if __name__ == "__main__":
    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=True)
