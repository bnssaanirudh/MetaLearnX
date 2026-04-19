"""
MetaLearnX — kNN-based Meta-Learner
Recommends models based on similarity to past datasets.
Also provides zero-shot model recommendation with rationale text.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from core.tracking.meta_db import (
    get_all_embeddings,
    get_all_meta_features,
    get_dataset,
    get_neighbors,
    list_datasets,
    upsert_neighbors,
)


# ─────────────────────────────────────────────
# Similarity computation
# ─────────────────────────────────────────────

def _build_combined_matrix(
    ids: List[str],
    mf_map: Dict[str, np.ndarray],
    emb_map: Dict[str, np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Combine handcrafted meta-features and learned embeddings.
    alpha=0 → only handcrafted; alpha=1 → only embedding; 0.5 → hybrid.
    """
    vecs = []
    for did in ids:
        parts = []
        if did in mf_map and alpha < 1.0:
            mf = normalize(mf_map[did].reshape(1, -1))[0]
            parts.append(mf * (1 - alpha))
        if did in emb_map and alpha > 0.0:
            emb = normalize(emb_map[did].reshape(1, -1))[0]
            # Pad or trim embedding to match meta-feature dim for simplicity
            emb_padded = emb
            parts.append(emb_padded * alpha)
        if parts:
            vecs.append(np.concatenate(parts))
        else:
            vecs.append(np.zeros(33, dtype=np.float32))  # fallback zero vector
    return np.vstack(vecs) if vecs else np.zeros((1, 33))


def compute_and_store_neighbors(
    dataset_id: str,
    query_mf: np.ndarray,
    query_emb: Optional[np.ndarray],
    top_k: int = 5,
    alpha: float = 0.5,
) -> List[Dict]:
    """
    Compute cosine similarity between the query dataset and all stored datasets.
    Stores and returns top-k neighbors.
    """
    # Build background corpus
    all_mf = dict(get_all_meta_features())  # {id: vec}
    all_emb = dict(get_all_embeddings())    # {id: vec}

    candidate_ids = [
        did for did in all_mf.keys() if did != dataset_id
    ]

    if not candidate_ids:
        logger.info("Meta-db empty — no neighbors to compute.")
        return []

    # Query vector
    q_parts = []
    if alpha < 1.0:
        q_mf = normalize(query_mf.reshape(1, -1))[0] * (1 - alpha)
        q_parts.append(q_mf)
    if alpha > 0.0 and query_emb is not None:
        q_emb = normalize(query_emb.reshape(1, -1))[0] * alpha
        q_parts.append(q_emb)
    q_vec = np.concatenate(q_parts).reshape(1, -1) if q_parts else np.zeros((1, 33))

    # Corpus matrix
    corpus_matrix = _build_combined_matrix(candidate_ids, all_mf, all_emb, alpha)

    # Ensure same dimensionality
    q_dim = q_vec.shape[1]
    c_dim = corpus_matrix.shape[1]
    if q_dim != c_dim:
        min_dim = min(q_dim, c_dim)
        q_vec = q_vec[:, :min_dim]
        corpus_matrix = corpus_matrix[:, :min_dim]

    try:
        sims = cosine_similarity(q_vec, corpus_matrix)[0]
    except Exception as e:
        logger.warning(f"Similarity computation failed: {e}")
        return []

    # Top-k
    top_indices = np.argsort(sims)[::-1][:top_k]
    neighbors = [
        (candidate_ids[i], float(sims[i])) for i in top_indices
    ]

    upsert_neighbors(dataset_id, neighbors, method="cosine_hybrid")

    # Enrich with dataset info
    result = []
    for nid, score in neighbors:
        ds = get_dataset(nid)
        if ds:
            result.append({
                "dataset_id": nid,
                "dataset_name": ds["name"],
                "similarity_score": round(score, 4),
                "task_type": ds["task_type"],
                "n_samples": ds["n_samples"],
                "n_features": ds["n_features"],
            })

    return result


# ─────────────────────────────────────────────
# Zero-shot model recommendation
# ─────────────────────────────────────────────

def zero_shot_recommend(
    dataset_id: str,
    task_type: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Zero-shot pipeline recommendation using kNN over meta-db.
    Returns ranked list of model suggestions with confidence + rationale.
    """
    from core.tracking.meta_db import (
        get_neighbors,
        list_experiments,
        get_experiment,
    )

    neighbors = get_neighbors(dataset_id, top_k=5)

    if not neighbors:
        # Cold start: return prior based on task type
        return _cold_start_recommendations(task_type)

    # Aggregate model frequency from neighbors' best experiments
    model_scores: Dict[str, List[float]] = {}
    model_contexts: Dict[str, List[str]] = {}

    for n in neighbors:
        nid = n["neighbor_id"]
        exps = list_experiments(dataset_id=nid)
        best_exp = next(
            (e for e in exps if e["status"] == "done"),
            None,
        )
        if not best_exp:
            continue
        metrics = best_exp.get("metrics", {})
        model = best_exp.get("pipeline_config", {}).get("model_name", "")
        score = metrics.get("primary_metric", 0.0)

        if model:
            similarity = n["similarity_score"]
            weighted_score = score * similarity
            model_scores.setdefault(model, []).append(weighted_score)
            ds = get_dataset(nid)
            if ds:
                model_contexts.setdefault(model, []).append(ds["name"])

    if not model_scores:
        return _cold_start_recommendations(task_type)

    # Rank models
    ranked = sorted(
        model_scores.items(),
        key=lambda x: np.mean(x[1]),
        reverse=True,
    )

    results = []
    for rank, (model, scores) in enumerate(ranked[:top_k], 1):
        avg = float(np.mean(scores))
        conf = min(avg, 1.0)
        ctx_datasets = model_contexts.get(model, [])
        rationale = _build_rationale(model, ctx_datasets, avg)

        results.append({
            "rank": rank,
            "model_name": model,
            "confidence": round(conf, 3),
            "avg_weighted_score": round(avg, 4),
            "supporting_datasets": ctx_datasets,
            "rationale": rationale,
        })

    return results


def _cold_start_recommendations(task_type: str) -> List[Dict]:
    """Prior recommendations when meta-db is empty."""
    if task_type == "classification":
        order = [
            ("random_forest", 0.82, "Strong default for tabular classification across diverse datasets."),
            ("xgboost", 0.80, "Top performer on structured data in competitions."),
            ("lightgbm", 0.79, "Fast gradient boosting with low memory footprint."),
            ("logistic_regression", 0.65, "Interpretable baseline; works well for linearly separable data."),
            ("mlp", 0.60, "Handles complex non-linear patterns with sufficient data."),
        ]
    else:
        order = [
            ("random_forest", 0.78, "Robust ensemble for regression; handles non-linearity well."),
            ("xgboost", 0.76, "State-of-the-art for tabular regression tasks."),
            ("lightgbm", 0.75, "Fast gradient boosting with excellent large-dataset scaling."),
            ("logistic_regression", 0.55, "Ridge regression baseline; interpretable and stable."),
            ("mlp", 0.58, "Neural network for non-linear regression tasks."),
        ]
    return [
        {
            "rank": i + 1,
            "model_name": m,
            "confidence": c,
            "avg_weighted_score": c,
            "supporting_datasets": [],
            "rationale": f"[Cold start prior] {r}",
        }
        for i, (m, c, r) in enumerate(order)
    ]


def _build_rationale(model: str, datasets: List[str], score: float) -> str:
    ds_str = (
        f"similar datasets ({', '.join(datasets[:3])})"
        if datasets
        else "similar past datasets"
    )
    model_label = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "logistic_regression": "Logistic Regression",
        "svm": "SVM",
        "mlp": "Neural Network (MLP)",
        "tabpfn": "TabPFN (Foundation Model)",
    }.get(model, model)

    return (
        f"{model_label} was recommended because it achieved an average weighted "
        f"score of {score:.3f} on {ds_str} retrieved from the meta-knowledge base. "
        f"The similarity-weighted performance across retrieved datasets supports "
        f"this as the most promising candidate to warm-start optimization."
    )
