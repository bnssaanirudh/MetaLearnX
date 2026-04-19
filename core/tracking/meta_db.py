"""
MetaLearnX — Meta-Knowledge Base
SQLite backend with WAL mode for concurrent reads.
Stores dataset profiles, embeddings, experiment results, and neighbor graphs.
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

DB_PATH = Path(__file__).parents[3] / "data" / "meta_db" / "metax.db"


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def get_db():
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_db() -> None:
    """Create all tables if they don't exist."""
    schema = """
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        task_type TEXT NOT NULL,          -- 'classification' | 'regression'
        n_samples INTEGER,
        n_features INTEGER,
        meta_features_json TEXT,           -- JSON: handcrafted meta-feature vector
        embedding_json TEXT,               -- JSON: learned embedding vector
        description TEXT,                  -- Raw dataset description
        semantic_embedding_json TEXT,      -- JSON: 384-dim semantic text embedding
        created_at REAL NOT NULL,
        source TEXT                        -- 'upload' | 'benchmark' | 'openml'
    );

    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL REFERENCES datasets(id),
        pipeline_config_json TEXT,         -- JSON: full pipeline spec
        hyperparams_json TEXT,             -- JSON: best hyperparameters
        metrics_json TEXT,                 -- JSON: all evaluation metrics
        reasoning_logs TEXT,               -- JSON: logs from Architect and Critic agents
        runtime_seconds REAL,
        model_size_bytes INTEGER,
        n_trials INTEGER,
        mlflow_run_id TEXT,
        status TEXT DEFAULT 'pending',     -- 'pending' | 'running' | 'done' | 'failed'
        error_msg TEXT,
        created_at REAL NOT NULL,
        completed_at REAL
    );

    CREATE TABLE IF NOT EXISTS model_performance (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL REFERENCES experiments(id),
        dataset_id TEXT NOT NULL REFERENCES datasets(id),
        model_name TEXT NOT NULL,
        primary_metric REAL,
        metric_name TEXT,
        rank_in_experiment INTEGER,
        created_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dataset_neighbors (
        dataset_id TEXT NOT NULL REFERENCES datasets(id),
        neighbor_id TEXT NOT NULL REFERENCES datasets(id),
        similarity_score REAL NOT NULL,
        similarity_method TEXT DEFAULT 'cosine',
        PRIMARY KEY (dataset_id, neighbor_id)
    );

    CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_id);
    CREATE INDEX IF NOT EXISTS idx_model_perf_dataset ON model_performance(dataset_id);
    CREATE INDEX IF NOT EXISTS idx_model_perf_model ON model_performance(model_name);
    CREATE INDEX IF NOT EXISTS idx_neighbors_dataset ON dataset_neighbors(dataset_id);
    """
    with get_db() as conn:
        conn.executescript(schema)
    logger.info(f"Meta-knowledge base initialized at {DB_PATH}")


# ─────────────────────────────────────────────
# Dataset CRUD
# ─────────────────────────────────────────────

def upsert_dataset(
    name: str,
    task_type: str,
    n_samples: int,
    n_features: int,
    meta_features: Dict[str, Any],
    embedding: Optional[List[float]] = None,
    source: str = "upload",
    dataset_id: Optional[str] = None,
    description: Optional[str] = None,
    semantic_embedding: Optional[List[float]] = None,
) -> str:
    did = dataset_id or str(uuid.uuid4())
    with get_db() as conn:
        # Schema migration check: Ensure columns exist
        try:
            conn.execute("ALTER TABLE datasets ADD COLUMN description TEXT;")
            conn.execute("ALTER TABLE datasets ADD COLUMN semantic_embedding_json TEXT;")
        except sqlite3.OperationalError:
            pass # Columns already exist

        conn.execute(
            """
            INSERT INTO datasets
                (id, name, task_type, n_samples, n_features, meta_features_json,
                 embedding_json, description, semantic_embedding_json, created_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                meta_features_json=excluded.meta_features_json,
                embedding_json=excluded.embedding_json,
                description=excluded.description,
                semantic_embedding_json=excluded.semantic_embedding_json,
                n_samples=excluded.n_samples,
                n_features=excluded.n_features
            """,
            (
                did,
                name,
                task_type,
                n_samples,
                n_features,
                json.dumps(meta_features),
                json.dumps(embedding) if embedding is not None else None,
                description,
                json.dumps(semantic_embedding) if semantic_embedding is not None else None,
                time.time(),
                source,
            ),
        )
    return did


def get_dataset(dataset_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM datasets WHERE id=?", (dataset_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["meta_features"] = json.loads(d.pop("meta_features_json") or "{}")
    d["embedding"] = json.loads(d.pop("embedding_json") or "null")
    if "semantic_embedding_json" in d:
        d["semantic_embedding"] = json.loads(d.pop("semantic_embedding_json") or "null")
    return d


def list_datasets() -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["meta_features"] = json.loads(d.pop("meta_features_json") or "{}")
        d["embedding"] = json.loads(d.pop("embedding_json") or "null")
        result.append(d)
    return result


def get_all_embeddings() -> List[Tuple[str, np.ndarray]]:
    """Return (dataset_id, embedding_vector) for all datasets with embeddings."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, embedding_json FROM datasets WHERE embedding_json IS NOT NULL"
        ).fetchall()
    return [
        (row["id"], np.array(json.loads(row["embedding_json"]), dtype=np.float32))
        for row in rows
    ]


def get_all_meta_features() -> List[Tuple[str, np.ndarray]]:
    """Return (dataset_id, meta_feature_vector) for all datasets."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, meta_features_json FROM datasets WHERE meta_features_json IS NOT NULL"
        ).fetchall()
    results = []
    for row in rows:
        mf = json.loads(row["meta_features_json"])
        vec = np.array(list(mf.values()), dtype=np.float32)
        results.append((row["id"], vec))
    return results


# ─────────────────────────────────────────────
# Experiment CRUD
# ─────────────────────────────────────────────

def create_experiment(dataset_id: str) -> str:
    eid = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO experiments (id, dataset_id, status, created_at)
            VALUES (?, ?, 'pending', ?)
            """,
            (eid, dataset_id, time.time()),
        )
    return eid


def update_experiment(experiment_id: str, **kwargs) -> None:
    allowed = {
        "pipeline_config_json", "hyperparams_json", "metrics_json", "reasoning_logs",
        "runtime_seconds", "model_size_bytes", "n_trials",
        "mlflow_run_id", "status", "error_msg", "completed_at",
    }
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    sets = ", ".join(f"{k}=?" for k in updates)
    with get_db() as conn:
        conn.execute(
            f"UPDATE experiments SET {sets} WHERE id=?",
            (*updates.values(), experiment_id),
        )


def get_experiment(experiment_id: str) -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id=?", (experiment_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    for field in ("pipeline_config_json", "hyperparams_json", "metrics_json", "reasoning_logs"):
        key = field.replace("_json", "")
        if field == "reasoning_logs":
            key = "reasoning_logs"
        d[key] = json.loads(d.pop(field) or "{}")
    return d


def list_experiments(dataset_id: Optional[str] = None) -> List[Dict]:
    with get_db() as conn:
        if dataset_id:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE dataset_id=? ORDER BY created_at DESC",
                (dataset_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        for field in ("pipeline_config_json", "hyperparams_json", "metrics_json"):
            key = field.replace("_json", "")
            d[key] = json.loads(d.pop(field) or "{}")
        results.append(d)
    return results


# ─────────────────────────────────────────────
# Model Performance
# ─────────────────────────────────────────────

def log_model_performance(
    experiment_id: str,
    dataset_id: str,
    model_name: str,
    primary_metric: float,
    metric_name: str,
    rank: int,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO model_performance
                (id, experiment_id, dataset_id, model_name, primary_metric,
                 metric_name, rank_in_experiment, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), experiment_id, dataset_id, model_name,
             primary_metric, metric_name, rank, time.time()),
        )


def get_model_rankings(model_name: str, limit: int = 20) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT mp.*, d.name as dataset_name
            FROM model_performance mp
            JOIN datasets d ON mp.dataset_id = d.id
            WHERE mp.model_name=?
            ORDER BY mp.primary_metric DESC
            LIMIT ?
            """,
            (model_name, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# Dataset Neighbors
# ─────────────────────────────────────────────

def upsert_neighbors(
    dataset_id: str,
    neighbors: List[Tuple[str, float]],
    method: str = "cosine",
) -> None:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM dataset_neighbors WHERE dataset_id=?", (dataset_id,)
        )
        conn.executemany(
            """
            INSERT INTO dataset_neighbors (dataset_id, neighbor_id, similarity_score, similarity_method)
            VALUES (?, ?, ?, ?)
            """,
            [(dataset_id, nid, score, method) for nid, score in neighbors],
        )


def get_neighbors(dataset_id: str, top_k: int = 5) -> List[Dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT dn.*, d.name as neighbor_name, d.task_type
            FROM dataset_neighbors dn
            JOIN datasets d ON dn.neighbor_id = d.id
            WHERE dn.dataset_id=?
            ORDER BY dn.similarity_score DESC
            LIMIT ?
            """,
            (dataset_id, top_k),
        ).fetchall()
    return [dict(r) for r in rows]


def get_meta_db_stats() -> Dict[str, Any]:
    with get_db() as conn:
        n_datasets = conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]
        n_experiments = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        n_done = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status='done'"
        ).fetchone()[0]
        top_models = conn.execute(
            """
            SELECT model_name, COUNT(*) as n_wins
            FROM model_performance WHERE rank_in_experiment=1
            GROUP BY model_name ORDER BY n_wins DESC LIMIT 5
            """
        ).fetchall()
    return {
        "n_datasets": n_datasets,
        "n_experiments": n_experiments,
        "n_completed": n_done,
        "top_model_families": [dict(r) for r in top_models],
    }
