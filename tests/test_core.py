import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture
def iris_df():
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="target")
    return pd.concat([X, y], axis=1)


@pytest.fixture
def regression_df():
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    X = pd.DataFrame(data["data"], columns=data["feature_names"])
    y = pd.Series(data["target"], name="target")
    return pd.concat([X, y], axis=1)


class TestProfiler:
    def test_basic_shape(self, iris_df):
        from core.dataset_intelligence.profiler import profile_dataset
        p = profile_dataset(iris_df, "target", task_type="classification", name="iris")
        assert p.n_samples == 150
        assert p.n_features == 4

    def test_classification_fields(self, iris_df):
        from core.dataset_intelligence.profiler import profile_dataset
        p = profile_dataset(iris_df, "target", task_type="classification", name="iris")
        assert p.n_classes == 3
        assert 0 < p.class_imbalance_ratio <= 1.0
        assert p.target_entropy > 0

    def test_regression_fields(self, regression_df):
        from core.dataset_intelligence.profiler import profile_dataset
        p = profile_dataset(regression_df, "target", task_type="regression")
        assert p.n_classes == 0
        assert p.target_std > 0

    def test_complexity_score_range(self, iris_df):
        from core.dataset_intelligence.profiler import profile_dataset
        p = profile_dataset(iris_df, "target")
        assert 0.0 <= p.complexity_score <= 1.0

    def test_missing_data_detection(self):
        from core.dataset_intelligence.profiler import profile_dataset
        df = pd.DataFrame({
            "a": [1, None, 3, 4, 5] * 10,
            "b": [1.0, 2.0, None, 4.0, 5.0] * 10,
            "y": [0, 1, 0, 1, 1] * 10,
        })
        p = profile_dataset(df, "y", task_type="classification")
        assert p.has_missing
        assert p.missing_rate_mean > 0

    def test_to_dict_has_all_fields(self, iris_df):
        from core.dataset_intelligence.profiler import profile_dataset
        p = profile_dataset(iris_df, "target")
        d = p.to_dict()
        assert "n_samples" in d
        assert "complexity_score" in d
        assert "alerts" in d


class TestMetaFeatures:
    def test_vector_length(self, iris_df):
        from core.dataset_intelligence.profiler import profile_dataset
        from core.meta_features.handcrafted import extract_meta_features, META_FEATURE_KEYS
        p = profile_dataset(iris_df, "target")
        vec = extract_meta_features(p)
        assert vec.shape == (len(META_FEATURE_KEYS),)

    def test_no_nan_or_inf(self, iris_df):
        from core.dataset_intelligence.profiler import profile_dataset
        from core.meta_features.handcrafted import extract_meta_features
        p = profile_dataset(iris_df, "target")
        vec = extract_meta_features(p)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_dtype(self, iris_df):
        from core.dataset_intelligence.profiler import profile_dataset
        from core.meta_features.handcrafted import extract_meta_features
        p = profile_dataset(iris_df, "target")
        vec = extract_meta_features(p)
        assert vec.dtype == np.float32


class TestPipelineBuilder:
    def test_basic_build(self, iris_df):
        from core.pipeline_builder.builder import build_pipeline, get_column_types
        num_cols, cat_cols = get_column_types(iris_df, "target")
        config = {
            "model_name": "random_forest",
            "imputer": "median",
            "scaler": "standard",
            "encoder": "ordinal",
            "feature_selector": "none",
            "model_params": {"n_estimators": 10},
        }
        pipeline = build_pipeline(config, num_cols, cat_cols, "classification")
        X = iris_df.drop(columns=["target"])
        y = iris_df["target"]
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)

    def test_column_types(self, iris_df):
        from core.pipeline_builder.builder import get_column_types
        num_cols, cat_cols = get_column_types(iris_df, "target")
        assert len(num_cols) == 4
        assert len(cat_cols) == 0


class TestMetaDB:
    def test_initialize_and_upsert(self, tmp_path, monkeypatch):
        import core.tracking.meta_db as mdb
        monkeypatch.setattr(mdb, "DB_PATH", tmp_path / "test.db")
        mdb.initialize_db()
        did = mdb.upsert_dataset(
            name="test",
            task_type="classification",
            n_samples=100,
            n_features=5,
            meta_features={"n_samples": 100.0},
            embedding=[0.1, 0.2, 0.3],
            source="test",
        )
        assert len(did) == 36
        ds = mdb.get_dataset(did)
        assert ds is not None
        assert ds["name"] == "test"

    def test_experiment_lifecycle(self, tmp_path, monkeypatch):
        import core.tracking.meta_db as mdb
        monkeypatch.setattr(mdb, "DB_PATH", tmp_path / "test2.db")
        mdb.initialize_db()
        did = mdb.upsert_dataset("ds", "classification", 50, 3, {}, source="test")
        eid = mdb.create_experiment(did)
        assert len(eid) == 36
        mdb.update_experiment(eid, status="done", n_trials=10)
        exp = mdb.get_experiment(eid)
        assert exp["status"] == "done"
        assert exp["n_trials"] == 10
