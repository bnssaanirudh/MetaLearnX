import sys
sys.path.insert(0, ".")
import numpy as np

# Test knowledge graph
from core.meta_learning.knowledge_graph import DatasetKnowledgeGraph
g = DatasetKnowledgeGraph()
g.add_or_update_node("id1", "iris", np.random.randn(33).astype(np.float32), "random_forest", 0.95, "classification")
g.add_or_update_node("id2", "wine", np.random.randn(33).astype(np.float32), "xgboost", 0.91, "classification")
g.add_or_update_node("id3", "diabetes", np.random.randn(33).astype(np.float32), "lightgbm", 0.82, "regression")
js = g.to_json()
print("OK: Knowledge Graph - nodes={}, edges={}".format(js["n_nodes"], js["n_edges"]))

# Test continual learner
from core.meta_learning.continual_learner import ContinualMetaLearner
cl = ContinualMetaLearner()
stats = cl.get_stats()
print("OK: Continual Learner - updates={}, buffer_size={}".format(stats["n_updates"], stats["replay_buffer"]["size"]))

# Test search space pruner
from core.optimization.search_space_learner import SearchSpacePruner
pruner = SearchSpacePruner()
pruned = pruner.get_pruned_space(np.zeros(33, dtype=np.float32), "random_forest")
print("OK: Search Space Pruner - rf params: {}".format(list(pruned.keys())))

# Test uncertainty
from core.meta_learning.uncertainty import uncertain_recommend
res = uncertain_recommend(np.zeros(33, dtype=np.float32), n_passes=10)
print("OK: Uncertainty - global={}, recs={}".format(res["global_uncertainty"], len(res["recommendations"])))

# Test few-shot
from core.meta_learning.few_shot import detect_few_shot_regime
regime = detect_few_shot_regime(30, 5)
print("OK: Few-Shot - regime={}, safe_models={}".format(regime["regime"], regime["safe_models"]))

# Test federated (just instantiation, not full run)
from core.meta_learning.federated import FederatedMetaLearner
fl = FederatedMetaLearner(n_rounds=2, n_orgs=3)
print("OK: Federated - ready, n_rounds={}, n_orgs={}".format(fl.n_rounds, fl.n_orgs))

print("\nAll advanced modules smoke-tested OK!")
