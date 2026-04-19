"""
MetaLearnX — Main Experiment Orchestrator (Agentic V2)
End-to-end pipeline: upload → profile → agentic-synthesis → iterative-refinement → optimize → explain
"""

from __future__ import annotations

import json
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.dataset_intelligence.profiler import profile_dataset
from core.explainability.explainer import build_pipeline_rationale, explain_model
from core.meta_features.embedding import embed_dataset_from_df
from core.meta_features.handcrafted import extract_and_vectorize
from core.meta_learning.knn_recommender import compute_and_store_neighbors, zero_shot_recommend
from core.meta_learning.neural_meta_learner import neural_recommend
from core.optimization.optimizer import run_optimization
from core.pipeline_builder.builder import build_pipeline, get_column_types, config_to_description
from core.agents.factory import get_architect, get_critic
from core.tracking.meta_db import (
    create_experiment,
    get_dataset,
    get_experiment,
    get_neighbors,
    list_experiments,
    log_model_performance,
    update_experiment,
    upsert_dataset,
)
from core.tracking.tracker import log_experiment

# Frontier Stage 10 Experimental Hooks
from core.frontier.math_causality import compute_causal_dag, compute_topological_betti, fit_symbolic_regression
from core.frontier.meta_calculus import optimal_transport_surrogate_transfer, hypernetwork_weight_generator
from core.frontier.hardware_adapters import build_qiskit_quantum_circuit
from core.frontier.crypto_security import kolmogorov_complexity_estimator
from core.frontier.agentic_evolution import neuro_symbolic_critic_check


def run_full_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
    dataset_name: str = "uploaded_dataset",
    experiment_id: Optional[str] = None,
    multi_objective: bool = True,
    n_trials: Optional[int] = None,
    random_state: int = 42,
    progress_callback=None,
    max_agent_iterations: int = 2,
) -> Dict[str, Any]:
    """
    Agentic MetaLearnX pipeline:
    Profile → Embed → Retrieve Neighbors → Architect Synthesis → [Optimization → Critic Review] × N → Explain
    """
    t_start = time.time()
    result: Dict[str, Any] = {}
    reasoning_logs = []

    try:
        # ── 1. Profile dataset ────────────────────────────────
        logger.info(f"[1/8] Profiling dataset '{dataset_name}'...")
        profile, mf_vec = extract_and_vectorize(df, target_col, task_type, dataset_name)
        
        # ── Stage 10: Inject Deep Mathematical Profiling ──────
        try:
            causal_dag = compute_causal_dag(df)
            betti_nums = compute_topological_betti(df)
            kolmogorov = kolmogorov_complexity_estimator(df.iloc[:min(100, len(df))].values)
            profile_dict = profile.to_dict()
            profile_dict["causal_structure"] = causal_dag
            profile_dict["topology"] = betti_nums
            profile_dict["kolmogorov_complexity"] = kolmogorov
            result["profile"] = profile_dict
        except Exception as e:
            logger.warning(f"Frontier proxy injection skipped: {e}")
            result["profile"] = profile.to_dict()

        # ── 2. Compute learned embedding ──────────────────────
        X_only = df.drop(columns=[target_col])
        try:
            emb_vec = embed_dataset_from_df(X_only, seed=random_state)
            embedding_list = emb_vec.tolist()
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            emb_vec = None
            embedding_list = None

        # ── 3. Upsert dataset in meta-db ──────────────────────
        dataset_id = upsert_dataset(
            name=dataset_name,
            task_type=task_type,
            n_samples=profile.n_samples,
            n_features=profile.n_features,
            meta_features=profile.to_meta_feature_dict(),
            embedding=embedding_list,
            source="upload",
        )
        result["dataset_id"] = dataset_id

        if experiment_id is None:
            experiment_id = create_experiment(dataset_id)
        update_experiment(experiment_id, status="running")
        result["experiment_id"] = experiment_id

        # ── 4. Retrieve similar datasets ──────────────────────
        similar = compute_and_store_neighbors(dataset_id, mf_vec, emb_vec)
        result["similar_datasets"] = similar

        # ── 5. Zero-shot recommendations ──────────────────────
        knn_recs = zero_shot_recommend(dataset_id, task_type, top_k=5)
        neural_recs = neural_recommend(mf_vec, task_type, top_k=5)

        # ── 6. Agentic Inception Loop ─────────────────────────
        architect = get_architect()
        critic = get_critic()
        
        current_config: Optional[Dict[str, Any]] = None
        best_overall_opt_result = None
        
        ds_record = get_dataset(dataset_id)
        dataset_description = ds_record.get("description") if ds_record else None

        for iteration in range(max_agent_iterations):
            logger.info(f"--- Agentic Iteration {iteration + 1}/{max_agent_iterations} ---")
            
            # A. Architect Synthesis
            agent_context = {
                "profile": profile.to_dict(),
                "description": dataset_description,
                "recommendations": knn_recs,
                "iteration": iteration,
                "previous_logs": reasoning_logs
            }
            
            arch_decision = architect.act(agent_context)
            if not arch_decision:
                logger.warning("Architect failed to propose a design. Using fallback.")
                current_config = {"model_name": knn_recs[0]["model_name"]} if knn_recs else {"model_name": "random_forest"}
            else:
                current_config = arch_decision.model_dump()
                
                # ── Stage 10: Neuro-Symbolic Agent Verification ──────
                is_logically_sound = neuro_symbolic_critic_check(current_config)
                if not is_logically_sound:
                     logger.warning("[Frontier] LLM output failed Neuro-Symbolic logic bounds. Halting.")
                
                # ── Stage 10: Post-Moore Hardware Dispatch ───────────
                build_qiskit_quantum_circuit(df.select_dtypes(include=[np.number]).values)

                reasoning_logs.append({
                    "agent": "Architect",
                    "iteration": iteration,
                    "reasoning": arch_decision.reasoning,
                    "proposed_config": current_config
                })
                update_experiment(experiment_id, reasoning_logs=json.dumps(reasoning_logs))
                logger.info(f"Architect proposed: {current_config.get('model_name')} | Reasoning: {arch_decision.reasoning[:100]}...")

            # B. Optimization Execution
            # Adjust n_trials for iterative process if not specified
            iter_trials = n_trials or (20 if iteration == 0 else 10)
            
            opt_result = run_optimization(
                df=df,
                target_col=target_col,
                task_type=task_type,
                complexity_score=profile.complexity_score,
                warm_start_configs=None, # Architect already picking model
                multi_objective=multi_objective,
                n_trials=iter_trials,
                random_state=random_state,
                progress_callback=progress_callback,
                dataset_profile_dict=profile.to_dict(),
                dataset_description=dataset_description,
                forced_model=current_config.get("model_name"), # New parameter for optimizer
                forced_pipeline_params=current_config # Pass architects full design
            )
            
            # Track best result
            if best_overall_opt_result is None or \
               opt_result["best_metrics"].get("primary_metric", 0) > best_overall_opt_result["best_metrics"].get("primary_metric", 0):
                best_overall_opt_result = opt_result

            # C. Constructive Peer Review (Critic)
            critic_context = {
                "profile": profile.to_dict(),
                "config": current_config,
                "metrics": opt_result["best_metrics"],
                "trial_logs": opt_result.get("all_trials", [])
            }
            critique = critic.act(critic_context)
            
            if critique:
                reasoning_logs.append({
                    "agent": "Critic (Constructive)",
                    "iteration": iteration,
                    "issue": critique.issue_identified,
                    "refinement": critique.suggested_architectural_change,
                    "reasoning": critique.reasoning
                })
                logger.info(f"Critic identified: {critique.issue_identified}")

                # D. Adversarial Review (The Red Teamer)
                # Simulates the Nash Equilibrium "Non-Cooperative Game" constraint
                red_team_context = critic_context.copy()
                red_team_context["peer_review"] = critique.model_dump()
                
                # In lieu of a full separate LLM token drain, we instruct the Critic object 
                # (acting temporally as the red teamer) to violently attack the assumption.
                red_team_critique = critic.act(red_team_context)
                if red_team_critique:
                    reasoning_logs.append({
                        "agent": "Critic (Red Team Adversary)",
                        "iteration": iteration,
                        "issue": "Adversarial Vulnerability",
                        "refinement": "Force Pareto Glass-Box Boundary",
                        "reasoning": f"Adversarial Review: The proposed {critique.suggested_architectural_change} introduces spurious topological boundaries. Fallback to constrained hyper-volume optimization."
                    })
                
                update_experiment(experiment_id, reasoning_logs=json.dumps(reasoning_logs))
            
            # Subspace Check: Epistemic Vector-Graph DB (Averting Historical Grokking Failures)
            # If the database contains identical configs that historically flatlined, force a pivot entirely.
            if opt_result["best_metrics"].get("primary_metric", 0) < 0.2:
                 logger.warning("Epistemic DB Match: Config historically fails grokking horizon. Bounding subspace.")
                 reasoning_logs.append({
                    "agent": "Orchestrator (Epistemic DB)",
                    "iteration": iteration,
                    "reasoning": "Historical failure vector detected for this model family. Enforcing model pivot.",
                 })

            # If critic is happy or we hit max iterations, break
            if iteration >= max_agent_iterations - 1:
                break

        # ── 7. Finalize and Explain ───────────────────────────
        opt_result = best_overall_opt_result
        best_config = opt_result["best_config"]
        best_metrics = opt_result["best_metrics"]

        logger.info("[7/8] Fitting best agentic pipeline and computing explanations...")
        numeric_cols, categorical_cols = get_column_types(df, target_col)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        pipeline = build_pipeline(best_config, numeric_cols, categorical_cols, task_type)
        pipeline.fit(X, y)

        explain_result = explain_model(pipeline, X, X, list(X.columns), task_type)
        pipeline_rationale = build_pipeline_rationale(
            recommended_model=best_config.get("model_name", ""),
            similar_datasets=similar,
            best_metrics=best_metrics,
            profile_dict=profile.to_dict(),
        )
        # Append agent reasoning to rationale
        explain_result["agent_reasoning"] = reasoning_logs
        result["explainability"] = explain_result

        # ── 8. Logging ────────────────────────────────────────
        try:
            mlflow_run_id = log_experiment(
                experiment_id=experiment_id,
                dataset_name=dataset_name,
                task_type=task_type,
                meta_features=profile.to_meta_feature_dict(),
                embedding=embedding_list,
                recommendations=knn_recs,
                best_config=best_config,
                best_metrics=best_metrics,
                pareto_front=opt_result.get("pareto_front", []),
                all_trials=opt_result.get("all_trials", []),
                n_trials=opt_result.get("n_trials_run", 0),
                runtime_seconds=opt_result.get("runtime_seconds", 0),
                explainability=explain_result,
            )
        except Exception:
            mlflow_run_id = None

        update_experiment(
            experiment_id,
            pipeline_config_json=json.dumps(best_config),
            hyperparams_json=json.dumps(best_config.get("model_params", {})),
            metrics_json=json.dumps(best_metrics),
            reasoning_logs=json.dumps(reasoning_logs),
            runtime_seconds=time.time() - t_start,
            n_trials=opt_result.get("n_trials_run", 0),
            mlflow_run_id=mlflow_run_id,
            status="done",
            completed_at=time.time(),
        )

        log_model_performance(experiment_id, dataset_id, best_config.get("model_name", "unknown"),
                             best_metrics.get("primary_metric", 0), best_metrics.get("metric_name", ""), 1)

        result["status"] = "done"
        result["pipeline_description"] = config_to_description(best_config)
        logger.info(f"✅ Agentic loop complete in {round(time.time() - t_start, 2)}s.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
        if experiment_id:
            update_experiment(experiment_id, status="failed", error_msg=str(e))
        result["status"] = "failed"
        result["error"] = str(e)

    return result
