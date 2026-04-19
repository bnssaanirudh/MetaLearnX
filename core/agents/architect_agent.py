"""
MetaLearnX — Architect Agent
Synthesizes the initial ML pipeline architecture using dataset intelligence.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.agents.agent_base import BaseAgent


class PipelineArchitectureSchema(BaseModel):
    """Schema for the full ML pipeline design."""
    imputer: str = Field(..., description="Imputation strategy: 'mean', 'median', 'most_frequent', 'constant'")
    scaler: str = Field(..., description="Scaling strategy: 'standard', 'robust', 'minmax', 'none'")
    encoder: str = Field(..., description="Categorical encoding: 'ordinal', 'onehot', 'target', 'none'")
    feature_selector: str = Field(..., description="Feature selection: 'kbest', 'from_model', 'none'")
    selector_k: int = Field(..., description="Number of features to keep (if kbest is used)")
    model_name: str = Field(..., description="Model family from registry: 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'svm', 'mlp'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Numerical confidence score (0.0-1.0) in this pipeline design")
    reasoning: str = Field(..., description="Technical justification for this specific architecture design")


class ArchitectAgent(BaseAgent):
    """
    Principal Architect responsible for designing the end-to-end ML pipeline.
    Combines statistical meta-features with research-grade heuristics.
    """

    def __init__(self, model_name: str = "groq/llama3-70b-8192"):
        super().__init__(
            name="Architect",
            role="Elite Lead Machine Learning Architect",
            model_name=model_name
        )

    def act(self, context: Dict[str, Any]) -> Optional[PipelineArchitectureSchema]:
        """
        Synthesize initial pipeline architecture based on dataset context.
        """
        dataset_profile = context.get("profile")
        description = context.get("description")
        recommendations = context.get("recommendations", [])

        if not dataset_profile:
            return None

        system_prompt = (
            "Design a high-performance scikit-learn compatible pipeline architecture.\n\n"
            "Guidelines:\n"
            "1. For high cardinality (>20 categories), use 'target' or 'onehot' carefully.\n"
            "2. For outliers, prioritize 'robust' scaling.\n"
            "3. For high-dimensional datasets (>100 features), implement 'from_model' or 'kbest' selection.\n"
            "4. Match the model choice to the task type and sample size.\n"
        )

        user_content = f"Dataset Profile:\n{json.dumps(dataset_profile, indent=2)}\n"
        if description:
            user_content += f"\nDataset Description: {description}\n"
        if recommendations:
            user_content += f"\nZero-Shot Recommendations: {json.dumps(recommendations)}\n"

        return self._call_llm(system_prompt, user_content, PipelineArchitectureSchema)
