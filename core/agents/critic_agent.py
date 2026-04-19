"""
MetaLearnX — Critic Agent
Analyzes trial performance and suggests architectural refinements.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.agents.agent_base import BaseAgent


class CritiqueSchema(BaseModel):
    """Schema for pipeline critique and refinement suggestions."""
    issue_identified: str = Field(..., description="Description of the performance bottleneck or failure mode identified")
    suggested_architectural_change: Optional[str] = Field(None, description="A conceptual change (e.g., 'Switch to TargetEncoder')")
    hyperparameter_adjustment: Optional[str] = Field(None, description="Advice for the optimizer (e.g., 'Increase max_depth to capture complexity')")
    should_pivot_model: bool = Field(..., description="True if the current model family is likely a poor fit for the manifold")
    alternative_model: Optional[str] = Field(None, description="Suggested alternative model family if pivoting")
    reasoning: str = Field(..., description="Academic justification for the critique")


class CriticAgent(BaseAgent):
    """
    Research Critic responsible for analyzing empirical results and suggesting pivots.
    Acts as a 'Peer Reviewer' for the Architect's designs.
    """

    def __init__(self, model_name: str = "groq/llama3-70b-8192"):
        super().__init__(
            name="Critic",
            role="Elite Research Peer Reviewer & Statistical Advisor",
            model_name=model_name
        )

    def act(self, context: Dict[str, Any]) -> Optional[CritiqueSchema]:
        """
        Analyze current experiment results and suggest refinements.
        """
        current_metrics = context.get("metrics")
        current_config = context.get("config")
        trial_logs = context.get("trial_logs", [])
        dataset_profile = context.get("profile")

        if not current_metrics or not current_config:
            return None

        system_prompt = (
            "Analyze the provided ML trial results and determine if the current architecture is optimal.\n"
            "If the score is low relative to the complexity or if the training time is excessive, "
            "recommend actionable pivots.\n\n"
            "You must decide if we should stay with the current model family or pivot to a different one."
        )

        user_content = (
            f"Dataset Profile:\n{json.dumps(dataset_profile, indent=2)}\n\n"
            f"Current Pipeline Architecture:\n{json.dumps(current_config, indent=2)}\n\n"
            f"Performance Metrics:\n{json.dumps(current_metrics, indent=2)}\n"
        )
        
        if trial_logs:
            user_content += f"\nTrial Log Summaries:\n{json.dumps(trial_logs[-5:], indent=2)}\n"

        return self._call_llm(system_prompt, user_content, CritiqueSchema)
