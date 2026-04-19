"""
MetaLearnX — Base Agent Interface
Provides unified LLM access with structured output support and retry logic.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

T = TypeVar("T", bound=BaseModel)

class BaseAgent(ABC):
    """
    Abstract base class for all research agents in the MetaLearnX swarm.
    """

    def __init__(
        self, 
        name: str, 
        role: str, 
        model_name: str = "groq/llama3-70b-8192",
        temperature: float = 0.1
    ):
        self.name = name
        self.role = role
        self.model_name = model_name
        self.temperature = temperature

    def _call_llm(
        self, 
        system_prompt: str, 
        user_content: str, 
        response_model: Type[T]
    ) -> Optional[T]:
        """
        Executes a structured LLM call via LiteLLM.
        """
        if not LITELLM_AVAILABLE:
            logger.warning(f"Agent '{self.name}': LiteLLM not available.")
            return None

        messages = [
            {"role": "system", "content": f"Role: {self.role}\n\n{system_prompt}"},
            {"role": "user", "content": user_content}
        ]

        max_retries = 3
        current_temp = self.temperature

        for attempt in range(max_retries):
            try:
                logger.info(f"Agent '{self.name}' invoking LLM ({self.model_name}) [Attempt {attempt+1}/{max_retries}]...")
                response = litellm.completion(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_model,
                    temperature=current_temp,
                    max_tokens=1000
                )

                content = response.choices[0].message.content
                return response_model.model_validate_json(content)

            except ValidationError as e:
                logger.warning(f"Agent '{self.name}' schema validation failed on attempt {attempt+1}: {e}")
                # Append error to context to force LLM to correct itself
                messages.append({"role": "assistant", "content": content if 'content' in locals() else "{}"})
                messages.append({"role": "user", "content": f"CRITICAL ERROR: Your output violated the schema format. Eradicate all non-JSON text. Details: {str(e)}"})
                current_temp *= 0.5  # Temperature Decay for determinism
            except Exception as e:
                logger.error(f"Agent '{self.name}' LLM API call failed: {e}")
                import time
                time.sleep(2) # Backoff for rate limits

        logger.error(f"Agent '{self.name}' failed to generate valid output after {max_retries} attempts.")
        return None

    @abstractmethod
    def act(self, context: Dict[str, Any]) -> Any:
        """Execute the agent's primary logic."""
        pass
