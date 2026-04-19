"""
MetaLearnX — Agent Factory
Centralized manager for the specialized research agent swarm.
"""

from __future__ import annotations

from typing import Optional
from core.agents.architect_agent import ArchitectAgent
from core.agents.critic_agent import CriticAgent


class AgentFactory:
    """
    Manages the lifecycle of specialized agents.
    """
    _instance: Optional[AgentFactory] = None

    def __init__(self):
        self.architect = ArchitectAgent()
        self.critic = CriticAgent()

    @classmethod
    def get_instance(cls) -> AgentFactory:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_architect() -> ArchitectAgent:
    return AgentFactory.get_instance().architect


def get_critic() -> CriticAgent:
    return AgentFactory.get_instance().critic
