from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseHarness(ABC):
    """
    The Interface for all Research Subcontractors.
    """
    
    def __init__(self, paper_context: Dict[str, Any]):
        self.context = paper_context

    @property
    @abstractmethod
    def mode(self) -> str:
        """
        Declares the training paradigm this harness operates in.
        Used by DataEngine to select the correct mapping strategy.
        Must be one of: "sft", "dpo", "ppo", "grpo"
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Returns the specific persona (e.g., 'You are an RL Engineer')"""
        pass

    @abstractmethod
    def get_template(self) -> str:
        """Returns the Python skeleton/boilerplate for this domain"""
        pass

    @abstractmethod
    def validate_logs(self, logs: str) -> bool:
        """
        Custom logic to check if a run is healthy.
        Returns True if healthy, False if crashed.
        """
        pass