"""Base template interface for all method implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AdapterCode:
    """Generated adapter code with metadata."""
    code: str
    method_type: str
    estimated_loc: int
    required_imports: List[str]
    config_params: Dict[str, any]


class BaseTemplate(ABC):
    """Base class for all method templates."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.max_tokens = 1000  # Default max for LLM generation
    
    @abstractmethod
    def generate_adapter(self, repo_map: str, context: Dict) -> AdapterCode:
        """
        Generate adapter code from repo map and context.
        
        Args:
            repo_map: Structured repository map from Cartographer
            context: Additional context (paper abstract, classification, etc.)
        
        Returns:
            AdapterCode with generated implementation
        """
        pass
    
    @abstractmethod
    def get_required_files(self, repo_map: str) -> List[str]:
        """
        Identify which files from repo are needed.
        
        Args:
            repo_map: Repository structure
        
        Returns:
            List of file paths to extract
        """
        pass
    
    def validate_generated_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Template-specific validation.
        
        Args:
            code: Generated adapter code
        
        Returns:
            (is_valid, error_message)
        """
        # Default: check for required functions
        required_functions = ['get_config', 'get_model']
        
        for func in required_functions:
            if f'def {func}(' not in code:
                return False, f"Missing required function: {func}"
        
        return True, None