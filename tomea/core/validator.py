"""Pre-flight validation for generated code."""

import ast
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidationResult:
    """Result of validation check."""
    passed: bool
    level: str
    error: Optional[str] = None


# Packages only available in Modal environment
MODAL_ONLY_PACKAGES = {
    'peft',
    'datasets',
    'transformers', 
    'torch',
    'accelerate',
    'evaluate',
    'sentencepiece',
    'protobuf'
}


class PreFlightValidator:
    """
    Pre-flight validation to catch errors before GPU execution.
    
    Validates through multiple levels:
    1. Syntax (ast.parse)
    2. Imports (skip Modal-only packages)
    3. Interface (has required functions)
    """
    
    def validate(self, code: str, levels: List[str] = None) -> ValidationResult:
        """
        Validate generated code.
        
        Args:
            code: Python code to validate
            levels: List of validation levels to run
                   Options: ['syntax', 'imports', 'interface']
                   Default: ['syntax'] (imports available in Modal)
        
        Returns:
            ValidationResult with pass/fail and error details
        """
        if levels is None:
            # Only check syntax - imports will be available in Modal
            levels = ['syntax']
        
        # Level 1: Syntax
        if 'syntax' in levels:
            result = self._validate_syntax(code)
            if not result.passed:
                return result
        
        # Level 2: Imports
        if 'imports' in levels:
            result = self._validate_imports(code)
            if not result.passed:
                return result
        
        # Level 3: Interface
        if 'interface' in levels:
            result = self._validate_interface(code)
            if not result.passed:
                return result
        
        return ValidationResult(passed=True, level='all')
    
    def _validate_syntax(self, code: str) -> ValidationResult:
        """Level 1: Check Python syntax."""
        try:
            ast.parse(code)
            return ValidationResult(passed=True, level='syntax')
        except SyntaxError as e:
            return ValidationResult(
                passed=False,
                level='syntax',
                error=f"Syntax error at line {e.lineno}: {e.msg}"
            )
    
    def _validate_imports(self, code: str) -> ValidationResult:
        """
        Level 2: Check if imports would work.
        
        Skips Modal-only packages that aren't available locally.
        """
        try:
            tree = ast.parse(code)
            imports = []
            
            # Extract all imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
            
            # Test imports (skip Modal-only)
            for module_name in set(imports):
                if module_name in MODAL_ONLY_PACKAGES:
                    continue  # Skip - available in Modal
                
                try:
                    __import__(module_name)
                except ImportError as e:
                    return ValidationResult(
                        passed=False,
                        level='imports',
                        error=f"Import error: {e}"
                    )
            
            return ValidationResult(passed=True, level='imports')
        
        except Exception as e:
            return ValidationResult(
                passed=False,
                level='imports',
                error=f"Import validation error: {e}"
            )
    
    def _validate_interface(self, code: str) -> ValidationResult:
        """Level 3: Check required functions exist."""
        try:
            tree = ast.parse(code)
            
            # Required functions for adapters
            required_functions = {'get_model'}
            found_functions = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    found_functions.add(node.name)
            
            missing = required_functions - found_functions
            if missing:
                return ValidationResult(
                    passed=False,
                    level='interface',
                    error=f"Missing required functions: {missing}"
                )
            
            return ValidationResult(passed=True, level='interface')
        
        except Exception as e:
            return ValidationResult(
                passed=False,
                level='interface',
                error=f"Interface validation error: {e}"
            )