"""
Local validation using tiny models.

Tests adapters locally before sending to expensive Modal GPU.
"""

import os
import torch
import logging
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import tempfile
import sys

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of local validation."""
    success: bool
    error: Optional[str] = None
    error_type: Optional[str] = None  # "NameError", "AttributeError", etc.


# Tiny models for testing
TINY_MODELS = {
    'bert': 'prajjwal1/bert-tiny',           # 4MB
    'gpt2': 'sshleifer/tiny-gpt2',           # 11MB  
    't5': 'google/t5-small',                 # 60MB
    'llama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # 1.1GB
}


class LocalValidator:
    """
    Validates adapter code locally using tiny models.
    
    This is much faster than sending to Modal (30s vs 5min).
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            cache_dir: Where to cache tiny models (default: ~/.cache/tomea)
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser('~/.cache/tomea')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track which models we've downloaded
        self._downloaded = set()
    
    def test_adapter(
        self,
        adapter_code: str,
        base_model_type: str = 'bert'
    ) -> ValidationResult:
        """
        Test adapter code with tiny model.
        
        Args:
            adapter_code: Generated adapter code (with get_model function)
            base_model_type: Architecture type ("bert", "gpt2", etc.)
        
        Returns:
            ValidationResult with success status
        """
        # Get tiny model name
        if base_model_type not in TINY_MODELS:
            logger.warning(f"No tiny model for {base_model_type}, using bert")
            base_model_type = 'bert'
        
        tiny_model_name = TINY_MODELS[base_model_type]
        
        # Download tiny model if needed
        self._ensure_model_downloaded(tiny_model_name)
        
        # Test in isolated environment
        try:
            # Create temp module
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                dir=self.cache_dir
            ) as f:
                f.write(adapter_code)
                temp_file = f.name
            
            # Try to import and execute
            result = self._execute_adapter(temp_file, tiny_model_name)
            
            # Cleanup
            os.unlink(temp_file)
            
            return result
        
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__
            )
    
    def _ensure_model_downloaded(self, model_name: str):
        """Download tiny model if not cached."""
        if model_name in self._downloaded:
            return
        
        try:
            logger.info(f"Downloading tiny model: {model_name}")
            from transformers import AutoModelForSequenceClassification
            
            # Download to cache
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                cache_dir=self.cache_dir
            )
            
            self._downloaded.add(model_name)
            logger.info(f"Downloaded {model_name}")
        
        except Exception as e:
            logger.warning(f"Failed to download {model_name}: {e}")
    
    def _execute_adapter(
        self,
        adapter_file: str,
        tiny_model_name: str
    ) -> ValidationResult:
        """
        Execute adapter code in isolated environment.
        
        Tests:
        1. Import succeeds
        2. get_model() exists
        3. Model loads
        4. Forward pass works
        """
        # Import adapter as module
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("test_adapter", adapter_file)
        if spec is None or spec.loader is None:
            return ValidationResult(
                success=False,
                error="Failed to load adapter module"
            )
        
        module = importlib.util.module_from_spec(spec)
        
        try:
            # Execute module (loads imports, defines functions)
            spec.loader.exec_module(module)
        except Exception as e:
            return ValidationResult(
                success=False,
                error=f"Import failed: {e}",
                error_type=type(e).__name__
            )
        
        # Check get_model exists
        if not hasattr(module, 'get_model'):
            return ValidationResult(
                success=False,
                error="Module does not define get_model() function"
            )
        
        try:
            # Call get_model
            model = module.get_model(tiny_model_name, num_labels=2)
        except Exception as e:
            return ValidationResult(
                success=False,
                error=f"get_model() failed: {e}",
                error_type=type(e).__name__
            )
        
        # Try forward pass with dummy data
        try:
            # Create dummy input
            dummy_input = {
                'input_ids': torch.randint(0, 1000, (1, 10)),
                'attention_mask': torch.ones(1, 10, dtype=torch.long)
            }
            
            # Forward pass
            with torch.no_grad():
                output = model(**dummy_input)

            has_logits = False
            
            # Case 1: Standard Output Object
            if hasattr(output, 'logits'):
                has_logits = True
                
            # Case 2: Tuple Output (common in older models or return_dict=False)
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                # Assuming first element is logits [batch, num_labels]
                if isinstance(output[0], torch.Tensor):
                    has_logits = True
                    
            if not has_logits:
                return ValidationResult(
                    success=False, 
                    error=f"Model output format unknown: {type(output)}. Expected 'logits' attribute or tuple."
                )
            
            
            
            return ValidationResult(success=True)
        
        except Exception as e:
            return ValidationResult(
                success=False,
                error=f"Forward pass failed: {e}",
                error_type=type(e).__name__
            )