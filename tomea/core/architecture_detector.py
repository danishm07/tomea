"""
Architecture detection from model names.

Detects which base architecture a model uses (BERT, GPT-2, etc.)
"""

from typing import Optional


def detect_architecture(model_name: str) -> str:
    """
    Detect architecture from model name.
    
    Args:
        model_name: HuggingFace model name (e.g., "bert-base-uncased")
    
    Returns:
        Architecture type: "bert", "gpt2", "t5", "llama", or "unknown"
    
    Examples:
        >>> detect_architecture("bert-base-uncased")
        'bert'
        >>> detect_architecture("gpt2-medium")
        'gpt2'
        >>> detect_architecture("roberta-base")
        'bert'  # RoBERTa uses BERT architecture
    """
    model_lower = model_name.lower()
    
    # BERT family (includes RoBERTa, DistilBERT, ALBERT)
    if any(x in model_lower for x in ['bert', 'roberta', 'distilbert', 'albert']):
        return 'bert'
    
    # GPT family
    elif any(x in model_lower for x in ['gpt', 'gpt2', 'gpt-2']):
        return 'gpt2'
    
    # T5 family
    elif 't5' in model_lower or 'flan' in model_lower:
        return 't5'
    
    # LLaMA family (includes Mistral, Qwen)
    elif any(x in model_lower for x in ['llama', 'mistral', 'qwen']):
        return 'llama'
    
    # Unknown
    else:
        return 'unknown'


def infer_from_config(config_dict: dict) -> Optional[str]:
    """
    Infer architecture from config dictionary.
    
    Fallback method when model name doesn't match patterns.
    
    Args:
        config_dict: Model config as dictionary
    
    Returns:
        Architecture type or None
    """
    # Check for architecture-specific attributes
    if 'num_hidden_layers' in config_dict and 'hidden_size' in config_dict:
        # BERT-style
        if 'encoder' in str(config_dict.get('architectures', [])):
            return 'bert'
    
    if 'n_layer' in config_dict and 'n_embd' in config_dict:
        # GPT-2 style
        return 'gpt2'
    
    if 'num_layers' in config_dict and 'd_model' in config_dict:
        # T5 style
        return 't5'
    
    return None