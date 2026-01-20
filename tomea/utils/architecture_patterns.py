"""
Architecture-specific patterns and metadata.

Centralizes knowledge about different model architectures.
"""

ARCHITECTURE_PATTERNS = {
    'bert': {
        'attention_path': 'encoder.layer[{i}].attention.self',
        'layer_count_attr': 'config.num_hidden_layers',
        'attention_module_names': ['query', 'key', 'value'],
        'model_prefix': 'bert',
        'config_class': 'BertConfig'
    },
    
    'roberta': {
        'attention_path': 'encoder.layer[{i}].attention.self',
        'layer_count_attr': 'config.num_hidden_layers',
        'attention_module_names': ['query', 'key', 'value'],
        'model_prefix': 'roberta',
        'config_class': 'RobertaConfig'
    },
    
    'gpt2': {
        'attention_path': 'h[{i}].attn',
        'layer_count_attr': 'config.n_layer',
        'attention_module_names': ['c_attn', 'c_proj'],
        'model_prefix': 'transformer',
        'config_class': 'GPT2Config'
    },
    
    't5': {
        'attention_path': 'encoder.block[{i}].layer[0].SelfAttention',
        'layer_count_attr': 'config.num_layers',
        'attention_module_names': ['q', 'k', 'v', 'o'],
        'model_prefix': 'encoder',
        'config_class': 'T5Config'
    },
    
    'llama': {
        'attention_path': 'model.layers[{i}].self_attn',
        'layer_count_attr': 'config.num_hidden_layers',
        'attention_module_names': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'model_prefix': 'model',
        'config_class': 'LlamaConfig'
    },
}


def get_architecture_pattern(arch_type: str) -> dict:
    """
    Get architecture pattern metadata.
    
    Args:
        arch_type: Architecture type ("bert", "gpt2", etc.)
    
    Returns:
        Dictionary with architecture metadata
    
    Raises:
        KeyError: If architecture not found
    """
    if arch_type not in ARCHITECTURE_PATTERNS:
        raise KeyError(f"Unknown architecture: {arch_type}")
    
    return ARCHITECTURE_PATTERNS[arch_type]


def get_attention_path(arch_type: str, layer_idx: int) -> str:
    """
    Get attention path for specific layer.
    
    Args:
        arch_type: Architecture type
        layer_idx: Layer index (0-based)
    
    Returns:
        Path to attention module (e.g., "encoder.layer[0].attention.self")
    """
    pattern = get_architecture_pattern(arch_type)
    return pattern['attention_path'].format(i=layer_idx)


def get_layer_count_attr(arch_type: str) -> str:
    """Get attribute name for layer count."""
    pattern = get_architecture_pattern(arch_type)
    return pattern['layer_count_attr']