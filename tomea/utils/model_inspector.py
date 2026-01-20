from transformers import AutoConfig
import logging

logger = logging.getLogger(__name__)

def inspect_model(model_name: str) -> dict:
    """
    Fetches architectural specs for a HuggingFace model.
    Returns a dictionary of critical dimensions.
    """
    try:
        config = AutoConfig.from_pretrained(model_name)
        
        specs = {
            "model_type": getattr(config, "model_type", "unknown"),
            "hidden_size": getattr(config, "hidden_size", getattr(config, "d_model", "UNKNOWN")),
            "num_attention_heads": getattr(config, "num_attention_heads", "UNKNOWN"),
            "vocab_size": getattr(config, "vocab_size", "UNKNOWN"),
            "intermediate_size": getattr(config, "intermediate_size", "UNKNOWN")
        }
        
        # Helper for LLM readability
        specs_str = (
            f"Model Type: {specs['model_type']}\n"
            f"Hidden Size (d_model): {specs['hidden_size']}\n"
            f"Attention Heads: {specs['num_attention_heads']}\n"
            f"Feed Forward Size: {specs['intermediate_size']}"
        )
        
        return {"dict": specs, "str": specs_str}
        
    except Exception as e:
        logger.warning(f"Failed to inspect model {model_name}: {e}")
        return {"str": "Could not fetch model specs. Assume standard BERT-base (768 hidden)."}