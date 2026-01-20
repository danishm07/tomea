import logging
from langchain_openai import ChatOpenAI
from tomea.utils.arxiv_parser import get_paper_data
from tomea.core.local_validator import LocalValidator
from tomea.core.architecture_detector import detect_architecture

logger = logging.getLogger(__name__)

class PaperSynthesizer:
    def __init__(self, llm_client: ChatOpenAI):
        self.llm = llm_client
        self.validator = LocalValidator()
        
    def synthesize_from_arxiv(
        self, 
        arxiv_id: str, 
        base_model: str = "bert-base-uncased",
        model_specs: str = "",  # <--- NEW: Accepts blueprints
        status_callback=None
    ) -> str:
        """
        Main entry point: ArXiv ID -> Working Code.
        """
        logger.info(f"🧪 Synthesizer: Processing {arxiv_id} for {base_model}...")
        
        # 1. Get Paper Text
        paper_data = get_paper_data(arxiv_id)
        if not paper_data:
            raise ValueError(f"Could not fetch paper {arxiv_id}")
        
        paper_text = paper_data['text']
        # Truncate to avoid context limit (adjust based on model)
        if len(paper_text) > 40000: 
            paper_text = paper_text[:40000] 

        # 2. Detect Arch
        arch_type = detect_architecture(base_model)
        
        # 3. Generate Initial Code (With Specs)
        logger.info("   🧠 Reading paper and writing code...")
        code = self._generate_initial_code(paper_text, base_model, arch_type, model_specs)
        
        # 4. Validation & Healing Loop (Internal Safety Net)
        max_retries = 2 # Keep it tight
        for attempt in range(max_retries):
            # Check if valid python/basic import
            result = self.validator.test_adapter(code, base_model_type=arch_type)
            
            if result.success:
                return code
            
            logger.warning(f"   ⚠️ Code failed local validation, calling healer: {result.error}")
            logger.info("   🩹 Healing code...")
            code = self._heal_code(code, result.error, model_specs)
            
        return code # Return best effort, Engine loop handles the rest

    def _generate_initial_code(self, paper_text: str, base_model: str, arch_type: str, specs: str) -> str:
        
        prompt = f"""
You are a Senior Research Engineer. Implement a novel mechanism from a paper.

**TARGET ARCHITECTURE:**
- Base Model: {base_model} ({arch_type})
{specs}  <--- CRITICAL: LLM now knows dimensions (e.g. 768)

**PAPER TEXT:**
{paper_text}

**YOUR GOAL:**
Write a complete, self-contained Python script (PyTorch) that:
1. Implements the core innovation as a `nn.Module`.
2. Implements `get_model(base_model_name, num_labels)` to inject it.

**CRITICAL RULES:**
1. **Inheritance:** Inherit from the specific HuggingFace layer you replace.
2. **Signature:** `forward` MUST accept `*args` and `**kwargs`.
3. **Dimensions:** USE THE SPECS ABOVE. Do not hardcode 128 if hidden_size is 768.
   - If the paper introduces a projection, ensure input/output matches the base model's stream.
4. **Integration:** Replace layers in-place in `get_model`.
5. **NO HALLUCINATIONS:** 
    - `config` objects DO NOT have attributes like `all_head_size` or `d_head`.
    - YOU MUST CALCULATE IT: `head_dim = config.hidden_size // config.num_attention_heads`.
    - `self.head_dim` DOES NOT EXIST. Use `self.attention_head_size`.
    - `self.all_head_size` DOES NOT EXIST. Use `self.num_attention_heads * self.attention_head_size`.
    - `config.d_model` DOES NOT EXIST on BERT. Use `config.hidden_size`.
    - Do NOT assume you inherit helper methods like `transpose_for_scores`.
    - YOU MUST COPY-PASTE THE IMPLEMENTATION of `transpose_for_scores` into your class.
    - If you use it, you must define it.
6. **Argument Safety:** Your `forward()` methods MUST accept `*args` and `**kwargs`.
           - Reason: The `transformers` library updates frequently and injects new arguments (like 'cache_position', 'past_key_values').
           - Failure to do this causes immediate crashes on newer versions.
7. **Return Type Safety (CRITICAL):**
           - If replacing a BERT SelfAttention layer, your `forward` method MUST return a TUPLE, not a Tensor.
           - Correct: `return (context_layer,)` or `return (context_layer, attention_probs)`
           - Incorrect: `return context_layer`
           - Why: HuggingFace layers expect to unpack the output. Returning a raw tensor causes a crash.
8.  **Trainer Compatibility (CRITICAL):**
           - If you wrap the entire model (e.g., creating a `MoEBERTModel` class), your `forward` method MUST accept `labels=None`.
           - If `labels` are provided, you MUST calculate the CrossEntropyLoss.
           - Return format with labels: `return (loss, logits)`
           - Return format without labels: `return (logits,)`
9. **Activation Safety (CRITICAL):**
           - Do NOT call `get_activation()` or `super().get_activation()`. These do not exist.
           - To get the activation function (EXAMPLE!):
             `from transformers.activations import ACT2FN`
             `self.act_fn = ACT2FN[config.hidden_act]`
10. **Hygiene (CRITICAL):**
    - The `transformers` Trainer injects internal arguments like `num_items_in_batch`, `cache_position` and `past_key_values`.
    - These WILL CRASH standard BERT layers.
    - **MANDATORY:** At the very top of `forward`, purely purge these args:
      ```python
      for bad_arg in ['num_items_in_batch', 'cache_position', 'past_key_values']:
          kwargs.pop(bad_arg, None)
      ```
    - Do this BEFORE calling `super().forward()` or `self.bert(...)`.
11. **Integer Casting (CRITICAL):**
    - HuggingFace utility functions (like `apply_chunking_to_forward`) expect INTEGER arguments.
    - If you extract `chunk_size` from config, you MUST cast it: `int(config.chunk_size_feed_forward)`
    - Never pass Tensors directly to functions expecting ints/bools.
12. **HuggingFace Inheritance (CRITICAL):**
    - If inheriting from BertAttention or BertSelfAttention, ALWAYS set config._attn_implementation FIRST:
```python
      def __init__(self, config, position_embedding_type=None):
          if not hasattr(config, '_attn_implementation') or config._attn_implementation is None:
              config._attn_implementation = 'eager'
          super().__init__(config, position_embedding_type)
```
    - This applies to EVERY class that inherits from HuggingFace attention layers.

**OUTPUT:**
Return ONLY raw Python code. No markdown.
"""
        response = self.llm.invoke(prompt)
        return self._clean_response(response.content)

    def _heal_code(self, broken_code: str, error: str, specs: str) -> str:
        """
        Use the SAME diagnostic logic as the main healer for consistency.
        """
        # Import or reuse the diagnostic logic
        hints = []
        
        # Copy the SAME diagnostics from fix_code_with_llm
        if "unexpected keyword" in error or "multiple values" in error:
            hints.append("CRITICAL: Pop 'num_items_in_batch', 'cache_position', 'past_key_values' from kwargs.")
        
        if "types match" in error and "Float" in error:
            hints.append("CRITICAL: Use .to(target.dtype) for indexed assignments.")
        
        if "has no attribute 'get_model'" in error:
            hints.append("CRITICAL: Restore the get_model() function.")
        
        if "Boolean value of Tensor" in error:
            hints.append("CRITICAL: Wrap Tensor in int() when passing to functions expecting integers.")
        
        if "KeyError: None" in error or "_attn_implementation" in error:
            hints.append("CRITICAL: Set config._attn_implementation = 'eager' before super().__init__()")
        
        hint_block = "\n".join([f"- {h}" for h in hints])

        prompt = f"""
    You are a Senior PyTorch Debugger.

    ERROR:
    {error}

    DIAGNOSTICS (MANDATORY):
    {hint_block}

    MODEL SPECS:
    {specs}

    BROKEN CODE:
    {broken_code}

    Fix the code. Return ONLY complete Python code.
    """
        response = self.llm.invoke(prompt)
        return self._clean_response(response.content)

    def _clean_response(self, text: str) -> str:
        text = text.strip()
        if "```python" in text: text = text.split("```python")[1]
        if "```" in text: text = text.split("```")[0]
        if "import torch" in text: text = text[text.find("import torch"):]
        return text