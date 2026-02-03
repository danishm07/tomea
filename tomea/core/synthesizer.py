import re
import json
import logging
from typing import Optional, Dict, Tuple

from tomea.harnesses.base import BaseHarness
from tomea.context.ingest import ContextIngestor
from tomea.utils.llm import LLMClient

try:
    from tomea.utils.arxiv_parser import get_paper_data
except ImportError:
    get_paper_data = None 

logger = logging.getLogger(__name__)

class Synthesizer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def generate_implementation(self, harness: BaseHarness, paper_text: str, arxiv_id: str, data_context: Dict) -> Tuple[str, Dict]:
        logger.info("---  Synthesizer: Gathering Context... ---")
        
        full_text = paper_text
        if get_paper_data and len(paper_text) < 500: 
            try:
                data = get_paper_data(arxiv_id)
                if data: full_text = data['text'][:30000] 
            except Exception: pass
        
        skeleton = "No official repo found."
        github_links = re.findall(r'github\.com/[\w-]+/[\w-]+', full_text)
        if github_links:
            try:
                ingested = await ContextIngestor.read_repo_async(f"https://{github_links[0]}")
                if "Error" not in ingested: skeleton = ingested
            except Exception: pass

        # Speed up extraction for smoke tests
        extraction_metadata = {"confidence": 0.9, "novel_contribution": "See paper", "smoke_test_feasible": "yes"}

        logger.info("--- ✍️ Synthesizer: Writing Code... ---")
        template = harness.get_template()
        system_prompt = harness.get_system_prompt()
        
        data_block = f"USER DATASET: {data_context.get('source_name')}\nCOLUMNS: {data_context.get('columns')}"

        user_message = f"""
PAPER CONTENT:
{full_text}

REPO SKELETON:
{skeleton}

{data_block}

TASK:
Implement the experiment using the TEMPLATE below.

CRITICAL TRL CONFIG RULES (DO NOT IGNORE):
1. **NO Custom Config Args:** `GRPOConfig` matches the standard library strictly.
   - ❌ IT WILL CRASH if you pass: `clip_range`, `clip_range_ratio`, `epsilon`, `beta_kl`, `reward_weights`.
   - ✅ If the paper needs these parameters, define them as **Class Constants** inside `CustomTrainer` (e.g. `CLIP_RANGE = 0.2`).

2. **Standard Arguments:**
   - Use `processing_class=tokenizer` (NOT tokenizer=).
   - Use `reward_funcs=[reward_function]` (list format).

CRITICAL SPEED RULES (SMOKE TEST):
1. Use `load_dataset(..., split="train[:200]")`.
2. Set `max_steps=5`, `num_train_epochs=1`, `num_generations=2`.

TEMPLATE:
{template}
""" 
        code = await self.llm.generate(system_prompt, user_message)  
        return self._strip_markdown(code), extraction_metadata

    def _strip_markdown(self, code: str) -> str:
        if not code: return ""
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        return code.strip()