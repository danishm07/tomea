import logging
from typing import List, Dict
from tomea.harnesses.base import BaseHarness
from tomea.utils.llm import LLMClient

logger = logging.getLogger(__name__)

class Healer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def heal_code(self, broken_code: str, error_logs: str, harness: BaseHarness, attempt_history: List[Dict] = None) -> str:
        logger.info("--- 🚑 Healer: Analyzing Crash... ---")
        
        max_log_chars = 15000 
        if len(error_logs) > max_log_chars:
            compact_logs = "...[TRUNCATED EARLY LOGS]...\n" + error_logs[-max_log_chars:]
        else:
            compact_logs = error_logs

        history_block = self._format_attempt_history(attempt_history)

        system_prompt = "You are an Expert Python Debugger for AI Training Scripts."
        
        user_message = f"""
THE SCRIPT CRASHED.

HARNESS MODE: {harness.mode}

{history_block}

CURRENT BROKEN CODE:
```python
{broken_code}
CURRENT ERROR LOGS (Tail): {compact_logs}

TASK:

Identify the API mismatch.

Fix the code.

Return the FULLY FIXED script.

SPECIFIC FIXES FOR TRL/GRPO:

Config Error: If GRPOConfig complains about clip_range, clip_range_ratio, or epsilon, DELETE THE ARGUMENT. Do not rename it. These arguments do not exist in the standard config.

Tokenizer Error: Rename tokenizer= to processing_class=.

Reward Error: Ensure reward_funcs=[...] is a list.

RETURN ONLY THE CODE. No markdown. """ 
        logger.info("--- 🚑 Healer: Patching Code... ---") 
        code = await self.llm.generate(system_prompt, user_message)

        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        
        if code_lines > 0 and comment_lines / len(lines) > 0.8:
            logger.warning("   ⚠️  Detected comment spiral. Regenerating...")
            system_prompt += "\n\nCRITICAL: Write WORKING CODE, not just comments."
            code = await self.llm.generate(system_prompt, user_message)

        return self._strip_markdown(code)

    def _format_attempt_history(self, attempt_history: List[Dict] = None) -> str:
        if not attempt_history or len(attempt_history) <= 1:
            return "PREVIOUS ATTEMPTS: None."
        
        blocks = ["PREVIOUS ATTEMPTS (DO NOT REPEAT THESE MISTAKES):"]
        for attempt in attempt_history[:-1]:
            log_snippet = attempt["logs"][-2000:] if len(attempt["logs"]) > 2000 else attempt["logs"]
            blocks.append(f"--- Attempt {attempt['attempt']} ---\nError Tail:\n{log_snippet}")
        return "\n".join(blocks)

    def _strip_markdown(self, code: str) -> str:
        if not code: return ""
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
        return code.strip()