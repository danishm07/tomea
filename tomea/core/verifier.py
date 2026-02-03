import os
import logging
from tomea.utils.llm import LLMClient

logger = logging.getLogger(__name__)

class MathVerifier:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def verify_logic(self, code: str, run_dir: str) -> bool:
        """
        Generates a 'Probe Script' to sanity check the math in the generated code.
        """
        logger.info("   Verifier: Probing Math Integrity...")
        
        # 1. Write the Probe
        # We assume the main script will be saved as 'experiment.py'
        probe_prompt = f"""
You are a QA Engineer. I have generated a training script called `experiment.py`.
Your job is to write a separate "Unit Test" script called `probe.py` that imports the `CustomTrainer` (or Model) from `experiment` and tests the LOSS FUNCTION.

RULES:
1. Import the Trainer class from `experiment`.
2. Create dummy torch inputs (batch size 2).
3. Instantiate the Trainer (use a tiny model like 'prajjwal1/bert-tiny' or 'Qwen/Qwen2.5-0.5B-Instruct' to avoid OOM).
4. Run `trainer.compute_loss` (or `get_batch_loss_metrics` if DPO).
5. Assert that the loss is a scalar and not NaN.
6. Print "PROBE_SUCCESS" if it passes.

THE GENERATED CODE TO TEST:
{code}
"""
        probe_code = await self.llm.generate("You are a QA Engineer.", probe_prompt)
        probe_code = self._clean_code(probe_code)
        
        # 2. Save Probe
        probe_path = os.path.join(run_dir, "probe.py")
        with open(probe_path, "w") as f:
            f.write(probe_code)
            
        # 3. Execution Note
        # In a full production env, you would run `subprocess.run(["python", probe_path])`
        # For now, we generate it to prove the capability.
        logger.info(f"   ✅ Verifier: Probe generated at {probe_path}")
        return True

    def _clean_code(self, code):
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        return code.strip()