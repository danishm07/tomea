from .base import BaseHarness

class RLHarness(BaseHarness):

    @property
    def mode(self) -> str:
        return "rl"

    def get_system_prompt(self) -> str:
        return """You are an Expert AI Research Engineer using the TRL library.

CRITICAL ARCHITECTURE RULES:
1. **Strict Configs:** `GRPOConfig` and `DPOConfig` are FROZEN. They DO NOT accept new arguments. 
   - IF the paper introduces new hyperparameters (e.g. `epsilon_low`, `alpha`), define them as **Class Constants** inside your `CustomTrainer` or as global constants. 
   - **NEVER** pass non-standard arguments to the Config constructor.

2. **Method Signatures:** The `compute_loss` method signature MUST match `transformers>=4.46`:
   - `def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):`
   - You MUST accept `num_items_in_batch`.

3. **TRL Compatibility:** - Use `processing_class=tokenizer` (NOT `tokenizer=tokenizer`).
   - Use `reward_funcs=[reward_function]` (list format).
"""

    def get_template(self) -> str:
        return """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig, DPOTrainer, DPOConfig
from datasets import load_dataset
from typing import List, Optional

# === REWARD FUNCTION ===
def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    # Placeholder: Length-based reward
    return [float(len(c.split())) / 10.0 for c in completions]

# === CUSTOM TRAINER ===
class CustomTrainer(GRPOTrainer): 
    # TODO: Define paper-specific hyperparameters here to avoid Config errors
    # EXAMPLE: EPSILON_LOW = 0.2 
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # CRITICAL: Preserve signature compatibility
        if num_items_in_batch is not None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

# === EXECUTION ===
def run_experiment():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # NOTE: Pass ONLY standard library arguments here. 
    # Do NOT pass paper-specific params like 'beta_high' or 'epsilon'.
    config = GRPOConfig(
        output_dir="runs/experiment",
        num_train_epochs=1,
        max_steps=5,               
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=128, 
        learning_rate=1e-5,
    )
    
    # Load and map dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split="train[:200]")
    
    def format_dataset(sample):
        # Basic prompt extraction
        return {"prompt": sample["chosen"].split("\\n\\nAssistant:")[0] + "\\n\\nAssistant:"}
    
    train_dataset = dataset.map(format_dataset)
    
    trainer = CustomTrainer(
        model=model_name,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,     # CORRECT ARG NAME
        reward_funcs=[reward_function], # CORRECT LIST FORMAT
    )
    
    print("--- 🚀 Starting Training (Smoke Test) ---")
    trainer.train()

if __name__ == "__main__":
    run_experiment()
"""

    def validate_logs(self, logs: str) -> bool:
        return "loss" in logs and "Traceback" not in logs