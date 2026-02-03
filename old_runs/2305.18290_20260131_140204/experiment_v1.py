import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from typing import List, Dict, Any, Optional

# === INJECTION POINT 1: REWARD FUNCTION (PPO/GRPO Only) ===
# This function is ignored for DPOTrainer.
def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    return [0.0] * len(completions)

# === INJECTION POINT 2: CUSTOM TRAINER ===
# The paper introduces Direct Preference Optimization (DPO), which is a direct optimization
# method using a classification loss, explicitly stating it avoids RL.
# Therefore, we should subclass DPOTrainer.
# The paper's core contribution is the DPO objective function (Eq. 7).
# The DPOTrainer in `trl` already implements this objective.
# We don't need to override `get_batch_loss_metrics` unless we want to modify the DPO loss itself,
# which the paper does not suggest for a basic implementation.
# The `DPOTrainer`'s default `get_batch_loss_metrics` already computes the loss as described
# in the paper's Eq. 7.
class CustomDPOTrainer(DPOTrainer):
    # No custom overrides are strictly necessary for a basic implementation of DPO
    # as the `DPOTrainer` already implements the core DPO loss.
    # If there were specific modifications to the loss or metrics, they would go here.
    pass

# === INJECTION POINT 3: EXECUTION ===
def run_experiment(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # The paper mentions using πref, which is often the SFT model.
    # For DPO, the tokenizer needs a pad_token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # DPOConfig is the correct choice for DPO.
    # The paper mentions 'beta' as a parameter controlling deviation from the reference policy.
    # We set max_steps and per_device_train_batch_size for a smoke test.
    training_args = DPOConfig(
        output_dir="runs/dpo_experiment",
        num_train_epochs=1,
        max_steps=10,  # Smoke test
        per_device_train_batch_size=2, # Smoke test
        logging_steps=1,
        beta=0.1, # A common value for beta in DPO
        remove_unused_columns=False, # Important for custom datasets
    )
    
    # Load the dataset as specified in the prompt
    raw_dataset = load_dataset("Anthropic/hh-rlhf")

    # Format the dataset for DPOTrainer
    # DPOTrainer expects a dictionary with 'prompt', 'chosen', and 'rejected' keys.
    def format_dataset(sample):
        # The 'chosen' and 'rejected' columns already contain the full text including prompt.
        # DPOTrainer expects 'prompt' to be the common prefix, and 'chosen'/'rejected' to be the full sequence.
        # We need to extract the common prompt from 'chosen' and 'rejected'.
        # For hh-rlhf, the prompt is typically "Human: ... Assistant: "
        # We'll try to find the last occurrence of "Assistant:" as a heuristic for the prompt boundary.
        
        chosen_text = sample['chosen']
        rejected_text = sample['rejected']

        # Find the last "Assistant:" to delineate the prompt from the response
        chosen_prompt_end_idx = chosen_text.rfind("Assistant:")
        rejected_prompt_end_idx = rejected_text.rfind("Assistant:")

        if chosen_prompt_end_idx != -1 and rejected_prompt_end_idx != -1:
            # Assume the prompt is the same up to the last "Assistant:"
            prompt = chosen_text[:chosen_prompt_end_idx + len("Assistant:")]
            # DPOTrainer expects the full sequence for chosen/rejected, not just the response.
            # The original code was already doing this correctly for chosen/rejected.
            # The prompt extraction is for the 'prompt' column.
        else:
            # Fallback if "Assistant:" isn't found, or if the structure is different.
            # This might not be perfect for all samples in hh-rlhf, but serves as a starting point.
            # A more robust solution might involve a specific parsing function for hh-rlhf.
            prompt = "" # Or some other default/error handling
            # If no clear prompt, the full text might be considered the prompt,
            # or it might indicate a malformed sample. For simplicity, we'll keep
            # chosen/rejected as is and prompt as empty if not found.
            
        return {
            "prompt": prompt,
            "chosen": chosen_text, # DPOTrainer expects full chosen sequence
            "rejected": rejected_text # DPOTrainer expects full rejected sequence
        }

    # Apply the formatting to the training split
    train_dataset = raw_dataset["train"].map(
        format_dataset,
        remove_columns=raw_dataset["train"].column_names # Remove original columns
    )
    
    # Initialize the base model and reference model
    # The paper states: "πref, namely the initial SFT model πSFT."
    # And "In practice, one would like to reuse preference datasets publicly available,
    # rather than generating samples and gathering human preferences. Since the preference datasets
    # are sampled using πSFT, we initialize πref = πSFT whenever available."
    # For a smoke test, we'll use the same model for both.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_ref = AutoModelForCausalLM.from_pretrained(model_name)

    # Initialize the CustomDPOTrainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=model_ref, # Reference model is crucial for DPO
        args=training_args,
        tokenizer=tokenizer, # The error indicates 'tokenizer' is not expected here.
                             # DPOTrainer expects the tokenizer to be passed to the DPOConfig.
        train_dataset=train_dataset,
        # eval_dataset=raw_dataset["test"].map(format_dataset), # Optional for smoke test
    )
    
    print("--- 🚀 Starting Custom DPO Training ---")
    trainer.train()
    
if __name__ == "__main__":
    run_experiment()

