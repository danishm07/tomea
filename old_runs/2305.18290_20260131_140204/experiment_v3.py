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
        # tokenizer=tokenizer, # The tokenizer should be passed to DPOConfig - THIS IS THE ERROR
    )
    
    # Load the dataset as specified in the prompt
    raw_dataset = load_dataset("Anthropic/hh-rlhf")

    # Format the dataset for DPOTrainer
    # DPOTrainer expects a dictionary with 'prompt', 'chosen', and 'rejected' keys.
    # We need to extract the common prompt from 'chosen' and 'rejected'.
    # For hh-rlhf, the prompt is typically "Human: ... Assistant: "
    # We'll try to find the last occurrence of "Assistant:" as a heuristic for the prompt boundary.
    def format_dataset(sample):
        chosen_text = sample['chosen']
        rejected_text = sample['rejected']

        # Find the last "Assistant:" to delineate the prompt from the response
        last_assistant_idx_chosen = chosen_text.rfind("Assistant:")
        last_assistant_idx_rejected = rejected_text.rfind("Assistant:")

        prompt = ""
        if last_assistant_idx_chosen != -1 and last_assistant_idx_rejected != -1:
            # Ensure the prompt is common to both chosen and rejected.
            # We take the part up to the *first* "Assistant:" if they are different,
            # or the common prefix if they are identical.
            # A more robust way for HH-RLHF is to find the last "Human: " and "Assistant: "
            # to isolate the turn.
            
            # For HH-RLHF, the prompt is typically everything up to the last "Assistant:"
            # that is common to both chosen and rejected.
            
            # Let's find the common prefix up to the last "Assistant:"
            common_prefix_len = 0
            for i in range(min(len(chosen_text), len(rejected_text))):
                if chosen_text[i] == rejected_text[i]:
                    common_prefix_len += 1
                else:
                    break
            
            # Now, find the last "Assistant:" within this common prefix
            common_text = chosen_text[:common_prefix_len]
            last_assistant_in_common = common_text.rfind("Assistant:")
            
            if last_assistant_in_common != -1:
                prompt = common_text[:last_assistant_in_common + len("Assistant:")]
            else:
                # If no "Assistant:" in common prefix, try to find the first "Assistant:"
                # This is a heuristic and might need adjustment based on dataset specifics.
                first_assistant_idx = chosen_text.find("Assistant:")
                if first_assistant_idx != -1:
                    prompt = chosen_text[:first_assistant_idx + len("Assistant:")]
                else:
                    prompt = "" # Fallback
        elif last_assistant_idx_chosen != -1: # Only chosen has Assistant:
            prompt = chosen_text[:last_assistant_idx_chosen + len("Assistant:")]
        elif last_assistant_idx_rejected != -1: # Only rejected has Assistant:
            prompt = rejected_text[:last_assistant_idx_rejected + len("Assistant:")]
        else:
            prompt = "" # No "Assistant:" found in either, treat as empty prompt
        
        return {
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text
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
        tokenizer=tokenizer, # The tokenizer should be passed directly to DPOTrainer, not DPOConfig
        train_dataset=train_dataset,
        # eval_dataset=raw_dataset["test"].map(format_dataset), # Optional for smoke test
    )
    
    print("--- 🚀 Starting Custom DPO Training ---")
    trainer.train()
    
if __name__ == "__main__":
    run_experiment()

