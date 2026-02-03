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
        tokenizer=tokenizer, # The tokenizer should be passed to DPOConfig
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
        chosen_prompt_end_idx = chosen_text.rfind("Assistant:")
        rejected_prompt_end_idx = rejected_text.rfind("Assistant:")

        # Use the minimum of the two indices to ensure the prompt is common to both
        # If one is not found, or if they differ, a more robust parsing might be needed.
        # For simplicity, we assume a common structure up to "Assistant:".
        if chosen_prompt_end_idx != -1 and rejected_prompt_end_idx != -1:
            # Take the shorter prompt if they differ, to ensure it's a common prefix
            prompt_end_idx = min(chosen_prompt_end_idx, rejected_prompt_end_idx)
            prompt = chosen_text[:prompt_end_idx + len("Assistant:")]
        else:
            # Fallback for cases where "Assistant:" isn't found or structure is unexpected
            # A more sophisticated parser for hh-rlhf might be needed for full robustness.
            # For now, we'll use a simple heuristic or default to empty prompt.
            # If the prompt cannot be reliably extracted, DPOTrainer might struggle.
            # For this dataset, the structure is usually consistent.
            # Let's assume the prompt is everything before the first "Assistant:"
            # and if not found, the whole chosen/rejected is the response.
            # A safer heuristic for hh-rlhf is to split on "Assistant:"
            
            # Let's try a more robust split based on the typical HH-RLHF format
            # "Human: <query>\n\nAssistant: <response>"
            
            # Find the last "Human:" and "Assistant:" to isolate the prompt part
            # This is a common pattern in HH-RLHF
            
            # Example:
            # chosen = "Human: What is the capital of France?\n\nAssistant: Paris is the capital of France."
            # rejected = "Human: What is the capital of France?\n\nAssistant: London is the capital of France."
            
            # We want prompt = "Human: What is the capital of France?\n\nAssistant:"
            
            # Find the last occurrence of "Assistant:" in the chosen text
            last_assistant_idx = chosen_text.rfind("Assistant:")
            if last_assistant_idx != -1:
                prompt = chosen_text[:last_assistant_idx + len("Assistant:")]
            else:
                # If "Assistant:" is not found, the whole text might be a prompt or malformed.
                # For robustness, we might need to inspect the dataset more closely.
                # For now, if no "Assistant:", assume the whole chosen is the prompt.
                # This might lead to issues if the model tries to generate from a full conversation.
                # A better approach for HH-RLHF is often to split on the last "Human: " and "Assistant: "
                # to get the turn-based prompt.
                
                # Let's simplify for the smoke test and assume the prompt is everything up to the first Assistant:
                # If the dataset is clean, this should work.
                first_assistant_idx = chosen_text.find("Assistant:")
                if first_assistant_idx != -1:
                    prompt = chosen_text[:first_assistant_idx + len("Assistant:")]
                else:
                    # If no "Assistant:" at all, this sample might be problematic for DPO.
                    # For a smoke test, we can set prompt to empty or handle it as an edge case.
                    prompt = "" # This might cause issues if prompt is empty.
                                # A more robust solution would be to filter such samples.
                                # For now, let's proceed with an empty prompt if not found.
        
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
        # The tokenizer should be passed to DPOConfig, not directly to DPOTrainer.__init__
        # tokenizer=tokenizer, # REMOVED: This was the cause of the TypeError
        train_dataset=train_dataset,
        # eval_dataset=raw_dataset["test"].map(format_dataset), # Optional for smoke test
    )
    
    print("--- 🚀 Starting Custom DPO Training ---")
    trainer.train()
    
if __name__ == "__main__":
    run_experiment()

