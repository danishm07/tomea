import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from typing import List, Dict, Any, Optional

# === INJECTION POINT 1: REWARD FUNCTION (PPO/GRPO Only) ===
# This function is not used for DPOTrainer, but kept for completeness of the template.
def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    return [0.0] * len(completions)

# === INJECTION POINT 2: CUSTOM TRAINER ===
# The paper introduces Direct Preference Optimization (DPO), which is implemented
# by subclassing DPOTrainer. The core logic of DPO is handled by the DPOTrainer
# itself, specifically through its `get_batch_loss_metrics` method which computes
# the DPO loss. The paper's algorithm directly optimizes the policy using a
# binary cross-entropy objective, which is what DPOTrainer does.
class CustomDPOTrainer(DPOTrainer):

    # The paper's algorithm is directly implemented by the DPOTrainer's
    # `get_batch_loss_metrics` method, which calculates the DPO loss
    # based on the chosen and rejected responses.
    # No explicit override is needed here unless there's a specific
    # modification to the DPO loss calculation or metrics.
    # The paper's Eq. 7 is the core DPO loss, which DPOTrainer implements.
    pass

# === INJECTION POINT 3: EXECUTION ===
def run_experiment(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # DPOConfig is used for DPOTrainer
    # The paper mentions beta as a parameter controlling deviation from the reference policy.
    # A common value for beta is 0.1, as often seen in DPO implementations.
    training_args = DPOConfig(
        output_dir="runs/dpo_experiment",
        num_train_epochs=1, # SMOKE TEST
        max_steps=10,       # SMOKE TEST
        per_device_train_batch_size=2, # SMOKE TEST
        logging_steps=1,
        beta=0.1, # A common value for beta in DPO
        remove_unused_columns=False, # Important for custom datasets
    )

    # Load the dataset
    raw_dataset = load_dataset("Anthropic/hh-rlhf")

    # Format the dataset for DPOTrainer
    def format_dataset(sample):
        # DPOTrainer expects 'prompt', 'chosen', 'rejected'
        # The Anthropic/hh-rlhf dataset already has 'chosen' and 'rejected'.
        # We need to extract the prompt from the 'chosen' and 'rejected' strings.
        # The format is "Human: ... Assistant: ..."
        # We'll assume the prompt is everything before the final Assistant response.
        
        # Find the last occurrence of "Assistant:" to split the prompt and response
        chosen_parts = sample["chosen"].rsplit("Assistant:", 1)
        rejected_parts = sample["rejected"].rsplit("Assistant:", 1)

        if len(chosen_parts) == 2 and len(rejected_parts) == 2:
            prompt = chosen_parts[0] + "Assistant:" # Re-add "Assistant:" to the prompt
            chosen = chosen_parts[1].strip()
            rejected = rejected_parts[1].strip()
        else:
            # Fallback if the expected format isn't found, or for single-turn prompts
            # For simplicity, if no "Assistant:" is found, we might treat the whole thing as a prompt
            # and the response as empty, or handle more complex multi-turn cases.
            # For hh-rlhf, it's usually multi-turn, so this fallback might need refinement
            # based on actual data inspection.
            prompt = sample["chosen"].rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:" if "\n\nAssistant:" in sample["chosen"] else sample["chosen"]
            chosen = sample["chosen"].replace(prompt, "").strip()
            rejected = sample["rejected"].replace(prompt, "").strip()

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    # Apply the formatting to the training split
    train_dataset = raw_dataset["train"].map(format_dataset, remove_columns=raw_dataset["train"].column_names)
    # Filter out samples where prompt, chosen, or rejected might be empty after formatting
    train_dataset = train_dataset.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])

    # Initialize the CustomDPOTrainer
    # CRITICAL: use processing_class=tokenizer
    trainer = CustomDPOTrainer(
        model=model_name,
        ref_model=None, # DPO can use the same model as ref_model or a separate one.
                        # For simplicity and common practice, we often use the initial model as ref_model.
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer, # This is still used for tokenization within the trainer
        processing_class=tokenizer, # CRITICAL: Updated argument name
    )
    
    print("--- 🚀 Starting Custom DPO Training ---")
    trainer.train()
    
if __name__ == "__main__":
    run_experiment()

