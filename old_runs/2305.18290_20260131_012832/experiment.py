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
# The `DPOTrainer` in `trl` already implements this objective.
# We don't need to override `get_batch_loss_metrics` unless we want to modify the DPO loss itself,
# which the paper does not suggest for a basic implementation.
# The `DPOTrainer`'s default `get_batch_loss_metrics` will compute the loss as described in the paper.
class CustomDPOTrainer(DPOTrainer):
    # The paper's algorithm is directly implemented by the DPOTrainer's default behavior.
    # No custom override of get_batch_loss_metrics is needed for a faithful implementation
    # of the core DPO algorithm as described by Eq. 7.
    pass

# === INJECTION POINT 3: EXECUTION ===
def run_experiment(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # The paper uses a reference model (πref). In trl's DPOTrainer,
    # this is typically handled by `model_ref`.
    # For DPO, it's common to set the pad_token to eos_token if not already set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # The paper uses a beta parameter (β) in its DPO objective (Eq. 7).
    # This corresponds directly to the `beta` parameter in DPOConfig.
    # The paper mentions "eliminating the need for sampling from the LM during fine-tuning or performing
    # significant hyperparameter tuning." and "substantially simpler to implement and train."
    # This implies DPOConfig is the correct choice.
    training_args = DPOConfig(
        output_dir="runs/dpo_experiment",
        num_train_epochs=1,  # Smoke test
        max_steps=10,        # Smoke test
        per_device_train_batch_size=2, # Smoke test
        logging_steps=1,
        beta=0.1, # A common value for beta in DPO, as used in many examples.
        # The paper mentions "maintaining the generation diversity and preventing mode-collapse to single high-reward answers."
        # This is handled by the KL constraint, which beta controls.
        # The paper also mentions "The DPO update increases the relative log probability of preferred to dispreferred responses,
        # but it incorporates a dynamic, per-example importance weight that prevents the model degeneration that we find occurs
        # with a naive probability ratio objective." This is inherently part of the DPO loss.
    )

    # Load the dataset
    raw_dataset = load_dataset("Anthropic/hh-rlhf")

    # Format the dataset for DPOTrainer
    # DPOTrainer expects a dictionary with 'prompt', 'chosen', and 'rejected' keys.
    def format_dataset(sample):
        # The 'chosen' and 'rejected' columns already contain the full dialogues.
        # For DPO, we need to split these into a common prompt and the chosen/rejected responses.
        # The Anthropic/hh-rlhf dataset has the format:
        # Human: <prompt>
        # Assistant: <response>
        # We need to find the last "Assistant:" to separate the prompt from the response.
        
        # Helper to split a dialogue into prompt and response
        def split_dialogue(dialogue_text):
            parts = dialogue_text.split("\n\nAssistant:", 1)
            if len(parts) == 2:
                prompt = parts[0] + "\n\nAssistant:" # Keep the Assistant: as part of the prompt for consistency
                response = parts[1]
                return prompt, response
            return dialogue_text, "" # Fallback if format is unexpected

        chosen_prompt, chosen_response = split_dialogue(sample['chosen'])
        rejected_prompt, rejected_response = split_dialogue(sample['rejected'])

        # Ensure prompts are consistent. If they are not, it indicates a problem with the dataset structure
        # or our splitting logic. For hh-rlhf, the prompts should be identical up to the last Assistant: turn.
        # We'll take the chosen_prompt as the canonical one.
        return {
            "prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }

    # Apply the formatting to the training split
    train_dataset = raw_dataset["train"].map(format_dataset, remove_columns=raw_dataset["train"].column_names)
    
    # Initialize the DPO Trainer
    # The paper states: "we initialize πref = πSFT whenever available."
    # In trl's DPOTrainer, `model` is the policy to be optimized (πθ), and `model_ref` is the reference policy (πref).
    # If `model_ref` is not provided, `model` is copied to `model_ref`.
    # For a true SFT-initialized πref, one would typically load the SFT model into `model_ref`.
    # For this smoke test, we'll let `model_ref` be a copy of the initial `model`.
    
    # Load the base model for fine-tuning
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # The reference model (pi_ref) is often the SFT model.
    # For a smoke test, we can initialize it from the same base model.
    model_ref = AutoModelForCausalLM.from_pretrained(model_name)

    trainer = CustomDPOTrainer(
        model=model,
        ref_model=model_ref, # This is pi_ref in the paper
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        # The paper mentions "Since the preference datasets are sampled using πSFT,
        # we initialize πref = πSFT whenever available."
        # And "However, when πSFT is not available, we initialize πref by maximizing likelihood of preferred completions (x, yw),
        # that is, πref = arg maxπ Ex,yw∼D [log π(yw | x)]."
        # The `ref_model` argument handles the πref.
    )
    
    print("--- 🚀 Starting Custom DPO Training ---")
    trainer.train()
    
if __name__ == "__main__":
    run_experiment()

