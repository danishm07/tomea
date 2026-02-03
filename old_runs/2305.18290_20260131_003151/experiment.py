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
# Therefore, we subclass DPOTrainer.
class CustomDPOTrainer(DPOTrainer): 
    
    # The paper's core contribution is the DPO objective function (Eq. 7).
    # The `get_batch_loss_metrics` method in DPOTrainer is where the loss calculation
    # for DPO happens. We will override this to ensure it aligns with the paper's
    # formulation, especially the implicit reward calculation and the binary cross-entropy.
    # The default DPOTrainer implementation already aligns well with the paper's Eq. 7.
    # However, if there were specific nuances in the loss calculation (e.g., a different
    # weighting scheme or a modified sigmoid argument), this is where we would implement it.
    # For this paper, the default DPOTrainer's `get_batch_loss_metrics` is a direct
    # implementation of Eq. 7, so we can largely rely on it.
    # We can add a print statement or minor modification to demonstrate the override capability.
    def get_batch_loss_metrics(self, model, batch: Dict[str, Any], train_eval: str = "train"):
        # The base DPOTrainer's get_batch_loss_metrics already implements the DPO loss
        # as described in the paper (Eq. 7).
        # It calculates: log_probs_chosen = log pi_theta(y_w | x)
        #               log_probs_rejected = log pi_theta(y_l | x)
        #               ref_log_probs_chosen = log pi_ref(y_w | x)
        #               ref_log_probs_rejected = log pi_ref(y_l | x)
        # And then computes the implicit reward difference:
        # r_hat_chosen = beta * (log_probs_chosen - ref_log_probs_chosen)
        # r_hat_rejected = beta * (log_probs_rejected - ref_log_probs_rejected)
        # loss = -log(sigmoid(r_hat_chosen - r_hat_rejected))
        
        # This directly corresponds to:
        # LDPO = -E[log sigma(beta * log(pi_theta(yw|x)/pi_ref(yw|x)) - beta * log(pi_theta(yl|x)/pi_ref(yl|x)))]
        
        metrics = super().get_batch_loss_metrics(model, batch, train_eval)
        # print(f"--- CustomDPOTrainer: Calculating DPO loss for {train_eval} batch ---")
        # You could inspect or modify metrics here if needed.
        return metrics

# === INJECTION POINT 3: EXECUTION ===
def run_experiment(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # The paper mentions using `πref = πSFT` or initializing `πref` by maximizing
    # likelihood of preferred completions. For `trl`'s DPOTrainer, the `ref_model`
    # is typically the SFT model.
    # For Qwen, it's important to set pad_token_id to eos_token_id for DPO.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # DPOConfig is the correct configuration for Direct Preference Optimization.
    # The paper mentions 'beta' as a parameter controlling deviation from the base policy.
    # A common value for beta is 0.1, as used in many DPO implementations.
    training_args = DPOConfig(
        output_dir="runs/dpo_experiment",
        num_train_epochs=1,  # Smoke test
        max_steps=10,        # Smoke test
        per_device_train_batch_size=2, # Smoke test
        logging_steps=1,
        beta=0.1, # As per DPO paper, controls the KL divergence penalty
        remove_unused_columns=False, # Important for custom datasets
    )
    
    # Load the dataset
    raw_dataset = load_dataset("Anthropic/hh-rlhf")

    # Format the dataset for DPOTrainer
    def format_dataset(sample):
        # DPOTrainer expects 'prompt', 'chosen', 'rejected'
        # The 'chosen' and 'rejected' columns from Anthropic/hh-rlhf map directly.
        # For the prompt, we need to extract it from the beginning of the chosen/rejected text.
        # The Anthropic dataset has a specific format: "\n\nHuman: ...\n\nAssistant: ..."
        # We need to separate the human prompt from the assistant's response.
        
        # Find the split point for the prompt (before the first Assistant response)
        # Assuming the prompt is everything before the Assistant's first turn.
        # This is a heuristic and might need refinement based on actual data inspection.
        
        # Let's try to find the last "Human: " and take everything before the first "Assistant: "
        # This is a common pattern in the Anthropic dataset.
        
        # Example: "\n\nHuman: What are some cuss words in english?\n\nAssistant: Here’s an incomplete list."
        # Prompt should be: "\n\nHuman: What are some cuss words in english?"
        # Chosen/Rejected should be: "Here’s an incomplete list."
        
        # A more robust way is to split by the first occurrence of "\n\nAssistant:"
        # and use the part before it as the prompt.
        
        prompt_chosen_parts = sample["chosen"].split("\n\nAssistant:", 1)
        prompt_rejected_parts = sample["rejected"].split("\n\nAssistant:", 1)

        if len(prompt_chosen_parts) > 1 and len(prompt_rejected_parts) > 1:
            prompt = prompt_chosen_parts[0] + "\n\nAssistant:" # Include the "Assistant:" to guide the model
            chosen = prompt_chosen_parts[1]
            rejected = prompt_rejected_parts[1]
        else:
            # Fallback if the split pattern isn't found, or if it's a very short entry.
            # This might indicate an issue with the prompt extraction logic for some samples.
            # For simplicity in a smoke test, we can just use the full text as prompt
            # and empty chosen/rejected, though this isn't ideal for DPO.
            # A better approach would be to filter these samples out.
            prompt = sample["chosen"] # Or some other default
            chosen = ""
            rejected = ""
            print(f"Warning: Could not parse prompt/response for sample. Chosen: {sample['chosen'][:100]}")


        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    # Apply the formatting function to the dataset
    train_dataset = raw_dataset["train"].map(format_dataset, remove_columns=raw_dataset["train"].column_names)
    eval_dataset = raw_dataset["test"].map(format_dataset, remove_columns=raw_dataset["test"].column_names)

    # Filter out samples where chosen or rejected is empty after parsing
    train_dataset = train_dataset.filter(lambda x: x["chosen"] != "" and x["rejected"] != "")
    eval_dataset = eval_dataset.filter(lambda x: x["chosen"] != "" and x["rejected"] != "")

    # Initialize the correct Trainer - DPOTrainer for DPO paper
    # The ref_model is typically the SFT model, which is the same as the policy model
    # at the beginning of DPO training.
    trainer = CustomDPOTrainer(
        model=model_name,
        ref_model=model_name, # The reference model is usually the SFT model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    print("--- 🚀 Starting Custom DPO Training ---")
    trainer.train()
    
if __name__ == "__main__":
    run_experiment()

