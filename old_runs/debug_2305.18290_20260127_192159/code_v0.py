import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from typing import List, Dict, Any, Optional

# === INJECTION POINT 1: REWARD FUNCTION (PPO/GRPO Only) ===
# DPO does not use an explicit reward function, so this is not needed.
def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    raise NotImplementedError("DPO does not use an explicit reward function.")

# === INJECTION POINT 2: CUSTOM TRAINER ===
# The paper introduces Direct Preference Optimization (DPO), which is a direct policy optimization
# method that avoids explicit reward modeling and reinforcement learning.
# Therefore, we subclass DPOTrainer.
class CustomDPOTrainer(DPOTrainer): 
    
    # For DPO, we usually override `get_batch_loss_metrics` to implement the specific DPO loss.
    # The paper's DPO loss is given by Equation 7:
    # LDPO(πθ; πref) = −E(x,yw,yl)∼D [ log σ( β log πθ(yw | x) / πref(yw | x) − β log πθ(yl | x) / πref(yl | x) ) ]
    # The `DPOTrainer` in `trl` already implements this loss function internally.
    # We just need to ensure the `beta` parameter is correctly passed via `DPOConfig`.
    # If there were modifications to the core DPO loss (e.g., a different weighting scheme),
    # we would override `get_batch_loss_metrics`.
    # However, the paper's core DPO loss is what `DPOTrainer` implements.
    # The "dynamic, per-example importance weight" mentioned in the paper is part of the
    # gradient calculation, which is handled by the standard DPO loss implementation.
    
    # If we needed to customize the loss calculation beyond what the base DPOTrainer offers,
    # we would implement it here. For this paper, the default DPOTrainer's `get_batch_loss_metrics`
    # is sufficient as it directly implements the DPO loss from Equation 7.
    pass

# === INJECTION POINT 3: EXECUTION ===
def run_experiment(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # The paper mentions using πSFT as πref. In trl's DPOTrainer, `model` is πθ and `ref_model` is πref.
    # If `ref_model` is not provided, it defaults to a copy of `model` at initialization.
    # For the DPO algorithm, it's crucial that `ref_model` remains static or is a SFT model.
    
    # Ensure pad_token is set for generation and batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # The paper uses a beta parameter in its DPO loss (Eq. 7).
    # This corresponds to the `beta` parameter in DPOConfig.
    # The paper also mentions initializing πref by maximizing likelihood of preferred completions
    # if πSFT is not available. In `trl`, this would typically mean training an SFT model first
    # and then using it as `ref_model`, or letting `DPOTrainer` handle the `ref_model` initialization.
    training_args = DPOConfig(
        output_dir="runs/dpo_experiment",
        num_train_epochs=1,
        logging_steps=1,
        beta=0.1,  # A common value for beta, as used in DPO papers.
        # The paper mentions `beta` as a parameter controlling deviation from πref.
        # It's a hyperparameter to tune.
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        max_length=512,
        max_prompt_length=256,
        # Other parameters can be set as needed for the specific experiment
    )
    
    # Mock Data: DPO requires a dataset of (prompt, preferred_completion, dispreferred_completion)
    # The paper uses D = {x(i), y(i)_w, y(i)_l}
    # For `trl.DPOTrainer`, the dataset should contain 'prompt', 'chosen', and 'rejected' keys.
    dataset = [
        {"prompt": "What is the capital of France?", "chosen": "Paris is the capital of France.", "rejected": "London is the capital of France."},
        {"prompt": "Tell me a short story.", "chosen": "Once upon a time, in a land far away, lived a brave knight.", "rejected": "The quick brown fox jumps over the lazy dog."},
        {"prompt": "Write a poem about nature.", "chosen": "The gentle breeze, a whispered song, through emerald leaves, where spirits throng.", "rejected": "I like to eat apples and bananas, they are very tasty."},
    ] * 10 # Repeat to make it a bit larger for demonstration

    # Initialize the base model (πθ) and the reference model (πref)
    # The paper states πθ is initialized to πSFT, and πref is also πSFT.
    # In trl, if `ref_model` is not provided, it's initialized as a copy of `model`.
    # If you have a specific SFT model, you would load it for both `model` and `ref_model`.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # For DPO, the reference model should ideally be the SFT model.
    # If not explicitly provided, DPOTrainer will create a copy of the initial model.
    # ref_model = AutoModelForCausalLM.from_pretrained(model_name) # Uncomment if you want a separate ref_model instance

    trainer = CustomDPOTrainer(
        model=model,
        ref_model=None, # Let DPOTrainer initialize ref_model as a copy of model
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print("--- 🚀 Starting Custom DPO Training ---")
    trainer.train()
    
    print("--- ✅ DPO Training Complete ---")
    # Optionally, save the fine-tuned model
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    run_experiment()
