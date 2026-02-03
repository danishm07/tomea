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
# The paper introduces Direct Preference Optimization (DPO), which is a DPO-based algorithm.
# Therefore, we subclass DPOTrainer.
class CustomDPOTrainer(DPOTrainer): 
    
    # DPO's core logic is in `get_batch_loss_metrics` where the DPO loss is computed.
    # The paper's Equation 7 defines the DPO loss.
    # LDPO(πθ; πref) = −E(x,yw,yl)∼D [ log σ( β log πθ(yw | x) / πref(yw | x) − β log πθ(yl | x) / πref(yl | x) ) ]
    # This corresponds to the `dpo_loss` in the base DPOTrainer.
    # We will override `get_batch_loss_metrics` to ensure the loss calculation precisely matches the paper's formula.
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Any],
        train_eval: str = "train",
    ) -> Dict[str, Any]:
        """
        Calculates the DPO loss and other metrics for a given batch.
        This method overrides the base DPOTrainer's method to ensure the loss
        calculation aligns precisely with Equation 7 from the paper.
        """
        # The base DPOTrainer's `get_batch_loss_metrics` already implements the DPO loss
        # as described in the paper (Equation 7).
        # We call the super method to leverage the existing implementation and
        # ensure all necessary components (log_probs, rewards, etc.) are computed.
        
        # The `DPOTrainer`'s `dpo_loss` function (called internally by `get_batch_loss_metrics`)
        # computes: -log(sigmoid(beta * (chosen_logps - rejected_logps)))
        # which is equivalent to: -log(sigmoid(beta * log(pi_theta(yw|x)/pi_ref(yw|x)) - beta * log(pi_theta(yl|x)/pi_ref(yl|x))))
        # when `chosen_logps` and `rejected_logps` are already adjusted for the reference model.
        # The `DPOTrainer` handles the `log(pi_theta/pi_ref)` part by calculating `policy_chosen_logps - ref_chosen_logps`
        # and `policy_rejected_logps - ref_rejected_logps`.
        
        # So, the default implementation of `get_batch_loss_metrics` in `DPOTrainer`
        # already correctly implements Equation 7 from the paper.
        
        metrics = super().get_batch_loss_metrics(model, batch, train_eval)
        
        # We can add custom logging or checks here if needed, but for direct implementation
        # of the paper's loss, the super method is sufficient.
        
        return metrics

# === INJECTION POINT 3: EXECUTION ===
def run_experiment(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # The paper uses DPO, so we use DPOConfig.
    # The beta parameter is crucial for DPO, as seen in Equation 7.
    # The paper mentions "β is a parameter controlling the deviation from the base reference policy πref"
    # and "scaled by β" in the gradient analysis. A common value is 0.1.
    training_args = DPOConfig(
        output_dir="runs/dpo_experiment",
        num_train_epochs=1,
        logging_steps=1,
        beta=0.1,  # As per DPO paper, a common value for beta
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        max_length=512,
        max_prompt_length=256,
        # The paper mentions initializing πref = πSFT.
        # In trl's DPOTrainer, the `ref_model` is used for πref.
        # If `ref_model` is None, it defaults to a frozen copy of the policy model.
        # For this example, we'll let it default, but in a real scenario,
        # you might load a specific SFT model as `ref_model`.
        # The paper also mentions "when πSFT is not available, we initialize πref by maximizing likelihood of preferred completions (x, yw)".
        # This is typically handled by the `ref_model` argument or by pre-training the `ref_model` appropriately.
    )
    
    # Mock Data for DPO:
    # DPO requires a dataset of (prompt, chosen_response, rejected_response) tuples.
    # The paper describes D = {x(i), y(i)_w, y(i)_l}N_i=1
    # where y_w is the preferred (chosen) and y_l is the dispreferred (rejected) completion.
    dataset = [
        {"prompt": "What is the capital of France?", "chosen": "Paris", "rejected": "London"},
        {"prompt": "Tell me about large language models.", "chosen": "LLMs are powerful AI models trained on vast amounts of text data.", "rejected": "They are just big text generators."},
        {"prompt": "Write a short poem about nature.", "chosen": "Green leaves dance in gentle breeze,\nSunlight filters through the trees.", "rejected": "Nature is good. I like it a lot."},
    ] * 5 # Repeat to have more data for a mock run

    # Initialize the policy model and a reference model (if different from policy)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # For DPO, the `ref_model` is typically a frozen SFT model.
    # If `ref_model` is not provided, DPOTrainer uses a frozen copy of the `model`.
    # This aligns with the paper's concept of πref being the SFT model.
    ref_model = None # Let DPOTrainer create a frozen copy of `model` as `ref_model`

    # Initialize the CustomDPOTrainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        # The paper does not use a separate `peft_config` in its core algorithm description,
        # but PEFT is commonly used for efficiency. We omit it for direct paper implementation.
        # peft_config=None,
    )
    
    print("--- 🚀 Starting Custom DPO Training ---")
    trainer.train()
    
if __name__ == "__main__":
    run_experiment()

