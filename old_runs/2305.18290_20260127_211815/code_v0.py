
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
# Helper imports - Synthesizer will ensure these are available
from trl import PPOTrainer, PPOConfig, GRPOTrainer, GRPOConfig, DPOTrainer, DPOConfig
from typing import List, Dict, Any, Optional

# === INJECTION POINT 1: REWARD FUNCTION (PPO/GRPO Only) ===
def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    # DPO does not use an explicit reward function, so this is a placeholder.
    return [0.0] * len(completions)

# === INJECTION POINT 2: CUSTOM TRAINER ===
# The paper introduces Direct Preference Optimization (DPO), which is a DPO-based algorithm.
# Therefore, we subclass DPOTrainer.
class CustomDPOTrainer(DPOTrainer): 
    
    # DPO typically overrides `get_batch_loss_metrics` to implement its specific loss.
    # The paper's core contribution is the DPO loss function itself.
    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, List[str]],
        train_eval: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the DPO loss for a given batch of preferences.
        
        The DPO loss is defined in Equation 7 of the paper:
        LDPO(πθ; πref) = −E(x,yw,yl)∼D [log σ(β log πθ(yw | x)/πref(yw | x) − β log πθ(yl | x)/πref(yl | x))]
        """
        
        # Extract preferred and rejected responses from the batch
        policy_chosen_logps = model.get_log_probs(batch["chosen_input_ids"], batch["chosen_labels"], average_log_probs=False)
        policy_rejected_logps = model.get_log_probs(batch["rejected_input_ids"], batch["rejected_labels"], average_log_probs=False)
        
        # Reference model log probabilities (πref)
        with torch.no_grad():
            reference_chosen_logps = self.ref_model.get_log_probs(batch["chosen_input_ids"], batch["chosen_labels"], average_log_probs=False)
            reference_rejected_logps = self.ref_model.get_log_probs(batch["rejected_input_ids"], batch["rejected_labels"], average_log_probs=False)

        # Calculate the implicit reward terms: β * log(πθ(y|x) / πref(y|x))
        # The paper defines ˆrθ(x, y) = β log πθ(y|x) / πref(y|x)
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # The difference in implicit rewards: ˆrθ(x, yw) - ˆrθ(x, yl)
        # This corresponds to the term inside the sigmoid in Eq. 7
        logits = chosen_logratios - rejected_logratios

        # DPO loss: -log(sigmoid(β * logits))
        # The paper uses β * (chosen_logratios - rejected_logratios)
        # The `beta` parameter is part of the DPOConfig and is applied here.
        losses = -torch.nn.functional.logsigmoid(self.beta * logits)
        
        # Average loss over the batch
        loss = losses.mean()

        # For metrics, we can also return the chosen and rejected rewards for analysis
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        return {
            f"{train_eval}_loss": loss,
            f"{train_eval}_rewards/chosen": chosen_rewards.mean(),
            f"{train_eval}_rewards/rejected": rejected_rewards.mean(),
            f"{train_eval}_rewards/accuracies": (chosen_rewards > rejected_rewards).float().mean(),
            f"{train_eval}_rewards/margins": (chosen_rewards - rejected_rewards).mean(),
            f"{train_eval}_log_ratios/chosen": chosen_logratios.mean(),
            f"{train_eval}_log_ratios/rejected": rejected_logratios.mean(),
        }

# === INJECTION POINT 3: EXECUTION ===
def run_experiment(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # The paper uses DPO, so we use DPOConfig.
    # Beta is a crucial hyperparameter in DPO, controlling the KL divergence constraint.
    # The paper mentions "β is a parameter controlling the deviation from the base reference policy πref".
    # A common starting value for beta is 0.1.
    training_args = DPOConfig(
        output_dir="runs/dpo_experiment",
        num_train_epochs=1,
        logging_steps=1,
        beta=0.1,  # As per the paper's formulation
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        max_length=512,
        max_prompt_length=256,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False, # Important for custom datasets
    )
    
    # Mock Data for DPO:
    # DPO requires a dataset of (prompt, chosen_response, rejected_response) tuples.
    # The paper states: "Given a dataset of human preferences over model responses,
    # DPO can therefore optimize a policy using a simple binary cross entropy objective"
    # and "D = {x(i), y(i)w, yl)(i)}N i=1"
    dataset = [
        {"prompt": "What is the capital of France?", "chosen": "Paris is the capital of France.", "rejected": "London is the capital of France."},
        {"prompt": "Tell me a joke.", "chosen": "Why don't scientists trust atoms? Because they make up everything!", "rejected": "I don't know any jokes."},
        {"prompt": "Write a short story about a cat.", "chosen": "Whiskers, a fluffy tabby, loved chasing sunbeams. One day, a particularly bright beam led her to a hidden stash of catnip.", "rejected": "The cat sat on the mat. It was a black cat."},
    ] * 10 # Repeat for a larger mock dataset

    # Initialize the correct Trainer: CustomDPOTrainer
    # DPO trainer signature is different (no reward_funcs argument, requires ref_model)
    # For DPO, the `model` is πθ and `ref_model` is πref.
    # The `ref_model` is typically a frozen copy of the initial SFT model.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name) # Initialize ref_model with the same base model

    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model, # Required for DPO
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print("--- 🚀 Starting Custom DPO Training ---")
    trainer.train()
    
if __name__ == "__main__":
    run_experiment()
