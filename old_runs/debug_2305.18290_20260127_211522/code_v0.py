import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from typing import List, Dict, Any, Optional

# === INJECTION POINT 1: REWARD FUNCTION (PPO/GRPO Only) ===
# DPO does not use an explicit reward function, so this is ignored.
def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    return [0.0] * len(completions)

# === INJECTION POINT 2: CUSTOM TRAINER ===
# The paper introduces Direct Preference Optimization (DPO), which is a DPO-based algorithm.
# Therefore, we subclass DPOTrainer.
class CustomDPOTrainer(DPOTrainer): 
    
    # DPO's core logic is in `get_batch_loss_metrics`, which calculates the DPO loss.
    # The paper's Equation (7) defines the DPO loss.
    # We will override `get_batch_loss_metrics` to ensure the loss calculation
    # precisely matches the paper's formulation, especially the implicit reward calculation
    # and the weighting factor.

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, List[torch.LongTensor]],
        train_eval: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the DPO loss for a given batch of preferences.
        This method implements Equation (7) from the paper:
        LDPO(πθ; πref) = −E(x,yw,yl)∼D [ log σ( β log πθ(yw | x) / πref(yw | x) − β log πθ(yl | x) / πref(yl | x) ) ]
        """
        policy_chosen_logps = model.get_log_probs(batch["chosen_input_ids"], batch["chosen_labels"], average_log_probs=False)
        policy_rejected_logps = model.get_log_probs(batch["rejected_input_ids"], batch["rejected_labels"], average_log_probs=False)

        # The reference model is typically the SFT model or the initial model before DPO.
        # In trl's DPOTrainer, `model.get_log_probs` for the reference model is handled internally
        # by `self.ref_model`.
        with torch.no_grad():
            reference_chosen_logps = self.ref_model.get_log_probs(batch["chosen_input_ids"], batch["chosen_labels"], average_log_probs=False)
            reference_rejected_logps = self.ref_model.get_log_probs(batch["rejected_input_ids"], batch["rejected_labels"], average_log_probs=False)

        # Calculate the implicit reward terms as defined in the paper:
        # r_hat_theta(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x))
        # Note: The `average_log_probs=False` means we get log_probs per token.
        # The paper's formulation implies a single log_prob for the entire sequence y given x.
        # We sum the log probabilities over the sequence to get the log_prob of the sequence.
        # We need to mask out padding tokens for correct summation.
        
        # Masking for chosen sequences
        chosen_mask = (batch["chosen_labels"] != -100).float()
        chosen_log_ratio = (policy_chosen_logps - reference_chosen_logps) * chosen_mask
        chosen_log_ratio_sum = chosen_log_ratio.sum(dim=-1) # Sum over sequence length

        # Masking for rejected sequences
        rejected_mask = (batch["rejected_labels"] != -100).float()
        rejected_log_ratio = (policy_rejected_logps - reference_rejected_logps) * rejected_mask
        rejected_log_ratio_sum = rejected_log_ratio.sum(dim=-1) # Sum over sequence length

        # The implicit reward difference term:
        # beta * (log(pi_theta(yw|x)/pi_ref(yw|x)) - log(pi_theta(yl|x)/pi_ref(yl|x)))
        # This is `beta * (chosen_log_ratio_sum - rejected_log_ratio_sum)`
        
        # The paper's loss is -E [log sigma(beta * (chosen_log_ratio_sum - rejected_log_ratio_sum))]
        # trl's DPO loss is -log(sigmoid(beta * (chosen_log_ratio_sum - rejected_log_ratio_sum)))
        # which is equivalent to the paper's formulation.
        
        # The `beta` parameter is already part of the DPOConfig and applied internally by trl's DPOTrainer
        # when it calls `dpo_loss`. We just need to provide the log ratios.
        
        # The `dpo_loss` function in trl's DPOTrainer expects `policy_chosen_logps`, `policy_rejected_logps`,
        # `reference_chosen_logps`, `reference_rejected_logps` as *summed* log probabilities for the sequence.
        # Let's re-calculate them as sequence log probabilities.
        
        # trl's `get_log_probs` already sums over the sequence and handles padding if `average_log_probs=False`
        # and `labels` are provided. Let's re-verify the internal implementation.
        # Looking at trl's `dpo_loss` function, it expects `logps_pi` and `logps_ref` to be
        # the sum of log probabilities for the entire sequence.
        
        # Let's re-fetch log_probs ensuring they are summed over the sequence as per trl's expectation
        # and the paper's implicit reward definition.
        policy_chosen_logps_sum = model.get_log_probs(batch["chosen_input_ids"], batch["chosen_labels"], average_log_probs=False).sum(dim=-1)
        policy_rejected_logps_sum = model.get_log_probs(batch["rejected_input_ids"], batch["rejected_labels"], average_log_probs=False).sum(dim=-1)
        
        with torch.no_grad():
            reference_chosen_logps_sum = self.ref_model.get_log_probs(batch["chosen_input_ids"], batch["chosen_labels"], average_log_probs=False).sum(dim=-1)
            reference_rejected_logps_sum = self.ref_model.get_log_probs(batch["rejected_input_ids"], batch["rejected_labels"], average_log_probs=False).sum(dim=-1)

        # Calculate the DPO loss using the internal `dpo_loss` method of DPOTrainer
        # This method applies the beta and sigmoid as per Equation (7)
        loss, stats = self.dpo_loss(
            policy_chosen_logps_sum,
            policy_rejected_logps_sum,
            reference_chosen_logps_sum,
            reference_rejected_logps_sum,
        )

        return loss, stats

# === INJECTION POINT 3: EXECUTION ===
def run_experiment(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # The paper uses DPO, so we use DPOConfig.
    # The beta parameter is crucial for DPO, controlling the KL-divergence constraint.
    # The paper mentions `beta` in Equation (3) and (7).
    # A common value for beta is 0.1, as seen in many DPO implementations.
    training_args = DPOConfig(
        output_dir="runs/dpo_experiment",
        num_train_epochs=1,
        logging_steps=1,
        beta=0.1, # From the paper's equations, beta is a key hyperparameter.
        per_device_train_batch_size=2, # Small batch size for example
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        max_length=512,
        max_prompt_length=256,
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False, # Important for custom datasets
    )
    
    # Mock Data for DPO. DPO requires pairs of chosen and rejected responses.
    # The paper describes D = {x(i), y(i)_w, y(i)_l}
    # where x is the prompt, y_w is the preferred (chosen) completion,
    # and y_l is the dispreferred (rejected) completion.
    # We need to format this as a list of dictionaries, each containing 'prompt', 'chosen', 'rejected'.
    
    # Example prompts and completions
    prompts = [
        "What is the capital of France?",
        "Write a short story about a brave knight.",
        "Explain the concept of quantum entanglement.",
    ]
    chosen_responses = [
        "The capital of France is Paris.",
        "Sir Reginald, a knight of unwavering courage, embarked on a perilous quest to save the kingdom.",
        "Quantum entanglement is a phenomenon where two or more particles become linked in such a way that they share the same fate, regardless of the distance between them.",
    ]
    rejected_responses = [
        "France's capital is Berlin.", # Incorrect
        "A knight was brave.", # Too short, not descriptive
        "It's when particles are connected, like magic.", # Too simplistic, inaccurate
    ]

    # Create a mock dataset in the format expected by DPOTrainer
    train_dataset = []
    for i in range(len(prompts)):
        train_dataset.append({
            "prompt": prompts[i],
            "chosen": chosen_responses[i],
            "rejected": rejected_responses[i],
        })
    
    # Initialize the correct Trainer: CustomDPOTrainer
    # DPOTrainer requires a reference model, which is typically the SFT model.
    # For this example, we'll use the same model as the policy model for simplicity,
    # but in a real scenario, `ref_model` would be the SFT model.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name) # In a real scenario, this would be the SFT model

    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model, # DPO requires a reference model
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    print("--- 🚀 Starting Custom DPO Training ---")
    trainer.train()
    
if __name__ == "__main__":
    # Ensure you have a model that can be loaded by AutoModelForCausalLM
    # For a quick test, you might use a smaller model like "gpt2" or "facebook/opt-125m"
    # model_name = "gpt2" 
    run_experiment()

