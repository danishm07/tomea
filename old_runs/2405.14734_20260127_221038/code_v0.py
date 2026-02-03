import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from typing import List, Dict, Any, Optional

# === INJECTION POINT 1: REWARD FUNCTION (PPO/GRPO Only) ===
# This is a DPO-style paper, so this function is not used.
def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    raise NotImplementedError("Reward function is not applicable for DPO-style trainers.")

# === INJECTION POINT 2: CUSTOM TRAINER ===
class SimPOTrainer(DPOTrainer):

    def __init__(self, *args, beta: float = 0.1, gamma: float = 0.0, **kwargs):
        super().__init__(*args, beta=beta, **kwargs)
        self.gamma = gamma

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, List[torch.LongTensor]],
        train_eval: str = "train",
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Calculates the SimPO loss for the given batch of preferences.

        Args:
            model: The policy model.
            batch: A dictionary containing the input data for the batch.
            train_eval: Whether the loss is calculated for training or evaluation.

        Returns:
            A dictionary containing the loss and metrics.
        """
        # Extract relevant tensors from the batch
        policy_chosen_logps = batch["policy_chosen_logps"]
        policy_rejected_logps = batch["policy_rejected_logps"]

        # Calculate length-normalized log probabilities
        # The paper defines p_theta(y|x) = 1/|y| * log pi_theta(y|x)
        # We need to get the length of the sequences to normalize.
        # The logps from DPOTrainer are already summed log probabilities.
        # We need to get the actual sequence lengths.
        # For `policy_chosen_logps`, `batch["chosen_attention_mask"]` can be used to get length.
        # The length is the sum of the attention mask, excluding padding.
        chosen_lengths = batch["chosen_attention_mask"].sum(dim=1)
        rejected_lengths = batch["rejected_attention_mask"].sum(dim=1)

        # Ensure lengths are at least 1 to avoid division by zero
        chosen_lengths = torch.max(chosen_lengths, torch.ones_like(chosen_lengths))
        rejected_lengths = torch.max(rejected_lengths, torch.ones_like(rejected_lengths))

        # Calculate length-normalized log probabilities
        # r_SimPO(x, y) = beta * (1/|y|) * log pi_theta(y|x)
        r_chosen = self.beta * (policy_chosen_logps / chosen_lengths)
        r_rejected = self.beta * (policy_rejected_logps / rejected_lengths)

        # SimPO Objective: -log sigma(r_SimPO(x, yw) - r_SimPO(x, yl) - gamma)
        # Equation (6) from the paper:
        # L_SimPO(pi_theta) = -E_{(x,yw,yl)~D} [log sigma(beta/|yw| log pi_theta(yw|x) - beta/|yl| log pi_theta(yl|x) - gamma)]
        
        # The `self.beta` from DPOConfig is used as the scaling factor for the log-probabilities.
        # In SimPO, this `beta` is part of the reward definition.
        # The `beta` in the DPO loss function is usually a temperature parameter.
        # Here, we are directly implementing the SimPO loss, where the `beta` from the paper
        # is integrated into `r_chosen` and `r_rejected`.
        # The `beta` parameter of the DPOTrainer is effectively the `beta` from SimPO's reward definition.

        logits = r_chosen - r_rejected - self.gamma
        losses = -F.logsigmoid(logits)

        # Calculate metrics
        chosen_rewards = r_chosen
        rejected_rewards = r_rejected
        
        # The original DPO trainer calculates chosen_rewards and rejected_rewards
        # using the reference model as well. SimPO is reference-free.
        # We can still report these for consistency, but they are based on the SimPO reward.
        
        # For logging, we can also compute the win rate based on the SimPO reward
        win_rate = (logits > 0).float().mean()

        return losses.mean(), {
            f"{train_eval}/loss": losses.mean(),
            f"{train_eval}/rewards/chosen": chosen_rewards.mean(),
            f"{train_eval}/rewards/rejected": rejected_rewards.mean(),
            f"{train_eval}/rewards/accuracies": win_rate,
            f"{train_eval}/rewards/margins": (chosen_rewards - rejected_rewards).mean(),
            f"{train_eval}/rewards/logits": logits.mean(),
        }


# === INJECTION POINT 3: EXECUTION ===
def run_experiment(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # SimPO uses DPOConfig as its base, but we add gamma.
    # The beta in DPOConfig corresponds to the beta in SimPO's reward formula (Eq. 4).
    # The paper suggests beta between 2.0 and 2.5, and gamma between 0.5 and 1.5.
    training_args = DPOConfig(
        output_dir="runs/simpo_experiment",
        num_train_epochs=1,
        logging_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        beta=2.0,  # Corresponds to beta in SimPO's reward (Eq. 4)
        max_length=512,
        max_prompt_length=256,
        # SimPO is reference-free, so we don't need a reference model.
        # However, DPOTrainer expects a ref_model. We can pass the same model
        # but SimPOTrainer's get_batch_loss_metrics does not use it.
        # Alternatively, we can set `ref_model_init_kwargs` to None or pass a dummy.
        # For simplicity and to align with the "reference-free" nature, we'll ensure
        # our custom loss doesn't use the reference model's logps.
        # The DPOTrainer's `compute_loss` method will still compute `reference_chosen_logps`
        # and `reference_rejected_logps`, but our overridden `get_batch_loss_metrics`
        # will ignore them.
    )

    # Mock Data for DPO (prompt, chosen, rejected)
    # In a real scenario, this would be loaded from a preference dataset.
    # The `chosen` and `rejected` fields are the full sequences (prompt + response).
    # The DPOTrainer will handle tokenization and splitting into prompt/response.
    dataset = [
        {"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "France's capital is Berlin."},
        {"prompt": "Tell me a short story.", "chosen": "Once upon a time, in a land far away, lived a brave knight.", "rejected": "Story: A knight lived in a land."},
        {"prompt": "Explain quantum physics simply.", "chosen": "Quantum physics studies the smallest particles and their strange behaviors, like existing in multiple places at once.", "rejected": "It's about really small stuff."},
    ] * 10

    # Initialize the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # SimPO is reference-free, so the reference model is not strictly needed for the loss calculation.
    # However, DPOTrainer's internal mechanisms might still try to load it.
    # We can pass the same model as the reference model, and our custom loss will ignore its outputs.
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)


    # Initialize the SimPO Trainer
    trainer = SimPOTrainer(
        model=model,
        ref_model=ref_model, # DPOTrainer expects a ref_model, even if not used in custom loss
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        # Pass the gamma parameter to the custom trainer
        gamma=1.0, # Example gamma value, paper suggests 0.5 to 1.5
    )

    print("--- 🚀 Starting SimPO Training ---")
    trainer.train()
    print("--- ✅ SimPO Training Complete ---")

if __name__ == "__main__":
    run_experiment()
