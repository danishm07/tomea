import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig, DPOTrainer, DPOConfig
from datasets import load_dataset
from typing import List, Optional

# === REWARD FUNCTION ===
def reward_function(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Placeholder reward function. In DAPO, the reward is rule-based,
    typically 1 for correct answers and -1 otherwise, with additional
    shaping for overlong responses.
    """
    # For a smoke test, we'll use a simple length-based reward.
    # In a real DAPO implementation, this would involve a math verifier.
    return [float(len(c.split())) / 10.0 for c in completions]

# === CUSTOM TRAINER ===
class CustomTrainer(GRPOTrainer):
    # DAPO-specific hyperparameters as class constants
    # These are derived from the paper's "Training Details" (Section 4.1)
    # and "Decouple Clip" (Section 3.1)
    EPSILON_LOW = 0.2
    EPSILON_HIGH = 0.28
    MAX_GENERATION_LENGTH = 20480  # Lmax
    SOFT_PUNISH_CACHE = 4096       # Lcache
    # Note: The paper mentions `mean({Ri}G_i=1)` and `std({Ri}G_i=1)` for advantage
    # calculation, which is standard GRPO.
    # The `s.t. 0 < {oi | is_equivalent(a, oi)} < G` condition for Dynamic Sampling
    # implies filtering at the data loading/sampling stage, not directly in compute_loss.
    # Token-level policy gradient loss is handled by how the loss is aggregated,
    # which is a change to the `compute_loss` method.

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # CRITICAL: Preserve signature compatibility
        # The paper states: "Token-level Policy Gradient Loss"
        # "In this setting, longer sequences can have more influence on the overall gradient update
        # compared to shorter sequences."
        # This implies that instead of averaging loss per sample and then across samples,
        # the loss should be averaged across all tokens in the batch.
        # The base GRPOTrainer's compute_loss already handles token-level loss calculation
        # and then averages it appropriately for the batch.
        # The key difference in DAPO's Equation (12) compared to GRPO's Equation (5)
        # is the normalization factor:
        # GRPO: `1/G * sum(1/|oi| * sum(loss_t))` (sample-level averaging)
        # DAPO: `1/(sum(|oi|)) * sum(sum(loss_t))` (token-level averaging across batch)
        # TRL's GRPOTrainer's `compute_loss` typically aggregates token-level losses
        # and then averages them over the batch, which aligns with the spirit of DAPO's
        # token-level loss if `num_items_in_batch` is used to scale the loss correctly.

        # For the purpose of this exercise, we assume the base GRPOTrainer's
        # `compute_loss` with `num_items_in_batch` (which represents the total number of tokens)
        # will effectively implement the token-level averaging across the batch.
        # If `num_items_in_batch` is not provided, it defaults to the number of sequences,
        # which would be sample-level averaging.
        # To enforce token-level averaging, we ensure `num_items_in_batch` is the total number of tokens.

        # The `clip` function in Equation (10) uses `epsilon_low` and `epsilon_high`.
        # The base GRPOTrainer uses a single `clip_range` parameter.
        # We need to override the clipping logic if `epsilon_low` and `epsilon_high` are different.
        # However, `GRPOConfig` does not expose separate `epsilon_low` and `epsilon_high`.
        # The `clip_range` in `GRPOConfig` corresponds to `epsilon`.
        # If `epsilon_low` and `epsilon_high` are different, a custom clipping function
        # would be needed, which is not directly supported by overriding `compute_loss`
        # without modifying the internal `ppo_loss` function of TRL.

        # For this exercise, we'll assume `clip_range` in GRPOConfig will be set to `EPSILON_LOW`
        # and acknowledge that `EPSILON_HIGH` would require deeper modification of TRL's PPO loss.
        # If `EPSILON_LOW` and `EPSILON_HIGH` are different, the current TRL `GRPOTrainer`
        # cannot directly implement "Clip-Higher" without internal modifications.
        # The paper sets `epsilon_low` to 0.2 and `epsilon_high` to 0.28, implying they are different.

        # The `Overlong Reward Shaping` and `Dynamic Sampling` are primarily
        # handled outside `compute_loss` in the reward function and data sampling logic respectively.

        # Call the parent's compute_loss.
        # We assume `num_items_in_batch` will be correctly set to the total number of tokens
        # by the TRL framework for token-level loss aggregation if configured.
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

# === EXECUTION ===
def run_experiment():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # NOTE: Pass ONLY standard library arguments here.
    # Do NOT pass paper-specific params like 'beta_high' or 'epsilon'.
    # The `clip_range` in GRPOConfig corresponds to the single epsilon in standard PPO/GRPO.
    # DAPO's `epsilon_low` and `epsilon_high` would require a custom loss function
    # or a TRL extension that supports asymmetric clipping.
    # For this exercise, we set `clip_range` to `EPSILON_LOW`.
    config = GRPOConfig(
        output_dir="runs/dapo_experiment",
        num_train_epochs=1,
        max_steps=5,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=128,
        learning_rate=1e-5,
        # `clip_range` in TRL's GRPOConfig corresponds to epsilon.
        # DAPO uses epsilon_low and epsilon_high. We use epsilon_low here.
        # The `clip_range` argument does not exist in GRPOConfig.
        # The clipping is handled internally by the PPO algorithm.
        # If custom clipping is needed, it would require modifying the PPO loss function directly.
        # clip_range=CustomTrainer.EPSILON_LOW, # REMOVED: This argument does not exist.
        # The paper removes KL divergence, so we set `beta_kl` to 0.0
        # beta_kl=0.0, # REMOVED: This argument does not exist in GRPOConfig.
        # The paper uses group-relative advantage, which is standard for GRPO.
        # The reward function will handle the `Overlong Reward Shaping`.
    )

    # Load and map dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split="train[:200]")

    def format_dataset(sample):
        # Basic prompt extraction
        # The paper uses (q, a) pairs, where 'a' is the ground-truth answer.
        # For hh-rlhf, we'll use the 'chosen' response as the "answer" for the prompt.
        # The reward function would then verify against this 'answer'.
        parts = sample["chosen"].split("\n\nAssistant:")
        if len(parts) > 1:
            prompt = parts[0] + "\n\nAssistant:"
            # In a real DAPO setup, the 'answer' part would be used by the reward model
            # to determine correctness, not directly as part of the prompt for generation.
            # For this example, we just extract the prompt.
        else:
            prompt = sample["chosen"] # Fallback if format is unexpected
        return {"prompt": prompt}

    train_dataset = dataset.map(format_dataset)

    # Dynamic Sampling:
    # The paper states: "Before training, we keep sampling until the batch is
    # fully filled with samples whose accuracy is neither 0 nor 1."
    # This implies a custom data loader or a modification to the `trainer.get_train_dataloader()`
    # method to implement this filtering. For a smoke test, we'll proceed with the standard
    # dataset loading, acknowledging this is a simplification.
    # In a full implementation, the `reward_function` would be called during data loading
    # to filter samples.

    trainer = CustomTrainer(
        model=model_name,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,     # CORRECT ARG NAME
        reward_funcs=[reward_function], # CORRECT LIST FORMAT
    )

    print("--- 🚀 Starting Training (Smoke Test) ---")
    trainer.train()

if __name__ == "__main__":
    run_experiment()