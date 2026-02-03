import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from experiment import CustomTrainer, GRPOConfig # Import CustomTrainer and GRPOConfig
import math

def probe_loss_function():
    # 1. Import the Trainer class from `experiment`. (Already done above)

    # 2. Create dummy torch inputs (batch size 2).
    model_name = "prajjwal1/bert-tiny" # Using a tiny model to avoid OOM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Dummy prompts
    prompts = [
        "What is the capital of France?",
        "Tell me a short story about a brave knight.",
    ]

    # Tokenize prompts
    # We need to simulate the input format that `compute_loss` expects.
    # For GRPO, this typically involves `input_ids`, `attention_mask`, `rewards`, etc.
    # Let's create dummy inputs that mimic what a batch from the DataLoader would look like.
    # The `compute_loss` method in TRL's GRPOTrainer expects specific keys.
    # We'll create dummy `input_ids`, `attention_mask`, `rewards`, `advantages`, `logprobs`, `values`.

    # Encode prompts
    prompt_encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=50)
    input_ids = prompt_encodings["input_ids"]
    attention_mask = prompt_encodings["attention_mask"]

    # Simulate generated sequences (for reward calculation context)
    # In a real scenario, these would come from the model's generation.
    # For `compute_loss`, we need the full sequence (prompt + generation).
    # Let's create dummy full sequences.
    dummy_generated_texts = [
        "What is the capital of France? Paris is the capital.",
        "Tell me a short story about a brave knight. Sir Lancelot fought a dragon.",
    ]
    full_encodings = tokenizer(dummy_generated_texts, return_tensors="pt", padding=True, truncation=True, max_length=100)
    full_input_ids = full_encodings["input_ids"]
    full_attention_mask = full_encodings["attention_mask"]

    # Dummy rewards, advantages, logprobs, values
    batch_size = len(prompts)
    sequence_length = full_input_ids.shape[1] # Max sequence length in the batch

    # Rewards are typically per sequence
    dummy_rewards = torch.tensor([0.5, -0.2], dtype=torch.float32).to("cpu") # Rewards for each sequence

    # Advantages, logprobs, values are typically per token
    dummy_advantages = torch.randn(batch_size, sequence_length, dtype=torch.float32).to("cpu")
    dummy_logprobs = torch.randn(batch_size, sequence_length, dtype=torch.float32).to("cpu")
    dummy_values = torch.randn(batch_size, sequence_length, dtype=torch.float32).to("cpu")

    # The `compute_loss` method expects a dictionary with specific keys.
    # These keys are usually populated by the `_prepare_inputs` method of the trainer
    # or directly from the `dataloader`.
    dummy_inputs = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "rewards": dummy_rewards,
        "advantages": dummy_advantages,
        "logprobs": dummy_logprobs,
        "values": dummy_values,
        # `num_items_in_batch` is used for token-level averaging.
        # It should be the total number of non-padded tokens.
        "num_items_in_batch": torch.sum(full_attention_mask).item()
    }

    # 3. Instantiate the Trainer
    # We need a dummy model for instantiation, but the actual forward pass
    # for loss calculation might use the model passed to `compute_loss`.
    # For `compute_loss`, the `model` argument is the policy model.
    dummy_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create a minimal GRPOConfig
    config = GRPOConfig(
        output_dir="probe_output",
        per_device_train_batch_size=batch_size,
        clip_range=CustomTrainer.EPSILON_LOW,
        beta_kl=0.0, # As per the experiment script
        # Other parameters can be default or minimal
    )

    # Instantiate CustomTrainer
    # We don't need a real dataset or reward_funcs for just testing compute_loss
    trainer = CustomTrainer(
        model=dummy_model,
        args=config,
        train_dataset=None, # Not needed for compute_loss directly
        processing_class=tokenizer,
        reward_funcs=None, # Not needed for compute_loss directly
    )

    # Move model to CPU for testing if not already there
    trainer.model.to("cpu")

    # 4. Run `trainer.compute_loss`
    # The `compute_loss` method expects the model as the first argument.
    # It also expects `inputs` which is the batch dictionary.
    loss, outputs = trainer.compute_loss(trainer.model, dummy_inputs, return_outputs=True)

    # 5. Assert that the loss is a scalar and not NaN.
    assert isinstance(loss, torch.Tensor), f"Loss is not a torch.Tensor, but {type(loss)}"
    assert loss.ndim == 0, f"Loss is not a scalar, but has {loss.ndim} dimensions"
    assert not torch.isnan(loss).any(), "Loss is NaN"
    assert not torch.isinf(loss).any(), "Loss is Inf"

    # Optionally, check outputs if `return_outputs=True`
    assert isinstance(outputs, dict), "Outputs should be a dictionary"
    assert "loss" in outputs, "Outputs dictionary should contain 'loss'"
    assert torch.isclose(loss, outputs["loss"]), "Returned loss and loss in outputs do not match"

    # 6. Print "PROBE_SUCCESS" if it passes.
    print("PROBE_SUCCESS")

if __name__ == "__main__":
    probe_loss_function()