import torch
from transformers import AutoTokenizer
from experiment import CustomTrainer, reward_function # Import CustomTrainer and reward_function
from trl import GRPOConfig
from datasets import Dataset

def probe_loss_function():
    # 1. Import the Trainer class from `experiment` (already done above)

    # 2. Create dummy torch inputs (batch size 2)
    model_name = "prajjwal1/bert-tiny" # Using a tiny model to avoid OOM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Dummy prompts and completions for the batch
    dummy_prompts = [
        "What is the capital of France?",
        "Tell me a short story about a cat.",
    ]
    dummy_completions = [
        "The capital of France is Paris.",
        "Once upon a time, there was a fluffy cat named Whiskers.",
    ]

    # Tokenize inputs
    # In GRPO, inputs typically contain 'input_ids', 'attention_mask', 'labels' (for reference model)
    # and potentially 'rewards' or 'advantages' which are computed internally or passed.
    # For `compute_loss`, we primarily need `input_ids` and `attention_mask` for the model,
    # and potentially `rewards` if the loss function directly uses them.
    # Since GRPO's compute_loss calculates rewards/advantages internally based on model outputs,
    # we just need valid tokenized prompts and completions.

    # Simulate the structure of inputs that `compute_loss` expects.
    # GRPO's `compute_loss` expects `inputs` to be a dictionary containing
    # `input_ids`, `attention_mask`, `rewards`, `advantages`, `logprobs`, etc.
    # For a basic test, we'll provide `input_ids` and `attention_mask` for the prompts
    # and let the trainer generate completions and compute rewards.

    # Create a dummy dataset for the trainer initialization
    dummy_dataset = Dataset.from_dict({"prompt": dummy_prompts})

    # 3. Instantiate the Trainer
    config = GRPOConfig(
        output_dir="runs/probe_experiment",
        num_train_epochs=1,
        max_steps=1, # Only need one step for the probe
        per_device_train_batch_size=2,
        num_generations=1,
        max_completion_length=10, # Keep completions short for speed
        learning_rate=1e-5,
        clip_range_ratio=CustomTrainer.EPSILON_LOW,
        beta=0.0,
        # Set logging_steps to a high value to avoid creating logs during probe
        logging_steps=999999,
        # Disable saving to speed up
        save_steps=999999,
    )

    trainer = CustomTrainer(
        model=model_name,
        args=config,
        train_dataset=dummy_dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_function],
    )

    # Prepare a dummy batch that mimics what `get_batch_loss_metrics` or `compute_loss` expects
    # The `compute_loss` method in GRPOTrainer typically gets its inputs from the data collator
    # which includes `input_ids`, `attention_mask`, `rewards`, `advantages`, `logprobs`, etc.
    # For a direct call to `compute_loss`, we need to simulate these.
    # The easiest way to get a valid batch is to use the trainer's internal data loader.
    # However, `compute_loss` can also be called with just `input_ids` and `attention_mask`
    # if it internally handles generation and reward calculation.
    # Let's create a minimal batch for the model.

    # Tokenize dummy prompts for the model's input
    tokenized_prompts = tokenizer(dummy_prompts, return_tensors="pt", padding=True, truncation=True, max_length=50)
    
    # In a real GRPO scenario, the `inputs` dict passed to `compute_loss` would contain
    # generated `response_input_ids`, `response_attention_mask`, `rewards`, `advantages`, etc.
    # Since we are testing `compute_loss` directly, and it's designed to be called
    # with a full batch (including generated responses and their associated metrics),
    # we need to simulate that.

    # The simplest way to test `compute_loss` in TRL trainers is often to
    # let the trainer's `get_batch_loss_metrics` or `training_step` prepare the batch.
    # However, the prompt asks to call `trainer.compute_loss` directly.
    # `compute_loss` in GRPOTrainer expects `inputs` to contain:
    # `query_input_ids`, `query_attention_mask`, `response_input_ids`, `response_attention_mask`,
    # `rewards`, `logprobs`, `advantages`, `returns`.
    # We will create dummy tensors for these.

    # Dummy generated responses (tokenized)
    tokenized_responses = tokenizer(dummy_completions, return_tensors="pt", padding=True, truncation=True, max_length=50)

    # Combine query and response input_ids and attention_mask
    # This is how TRL typically structures the full sequence
    full_input_ids = torch.cat([tokenized_prompts.input_ids, tokenized_responses.input_ids], dim=-1)
    full_attention_mask = torch.cat([tokenized_prompts.attention_mask, tokenized_responses.attention_mask], dim=-1)

    # Dummy rewards, logprobs, advantages, returns
    # These are usually computed by the trainer after generation.
    # For a probe, we can use placeholder values.
    batch_size = 2
    sequence_length = full_input_ids.shape[1] # Max length of combined sequence
    
    dummy_rewards = torch.randn(batch_size, device=trainer.accelerator.device) # Per sequence reward
    dummy_logprobs = torch.randn(batch_size, sequence_length, device=trainer.accelerator.device) # Logprobs for each token
    dummy_advantages = torch.randn(batch_size, sequence_length, device=trainer.accelerator.device) # Advantages for each token
    dummy_returns = torch.randn(batch_size, sequence_length, device=trainer.accelerator.device) # Returns for each token

    # Create the dummy inputs dictionary
    dummy_inputs = {
        "query_input_ids": tokenized_prompts.input_ids.to(trainer.accelerator.device),
        "query_attention_mask": tokenized_prompts.attention_mask.to(trainer.accelerator.device),
        "response_input_ids": tokenized_responses.input_ids.to(trainer.accelerator.device),
        "response_attention_mask": tokenized_responses.attention_mask.to(trainer.accelerator.device),
        "rewards": dummy_rewards,
        "logprobs": dummy_logprobs,
        "advantages": dummy_advantages,
        "returns": dummy_returns,
        # Add other potential inputs if the specific GRPO compute_loss requires them,
        # e.g., `ref_logprobs` if a reference model is used for KL.
        # Since beta=0, ref_logprobs might not be strictly necessary for loss computation,
        # but the method might still expect it. Let's add a dummy one.
        "ref_logprobs": torch.randn(batch_size, sequence_length, device=trainer.accelerator.device),
    }

    # 4. Run `trainer.compute_loss`
    print("Attempting to compute loss...")
    loss, outputs = trainer.compute_loss(trainer.model, dummy_inputs, return_outputs=True)

    # 5. Assert that the loss is a scalar and not NaN.
    assert isinstance(loss, torch.Tensor), f"Loss is not a torch.Tensor, but {type(loss)}"
    assert loss.ndim == 0, f"Loss is not a scalar, but has {loss.ndim} dimensions"
    assert not torch.isnan(loss), "Loss is NaN"
    assert torch.isfinite(loss), "Loss is not finite (inf)"

    # Optional: Check outputs if return_outputs=True
    assert isinstance(outputs, dict), "Outputs should be a dictionary"
    assert "loss" in outputs, "Outputs dictionary should contain 'loss'"
    assert torch.isclose(outputs["loss"], loss), "Loss in outputs does not match returned loss"

    # 6. Print "PROBE_SUCCESS" if it passes.
    print("PROBE_SUCCESS")

if __name__ == "__main__":
    probe_loss_function()