import torch
import torch.nn as nn
from transformers import AutoTokenizer
from experiment import CustomDPOTrainer, DPOConfig # Import CustomDPOTrainer from experiment
from typing import Dict, Any

# Rule 3: Use a tiny model to avoid OOM
MODEL_NAME = "prajjwal1/bert-tiny" 

def probe_loss_function():
    print("Starting probe.py for CustomDPOTrainer loss function...")

    # Rule 2: Create dummy torch inputs (batch size 2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dummy data mimicking the format expected by DPOTrainer
    # DPOTrainer expects 'prompt', 'chosen', 'rejected'
    dummy_prompts = [
        "Human: What is the capital of France?",
        "Human: Tell me a short story.",
    ]
    dummy_chosen = [
        "Assistant: The capital of France is Paris.",
        "Assistant: Once upon a time, in a land far away, lived a brave knight.",
    ]
    dummy_rejected = [
        "Assistant: France is a country in Europe.",
        "Assistant: The sky is blue.",
    ]

    # Tokenize the dummy data
    # DPOTrainer's get_batch_loss_metrics expects tokenized inputs
    # It will tokenize internally if not already done, but we can prepare it.
    # The trainer will handle the full tokenization and padding.
    
    # The actual batch passed to get_batch_loss_metrics will contain tokenized versions
    # of prompt_input_ids, chosen_input_ids, rejected_input_ids, etc.
    # We need to simulate this structure.

    # Let's create a more realistic dummy batch that the trainer would generate
    # after tokenization and processing.
    
    # First, tokenize the raw strings to get an idea of the structure
    tokenized_prompts = tokenizer(dummy_prompts, return_tensors="pt", padding=True, truncation=True, max_length=50)
    tokenized_chosen = tokenizer(dummy_chosen, return_tensors="pt", padding=True, truncation=True, max_length=50)
    tokenized_rejected = tokenizer(dummy_rejected, return_tensors="pt", padding=True, truncation=True, max_length=50)

    # Construct a dummy batch similar to what DPOTrainer's collator would produce
    dummy_batch: Dict[str, Any] = {
        "prompt_input_ids": tokenized_prompts.input_ids,
        "prompt_attention_mask": tokenized_prompts.attention_mask,
        "chosen_input_ids": tokenized_chosen.input_ids,
        "chosen_attention_mask": tokenized_chosen.attention_mask,
        "rejected_input_ids": tokenized_rejected.input_ids,
        "rejected_attention_mask": tokenized_rejected.attention_mask,
        # DPOTrainer also adds labels, but for loss calculation, it often derives them
        # or uses the input_ids themselves. Let's keep it minimal for a probe.
        # The `get_batch_loss_metrics` method primarily uses the `_input_ids` and `_attention_mask`.
    }

    # Rule 3: Instantiate the Trainer
    # We need a dummy DPOConfig for instantiation
    training_args = DPOConfig(
        output_dir="./dummy_output",
        per_device_train_batch_size=2,
        max_steps=1, # Not actually training, just need config
        beta=0.1,
        remove_unused_columns=False,
    )

    # Instantiate CustomDPOTrainer
    # For the probe, we don't need actual datasets, just the model and tokenizer.
    # The `model` and `ref_model` are required.
    # We'll use a simple dummy model for the probe.
    try:
        # DPOTrainer expects a causal language model
        # For bert-tiny, we might need to wrap it or use a different tiny model
        # Let's use a model that is typically used for causal LM tasks, even if tiny.
        # 'gpt2' is a good small causal LM. 'prajjwal1/bert-tiny' is not a causal LM.
        # Let's switch to a tiny causal LM.
        causal_lm_model_name = "gpt2" # A small causal LM
        tokenizer_causal = AutoTokenizer.from_pretrained(causal_lm_model_name)
        if tokenizer_causal.pad_token is None:
            tokenizer_causal.pad_token = tokenizer_causal.eos_token

        trainer = CustomDPOTrainer(
            model=causal_lm_model_name,
            ref_model=causal_lm_model_name,
            args=training_args,
            tokenizer=tokenizer_causal,
            # No datasets needed for just calling get_batch_loss_metrics directly
            train_dataset=None,
            eval_dataset=None,
        )
    except Exception as e:
        print(f"Error instantiating CustomDPOTrainer: {e}")
        print("Ensure 'gpt2' is installed or choose another small causal LM.")
        return

    # Move dummy batch to the correct device if trainer uses one
    if trainer.args.device.type == "cuda":
        for key in dummy_batch:
            if isinstance(dummy_batch[key], torch.Tensor):
                dummy_batch[key] = dummy_batch[key].to(trainer.args.device)

    # Rule 4: Run trainer.get_batch_loss_metrics
    print("Calling trainer.get_batch_loss_metrics...")
    try:
        metrics = trainer.get_batch_loss_metrics(trainer.model, dummy_batch, train_eval="train")
        loss = metrics["loss"]
        print(f"Computed loss: {loss}")

        # Rule 5: Assert that the loss is a scalar and not NaN
        assert isinstance(loss, torch.Tensor), f"Loss is not a torch.Tensor, but {type(loss)}"
        assert loss.ndim == 0, f"Loss is not a scalar, but has {loss.ndim} dimensions"
        assert not torch.isnan(loss).any(), "Loss is NaN"
        assert torch.isfinite(loss).all(), "Loss is infinite"

        # Rule 6: Print "PROBE_SUCCESS" if it passes
        print("PROBE_SUCCESS")

    except Exception as e:
        print(f"PROBE_FAILURE: An error occurred during loss computation or assertion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    probe_loss_function()