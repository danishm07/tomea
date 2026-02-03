import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig
from experiment import CustomDPOTrainer # Import the CustomDPOTrainer from your experiment script

# Rule 2: Create dummy torch inputs (batch size 2)
def create_dummy_inputs(tokenizer, batch_size=2):
    # DPO Trainer expects 'prompt_input_ids', 'chosen_input_ids', 'rejected_input_ids'
    # and their corresponding attention masks.
    # Let's create some simple dummy text.
    prompts = ["Human: What is your favorite color?", "Human: Tell me a joke."]
    chosen_responses = ["Assistant: I do not have a favorite color.", "Assistant: Why did the scarecrow win an award? Because he was outstanding in his field!"]
    rejected_responses = ["Assistant: Blue.", "Assistant: I don't know any jokes."]

    # Tokenize prompts
    prompt_encodings = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=50)
    prompt_input_ids = prompt_encodings['input_ids']
    prompt_attention_mask = prompt_encodings['attention_mask']

    # Tokenize chosen responses
    chosen_encodings = tokenizer(chosen_responses, return_tensors="pt", padding=True, truncation=True, max_length=100)
    chosen_input_ids = chosen_encodings['input_ids']
    chosen_attention_mask = chosen_encodings['attention_mask']

    # Tokenize rejected responses
    rejected_encodings = tokenizer(rejected_responses, return_tensors="pt", padding=True, truncation=True, max_length=100)
    rejected_input_ids = rejected_encodings['input_ids']
    rejected_attention_mask = rejected_encodings['attention_mask']

    # DPOTrainer's get_batch_loss_metrics expects a dictionary with specific keys
    # that are typically prepared by its data collator.
    # We need to mimic this structure for a direct call to compute_loss or get_batch_loss_metrics.
    # The DPO trainer internally concatenates prompt and response for chosen/rejected.
    # For `get_batch_loss_metrics`, it expects `chosen_logps`, `rejected_logps`, etc.
    # Let's simplify and directly call `compute_loss` which takes model outputs.

    # For `compute_loss`, we need the raw input_ids and attention_mask for chosen/rejected
    # and the corresponding labels.
    # The `DPOTrainer`'s `compute_loss` method expects `model`, `inputs`, `return_outputs`.
    # The `inputs` dictionary for `compute_loss` (when called internally by `training_step`)
    # contains `input_ids`, `attention_mask`, `labels` for chosen and rejected.
    # Let's prepare inputs that resemble what `DPOTrainer`'s data collator would produce.

    # Concatenate prompt and chosen/rejected responses
    chosen_full_input_ids = []
    chosen_full_attention_mask = []
    rejected_full_input_ids = []
    rejected_full_attention_mask = []

    for i in range(batch_size):
        # Concatenate prompt and chosen
        full_chosen_ids = torch.cat((prompt_input_ids[i], chosen_input_ids[i][1:])) # Remove BOS from response
        full_chosen_mask = torch.cat((prompt_attention_mask[i], chosen_attention_mask[i][1:]))
        chosen_full_input_ids.append(full_chosen_ids)
        chosen_full_attention_mask.append(full_chosen_mask)

        # Concatenate prompt and rejected
        full_rejected_ids = torch.cat((prompt_input_ids[i], rejected_input_ids[i][1:])) # Remove BOS from response
        full_rejected_mask = torch.cat((prompt_attention_mask[i], rejected_attention_mask[i][1:]))
        rejected_full_input_ids.append(full_rejected_ids)
        rejected_full_attention_mask.append(full_rejected_mask)

    # Pad to the longest sequence in the batch for chosen and rejected separately
    max_chosen_len = max(len(x) for x in chosen_full_input_ids)
    max_rejected_len = max(len(x) for x in rejected_full_input_ids)

    padded_chosen_input_ids = torch.full((batch_size, max_chosen_len), tokenizer.pad_token_id, dtype=torch.long)
    padded_chosen_attention_mask = torch.zeros((batch_size, max_chosen_len), dtype=torch.long)
    padded_rejected_input_ids = torch.full((batch_size, max_rejected_len), tokenizer.pad_token_id, dtype=torch.long)
    padded_rejected_attention_mask = torch.zeros((batch_size, max_rejected_len), dtype=torch.long)

    for i in range(batch_size):
        padded_chosen_input_ids[i, :len(chosen_full_input_ids[i])] = chosen_full_input_ids[i]
        padded_chosen_attention_mask[i, :len(chosen_full_attention_mask[i])] = chosen_full_attention_mask[i]
        padded_rejected_input_ids[i, :len(rejected_full_input_ids[i])] = rejected_full_input_ids[i]
        padded_rejected_attention_mask[i, :len(rejected_full_attention_mask[i])] = rejected_full_attention_mask[i]

    # Labels are typically the same as input_ids for language modeling tasks,
    # but masked for the prompt part in DPO.
    # DPOTrainer's `get_batch_loss_metrics` handles label creation internally.
    # For `compute_loss`, we need to provide `labels`.
    # The `labels` for DPO are the response tokens, with prompt tokens set to -100.
    chosen_labels = padded_chosen_input_ids.clone()
    rejected_labels = padded_rejected_input_ids.clone()

    # Mask out prompt tokens in labels
    for i in range(batch_size):
        chosen_labels[i, :len(prompt_input_ids[i])] = -100
        rejected_labels[i, :len(prompt_input_ids[i])] = -100

    return {
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "chosen_input_ids": padded_chosen_input_ids,
        "chosen_attention_mask": padded_chosen_attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": padded_rejected_input_ids,
        "rejected_attention_mask": padded_rejected_attention_mask,
        "rejected_labels": rejected_labels,
    }


def run_probe():
    # Rule 3: Instantiate the Trainer (use a tiny model)
    model_name = "prajjwal1/bert-tiny" # Using a very small model to avoid OOM
    # Note: bert-tiny is a BERT model, not a CausalLM.
    # For DPO, we need a CausalLM. Let's use a small CausalLM.
    # "Qwen/Qwen2.5-0.5B-Instruct" is a good choice, but might still be large for very limited memory.
    # Let's try a smaller one if available or stick to Qwen and hope for the best on a typical dev machine.
    # If OOM, one might need to use a dummy model or mock the model.
    # For this probe, we'll use a very small CausalLM if possible, or mock if necessary.
    # Let's try 'facebook/opt-125m' as it's a small CausalLM.
    model_name = "facebook/opt-125m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Dummy DPOConfig
    training_args = DPOConfig(
        output_dir="./probe_output",
        per_device_train_batch_size=2,
        max_steps=1, # Only need one step for loss calculation
        beta=0.1,
        report_to="none", # Disable reporting
    )

    # DPOTrainer requires a train_dataset, even if we don't call .train()
    # We'll provide a minimal dummy dataset.
    dummy_dataset = [
        {"prompt": "dummy prompt 1", "chosen": "dummy chosen 1", "rejected": "dummy rejected 1"},
        {"prompt": "dummy prompt 2", "chosen": "dummy chosen 2", "rejected": "dummy rejected 2"},
    ]

    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dummy_dataset, # Provide a minimal dummy dataset
    )

    # Rule 2: Create dummy torch inputs (batch size 2)
    dummy_inputs = create_dummy_inputs(tokenizer, batch_size=training_args.per_device_train_batch_size)

    # Move inputs to the same device as the model
    device = trainer.model.device
    for key in dummy_inputs:
        if isinstance(dummy_inputs[key], torch.Tensor):
            dummy_inputs[key] = dummy_inputs[key].to(device)

    # Rule 4: Run trainer.compute_loss (or get_batch_loss_metrics if DPO)
    # DPOTrainer overrides `compute_loss` which then calls `get_batch_loss_metrics`.
    # We can directly call `get_batch_loss_metrics` for a more direct test of the loss function.
    # `get_batch_loss_metrics` expects `model_output`, `ref_model_output`, `chosen_logps`, etc.
    # It's easier to call `compute_loss` as it handles the model forward passes.

    # The `compute_loss` method in DPOTrainer expects `model`, `inputs`, `return_outputs`.
    # The `inputs` dictionary should contain the necessary tensors for chosen and rejected sequences.
    # Let's prepare the inputs as `DPOTrainer`'s `training_step` would.
    # The `training_step` calls `self.get_batch_loss_metrics` with `(model, inputs)`.
    # `get_batch_loss_metrics` then performs the forward passes.

    # Let's call `get_batch_loss_metrics` directly, as it's the core loss calculation.
    # We need to simulate the outputs of the model and ref_model.
    # The `get_batch_loss_metrics` function expects:
    # model_output: CausalLMOutputWithPast (for chosen)
    # ref_model_output: CausalLMOutputWithPast (for chosen)
    # model_output_rejected: CausalLMOutputWithPast (for rejected)
    # ref_model_output_rejected: CausalLMOutputWithPast (for rejected)
    # chosen_labels: tensor
    # rejected_labels: tensor

    # Perform forward passes to get model outputs
    with torch.no_grad():
        model_output_chosen = trainer.model(
            input_ids=dummy_inputs["chosen_input_ids"],
            attention_mask=dummy_inputs["chosen_attention_mask"],
            labels=dummy_inputs["chosen_labels"] # Labels are used to compute loss internally by model
        )
        ref_model_output_chosen = trainer.ref_model(
            input_ids=dummy_inputs["chosen_input_ids"],
            attention_mask=dummy_inputs["chosen_attention_mask"],
            labels=dummy_inputs["chosen_labels"]
        )
        model_output_rejected = trainer.model(
            input_ids=dummy_inputs["rejected_input_ids"],
            attention_mask=dummy_inputs["rejected_attention_mask"],
            labels=dummy_inputs["rejected_labels"]
        )
        ref_model_output_rejected = trainer.ref_model(
            input_ids=dummy_inputs["rejected_input_ids"],
            attention_mask=dummy_inputs["rejected_attention_mask"],
            labels=dummy_inputs["rejected_labels"]
        )

    # Now call get_batch_loss_metrics
    # The method signature is `get_batch_loss_metrics(model, inputs)`
    # where `inputs` contains `chosen_input_ids`, `rejected_input_ids`, etc.
    # Let's reconstruct the `inputs` dictionary as expected by `get_batch_loss_metrics`.
    # It expects `input_ids`, `attention_mask`, `labels` for chosen and rejected.
    inputs_for_loss_metrics = {
        "chosen_input_ids": dummy_inputs["chosen_input_ids"],
        "chosen_attention_mask": dummy_inputs["chosen_attention_mask"],
        "chosen_labels": dummy_inputs["chosen_labels"],
        "rejected_input_ids": dummy_inputs["rejected_input_ids"],
        "rejected_attention_mask": dummy_inputs["rejected_attention_mask"],
        "rejected_labels": dummy_inputs["rejected_labels"],
        "prompt_input_ids": dummy_inputs["prompt_input_ids"], # Needed for masking
        "prompt_attention_mask": dummy_inputs["prompt_attention_mask"], # Needed for masking
    }

    # The `get_batch_loss_metrics` method in DPOTrainer is designed to be called by `training_step`.
    # It takes `model` and `inputs` as arguments.
    # It internally computes `chosen_logps`, `rejected_logps`, etc.
    # Let's call it as it would be called during training.
    # The `get_batch_loss_metrics` returns a tuple: (loss, metrics_dict).
    loss, metrics = trainer.get_batch_loss_metrics(trainer.model, inputs_for_loss_metrics)

    # Rule 5: Assert that the loss is a scalar and not NaN.
    assert isinstance(loss, torch.Tensor), f"Loss is not a torch.Tensor, but {type(loss)}"
    assert loss.ndim == 0, f"Loss is not a scalar, but has {loss.ndim} dimensions"
    assert not torch.isnan(loss), "Loss is NaN"
    assert torch.isfinite(loss), "Loss is infinite"

    # Rule 6: Print "PROBE_SUCCESS" if it passes.
    print("PROBE_SUCCESS")

if __name__ == "__main__":
    run_probe()