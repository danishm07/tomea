import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from typing import Optional, List, Dict, Union, Tuple
from trl import DPOTrainer, ORPOTrainer # Assuming ORPOTrainer is a custom trainer or DPOTrainer is adapted
from peft import LoraConfig, get_peft_model

# === INJECTION POINT: ARCHITECTURE ===
# ORPO is an optimization algorithm, not a change to the model architecture itself.
# The core model remains a standard CausalLM.
# We will define the ORPO loss function and integrate it via a custom Trainer.

class ORPOLoss:
    """
    Odds Ratio Preference Optimization (ORPO) loss function.
    This loss combines a standard Causal Language Modeling (CLM) loss
    with an odds ratio preference loss.
    """
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        """
        Initializes the ORPO loss.

        Args:
            beta (float): Weight for the odds ratio preference loss.
            label_smoothing (float): Label smoothing factor for the CLM loss.
        """
        self.beta = beta
        self.label_smoothing = label_smoothing

    def __call__(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_chosen_logits: torch.FloatTensor,
        policy_rejected_logits: torch.FloatTensor,
        chosen_labels: torch.LongTensor,
        rejected_labels: torch.LongTensor,
        average_log_likelihood: bool = True,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Calculates the ORPO loss.

        Args:
            policy_chosen_logps (torch.FloatTensor): Log probabilities of the chosen responses
                                                    under the policy model. Shape: (batch_size,)
            policy_rejected_logps (torch.FloatTensor): Log probabilities of the rejected responses
                                                       under the policy model. Shape: (batch_size,)
            policy_chosen_logits (torch.FloatTensor): Logits of the chosen responses
                                                      under the policy model. Shape: (batch_size, sequence_length, vocab_size)
            policy_rejected_logits (torch.FloatTensor): Logits of the rejected responses
                                                        under the policy model. Shape: (batch_size, sequence_length, vocab_size)
            chosen_labels (torch.LongTensor): Token IDs of the chosen responses. Shape: (batch_size, sequence_length)
            rejected_labels (torch.LongTensor): Token IDs of the rejected responses. Shape: (batch_size, sequence_length)
            average_log_likelihood (bool): Whether to average log likelihoods over sequence length.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                - loss (torch.FloatTensor): The total ORPO loss.
                - sft_loss (torch.FloatTensor): The Supervised Fine-Tuning (SFT) loss.
                - or_loss (torch.FloatTensor): The Odds Ratio (OR) preference loss.
        """
        # LSFT: Supervised Fine-Tuning Loss (Negative Log-Likelihood)
        # The paper defines LSFT as conventional causal language modeling NLL loss.
        # We need to calculate NLL for the chosen responses.
        # For NLL, we need to mask out padding tokens and average over non-padding tokens.

        # Shift logits and labels for causal language modeling
        shift_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
        shift_chosen_labels = chosen_labels[..., 1:].contiguous()

        # Calculate NLL loss
        sft_loss = F.cross_entropy(
            shift_chosen_logits.view(-1, shift_chosen_logits.size(-1)),
            shift_chosen_labels.view(-1),
            ignore_index=-100, # Assuming -100 is the ignore index for padding
            label_smoothing=self.label_smoothing,
        )

        # LOR: Odds Ratio Preference Loss
        # LOR = -log sigma(log odds_theta(yw|x) / odds_theta(yl|x))
        # odds_theta(y|x) = P_theta(y|x) / (1 - P_theta(y|x))
        # log odds_theta(y|x) = log P_theta(y|x) - log (1 - P_theta(y|x))

        # Ensure log probabilities are averaged over the sequence length if specified
        if average_log_likelihood:
            # The paper defines log P_theta(y|x) as 1/m * sum(log P_theta(yt|x, y<t))
            # policy_chosen_logps and policy_rejected_logps are already averaged per sequence by TRL's DPO trainer
            # if `average_log_likelihood` is True in `compute_log_probs`
            pass
        else:
            # If not averaged, we need to sum them up for the full sequence probability
            # This part depends on how policy_chosen_logps are computed in the trainer.
            # Assuming they are already sequence-level log-probabilities.
            pass

        # Calculate log odds for chosen and rejected responses
        # To avoid log(0) or log(negative), we need to clamp probabilities.
        # P_theta(y|x) = exp(log P_theta(y|x))
        # Clamp probabilities to a small epsilon to avoid numerical issues
        epsilon = 1e-6
        chosen_probs = torch.exp(policy_chosen_logps).clamp(min=epsilon, max=1.0 - epsilon)
        rejected_probs = torch.exp(policy_rejected_logps).clamp(min=epsilon, max=1.0 - epsilon)

        log_odds_chosen = torch.log(chosen_probs) - torch.log(1.0 - chosen_probs)
        log_odds_rejected = torch.log(rejected_probs) - torch.log(1.0 - rejected_probs)

        # Calculate log odds ratio
        log_odds_ratio = log_odds_chosen - log_odds_rejected

        # LOR = -log sigma(log_odds_ratio)
        or_loss = -F.logsigmoid(log_odds_ratio)

        # Total ORPO Loss
        loss = sft_loss + self.beta * or_loss.mean() # Mean over batch for OR loss

        return loss, sft_loss, or_loss.mean()


class ORPOTrainer(DPOTrainer):
    """
    Custom TRL Trainer for ORPO.
    Inherits from DPOTrainer and overrides the loss calculation.
    """
    def __init__(self, *args, beta: float = 0.1, label_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.orpo_loss_fn = ORPOLoss(beta=beta, label_smoothing=label_smoothing)
        self.beta = beta # Store beta for logging or access

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        policy_chosen_logits: torch.FloatTensor,
        policy_rejected_logits: torch.FloatTensor,
        chosen_labels: torch.LongTensor,
        rejected_labels: torch.LongTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Calculate the ORPO loss. This method replaces the standard DPO loss calculation.
        The `reference_chosen_logps` and `reference_rejected_logps` are not used in ORPO,
        but are kept in the signature for compatibility with DPOTrainer's `dpo_loss` method.
        """
        total_loss, sft_loss, or_loss = self.orpo_loss_fn(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_rejected_logits,
            chosen_labels=chosen_labels,
            rejected_labels=rejected_labels,
            average_log_likelihood=self.args.average_log_likelihood,
        )

        # For logging purposes, we can return the components
        # DPOTrainer expects a tuple of (loss, chosen_rewards, rejected_rewards)
        # We can adapt by returning dummy reward values or actual log_odds_ratio for insight.
        # Let's return the log_odds_ratio as a proxy for "reward" difference for logging.
        epsilon = 1e-6
        chosen_probs = torch.exp(policy_chosen_logps).clamp(min=epsilon, max=1.0 - epsilon)
        rejected_probs = torch.exp(policy_rejected_logps).clamp(min=epsilon, max=1.0 - epsilon)
        log_odds_chosen = torch.log(chosen_probs) - torch.log(1.0 - chosen_probs)
        log_odds_rejected = torch.log(rejected_probs) - torch.log(1.0 - rejected_probs)
        log_odds_ratio = log_odds_chosen - log_odds_rejected

        return total_loss, log_odds_chosen.mean(), log_odds_rejected.mean() # Returning means for logging

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the ORPO loss for a batch of inputs.
        This method is called by the Trainer's training loop.
        It largely reuses the logic from DPOTrainer's compute_loss but calls our custom dpo_loss.
        """
        # Extract necessary inputs for ORPO
        # The input format is expected to be similar to DPO, with chosen/rejected pairs.
        # The `inputs` dictionary should contain:
        # 'input_ids_chosen', 'attention_mask_chosen', 'labels_chosen'
        # 'input_ids_rejected', 'attention_mask_rejected', 'labels_rejected'
        # 'prompt_input_ids', 'prompt_attention_mask' (if using prompt-based generation)

        # Ensure that the model is a CausalLM
        if not isinstance(model, AutoModelForCausalLM):
            raise ValueError("ORPOTrainer expects a CausalLM model.")

        # Forward pass for chosen responses
        policy_chosen_outputs = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
            output_hidden_states=True, # Needed for some models, safe to include
        )
        policy_chosen_logits = policy_chosen_outputs.logits

        # Forward pass for rejected responses
        policy_rejected_outputs = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
            output_hidden_states=True,
        )
        policy_rejected_logits = policy_rejected_outputs.logits

        # Compute log probabilities for chosen and rejected responses
        # This part is crucial and needs to align with how the paper defines P_theta(y|x)
        # The paper uses 1/m * sum(log P_theta(yt|x, y<t))
        # TRL's `compute_log_probs` function does this, we should reuse it.
        policy_chosen_logps = self.get_batch_log_probs(
            policy_chosen_logits,
            inputs["labels_chosen"],
            average_log_likelihood=self.args.average_log_likelihood,
            is_encoder_decoder=model.config.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        policy_rejected_logps = self.get_batch_log_probs(
            policy_rejected_logits,
            inputs["labels_rejected"],
            average_log_likelihood=self.args.average_log_likelihood,
            is_encoder_decoder=model.config.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        # Call the ORPO loss function
        total_loss, chosen_log_odds, rejected_log_odds = self.dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=None,  # Not used in ORPO
            reference_rejected_logps=None, # Not used in ORPO
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_rejected_logits,
            chosen_labels=inputs["labels_chosen"],
            rejected_labels=inputs["labels_rejected"],
        )

        if return_outputs:
            return total_loss, {
                "loss": total_loss,
                "chosen_log_odds": chosen_log_odds,
                "rejected_log_odds": rejected_log_odds,
                "log_odds_ratio": chosen_log_odds - rejected_log_odds,
            }
        return total_loss


def get_model(base_model_name: str, lora_config: Optional[LoraConfig] = None):
    """
    Loads a causal language model and optionally applies LoRA.

    Args:
        base_model_name (str): The name or path of the base pre-trained model.
        lora_config (Optional[LoraConfig]): LoRA configuration for PEFT.

    Returns:
        PreTrainedModel: The loaded model, potentially with LoRA adapters.
    """
    # ORPO does not modify the base architecture, it's a training objective.
    # We load a standard CausalLM.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if supported
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager", # Use Flash Attention if available
    )

    if lora_config:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model

