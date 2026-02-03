import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from peft import get_peft_model, LoraConfig
from trl import DPOTrainer
import math
from typing import Optional, List, Dict, Any

# === INJECTION POINT: ARCHITECTURE ===
# The paper describes a fine-tuning method (SPIN) rather than a novel architecture.
# The core idea is a specific loss function and iterative training.
# Therefore, we will implement the SPIN loss within a custom DPOTrainer.
# The base model itself remains a standard AutoModelForCausalLM.

class SPINTrainer(DPOTrainer):
    """
    SPIN (Self-Play fIne-tuNing) Trainer.

    This trainer implements the Self-Play Fine-Tuning method as described in the paper.
    It extends DPOTrainer to leverage its infrastructure for preference-style optimization,
    but replaces the DPO loss with the SPIN loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The paper uses a regularization parameter lambda (λ)
        # We'll assume it's passed as part of the model_args or training_args,
        # or set a default.
        self.lambda_param = kwargs.pop("lambda_param", 0.1) # Default value, adjust as needed

        # The reference model (p_theta_t in the paper) is crucial for SPIN.
        # In DPO, this is typically the SFT model. For SPIN, it's the model
        # from the previous iteration. We'll use the `ref_model` from DPOTrainer
        # to represent p_theta_t.
        if self.ref_model is None:
            raise ValueError("SPINTrainer requires a `ref_model` to represent p_theta_t.")

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        # The original DPO loss has beta. SPIN uses lambda.
        # We'll use lambda_param from our init.
        # beta: float,
        # label_smoothing: float = 0.0,
        # a_min: Optional[float] = None,
        # a_max: Optional[float] = None,
        # loss_type: str = "sigmoid",
    ) -> (torch.FloatTensor, torch.FloatTensor, torch.FloatTensor):
        """
        Calculate the SPIN loss as described in the paper (Equation 4.7).

        L_SPIN = E [ℓ(λ log(p_theta(y|x)/p_theta_t(y|x)) - λ log(p_theta(y'|x)/p_theta_t(y'|x)))]

        Here:
        - policy_chosen_logps: log p_theta(y|x)
        - policy_rejected_logps: log p_theta(y'|x)
        - reference_chosen_logps: log p_theta_t(y|x)
        - reference_rejected_logps: log p_theta_t(y'|x)
        - lambda_param: λ from the paper

        The paper suggests logistic loss ℓ(t) = log(1 + exp(-t)).
        """
        # Calculate the terms inside the logistic loss
        # term_y = λ log(p_theta(y|x) / p_theta_t(y|x))
        term_y = self.lambda_param * (policy_chosen_logps - reference_chosen_logps)

        # term_y_prime = λ log(p_theta(y'|x) / p_theta_t(y'|x))
        term_y_prime = self.lambda_param * (policy_rejected_logps - reference_rejected_logps)

        # The argument to the loss function ℓ is (term_y - term_y_prime)
        loss_argument = term_y - term_y_prime

        # Apply the logistic loss function: ℓ(t) = log(1 + exp(-t))
        # This is equivalent to F.logsigmoid(-t)
        losses = -nn.functional.logsigmoid(loss_argument)

        # The paper's objective is to minimize L_SPIN, so we return the mean of these losses.
        # The DPOTrainer expects a positive loss for minimization.
        # The paper's formulation is argmin_theta E[ℓ(f(x,y) - f(x,y'))]
        # where f(x,y) = λ log(p_theta(y|x)/p_theta_t(y|x))
        # So, the loss is indeed -logsigmoid(f(x,y) - f(x,y'))

        # For logging purposes, we can return the mean of the individual terms
        # The DPOTrainer expects chosen_rewards and rejected_rewards for logging.
        # In SPIN, these are not direct rewards but components of the loss.
        # We can approximate them for logging or set them to 0 if not directly applicable.
        # For now, let's return the loss and dummy values for chosen/rejected rewards.
        chosen_rewards = term_y # This is not a reward, but a component of the loss
        rejected_rewards = term_y_prime # This is not a reward, but a component of the loss

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def get_model(base_model_name: str, num_labels: int = None):
    """
    Loads a causal language model suitable for SFT and then for SPIN.
    No custom architecture injection is needed as SPIN modifies the training objective,
    not the model's internal layers.
    """
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # The paper mentions fine-tuning a Mistral-7B based model.
    # Often, LoRA is used for efficient fine-tuning.
    # We can optionally add LoRA adapters here.
    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=16,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    return model

