import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig, PreTrainedModel
import math
from typing import Optional, List, Dict, Any

# === INJECTION POINT: ARCHITECTURE ===
class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) module for linear layers.
    Implements the reparametrization h = W0x + BAx as described in the paper.
    """
    def __init__(
        self,
        linear_layer: nn.Module,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.merged = False
        self.merge_weights = merge_weights

        # Original pre-trained weight matrix W0
        self.W0 = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        if r > 0:
            # LoRA A matrix (A in R^(r x k))
            self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
            # LoRA B matrix (B in R^(d x r))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.r))
            self.scaling = self.lora_alpha / self.r

            # Initialize A with Gaussian and B with zeros
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None
            self.scaling = 0.0

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            # h = W0x + BAx
            # W0x
            original_output = self.W0(x)
            # BAx
            lora_output = self.lora_B @ self.lora_A @ x.transpose(-1, -2)
            lora_output = lora_output.transpose(-1, -2) * self.scaling
            return original_output + self.lora_dropout(lora_output)
        else:
            return self.W0(x)

    def merge(self):
        if self.r > 0 and not self.merged:
            # Explicitly compute and store W = W0 + BA
            # This is for deployment, introducing no inference latency.
            # W0 is assumed to be a linear layer with .weight attribute
            if isinstance(self.W0, nn.Linear):
                delta_W = (self.lora_B @ self.lora_A) * self.scaling
                self.W0.weight.data += delta_W
                self.merged = True
            else:
                raise TypeError("W0 must be an instance of nn.Linear to merge weights.")

    def unmerge(self):
        if self.r > 0 and self.merged:
            if isinstance(self.W0, nn.Linear):
                delta_W = (self.lora_B @ self.lora_A) * self.scaling
                self.W0.weight.data -= delta_W
                self.merged = False
            else:
                raise TypeError("W0 must be an instance of nn.Linear to unmerge weights.")


def get_model(base_model_name: str, num_labels: int, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.0):
    """
    Loads a pre-trained model and injects LoRA modules into its attention layers.
    """
    config = AutoConfig.from_pretrained(base_model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, config=config)

    # Freeze all parameters of the base model
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to Wq and Wv in all self-attention modules
    # This assumes a standard Transformer architecture where attention layers are accessible.
    # The exact path might vary depending on the specific model (e.g., RoBERTa, GPT-2).
    # This example targets common Transformer structures.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Heuristic to identify query and value projection matrices in attention
            # This needs to be adapted based on the specific model's architecture.
            # For many HuggingFace models, these are often named 'q_proj', 'v_proj', 'query', 'value'.
            if any(attn_name in name for attn_name in ["query", "value", "q_proj", "v_proj"]):
                parent_module = model
                name_parts = name.split('.')
                for part in name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                
                # Replace the original linear layer with the LoRALinear module
                setattr(parent_module, name_parts[-1], LoRALinear(
                    linear_layer=module,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=False # We handle merging manually if needed
                ))
                # Ensure the newly injected LoRA parameters are trainable
                for param in getattr(parent_module, name_parts[-1]).parameters():
                    param.requires_grad = True
    
    return model

