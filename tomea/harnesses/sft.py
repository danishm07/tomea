from .base import BaseHarness

class SFTHarness(BaseHarness):

    @property
    def mode(self) -> str:
        return "sft"

    def get_system_prompt(self) -> str:
        return """You are an expert Research Engineer specializing in Transformers and SFT.
Your goal is to implement a novel architecture or adapter based on the user's paper.
You MUST use the provided template and only fill in the 'TODO' sections.
You must ensure strict adherence to PyTorch shape compatibility."""

    def get_template(self) -> str:
        return """
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig, PreTrainedModel
import math

# === INJECTION POINT: ARCHITECTURE ===
# TODO: Implement the custom class here (e.g., custom SelfAttention)
# Ensure you handle 'kwargs' in forward to avoid crashes from Trainer injection.

def get_model(base_model_name: str, num_labels: int):
    # TODO: Load the model and inject the custom architecture
    # return model
    pass
"""

    def validate_logs(self, logs: str) -> bool:
        if "CUDA out of memory" in logs:
            return False
        if "RuntimeError" in logs:
            return False
        if "Traceback" in logs:
            return False
        if "nan" in logs.lower() and "loss" in logs.lower():
            return False
        return "loss=" in logs or "Accuracy:" in logs