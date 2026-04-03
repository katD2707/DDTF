"""
Exponential Moving Average (EMA) for model weights.

Maintains a shadow copy of model weights updated as:
    shadow = decay * shadow + (1 - decay) * param

The EMA weights typically produce better/smoother samples.
"""

import torch
import torch.nn as nn


class EMA:
    """
    EMA tracker with apply/restore, state_dict support.

    Args:
        model: Model to track.
        decay: EMA decay rate (higher = slower updates, typically 0.99-0.9999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1 - self.decay) * param.data
                )

    def apply_shadow(self) -> None:
        """Apply EMA weights to model (backs up current weights)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self) -> None:
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]
