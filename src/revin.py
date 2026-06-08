"""
Reversible Instance Normalization (RevIN) for Time Series.

Normalizes time series data on a per-sample, per-channel basis,
which helps handle distribution shifts common in time series forecasting.

Reference:
    Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting
    against Distribution Shift" (ICLR 2022)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List

class Projector(nn.Module):
    """
    MLP projector that learns tau (scale) and delta (shift) factors
    from raw (unnormalized) data and its statistics.
    
    Adapted from Nonstationary_Transformers: takes raw input series x and
    its channel-wise statistics (mean/std) to produce learning factors.
    """
    def __init__(self, seq_len: int, output_dim: int, hidden_dims: List[int] = [128, 128], hidden_layers: int = 2):
        super().__init__()
        # Input: [mean (C) | std (C)] concatenated, total 2*C dimensions
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        layers = [nn.Linear(2, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x_raw: Tensor, stats: Tensor) -> Tensor:
        """
        Args:
            x_raw: [B, L, C] raw (unnormalized) data
            stats: [B, 1, C] either mean or std from the raw data
        Returns:
            [B, output_dim, C] learned factors
        """
        # Compute global mean from raw data: [B, L, C] -> [B, C]
        x = self.series_conv(x_raw)  # [B, 1, C]
        # Concatenate mean of raw data with provided stats (std or mean)
        feat = torch.cat([x, stats], dim=-1)  # [B, 2, C]
        return self.backbone(feat.transpose(1, 2)).transpose(1, 2)  # [B, output_dim, C]


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.

    Normalizes each sample independently along the temporal dimension,
    preserving per-channel statistics for denormalization.

    Args:
        num_features: Number of features/channels in the time series
        eps: Small constant for numerical stability
        affine: If True, learns scale and bias parameters per channel

    Shape:
        Input / Output: (B, L, C)
    """

    def __init__(self, num_features: int, seq_len: int, eps: float = 1e-5, affine: bool = False,
                 use_de_stationary: bool = False, p_hidden_dims: List[int] = [128, 128], p_hidden_layers: int = 2):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.use_de_stationary = use_de_stationary

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        if self.use_de_stationary:
            self.tau_learner = Projector(
                    seq_len=seq_len, output_dim=1,
                    hidden_dims=p_hidden_dims,
                    hidden_layers=p_hidden_layers
                )
            self.delta_learner = Projector(
                seq_len=seq_len, output_dim=seq_len,
                hidden_dims=p_hidden_dims,
                hidden_layers=p_hidden_layers,
            )

        self.register_buffer("mean", None)
        self.register_buffer("stdev", None)

    def forward(self, x: Tensor, mode: str, reuse_stats: bool = False) -> Tensor:
        if mode == "norm":
            x_norm = self._normalize(x, reuse_stats)
            if self.use_de_stationary:
                return x_norm, self.tau_learner(x, self.stdev).exp(), self.delta_learner(x, self.mean)
            return x_norm, None, None
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'norm' or 'denorm'.")

    def _compute_statistics(self, x: Tensor) -> None:
        dims_to_reduce = tuple(range(1, x.ndim - 1))  # (1,) for 3D
        self.mean = x.mean(dim=dims_to_reduce, keepdim=True).detach()
        var = x.var(dim=dims_to_reduce, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x: Tensor, reuse_stats: bool) -> Tensor:
        if not reuse_stats:
            self._compute_statistics(x)
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.mean is None or self.stdev is None:
            raise RuntimeError("Statistics not computed. Call forward with mode='norm' first.")
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x