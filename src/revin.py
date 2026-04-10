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

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("mean", None)
        self.register_buffer("stdev", None)

    def forward(self, x: Tensor, mode: str, reuse_stats: bool = False) -> Tensor:
        if mode == "norm":
            return self._normalize(x, reuse_stats)
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


class Projector(nn.Module):
    """
    MLP that learns de-stationary factors (tau or delta) from raw input
    and its normalization statistics.

    Args:
        enc_in:      number of input features (channels)
        seq_len:     input sequence length
        hidden_dim:  MLP hidden dimension
        output_dim:  output dimension (n_heads for tau, seq_len*n_heads for delta)
    """

    def __init__(self, enc_in: int, seq_len: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.series_conv = nn.Conv1d(
            in_channels=seq_len, out_channels=1,
            kernel_size=3, padding=1, padding_mode="circular", bias=False,
        )
        self.backbone = nn.Sequential(
            nn.Linear(2 * enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: Tensor, stats: Tensor) -> Tensor:
        """
        Args:
            x:     [B, S, C]  raw (un-normalized) series
            stats: [B, 1, C]  normalization statistic (mean or std)
        Returns:
            y:     [B, output_dim]
        """
        B = x.shape[0]
        h = self.series_conv(x)               # [B, 1, C]
        h = torch.cat([h, stats], dim=1)      # [B, 2, C]
        h = h.reshape(B, -1)                  # [B, 2*C]
        return self.backbone(h)