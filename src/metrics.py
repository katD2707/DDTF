"""Metrics for time series forecasting evaluation."""

import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor


def mse(pred: Tensor, target: Tensor) -> float:
    return F.mse_loss(pred, target).item()


def mae(pred: Tensor, target: Tensor) -> float:
    return F.l1_loss(pred, target).item()


def rmse(pred: Tensor, target: Tensor) -> float:
    return math.sqrt(mse(pred, target))


def crps(samples: Tensor, target: Tensor) -> float:
    """Compute Continuous Ranked Probability Score (CRPS) using PyTorch.

    Args:
        samples: [B, N, L, C] - Ensemble predictions
        target:  [B, L, C]    - Ground truth targets

    Returns:
        float: Mean CRPS value over all batch elements, timesteps, and channels.
    """
    samples_double = samples.to(dtype=torch.float64)
    target_double = target.to(dtype=torch.float64)
    num_samples = samples_double.shape[1]

    # First empirical CRPS term: E|X - y|
    term1 = torch.abs(samples_double - target_double.unsqueeze(1)).mean(dim=1)

    # Efficiently compute 0.5 * E|X - X'| from sorted ensemble samples
    weights = (
        2.0 * torch.arange(1, num_samples + 1, device=samples.device, dtype=torch.float64)
        - num_samples
        - 1.0
    ).view(1, num_samples, 1, 1)

    sorted_samples, _ = torch.sort(samples_double, dim=1)
    term2 = torch.sum(sorted_samples * weights, dim=1) / (num_samples ** 2)

    crps_values = term1 - term2
    result = float(crps_values.mean().item())

    if not math.isfinite(result):
        raise ValueError("CRPS result is non-finite")

    return result

def compute_metrics(
    pred: Tensor,
    target: Tensor,
    samples: Tensor = None,
) -> Dict[str, float]:
    """Compute forecasting metrics.

    Args:
        pred:    [B, L, C]           point forecast (e.g. ensemble mean)
        target:  [B, L, C]           ground truth
        samples: [B, N, L, C]        ensemble members for CRPS
    """
    return {
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
        "crps": crps(samples, target) if samples is not None else -1
    }


class MetricTracker:
    """Track and aggregate metrics over training batches."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], count: int = 1) -> None:
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
                self.counts[name] = 0
            self.metrics[name].append(value * count)
            self.counts[name] += count

    def get_average(self, name: str) -> float:
        if name not in self.metrics or self.counts[name] == 0:
            return 0.0
        return sum(self.metrics[name]) / self.counts[name]

    def get_all_averages(self) -> Dict[str, float]:
        return {name: self.get_average(name) for name in self.metrics}

    def reset(self) -> None:
        self.metrics = {}
        self.counts = {}

    def __str__(self) -> str:
        avgs = self.get_all_averages()
        return " | ".join(f"{k}: {v:.4f}" for k, v in avgs.items())
