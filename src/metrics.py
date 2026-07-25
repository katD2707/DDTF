"""Metrics for time series forecasting evaluation."""

import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import CRPS.CRPS as pscore
from multiprocessing import Pool, cpu_count


def mse(pred: Tensor, target: Tensor) -> float:
    return F.mse_loss(pred, target).item()


def mae(pred: Tensor, target: Tensor) -> float:
    return F.l1_loss(pred, target).item()


def rmse(pred: Tensor, target: Tensor) -> float:
    return math.sqrt(mse(pred, target))


def _crps_worker(args):
    """Compute CRPS for a single batch element (all timesteps & channels)."""
    samples_b, target_b = args  # samples_b: [N, L, C], target_b: [L, C]
    L, C = target_b.shape
    total = 0.0
    for l in range(L):
        for c in range(C):
            result = pscore(samples_b[:, l, c], target_b[l, c]).compute()
            total += result[0]
    return total, L * C


def crps(samples: Tensor, target: Tensor) -> float:
    samples_np = samples.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    B = samples_np.shape[0]
    worker_args = [(samples_np[b], target_np[b]) for b in range(B)]

    n_workers = min(B, cpu_count())
    with Pool(n_workers) as pool:
        results = pool.map(_crps_worker, worker_args)

    total = sum(r[0] for r in results)
    count = sum(r[1] for r in results)
    return total / count


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
