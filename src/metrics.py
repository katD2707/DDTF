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


def compute_metrics(pred: Tensor, target: Tensor) -> Dict[str, float]:
    return {
        "mse": mse(pred, target),
        "mae": mae(pred, target),
        "rmse": rmse(pred, target),
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
