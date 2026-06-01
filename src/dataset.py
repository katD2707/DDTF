"""Time series dataset with sliding windows and standard benchmark splits."""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from src.perturbation import apply_perturbations


# ═════════════════════════════════════════════════════════════════════════════
# Standard benchmark split definitions (fixed indices, no ratios)
# ═════════════════════════════════════════════════════════════════════════════

# ETT datasets: 12/4/4 months  (hourly → *24, 15-min → *24*4)
_H = 30 * 24          # ~hours per month
_M = 30 * 24 * 4      # ~15-min intervals per month

DATASET_SPLITS: Dict[str, Dict[str, Tuple[int, int]]] = {
    # ── ETT (Electricity Transformer Temperature) ────────────────────────
    "ETTh1": {"train": (0, 12 * _H), "val": (12 * _H, 16 * _H), "test": (16 * _H, 20 * _H)},
    "ETTh2": {"train": (0, 12 * _H), "val": (12 * _H, 16 * _H), "test": (16 * _H, 20 * _H)},
    "ETTm1": {"train": (0, 12 * _M), "val": (12 * _M, 16 * _M), "test": (16 * _M, 20 * _M)},
    "ETTm2": {"train": (0, 12 * _M), "val": (12 * _M, 16 * _M), "test": (16 * _M, 20 * _M)},

    # ── Other standard LTSF benchmarks ───────────────────────────────────
    # Original splits from Autoformer / FEDformer / PatchTST papers
    # (num_train = int(N*0.7), num_test = int(N*0.2), num_val = N-train-test)
    # Weather  (N=52696)
    "weather":      {"train": (0, 36887), "val": (36887, 42157), "test": (42157, 52696)},
    # Electricity / ECL  (N=26304)
    "electricity":  {"train": (0, 18412), "val": (18412, 21044), "test": (21044, 26304)},
    "ECL":          {"train": (0, 18412), "val": (18412, 21044), "test": (21044, 26304)},
    # Traffic  (N=17544)
    "traffic":      {"train": (0, 12280), "val": (12280, 14036), "test": (14036, 17544)},
    # Exchange  (N=7588)
    "exchange":     {"train": (0, 5311),  "val": (5311, 6071),   "test": (6071, 7588)},
    "exchange_rate": {"train": (0, 5311), "val": (5311, 6071),   "test": (6071, 7588)},
    # ILI  (N=966)
    "national_illness": {"train": (0, 676), "val": (676, 773), "test": (773, 966)},
    "ILI":              {"train": (0, 676), "val": (676, 773), "test": (773, 966)},
}


# ═════════════════════════════════════════════════════════════════════════════
# Base dataset
# ═════════════════════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for a single time series file.

    Args:
        data: Array of shape [T, C].
        context_length: Historical window length.
        prediction_length: Forecast horizon length.
        stride: Step between consecutive windows.
        scale: Whether to standardise.
        scaler: Pre-fitted scaler (for val/test).
    """

    def __init__(
        self,
        data: np.ndarray,
        context_length: int = 96,
        prediction_length: int = 96,
        stride: int = 1,
        scale: bool = True,
        scaler: Optional[StandardScaler] = None,
    ):
        super().__init__()
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.num_channels = data.shape[1]

        if scale:
            if scaler is None:
                self.scaler = StandardScaler()
                self.data = self.scaler.fit_transform(data).astype(np.float32)
            else:
                self.scaler = scaler
                self.data = self.scaler.transform(data).astype(np.float32)
        else:
            self.scaler = None
            self.data = data.astype(np.float32)

        total = context_length + prediction_length
        self.num_samples = max(0, (len(self.data) - total) // stride + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start = idx * self.stride
        ctx_end = start + self.context_length
        tgt_end = ctx_end + self.prediction_length
        context = torch.from_numpy(self.data[start:ctx_end])
        target = torch.from_numpy(self.data[ctx_end:tgt_end])
        return context, target


# ═════════════════════════════════════════════════════════════════════════════
# Public loader
# ═════════════════════════════════════════════════════════════════════════════

def load_csv_dataset(
    csv_path: str,
    context_length: int = 96,
    prediction_length: int = 1056,
    eval_prediction_length: int = 96,
    stride: int = 1,
    target_col: Optional[str] = None,
    perturbation_cfg=None,
):
    """
    Load a CSV file and return train / val / test datasets.

    The function detects the dataset name from the filename (stem, case-
    insensitive) and looks up its fixed split boundaries in
    ``DATASET_SPLITS``.  An error is raised for unregistered datasets —
    add a new entry to ``DATASET_SPLITS`` to support them.

    Args:
        csv_path: Path to a CSV file.  An optional ``date`` column is
            dropped automatically.
        context_length: Historical context window length.
        prediction_length: Forecast horizon length.
        stride: Step between consecutive sliding windows.
        target_col: If given, only this column is kept (univariate).
        perturbation_cfg: Optional ``PerturbationConfig`` instance.  When
            ``perturbation_cfg.enabled`` is ``True``, the specified
            perturbation types (Algorithms 1–4 from SEER Appendix B) are
            applied to the raw *training* split only, before
            ``StandardScaler`` fitting.  Validation and test splits are
            never perturbed.
    """
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    if target_col is not None:
        df = df[[target_col]]

    data = df.values.astype(np.float32)
    n = len(data)

    # ── Resolve split boundaries ──────────────────────────────────────────
    dataset_name = Path(csv_path).stem            # e.g. "ETTh1"
    splits = DATASET_SPLITS.get(dataset_name)
    if splits is None:
        available = ", ".join(sorted(DATASET_SPLITS.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Add an entry to DATASET_SPLITS in src/dataset.py.\n"
            f"Available: {available}"
        )
    # Clamp indices to actual data length
    splits = {k: (min(s, n), min(e, n)) for k, (s, e) in splits.items()}

    train_data = data[splits["train"][0] : splits["train"][1]]
    val_data   = data[splits["val"][0]   : splits["val"][1]]
    test_data  = data[splits["test"][0]  : splits["test"][1]]

    # ── Apply training-only data perturbation (SEER Appendix B) ──────────
    if (
        perturbation_cfg is not None
        and perturbation_cfg.enabled
        and perturbation_cfg.types
    ):
        train_data = apply_perturbations(
            train_data,
            types=perturbation_cfg.types,
            noise_ratio=perturbation_cfg.noise_ratio,
            noise_scale=perturbation_cfg.noise_scale,
            cont_anomaly_ratio=perturbation_cfg.cont_anomaly_ratio,
            cont_segment_len=perturbation_cfg.cont_segment_len,
            outlier_ratio=perturbation_cfg.outlier_ratio,
            anomaly_scale=perturbation_cfg.anomaly_scale,
            missing_ratio=perturbation_cfg.missing_ratio,
            missing_segment_len=perturbation_cfg.missing_segment_len,
            num_shift_segments=perturbation_cfg.num_shift_segments,
            shift_scale=perturbation_cfg.shift_scale,
            seed=perturbation_cfg.seed,
        )

    train_ds = TimeSeriesDataset(train_data, context_length, prediction_length, stride)
    val_ds   = TimeSeriesDataset(val_data,   context_length, eval_prediction_length, stride, scaler=train_ds.scaler)
    test_ds  = TimeSeriesDataset(test_data,  context_length, eval_prediction_length, stride, scaler=train_ds.scaler)
    return train_ds, val_ds, test_ds
