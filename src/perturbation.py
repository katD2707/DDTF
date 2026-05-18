"""Data perturbation functions from SEER (Qiu et al., 2026), Appendix B.

Implements the four synthetic perturbation types used in SEER's robustness
experiments:

  1. White Noise         – superimpose Gaussian noise on random time points
  2. Anomalies           – inject contiguous segments and point outliers
  3. Missing Values      – zero-out contiguous segments
  4. Distribution Shift  – add per-segment constant offsets

All functions operate on raw numpy arrays of shape [T, N] (time × channels)
and return a modified copy, leaving the input unchanged.

These are applied to the training split only, before StandardScaler fitting,
so that validation/test splits are never perturbed.

Reference:
    SEER: Transformer-based Robust Time Series Forecasting via Automated
    Patch Enhancement and Replacement. Qiu et al., arXiv:2602.00589, 2026.
    Appendix B, Algorithms 1–4.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm 1 – White Noise  (Appendix B.1)
# ═════════════════════════════════════════════════════════════════════════════

def white_noise(
    X: np.ndarray,
    noise_ratio: float = 0.05,
    noise_scale: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Superimpose Gaussian noise onto a random subset of time points.

    For each channel c:
        * σ_c = std(X[:, c])
        * Sample floor(T * noise_ratio) indices I_noise without replacement
        * X_noisy[I_noise, c] += N(0, 1) * noise_scale * σ_c

    Args:
        X: Raw time series of shape [T, N].
        noise_ratio: Fraction of time points to perturb (r_noise).
        noise_scale: Noise magnitude multiplier relative to channel std (α_noise).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Perturbed copy of X with shape [T, N].
    """
    if rng is None:
        rng = np.random.default_rng()

    T, N = X.shape
    X_noisy = X.copy()
    n_noise = int(np.floor(T * noise_ratio))
    if n_noise == 0:
        return X_noisy

    for c in range(N):
        sigma_c = X[:, c].std()
        indices = rng.choice(T, size=n_noise, replace=False)
        X_noisy[indices, c] += rng.standard_normal(n_noise) * noise_scale * sigma_c

    return X_noisy


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm 2 – Anomalies  (Appendix B.2)
# ═════════════════════════════════════════════════════════════════════════════

def anomalies(
    X: np.ndarray,
    cont_anomaly_ratio: float = 0.05,
    cont_segment_len: int = 12,
    outlier_ratio: float = 0.005,
    anomaly_scale: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Inject two forms of anomalies into the time series.

    1. Contiguous segments – sustained system faults:
       * n_seg = floor((T * cont_anomaly_ratio) / cont_segment_len) segments
       * Each segment gets a random-direction additive spike of
         anomaly_scale * σ_c (same direction for the whole segment)

    2. Point outliers – transient shocks:
       * n_out = floor(T * outlier_ratio) points
       * Each point gets anomaly_scale * σ_c * N(0, 1)

    Args:
        X: Raw time series of shape [T, N].
        cont_anomaly_ratio: Fraction of time covered by contiguous segments (r_cont).
        cont_segment_len: Length of each contiguous anomaly segment (L_cont).
        outlier_ratio: Fraction of time points that are point outliers (r_out).
        anomaly_scale: Magnitude multiplier relative to channel std (α_anom).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Perturbed copy of X with shape [T, N].
    """
    if rng is None:
        rng = np.random.default_rng()

    T, N = X.shape
    X_noisy = X.copy()

    # ── Contiguous anomalous segments ────────────────────────────────────────
    n_seg = int(np.floor((T * cont_anomaly_ratio) / cont_segment_len))
    if n_seg > 0:
        max_start = T - cont_segment_len
        if max_start >= n_seg:
            # Sample non-overlapping start positions
            starts = _sample_nonoverlapping_starts(
                n_seg, cont_segment_len, T, rng
            )
            for c in range(N):
                sigma_c = X[:, c].std()
                for s in starts:
                    direction = rng.choice([-1.0, 1.0])
                    X_noisy[s : s + cont_segment_len, c] += (
                        direction * anomaly_scale * sigma_c
                    )

    # ── Point outliers ────────────────────────────────────────────────────────
    n_out = int(np.floor(T * outlier_ratio))
    if n_out > 0:
        for c in range(N):
            sigma_c = X[:, c].std()
            indices = rng.choice(T, size=n_out, replace=False)
            X_noisy[indices, c] += (
                rng.standard_normal(n_out) * anomaly_scale * sigma_c
            )

    return X_noisy


def _sample_nonoverlapping_starts(
    n: int,
    seg_len: int,
    T: int,
    rng: np.random.Generator,
) -> List[int]:
    """Sample n non-overlapping segment start positions in [0, T - seg_len].

    Falls back to as many as possible without overlap if n is too large.
    """
    candidates = list(range(0, T - seg_len + 1))
    rng.shuffle(candidates)

    chosen: List[int] = []
    occupied = np.zeros(T, dtype=bool)
    for s in candidates:
        if not occupied[s : s + seg_len].any():
            chosen.append(s)
            occupied[s : s + seg_len] = True
            if len(chosen) == n:
                break
    return chosen


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm 3 – Missing Values  (Appendix B.3)
# ═════════════════════════════════════════════════════════════════════════════

def missing_values(
    X: np.ndarray,
    missing_ratio: float = 0.05,
    missing_segment_len: int = 12,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Zero-out contiguous segments to simulate missing data.

    For each channel c:
        * n_seg = floor((T * missing_ratio) / missing_segment_len)
        * Sample n_seg start positions from {0, ..., T − L_miss}
        * X_noisy[start : start + L_miss, c] = 0

    Args:
        X: Raw time series of shape [T, N].
        missing_ratio: Fraction of time to zero out (r_miss).
        missing_segment_len: Length of each missing segment (L_miss).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Perturbed copy of X with shape [T, N].
    """
    if rng is None:
        rng = np.random.default_rng()

    T, N = X.shape
    X_noisy = X.copy()
    n_seg = int(np.floor((T * missing_ratio) / missing_segment_len))
    if n_seg == 0 or missing_segment_len > T:
        return X_noisy

    max_start = T - missing_segment_len  # inclusive upper bound for start
    for c in range(N):
        starts = rng.choice(max_start + 1, size=min(n_seg, max_start + 1), replace=False)
        for s in starts:
            X_noisy[s : s + missing_segment_len, c] = 0.0

    return X_noisy


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm 4 – Distribution Shift  (Appendix B.4)
# ═════════════════════════════════════════════════════════════════════════════

def distribution_shift(
    X: np.ndarray,
    num_segments: int = 5,
    shift_scale: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply a distinct constant offset to each segment of the time series.

    For each channel c:
        * L_seg = floor(T / K_shift)
        * σ_c = std(X[:, c])
        * For each segment k: add shift_scale * σ_c * z_k, z_k ~ N(0, 1)

    The last segment extends to the end of the series to cover any remainder.

    Args:
        X: Raw time series of shape [T, N].
        num_segments: Number of segments to partition the series into (K_shift).
        shift_scale: Offset magnitude multiplier relative to channel std (α_shift).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Perturbed copy of X with shape [T, N].
    """
    if rng is None:
        rng = np.random.default_rng()

    if num_segments <= 0:
        return X.copy()

    T, N = X.shape
    X_noisy = X.copy()
    L_seg = max(1, int(np.floor(T / num_segments)))

    for c in range(N):
        sigma_c = X[:, c].std()
        for k in range(num_segments):
            start = k * L_seg
            end = (k + 1) * L_seg if k < num_segments - 1 else T
            if start >= T:
                break
            offset = rng.standard_normal() * shift_scale * sigma_c
            X_noisy[start:end, c] += offset

    return X_noisy


# ═════════════════════════════════════════════════════════════════════════════
# Combined entry point
# ═════════════════════════════════════════════════════════════════════════════

def apply_perturbations(
    X: np.ndarray,
    types: List[str],
    # White Noise
    noise_ratio: float = 0.05,
    noise_scale: float = 1.0,
    # Anomalies
    cont_anomaly_ratio: float = 0.05,
    cont_segment_len: int = 12,
    outlier_ratio: float = 0.005,
    anomaly_scale: float = 2.0,
    # Missing Values
    missing_ratio: float = 0.05,
    missing_segment_len: int = 12,
    # Distribution Shift
    num_shift_segments: int = 5,
    shift_scale: float = 5.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply one or more perturbation types to the training data array.

    Each enabled type is applied sequentially to the same copy of X.
    Valid type strings: ``"white_noise"``, ``"anomalies"``,
    ``"missing_values"``, ``"distribution_shift"``.

    Args:
        X: Raw training array of shape [T, N].
        types: List of perturbation type names to apply.
        seed: Optional integer seed for reproducibility.
        (remaining kwargs) – per-algorithm hyper-parameters (see individual
        functions for details).

    Returns:
        Perturbed array of shape [T, N].

    Raises:
        ValueError: If an unknown perturbation type string is given.
    """
    valid = {"white_noise", "anomalies", "missing_values", "distribution_shift"}
    unknown = set(types) - valid
    if unknown:
        raise ValueError(
            f"Unknown perturbation type(s): {unknown}. "
            f"Valid choices: {sorted(valid)}"
        )

    rng = np.random.default_rng(seed)
    X_out = X.copy()

    for ptype in types:
        if ptype == "white_noise":
            X_out = white_noise(
                X_out,
                noise_ratio=noise_ratio,
                noise_scale=noise_scale,
                rng=rng,
            )
        elif ptype == "anomalies":
            X_out = anomalies(
                X_out,
                cont_anomaly_ratio=cont_anomaly_ratio,
                cont_segment_len=cont_segment_len,
                outlier_ratio=outlier_ratio,
                anomaly_scale=anomaly_scale,
                rng=rng,
            )
        elif ptype == "missing_values":
            X_out = missing_values(
                X_out,
                missing_ratio=missing_ratio,
                missing_segment_len=missing_segment_len,
                rng=rng,
            )
        elif ptype == "distribution_shift":
            X_out = distribution_shift(
                X_out,
                num_segments=num_shift_segments,
                shift_scale=shift_scale,
                rng=rng,
            )

    return X_out
