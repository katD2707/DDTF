"""
Global early stopping for diffusion sampling.

Uses pilot samples to find a single optimal stopping step via
piecewise-linear breakpoint detection on smoothed score trajectories.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _make_odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _smooth_1d_batch(x: torch.Tensor, window: int) -> torch.Tensor:
    """Moving-average smoothing.  x: [N, T] -> [N, T]."""
    if window <= 1:
        return x
    pad = window // 2
    kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / window
    x = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
    return F.conv1d(x, kernel).squeeze(1)


def _segment_sse(n, sx, sy, sxx, sxy, syy, eps=1e-12):
    """SSE of best linear fit for a segment given sufficient statistics."""
    denom = n * sxx - sx * sx
    valid = n >= 2
    good = valid & (denom.abs() > eps)
    fallback = valid & ~good

    denom_safe = torch.where(denom.abs() > eps, denom, torch.ones_like(denom))
    n_safe = torch.where(n > 0, n, torch.ones_like(n))

    a_calc = (n * sxy - sx * sy) / denom_safe
    a = torch.where(good, a_calc, torch.zeros_like(sy))

    b_calc = (sy - a * sx) / n_safe
    b_fallback = sy / n_safe
    b = torch.where(good, b_calc, torch.where(fallback, b_fallback, torch.zeros_like(sy)))

    sse = syy - 2 * a * sxy - 2 * b * sy + a * a * sxx + 2 * a * b * sx + n * b * b
    sse = torch.where(valid, sse, torch.full_like(sse, float("inf")))
    return sse


@torch.no_grad()
def find_global_early_stop(
    scores: torch.Tensor,
    steps: Optional[torch.Tensor] = None,
    smooth_window: Optional[int] = None,
    min_frac: float = 0.1,
    safety_margin: int = 0,
) -> dict:
    """
    Find one global early-stopping step from pilot score trajectories.

    Args:
        scores: [N, T] — per-sample mean predictions at each sampling step.
        steps:  [T] — diffusion timestep values (descending). ``None`` → T-1..0.
        smooth_window: moving-average width (``None`` = auto).
        min_frac: minimum fraction of trajectory on each side of breakpoint.
        safety_margin: stop this many *indices* before the breakpoint.

    Returns:
        dict with ``stop_step``, ``breakpoint_step``, ``median_curve``, etc.
    """
    assert scores.ndim == 2
    N, T = scores.shape
    device, dtype = scores.device, scores.dtype

    if steps is None:
        steps = torch.arange(T - 1, -1, -1, device=device, dtype=dtype)
    else:
        steps = steps.to(device=device, dtype=dtype)

    if smooth_window is None:
        smooth_window = _make_odd(max(5, min(int(round(T * 0.03)), max(5, T // 4))))

    # Smooth score trajectories per sample: shape [N, T]
    smoothed_scores = _smooth_1d_batch(scores, smooth_window)

    x = steps

    # Prefix sums for piecewise-linear fit
    ones = torch.ones_like(x)
    cn = torch.cumsum(ones, 0)
    cx = torch.cumsum(x, 0)
    cxx = torch.cumsum(x * x, 0)

    cy = torch.cumsum(smoothed_scores, 1)
    cxy = torch.cumsum(x.unsqueeze(0) * smoothed_scores, 1)
    cyy = torch.cumsum(smoothed_scores * smoothed_scores, 1)

    min_seg = max(2, int(round(T * min_frac)))
    taus = torch.arange(min_seg, T - min_seg, device=device)

    ln = cn[taus - 1]
    lx = cx[taus - 1]
    lxx = cxx[taus - 1]
    ly = cy[:, taus - 1]
    lxy = cxy[:, taus - 1]
    lyy = cyy[:, taus - 1]

    rn = cn[-1] - ln
    rx = cx[-1] - lx
    rxx = cxx[-1] - lxx
    ry = cy[:, -1:] - ly
    rxy = cxy[:, -1:] - lxy
    ryy = cyy[:, -1:] - lyy

    left_sse = _segment_sse(ln, lx, ly, lxx, lxy, lyy)
    right_sse = _segment_sse(rn, rx, ry, rxx, rxy, ryy)
    total_sse = left_sse + right_sse  # [N, num_taus]

    # Best breakpoint index per sample
    sample_best_tau_indices = torch.argmin(total_sse, dim=1)  # [N]
    sample_best_indices = taus[sample_best_tau_indices]       # [N]
    sample_stop_indices = torch.clamp(sample_best_indices - safety_margin, min=0)  # [N]

    sample_breakpoint_steps = steps[sample_best_indices]     # [N]
    sample_stop_steps = steps[sample_stop_indices]           # [N]

    # Take median across all samples
    breakpoint_step = sample_breakpoint_steps.float().median().item()
    stop_step = sample_stop_steps.float().median().item()

    # Median curve across samples for reference/visualization
    median_curve = smoothed_scores.median(dim=0).values

    return {
        "breakpoint_step": breakpoint_step,
        "stop_step": stop_step,
        "median_curve": median_curve,
        "smooth_window": smooth_window,
        "sample_breakpoint_steps": sample_breakpoint_steps,
        "sample_stop_steps": sample_stop_steps,
    }

