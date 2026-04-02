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
    a = torch.zeros_like(sx)
    b = torch.zeros_like(sx)
    good = valid & (denom.abs() > eps)
    a[good] = (n[good] * sxy[good] - sx[good] * sy[good]) / denom[good]
    b[good] = (sy[good] - a[good] * sx[good]) / n[good]
    fallback = valid & ~good
    b[fallback] = sy[fallback] / n[fallback]
    sse = syy - 2 * a * sxy - 2 * b * sy + a * a * sxx + 2 * a * b * sx + n * b * b
    sse[~valid] = torch.inf
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

    y = _smooth_1d_batch(scores, smooth_window).median(dim=0).values  # [T]
    x = steps

    # Prefix sums for piecewise-linear fit
    ones = torch.ones_like(x)
    cn, cx, cy = torch.cumsum(ones, 0), torch.cumsum(x, 0), torch.cumsum(y, 0)
    cxx, cxy, cyy = torch.cumsum(x * x, 0), torch.cumsum(x * y, 0), torch.cumsum(y * y, 0)

    min_seg = max(2, int(round(T * min_frac)))
    taus = torch.arange(min_seg, T - min_seg, device=device)

    ln, lx, ly = cn[taus - 1], cx[taus - 1], cy[taus - 1]
    lxx, lxy, lyy = cxx[taus - 1], cxy[taus - 1], cyy[taus - 1]
    rn, rx, ry = cn[-1] - ln, cx[-1] - lx, cy[-1] - ly
    rxx, rxy, ryy = cxx[-1] - lxx, cxy[-1] - lxy, cyy[-1] - lyy

    total = _segment_sse(ln, lx, ly, lxx, lxy, lyy) + _segment_sse(rn, rx, ry, rxx, rxy, ryy)

    best = taus[torch.argmin(total)].item()
    stop_idx = max(0, best - safety_margin)

    return {
        "breakpoint_step": steps[best].item(),
        "stop_step": steps[stop_idx].item(),
        "median_curve": y,
        "smooth_window": smooth_window,
    }
