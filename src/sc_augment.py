"""
Self-conditioning augmentations for the diffusion model.

Implements three data augmentation strategies from TiRex (Auer et al., 2025)
Appendix B, applied to the self-conditioning signal during training:

1. Amplitude Modulation  – piecewise linear trend multiplication
2. Censor Augmentation   – random clipping from above or below
3. Spike Injection       – additive periodic spike signals
"""

import math
import torch
from torch import Tensor


# ── 1. Amplitude Modulation ─────────────────────────────────────────────────

def amplitude_modulation(y: Tensor, max_changepoints: int = 5) -> Tensor:
    """
    Multiply signal by a random piecewise-linear trend.

    Algorithm 1 from TiRex Appendix B:
        - Sample k ~ Uniform(0, 5) changepoints
        - Interpolate amplitudes ~ N(1, 1) between changepoints
        - y_aug = y * trend

    Args:
        y: [B, L, C]
        max_changepoints: upper bound on changepoints per sample
    Returns:
        augmented y: [B, L, C]
    """
    B, L, C = y.shape
    device = y.device
    N = max_changepoints  # max nodes between endpoints

    # Number of active changepoints per sample: k ~ U(0, N)
    k = torch.randint(0, N + 1, (B,), device=device)  # [B]

    # Fixed grid of N candidate changepoint positions in (0, L)
    # Each sample gets N random positions, sorted
    cps = torch.rand(B, N, device=device) * (L - 2) + 1.0  # [B, N] in (1, L-1)
    cps, _ = cps.sort(dim=1)

    # Build full node positions: [0, cp1, ..., cpN, L]  shape [B, N+2]
    zeros = torch.zeros(B, 1, device=device)
    ends = torch.full((B, 1), float(L), device=device)
    positions = torch.cat([zeros, cps, ends], dim=1)  # [B, N+2]

    # Sample amplitudes at each node ~ N(1, 1)
    amplitudes = torch.randn(B, N + 2, device=device) + 1.0

    # Mask out inactive changepoints: if idx > k, set amplitude to 1.0
    # Active nodes: index 0 (start), indices 1..k (changepoints), index k+1 mapped to end
    node_idx = torch.arange(N + 2, device=device).unsqueeze(0)  # [1, N+2]
    inactive = node_idx > (k.unsqueeze(1) + 1)  # nodes beyond the last active
    amplitudes = torch.where(inactive, torch.ones_like(amplitudes), amplitudes)

    # For inactive changepoints, collapse their position to L (so they don't
    # create extra segments). Active: 0, cp1..cpk, L.  Inactive cps → L.
    cp_inactive = node_idx[:, 1:-1] > k.unsqueeze(1)  # [B, N]
    cps_masked = torch.where(cp_inactive, torch.full_like(cps, float(L)), cps)
    positions = torch.cat([zeros, cps_masked, ends], dim=1)  # [B, N+2]

    # Vectorized piecewise-linear interpolation
    t_axis = torch.arange(L, device=device, dtype=torch.float32)  # [L]
    # For each timestep find which segment it belongs to
    # positions[b, :] is sorted → use searchsorted
    seg = torch.searchsorted(positions, t_axis.unsqueeze(0).expand(B, -1).contiguous(), right=True) - 1
    seg = seg.clamp(0, N)  # [B, L]

    # Gather left/right boundaries and amplitudes
    left_pos = positions.gather(1, seg)                    # [B, L]
    right_pos = positions.gather(1, (seg + 1).clamp(max=N + 1))
    left_amp = amplitudes.gather(1, seg)
    right_amp = amplitudes.gather(1, (seg + 1).clamp(max=N + 1))

    span = (right_pos - left_pos).clamp(min=1e-6)
    frac = (t_axis.unsqueeze(0) - left_pos) / span        # [B, L]
    frac = frac.clamp(0, 1)
    trend = left_amp * (1 - frac) + right_amp * frac       # [B, L]

    return y * trend.unsqueeze(-1)  # [B, L, 1] broadcast over C


# ── 2. Censor Augmentation ──────────────────────────────────────────────────

def censor_augmentation(y: Tensor) -> Tensor:
    """
    Clip signal from above or below at a random quantile threshold.

    Algorithm 2 from TiRex Appendix B:
        - Random direction: 'above' or 'below'
        - Sample quantile from empirical distribution
        - y_aug = max(y, c) or min(y, c)

    Args:
        y: [B, L, C]
    Returns:
        augmented y: [B, L, C]
    """
    B, L, C = y.shape
    device = y.device

    # Random direction per sample
    clip_above = torch.rand(B, device=device) > 0.5  # [B]

    # Random quantile per sample
    q = torch.rand(B, device=device)  # [B]

    # Compute per-sample thresholds
    flat = y.reshape(B, -1)  # [B, L*C]
    sorted_vals, _ = flat.sort(dim=1)
    # Map quantile to index
    idx = (q * (L * C - 1)).long().clamp(0, L * C - 1)
    thresholds = sorted_vals.gather(1, idx.unsqueeze(1))  # [B, 1]
    thresholds = thresholds.unsqueeze(-1)                  # [B, 1, 1]

    # Apply clamp: above → min(y, thresh), below → max(y, thresh)
    clamped_above = torch.min(y, thresholds)
    clamped_below = torch.max(y, thresholds)
    mask = clip_above.view(B, 1, 1)
    return torch.where(mask, clamped_above, clamped_below)


# ── 3. Spike Injection ──────────────────────────────────────────────────────

def spike_injection(y: Tensor) -> Tensor:
    """
    Add structured periodic spike signals.

    Algorithm 3 from TiRex Appendix B:
        - Sample a periodic pattern (simple, 3-periodic, 4-periodic, weekly-like)
        - Sample kernel type (tophat, RBF, linear)
        - Tile spikes across time axis
        - y_aug = y + spike_signal

    Args:
        y: [B, L, C]
    Returns:
        augmented y: [B, L, C]
    """
    B, L, C = y.shape
    device = y.device

    # Pre-define pattern templates (padded to max length 7 for batched indexing)
    PATTERNS = [
        [1, 0, 0, 0, 0, 0, 0],  # simple (period-2 effective)
        [1, 0, 0, 0, 0, 0, 0],  # 3-periodic
        [1, 0, 0, 0, 0, 0, 0],  # 4-periodic
        [1, 0, 0, 0, 0, 1, 0],  # weekly-like
    ]
    PAT_LENS = [2, 3, 4, 7]

    # Sample pattern type per sample
    p = torch.rand(B, device=device)
    pat_type = torch.zeros(B, dtype=torch.long, device=device)
    pat_type = torch.where(p < 0.75, torch.tensor(0, device=device), pat_type)
    pat_type = torch.where((p >= 0.75) & (p < 0.85), torch.tensor(1, device=device), pat_type)
    pat_type = torch.where((p >= 0.85) & (p < 0.95), torch.tensor(2, device=device), pat_type)
    pat_type = torch.where(p >= 0.95, torch.tensor(3, device=device), pat_type)

    # Sample period per sample
    lo_period = max(2, L // 20)
    hi_period = max(lo_period + 1, L // 4)
    periods = torch.randint(lo_period, hi_period, (B,), device=device).float()  # [B]

    # Sample kernel type per sample: 0=tophat, 1=rbf, 2=linear
    kernel_types = torch.randint(0, 3, (B,), device=device)  # [B]

    # Kernel parameters
    signal_std = y.reshape(B, -1).std(dim=1)  # [B]
    heights = signal_std * (0.5 + torch.rand(B, device=device) * 2.0)  # [B]
    widths = (periods * (0.05 + torch.rand(B, device=device) * 0.15)).clamp(min=1.0)  # [B]

    # Build spike signals for all samples at once
    t_axis = torch.arange(L, device=device, dtype=torch.float32)  # [L]
    spike_signals = torch.zeros(B, L, device=device)

    # Maximum number of spikes per sample
    max_spikes = int((L / lo_period) + 1)

    # Spike center indices: [B, max_spikes]
    spike_idx = torch.arange(max_spikes, device=device).unsqueeze(0).expand(B, -1).float()
    centers = spike_idx * periods.unsqueeze(1)  # [B, max_spikes]

    # Pattern mask: whether each spike is "active" based on its pattern
    pat_lens_t = torch.tensor(PAT_LENS, device=device)[pat_type]  # [B]
    pat_positions = spike_idx.long() % pat_lens_t.unsqueeze(1)  # [B, max_spikes]

    # Build pattern lookup: for each sample, check if pattern[pos] == 1
    patterns_t = torch.tensor(PATTERNS, device=device, dtype=torch.float32)  # [4, 7]
    per_sample_pat = patterns_t[pat_type]  # [B, 7]
    active = per_sample_pat.gather(1, pat_positions.clamp(max=6))  # [B, max_spikes]

    # Also mask out centers >= L
    active = active * (centers < L).float()

    # Compute spike contributions: [B, max_spikes, L]
    # dist[b, s, l] = |t_axis[l] - centers[b, s]|
    dist = (t_axis.unsqueeze(0).unsqueeze(0) - centers.unsqueeze(2)).abs()  # [B, S, L]

    # Tophat kernel
    tophat = torch.where(dist <= widths.view(B, 1, 1) / 2,
                         heights.view(B, 1, 1), torch.zeros(1, device=device))

    # RBF kernel
    variance = (widths ** 2).view(B, 1, 1).clamp(min=1e-6)
    rbf = heights.view(B, 1, 1) * torch.exp(-0.5 * dist ** 2 / variance)

    # Linear (triangular) kernel
    w_expanded = widths.view(B, 1, 1)
    linear = torch.where(dist < w_expanded,
                         heights.view(B, 1, 1) * (1 - dist / w_expanded),
                         torch.zeros(1, device=device))

    # Select kernel per sample
    is_tophat = (kernel_types == 0).view(B, 1, 1).float()
    is_rbf = (kernel_types == 1).view(B, 1, 1).float()
    is_linear = (kernel_types == 2).view(B, 1, 1).float()
    kernels = tophat * is_tophat + rbf * is_rbf + linear * is_linear  # [B, S, L]

    # Mask by active spikes and sum over spike dimension
    spike_signals = (kernels * active.unsqueeze(2)).sum(dim=1)  # [B, L]

    return y + spike_signals.unsqueeze(-1)  # [B, L, C]


# ── Combined augmentation ────────────────────────────────────────────────────

def apply_sc_augmentation(y: Tensor) -> Tensor:
    """
    Apply one of the three augmentations randomly to the self-conditioning signal.

    Probabilities follow TiRex defaults:
        - Amplitude modulation: 0.5
        - Censor augmentation:  0.5
        - Spike injection:      0.05

    Each augmentation is applied independently with its own probability.

    Args:
        y: [B, L, C] self-conditioning signal
    Returns:
        augmented y: [B, L, C]
    """
    out = y
    if torch.rand(1).item() < 0.5:
        out = amplitude_modulation(out)
    if torch.rand(1).item() < 0.5:
        out = censor_augmentation(out)
    if torch.rand(1).item() < 0.05:
        out = spike_injection(out)
    return out
