"""
Staged diffusion model for time series forecasting.

Implements a staged diffusion process where the signal is progressively
degraded via K stages (either FFT frequency filtering or repeated moving
average smoothing), decoupled from the T diffusion timesteps.

Reference: f-DM (Gu et al., 2022) — arXiv:2210.04955
"""

import math
from typing import Optional, Tuple
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from tqdm.auto import tqdm

from src.conditioner import ConditionEncoder
from src.dit_denoiser import DiTDenoiser
from src.early_stopping import find_global_early_stop
from src.utils import default, extract


# ═════════════════════════════════════════════════════════════════════════════
# Beta / Stage Schedules
# ═════════════════════════════════════════════════════════════════════════════

def make_beta_schedule(schedule: str, T: int, start: float = 1e-4, end: float = 0.02) -> Tensor:
    if schedule == "linear":
        return torch.linspace(start, end, T)
    elif schedule == "quad":
        return torch.linspace(start ** 0.5, end ** 0.5, T) ** 2
    elif schedule == "sigmoid":
        b = torch.linspace(-6, 6, T)
        return torch.sigmoid(b) * (end - start) + start
    elif schedule == "cosine":
        s = 0.008
        ab = lambda t: math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        return torch.tensor([min(1 - ab((i + 1) / T) / ab(i / T), 0.999) for i in range(T)])
    elif schedule == "exp":
        lambd = 5.0
        gamma = 0.9
        t = torch.linspace(0, 1, T + 1)

        alpha_bar = torch.exp(-lambd * (t ** gamma))
        alpha_bar[0] = 1.0

        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clamp(betas, min=start, max=0.999)

        return betas
    raise ValueError(f"Unknown schedule: {schedule}")


def make_stage_schedule(K: int, T: int, schedule: str = "linear") -> Tensor:
    """Compute K+1 stage boundaries in [0, T]."""
    k = torch.arange(K + 1, dtype=torch.float32)
    if schedule == "linear":
        b = T * k / K
    elif schedule == "cosine":
        b = T * 0.5 * (1 - torch.cos(math.pi * k / K))
    elif schedule == "quadratic":
        b = T * (k / K) ** 2
    elif schedule == "sqrt":
        b = T * (k / K).sqrt()
    else:
        raise ValueError(f"Unknown stage schedule: {schedule}")
    b[0], b[-1] = 0.0, float(T)
    return b


def select_sampling_timesteps(boundaries: Tensor, T: int, S: int) -> list:
    """Select S+1 sampling timesteps that include all stage boundaries."""
    T_max = T - 1
    bnd_set = set(boundaries[1:-1].round().long().clamp(0, T_max).tolist())
    bnd_set.discard(0)
    bnd_set.add(T_max)
    remaining = S + 1 - len(bnd_set)
    if remaining > 0:
        cands = torch.linspace(-1, T_max, remaining + len(bnd_set) + 10).round().long().tolist()
        extras = [c for c in cands if c not in bnd_set]
        if len(extras) > remaining:
            idx = torch.linspace(0, len(extras) - 1, remaining).round().long().tolist()
            extras = [extras[i] for i in idx]
        bnd_set.update(extras)
    times = sorted(bnd_set, reverse=True)
    if len(times) > S + 1:
        idx = torch.linspace(0, len(times) - 1, S + 1).round().long().tolist()
        times = [times[i] for i in idx]
    return times


# ═════════════════════════════════════════════════════════════════════════════
# Staged FFT Filtering
# ═════════════════════════════════════════════════════════════════════════════

class StagedFFT(nn.Module):
    """
    Progressive frequency-domain filtering with K stages.

    Stage 0: keep all frequencies (identity).
    Stage K: keep only the most dominant frequencies.
    """

    def __init__(self, K: int, seq_len: int, schedule: str = "linear"):
        super().__init__()
        self.K = K
        self.seq_len = seq_len
        self.freq_dim = seq_len // 2 + 1
        min_freq = max(1, self.freq_dim // (K + 1))
        k = torch.arange(K + 1, dtype=torch.float32)

        if schedule == "linear":
            progress = k / K
        elif schedule == "cosine":
            progress = 0.5 * (1 - torch.cos(math.pi * k / K))
        elif schedule == "quadratic":
            progress = (k / K) ** 2
        elif schedule == "sqrt":
            progress = (k / K).sqrt()
        else:
            raise ValueError(f"Unknown topk schedule: {schedule}")

        num_keep = (self.freq_dim + progress * (min_freq - self.freq_dim)).round().long()
        num_keep[0] = self.freq_dim
        num_keep[-1] = min_freq
        self.register_buffer("num_keep", num_keep)

    def apply_stage(self, x: Tensor, stage: Tensor) -> Tensor:
        """
        x: [B, L, D],  stage: [B]  →  filtered x: [B, L, D]
        """
        B, L, D = x.shape
        stage = stage.clamp(0, self.K)
        if (stage == 0).all():
            return x

        x_flat = rearrange(x, "b l v -> (b v) l")
        stage_exp = repeat(stage, "b -> (b v)", v=D)

        x_fft = torch.fft.rfft(x_flat, dim=1)
        mag = x_fft.abs()
        k = self.num_keep[stage_exp]
        k_max = min(k.max().item(), self.freq_dim)

        _, topk_idx = torch.topk(mag, k=k_max, dim=1, largest=True)
        arange = torch.arange(k_max, device=x.device).unsqueeze(0)
        valid = arange < k.unsqueeze(1)

        mask = torch.zeros_like(mag, dtype=torch.bool)
        mask = torch.scatter_reduce(mask, 1, topk_idx, valid, reduce="sum", include_self=True)

        x_recon = torch.fft.irfft(x_fft * mask, n=L, dim=1)
        return rearrange(x_recon, "(b v) l -> b l v", v=D)

    def apply_incremental(self, x_k: Tensor, k: Tensor) -> Tensor:
        """
        Incremental transform: given x at stage k, produce x at stage k+1.

        Removes the bottom (n_k - n_{k+1}) frequencies by magnitude from x_k,
        keeping only the top n_{k+1} frequencies.
        """
        k_next = (k + 1).clamp(max=self.K)
        return self.apply_stage(x_k, k_next)
        


# ═════════════════════════════════════════════════════════════════════════════
# Staged Moving-Average Smoothing
# ═════════════════════════════════════════════════════════════════════════════

class StagedMA(nn.Module):
    """
    Progressive moving-average smoothing with K stages.

    Stage 0: identity (no smoothing).
    Stage k: apply a box kernel of size `kernel_size` repeated k times,
             equivalent to a B-spline-like smoothing filter.

    All effective kernels are precomputed at init for efficient training.
    """

    def __init__(self, K: int, kernel_size: int = 3):
        super().__init__()
        self.K = K
        self.kernel_size = kernel_size
        self._precompute_kernels(K, kernel_size)

    def _precompute_kernels(self, K: int, ks: int):
        """Build the effective convolution kernel for each stage 1..K."""
        base = torch.ones(1, 1, ks, dtype=torch.float32) / ks
        # The incremental kernel (k → k+1) is always the base kernel,
        # since cumulative_{k+1} = conv(cumulative_k, base).
        self.register_buffer("_inc_kernel", base)
        kernels = []
        eff = base.clone()
        for stage in range(1, K + 1):
            if stage == 1:
                eff = base.clone()
            else:
                eff = F.conv1d(eff, base.flip(-1), padding=ks - 1)
            kernels.append(eff.clone())
        # Store as list of buffers so they move with .to(device)
        for i, k in enumerate(kernels):
            self.register_buffer(f"_kernel_{i}", k)

    def _get_kernel(self, stage_idx: int) -> Tensor:
        return getattr(self, f"_kernel_{stage_idx}")

    def apply_stage(self, x: Tensor, stage: Tensor) -> Tensor:
        """
        x: [B, L, D],  stage: [B] (int, 0..K)  →  smoothed x: [B, L, D]
        """
        B, L, D = x.shape
        stage = stage.clamp(0, self.K)
        if (stage == 0).all():
            return x

        x_flat = rearrange(x, "b l v -> (b v) l")
        stage_exp = repeat(stage, "b -> (b v)", v=D)   # [(B*D)]

        # Group by unique stage value for batched conv
        out = torch.zeros_like(x_flat)
        unique_stages = stage_exp.unique()
        for s in unique_stages:
            mask = stage_exp == s
            s_int = s.item()
            if s_int == 0:
                out[mask] = x_flat[mask]
            else:
                kernel = self._get_kernel(s_int - 1)            # 0-indexed
                xs = x_flat[mask].unsqueeze(1)                  # [N, 1, L]
                pad = kernel.shape[-1] // 2
                ys = F.conv1d(xs, kernel.to(xs.device), padding=pad)
                out[mask] = ys.squeeze(1)[:, :L]

        return rearrange(out, "(b v) l -> b l v", v=D)

    def apply_incremental(self, x_k: Tensor, k: Tensor) -> Tensor:
        """
        Incremental transform: given x at stage k, produce x at stage k+1.

        Applies a single base kernel convolution (one MA pass).
        Since cumulative_{k+1} = conv(cumulative_k, base), convolving x_k
        with base gives x_{k+1}.
        """
        B, L, D = x_k.shape
        k_next = (k + 1).clamp(max=self.K)
        # If already at max stage, return as-is
        if (k_next == k).all():
            return x_k

        kernel = self._inc_kernel.to(x_k.device)  # [1, 1, ks]
        pad = kernel.shape[-1] // 2
        x_flat = rearrange(x_k, "b l v -> (b v) 1 l")
        out = F.conv1d(x_flat, kernel, padding=pad)
        out = out[:, 0, :L]
        return rearrange(out, "(b v) l -> b l v", v=D)


# ═════════════════════════════════════════════════════════════════════════════
# Main Model: FFTDiffusion
# ═════════════════════════════════════════════════════════════════════════════

class FFTDiffusion(nn.Module):
    """
    Staged diffusion model for time series forecasting.

    Combines:
      - ConditionEncoder  (encodes historical context)
      - Denoiser          (DiT-style, predicts y_0 + delta_t)
      - Staged transform  (FFT frequency filtering or MA smoothing)
      - Standard DDPM/DDIM diffusion schedule
    """

    def __init__(
        self,
        context_length: int = 96,
        max_prediction_length: int = 1024,
        patch_size: int = 24,
        d_model_encoder: int = 128,
        d_model_denoiser: int = 128,
        n_encoder_layers: int = 2,
        n_denoiser_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        num_diffusion_steps: int = 50,
        num_sampling_steps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.1,
        noise_schedule: str = "linear",
        num_fft_stages: int = 8,
        stage_schedule: str = "linear",
        topk_schedule: str = "linear",
        eta: float = 1.0,
        transform_type: str = "fft",
        ma_kernel_size: int = 3,
        # Block diffusion
        block_diffusion_mode: str = "full",
        block_size: Optional[int] = None,
        # Bernoulli timestep sampling
        bernoulli_sampling: bool = False,
        bernoulli_p: float = 0.7,
    ):
        super().__init__()
        self.context_length = context_length
        self.max_prediction_length = max_prediction_length
        self.num_diffusion_steps = num_diffusion_steps
        self.num_sampling_steps = num_sampling_steps
        self.num_fft_stages = num_fft_stages
        self.eta = eta
        self.patch_size = patch_size
        self.transform_type = transform_type
        self._topk_schedule = topk_schedule
        self.causal = block_diffusion_mode != "full"
        self.bernoulli_sampling = bernoulli_sampling
        self.bernoulli_p = bernoulli_p
        self._t_delta: Optional[int] = None  # set dynamically from stop_step

        # ── Block diffusion configuration ───────────────────────────────────
        assert block_diffusion_mode in ("full", "fixed"), (
            f"Unknown block_diffusion_mode: {block_diffusion_mode}")
        self.block_diffusion_mode = block_diffusion_mode
        self._fixed_block_size = block_size or patch_size

        if block_diffusion_mode == "fixed":
            assert max_prediction_length % self._fixed_block_size == 0, (
                f"max_prediction_length ({max_prediction_length}) must be divisible by block_size ({self._fixed_block_size})")
            # assert self._fixed_block_size % patch_size == 0, (
            #     f"block_size ({self._fixed_block_size}) must be divisible by patch_size ({patch_size})")
        self.block_size = self._get_block_size()

        # Sub-modules
        self.condition_encoder = ConditionEncoder(
            context_length, patch_size, d_model_encoder, n_encoder_layers, n_heads, mlp_ratio, dropout,
        )
        self.denoiser = DiTDenoiser(
            max_prediction_length=max_prediction_length,
            patch_size=patch_size,
            d_model=d_model_denoiser,
            d_model_condition=d_model_encoder,
            n_layers=n_denoiser_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_causal_block_attn=self.causal,
            block_size=self.block_size if self.causal else None,
        )

        # ── Staged transform ──────────────────────────────────────────────────
        if transform_type == "ma":
            self.staged_transform = StagedMA(K=num_fft_stages, kernel_size=ma_kernel_size)
        else:
            self.staged_transform = StagedFFT(
                K=num_fft_stages, seq_len=self.block_size, schedule=topk_schedule)

        boundaries = make_stage_schedule(num_fft_stages, num_diffusion_steps, stage_schedule)
        self.register_buffer("stage_boundaries", boundaries)

        # Standard DDPM schedule (used for the Gaussian noise component)
        self._setup_schedule(num_diffusion_steps, beta_start, beta_end, noise_schedule)

    # ── schedule helpers ──────────────────────────────────────────────────

    def _setup_schedule(self, T, beta_start, beta_end, schedule):
        betas = make_beta_schedule(schedule, T, beta_start, beta_end)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, 0)
        ac_prev = F.pad(ac[:-1], (1, 0), value=1.0)

        def reg(name, val):
            self.register_buffer(name, val.float())

        reg("betas", betas)
        reg("alphas_cumprod", ac)
        reg("alphas_cumprod_prev", ac_prev)
        reg("sqrt_ac", ac.sqrt())
        reg("sqrt_1m_ac", (1 - ac).sqrt())
        reg("sqrt_recip_ac", (1 / ac).sqrt())
        reg("sqrt_recipm1_ac", (1 / ac - 1).sqrt())
        post_var = betas * (1 - ac_prev) / (1 - ac)
        reg("post_var", post_var)
        reg("post_log_var", torch.log(post_var.clamp(min=1e-20)))
        reg("post_coef1", betas * ac_prev.sqrt() / (1 - ac))
        reg("post_coef2", (1 - ac_prev) * alphas.sqrt() / (1 - ac))

    # ── Block diffusion helpers ───────────────────────────────────────────

    def _get_block_size(self) -> int:
        """Return the block size for the current batch."""
        # "full"  → seq_len = prediction_length  (original)
        # "fixed" → seq_len = block_size         (per-block transforms)
        if self.block_diffusion_mode == "fixed":
            return self._fixed_block_size
        return self.max_prediction_length

    def _sample_timestep_bernoulli(self, batch_size: int, device) -> Tensor:
        """Bernoulli-gated non-uniform timestep sampling (flipped ranges).

        Adapted from DDPM-BS (Zheng et al., ACM MM 2024) with flipped
        interval assignment:
          - With prob p:   t ~ Uniform[t_delta, T-1]   (high-noise)
          - With prob 1-p: t ~ Uniform[1, t_delta]     (low-noise)

        ``_t_delta`` is derived from the early-stopping stop_step and
        updated externally by the training loop after each evaluation.
        """
        T = self.num_diffusion_steps
        t_delta = self._t_delta  # guaranteed not None by caller

        u = torch.rand(batch_size, device=device)

        # High-noise branch: [t_delta, T-1]
        t_high = torch.randint(t_delta, T, (batch_size,), device=device)
        # Low-noise branch: [0, t_delta]  (inclusive, so upper bound is t_delta + 1 for randint)
        t_low = torch.randint(0, t_delta, (batch_size,), device=device)

        t = torch.where(u <= self.bernoulli_p, t_high, t_low)
        return t

    def q_sample(self, y0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """
        q_sample with per-position (per-block) diffusion timestep.

        Args:
            y0:    [B, L, C]   clean (degraded) signal
            t:     [B, L]      integer timestep per position
            noise: [B, L, C]
        Returns:
            y_t:   [B, L, C]
        """
        noise = default(noise, lambda: torch.randn_like(y0))
        sqrt_ac    = self.sqrt_ac[t]     # [B, L]
        sqrt_1m_ac = self.sqrt_1m_ac[t]  # [B, L]
        return sqrt_ac.unsqueeze(-1) * y0 + sqrt_1m_ac.unsqueeze(-1) * noise

    # ── Stage info ────────────────────────────────────────────────────────

    def _stage_info(self, t: Tensor):
        t_f = t.float()
        k = torch.searchsorted(self.stage_boundaries, t_f, right=True) - 1
        k = k.clamp(0, self.num_fft_stages - 1)
        tau_k = self.stage_boundaries[k]
        tau_k1 = self.stage_boundaries[(k + 1).clamp(max=self.num_fft_stages)]
        span = (tau_k1 - tau_k).float().clamp(min=1e-6)
        w = ((t_f - tau_k.float()) / span).clamp(0, 1)
        return k, tau_k, tau_k1, w

    def _compute_xt(self, y0: Tensor, t: Tensor):
        """Compute interpolated transformed signal x_t and stage-k version x_k."""
        k, _, _, w = self._stage_info(t)
        x_k = self.staged_transform.apply_stage(y0, k)
        x_k1 = self.staged_transform.apply_stage(y0, (k + 1).clamp(max=self.num_fft_stages))
        w = w.view(-1, 1, 1)
        x_t = (1 - w) * x_k + w * x_k1
        return x_t, x_k

    def _noise_from_y0(self, yt, t, y0):
        return (extract(self.sqrt_recip_ac, t, yt.shape) * yt - y0) / extract(self.sqrt_recipm1_ac, t, yt.shape)

    # ── Training forward ─────────────────────────────────────────────────

    def forward(self, y: Tensor, context: Tensor, tau: Optional[Tensor] = None, delta: Optional[Tensor] = None):
        """
        Training step — block-diffusion aware.

        In "full" mode this is identical to the original single-block behaviour.
        In "fixed" mode the prediction sequence is split into
        ``num_blocks`` blocks; each block gets an independently sampled
        diffusion timestep.  With 50 % probability the KV source for the
        causal self-attention is the *noisy* y_t (noised-prev training);
        with 50 % probability it is the clean FFT-degraded y_target
        (unnoised-prev training, corresponding to BD3LM's unmasked conditioning).
        """
        B, _, C = y.shape
        device = y.device

        cond = self.condition_encoder(context, tau=tau, delta=delta)

        # ── Timestep sampling ───────────────────────
        _use_bernoulli = (self.bernoulli_sampling and self._t_delta is not None)

        if self.causal:
            num_blocks = self.max_prediction_length // self.block_size
            if _use_bernoulli:
                # Sample each block's timestep via Bernoulli gating
                t_per_block = self._sample_timestep_bernoulli(
                    B * num_blocks, device).reshape(B, num_blocks)
            else:
                t_per_block = torch.randint(
                    0, self.num_diffusion_steps, (B, num_blocks), device=device)
            t = t_per_block.repeat_interleave(self.block_size, dim=1)  # [B, pred_len]
            
            y_flat = y.reshape(B * num_blocks, self.block_size, C)
            t_flat = t_per_block.reshape(B * num_blocks)
            y_target_flat, y_k_flat = self._compute_xt(y_flat, t_flat)
            
            y_target = y_target_flat.reshape(B, self.max_prediction_length, C)
            y_k      = y_k_flat.reshape(B, self.max_prediction_length, C)
        else:
            if _use_bernoulli:
                t_scalar = self._sample_timestep_bernoulli(B, device)
            else:
                t_scalar = torch.randint(
                    0, self.num_diffusion_steps, (B,), device=device)
            y_target, y_k = self._compute_xt(y, t_scalar)
            t = t_scalar.unsqueeze(1).expand(B, self.max_prediction_length)
        delta_target = y_k - y_target

        # ── Add Gaussian noise ────────────────────────────────────────────
        y_t = self.q_sample(y_target, t)
        if self.causal:
            y_kv = y
            t_kv = torch.full_like(t, self.num_diffusion_steps)

            y_input = torch.cat([y_t, y_kv], dim=1)   # [B, 2L, C]
            t_input = torch.cat([t, t_kv], dim=1)      # [B, 2L]
        else:
            y_input = y_t
            t_input = t

        # ── Main denoiser forward ─────────────────────────────────────────
        # loss_fn = F.mse_loss
        loss_fn = F.l1_loss
        y0_pred, delta_pred = self.denoiser(y_input, t_input, cond, tau=tau, delta=delta)
        rec_loss = torch.zeros_like(y)
        if self.causal:
            rec_loss = loss_fn(y0_pred[:, self.max_prediction_length:, :], y, reduction="none")
            y0_pred = y0_pred[:, :self.max_prediction_length, :]
            delta_pred = delta_pred[:, :self.max_prediction_length, :]
        
        # Losses
        stage_loss = loss_fn(y0_pred, y_target, reduction="none")
        fft1 = torch.fft.fft(y0_pred.transpose(1, 2), norm='forward')
        fft2 = torch.fft.fft(y_target.transpose(1, 2), norm='forward')
        fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
        fourier_loss = loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') \
                        + loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
        stage_loss +=  1.0 * fourier_loss
        degrad_loss = loss_fn(delta_pred, delta_target, reduction="none")

        # Filter outliers
        stage_loss = stage_loss[stage_loss < 100].mean()
        degrad_loss = degrad_loss[degrad_loss < 100].mean()
        rec_loss = rec_loss[rec_loss < 100].mean()

        total = stage_loss + degrad_loss + rec_loss
        return {
            "total": total,
            "stage": stage_loss,
            "degrad": degrad_loss,
        }

    # ── Inference (DDIM-style fast sampling) ──────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        context: Tensor,
        eval_prediction_length: Optional[int] = None,
        num_samples: int = 1,
        stop_step: Optional[int] = None,
        return_scores: bool = False,
        use_early_stop: bool = False,
        early_stop_safety_margin: int = 0,
        tau: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ):
        """
        Generate forecasts.

        In **full** mode: original single-block DDIM sampling (unchanged).
        In **block** mode: semi-AR block sampling (BD3LM-style).
          - The full prediction sequence is initialised as pure Gaussian noise.
          - Blocks are denoised left-to-right.
          - Each block runs ``num_sampling_steps`` reverse diffusion steps.
          - Previously generated (clean) blocks remain frozen and serve as
            causal conditioning for the current block via the block-causal
            self-attention.

        Args:
            context:                [B, ctx_len, C]
            eval_prediction_length: forecast horizon to generate; must be
                                    ≤ max_prediction_length (training length).
                                    In block mode must be divisible by block_size.
                                    Defaults to the training prediction_length.
            num_samples:            ensemble size
            stop_step:              if set, stop full-mode sampling at this t
            return_scores:          if True, return (pred, scores, steps)
            use_early_stop:         block mode only — each block independently
                                    detects its own optimal stop step inline;
                                    requires num_samples > 1.
            early_stop_safety_margin: steps before detected breakpoint to stop.
        Returns:
            predictions [B, eval_len, C]  (or tuple when return_scores=True)
        """
        B_orig, _, C = context.shape
        device = context.device

        eval_len = eval_prediction_length if eval_prediction_length is not None \
            else self.max_prediction_length
        assert eval_len <= self.max_prediction_length, (
            f"eval_prediction_length ({eval_len}) exceeds "
            f"max_prediction_length ({self.max_prediction_length})")

        cond = self.condition_encoder(context, tau=tau, delta=delta)
        if num_samples > 1:
            cond = repeat(cond, "b c p d -> (b n) c p d", n=num_samples)
            tau = repeat(tau, "b 1 c -> (b n) 1 c", n=num_samples) if tau is not None else None
            delta = repeat(delta, "b l c -> (b n) l c", n=num_samples) if delta is not None else None

        B_eff = B_orig * num_samples

        if not self.causal:
            return self._sample_full(
                cond, B_eff, C, device, eval_len, num_samples, stop_step, return_scores, tau, delta)

        return self._sample_block_semi_ar(
            cond, B_eff, C, device, eval_len, num_samples,
            return_scores=return_scores,
            use_early_stop=use_early_stop,
            early_stop_safety_margin=early_stop_safety_margin,
            tau=tau, delta=delta,
        )

    def _sample_full(self, cond, B_eff, C, device, eval_len, num_samples, stop_step, return_scores, tau, delta):
        """Full-sequence DDIM sampling; eval_len may be < max_prediction_length."""
        y_t = torch.randn(B_eff, eval_len, C, device=device)

        times = select_sampling_timesteps(self.stage_boundaries, self.num_diffusion_steps, self.num_sampling_steps)
        pairs = list(zip(times[:-1], times[1:]))

        all_scores: list = []
        all_steps: list = []

        for t_now, t_next in pairs:
            t_b = torch.full((B_eff,), t_now, device=device, dtype=torch.long)
            t_nb = torch.full((B_eff,), t_next, device=device, dtype=torch.long)

            t_input = t_b.unsqueeze(1).expand(B_eff, eval_len)
            y0_pred, delta_pred = self.denoiser(y_t, t_input, cond, tau=tau, delta=delta)
            noise_pred = self._noise_from_y0(y_t, t_b, y0_pred)
            candidate = y0_pred + delta_pred

            if return_scores:
                c_avg = rearrange(candidate, "(b n) l c -> b n l c", n=num_samples).mean(1)
                all_scores.append(c_avg.mean(dim=(-2, -1)))
                all_steps.append(t_now)

            # Early stopping
            if not return_scores and stop_step is not None and t_now <= stop_step:
                break

            if t_next < 0:
                break

            # Apply delta correction scaled for step size
            _, tau_k, _, _ = self._stage_info(t_b)
            denom = (t_b.float() - tau_k.float())
            safe = denom.abs() > 0.5
            scale = torch.where(safe, (t_b - t_nb).float() / denom, torch.zeros_like(denom)).view(-1, 1, 1)
            y0_corr = y0_pred + delta_pred * scale

            alpha_t = extract(self.alphas_cumprod, t_b, y_t.shape)
            alpha_s = extract(self.alphas_cumprod, t_nb, y_t.shape)
            sigma = self.eta * ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt()
            c_coef = (1 - alpha_s - sigma ** 2).sqrt()
            y_t = y0_corr * alpha_s.sqrt() + c_coef * noise_pred + sigma * torch.randn_like(y_t)

        if num_samples > 1:
            candidate = rearrange(candidate, "(b n) l c -> b n l c", n=num_samples).mean(1)

        if return_scores:
            scores = torch.stack(all_scores, dim=1)
            steps_t = torch.tensor(all_steps, device=candidate.device, dtype=torch.float)
            return candidate, scores, steps_t
        return candidate

    @torch.no_grad()
    def _sample_block_semi_ar(
        self, cond, B_eff, C, device, eval_len, num_samples,
        return_scores: bool = False,
        use_early_stop: bool = False,
        early_stop_safety_margin: int = 0,
        tau: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ):
        """
        Semi-AR block sampling — faithful BD3LM-style.

        Mirrors BD3LMS ``_semi_ar_sampler`` (no KV-cache path):
          - Maintain a growing buffer ``y_gen`` of shape [B, eval_len, C].
          - For each block b (left-to-right):
              * Pass the accumulated window ``y_gen[:, 0:b_end]`` as a
                SINGLE sequence to the denoiser (not doubled).
              * The denoiser uses the block-causal mask (M_BC quadrant of
                the full training mask), which allows block b to attend
                to all previously generated clean blocks 0..b-1.
              * t-encoding marks past blocks as clean (t = 2T) and the
                current block as actively noisy (t = t_now).
              * Run ``num_sampling_steps`` DDIM reverse steps on block b.
              * Write the denoised block b back into ``y_gen``.

        Args:
            return_scores: collect per-step ensemble-mean scores averaged
                           across blocks and return them.  Used by
                           find_optimal_stop_step for full-mode calibration.
            use_early_stop: per-block inline early stopping — each block
                            independently runs all S steps, detects its own
                            breakpoint via find_global_early_stop, and
                            commits the candidate at that step.
                            Requires num_samples > 1.
            early_stop_safety_margin: safety_margin passed to
                            find_global_early_stop for each block.

        eval_len must be divisible by block_size (asserted in sample()).
        """
        num_blocks = math.ceil(eval_len / self.block_size)
        padded_eval_len = num_blocks * self.block_size

        # Initialise the full buffer as Gaussian noise
        y_gen = torch.randn(B_eff, padded_eval_len, C, device=device)

        times = select_sampling_timesteps(
            self.stage_boundaries, self.num_diffusion_steps, self.num_sampling_steps)
        pairs = list(zip(times[:-1], times[1:]))

        all_block_scores: list = []  # for return_scores: [num_blocks] × Tensor[B, S]
        steps_recorded: list = []    # for return_scores: populated from first block

        for b_idx in range(num_blocks):
            b_start = b_idx * self.block_size
            b_end   = (b_idx + 1) * self.block_size

            block_score_list: list = []  # for return_scores: [S] × Tensor[B]
            cand_buf: list = []           # for inline ES: [S] × Tensor[B_eff, bs, C]
            es_score_list: list = []      # for inline ES: [S] × Tensor[B]
            es_steps_list: list = []      # for inline ES: [S] float timesteps

            for it, (t_now, t_next) in enumerate(pairs):
                t_b  = torch.full((B_eff,), t_now,  device=device, dtype=torch.long)
                t_nb = torch.full((B_eff,), t_next, device=device, dtype=torch.long)

                # ── BD3LMS-style single growing window ───────────────────────
                # Pass y_gen[:, 0:b_end] — only the accumulated (clean) prefix
                # + current noisy block.  The denoiser applies a block-causal
                # mask (M_BC) so that block b can attend to blocks 0..b-1.
                y_input = y_gen[:, :b_end]   # [B_eff, b_end, C]

                # t-encoding: clean past blocks → 2T, current block → t_now.
                # (Same convention as training: 2T = "already clean" sentinel.)
                t_input = torch.full((B_eff, b_end), self.num_diffusion_steps,
                                     device=device, dtype=torch.long)
                t_input[:, b_start:b_end] = t_now

                y0_pred, delta_pred = self.denoiser(
                    y_input, t_input, cond, tau=tau, delta=delta, sample_mode=True)
                # y0_pred / delta_pred: [B_eff, b_end, C] — use last block only
                candidate_b = y0_pred[:, b_start:b_end] + delta_pred[:, b_start:b_end]

                # ── Score collection (shared for return_scores and inline ES) ─
                if return_scores or use_early_stop:
                    c_avg = rearrange(candidate_b, "(b n) l c -> b n l c", n=num_samples).mean(1)  # [B, bs, C]
                    score = c_avg.mean(dim=(-2, -1))  # [B]
                    if return_scores:
                        block_score_list.append(score)
                        if b_idx == 0:
                            steps_recorded.append(t_now)
                    if use_early_stop:
                        cand_buf.append(candidate_b.detach().clone())
                        es_score_list.append(score)
                        es_steps_list.append(float(t_now))

                if it == self.num_sampling_steps - 1:
                    y_gen[:, b_start:b_end] = candidate_b
                    break

                # ── DDIM update (current block only) ─────────────────────────
                noise_pred_b = self._noise_from_y0(
                    y_gen[:, b_start:b_end], t_b, y0_pred[:, b_start:b_end])

                _, tau_k, _, _ = self._stage_info(t_b)
                denom = (t_b.float() - tau_k.float())
                safe  = denom.abs() > 0.5
                scale = torch.where(
                    safe, (t_b - t_nb).float() / denom,
                    torch.zeros_like(denom)).view(-1, 1, 1)
                y0_corr = y0_pred[:, b_start:b_end] + delta_pred[:, b_start:b_end] * scale
                alpha_t = extract(self.alphas_cumprod, t_b,  y_gen[:, b_start:b_end].shape)
                alpha_s = extract(self.alphas_cumprod, t_nb, y_gen[:, b_start:b_end].shape)
                sigma   = self.eta * (
                    (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt()
                c_coef  = (1 - alpha_s - sigma ** 2).sqrt()
                y_gen[:, b_start:b_end] = (
                    y0_corr * alpha_s.sqrt()
                    + c_coef * noise_pred_b
                    + sigma * torch.randn_like(y0_corr)
                )
            # block b_idx is now fully denoised; it stays frozen in y_gen
            # and will serve as causal context for subsequent blocks.

            # ── Per-block inline early stopping ───────────────────────────────
            # Each block independently detects its own breakpoint from the
            # full S-step score trajectory collected above, then overwrites
            # y_gen with the candidate at that block-specific step.
            if use_early_stop:
                if len(cand_buf) > 1:
                    scores_b  = torch.stack(es_score_list, dim=1)   # [B, S]
                    steps_loc = torch.tensor(es_steps_list, dtype=torch.float,
                                             device=device)          # descending
                    es = find_global_early_stop(
                        scores_b, steps=steps_loc,
                        safety_margin=early_stop_safety_margin,
                    )
                    stop_s = len(cand_buf) - 1   # fallback: last step
                    for k, t_k in enumerate(es_steps_list):
                        if t_k <= es["stop_step"]:
                            stop_s = k
                            break
                    y_gen[:, b_start:b_end] = cand_buf[stop_s]
                elif cand_buf:  # only 1 step collected (degenerate)
                    y_gen[:, b_start:b_end] = cand_buf[0]

            # Synchronize the committed block across replicas so that block b+1
            # sees a clean, averaged causal context instead of diverse per-replica noise.
            if num_samples > 1:
                avg_b = rearrange(
                    y_gen[:, b_start:b_end], "(b n) l c -> b n l c", n=num_samples
                ).mean(1)
                y_gen[:, b_start:b_end] = avg_b.repeat_interleave(num_samples, dim=0)

            if return_scores and block_score_list:
                all_block_scores.append(torch.stack(block_score_list, dim=1))  # [B, S]

        y_gen = y_gen[:, :eval_len, :]
        if num_samples > 1:
            y_gen = rearrange(y_gen, "(b n) l c -> b n l c", n=num_samples).mean(1)

        if return_scores and all_block_scores:
            # Average per-block score trajectories → single [B, S] tensor
            scores = torch.stack(all_block_scores, dim=0).mean(0)
            steps_t = torch.tensor(steps_recorded, dtype=torch.float, device=y_gen.device)
            return y_gen, scores, steps_t
        return y_gen

    @torch.no_grad()
    def find_optimal_stop_step(
        self,
        pilot_contexts: Tensor,
        num_samples: int = 1,
        smooth_window: Optional[int] = None,
        min_frac: float = 0.1,
        safety_margin: int = 0,
        tau: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ) -> dict:
        """
        Run pilot samples through full diffusion to find the optimal
        global stopping step via piecewise-linear breakpoint detection.

        Note: in block-diffusion mode (block_size < prediction_length)
        ``return_scores`` is not supported; the method falls back to full
        sampling and returns an empty dict (early stopping disabled).
        """
        if self.block_diffusion_mode != "full":
            # Early stopping not yet supported for block mode
            return {"stop_step": None}

        _, scores, steps = self.sample(
            pilot_contexts, num_samples=num_samples, return_scores=True, tau=tau, delta=delta,
        )
        return find_global_early_stop(
            scores=scores, steps=steps,
            smooth_window=smooth_window,
            min_frac=min_frac,
            safety_margin=safety_margin,
        )

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg) -> "FFTDiffusion":
        return cls(
            context_length=cfg.context_length,
            max_prediction_length=cfg.prediction_length,
            patch_size=cfg.patch_size,
            d_model_encoder=cfg.d_model_encoder,
            d_model_denoiser=cfg.d_model_denoiser,
            n_encoder_layers=cfg.n_encoder_layers,
            n_denoiser_layers=cfg.n_denoiser_layers,
            n_heads=cfg.n_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            num_diffusion_steps=cfg.num_diffusion_steps,
            num_sampling_steps=cfg.num_sampling_steps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            noise_schedule=cfg.noise_schedule,
            num_fft_stages=cfg.num_fft_stages,
            stage_schedule=cfg.stage_schedule,
            topk_schedule=cfg.topk_schedule,
            eta=cfg.eta,
            transform_type=cfg.transform_type,
            ma_kernel_size=cfg.ma_kernel_size,
            block_diffusion_mode=getattr(cfg, "block_diffusion_mode", "full"),
            block_size=getattr(cfg, "block_size", None),
            bernoulli_sampling=getattr(cfg, "bernoulli_sampling", False),
            bernoulli_p=getattr(cfg, "bernoulli_p", 0.7),
        )
