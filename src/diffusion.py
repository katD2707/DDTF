"""
Diffusion model for time series forecasting.

Implements a standard DDPM/DDIM diffusion process for time series,
with optional Bernoulli-gated non-uniform timestep sampling.
"""

import math
from typing import Optional, Tuple
import random
from functools import partial
import numpy as np
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
# Beta Schedule
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

# ═════════════════════════════════════════════════════════════════════════════
# Main Model: FFTDiffusion
# ═════════════════════════════════════════════════════════════════════════════

class FFTDiffusion(nn.Module):
    """
    Diffusion model for time series forecasting.

    Combines:
      - ConditionEncoder  (encodes historical context)
      - Denoiser          (DiT-style, predicts y_0)
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
        eta: float = 1.0,
        bernoulli_sampling: bool = False,
        bernoulli_p: float = 0.7,
    ):
        super().__init__()
        self.context_length = context_length
        self.max_prediction_length = max_prediction_length
        self.num_diffusion_steps = num_diffusion_steps
        self.num_sampling_steps = num_sampling_steps
        self.eta = eta
        self.patch_size = patch_size
        self.bernoulli_sampling = bernoulli_sampling
        self.bernoulli_p = bernoulli_p
        self._t_delta: Optional[int] = None  # set dynamically from stop_step

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
        )

        # Standard DDPM schedule
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

    # ── Bernoulli timestep sampling ──────────────────────────────────────────

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
        Forward diffusion: add noise to the clean signal.

        Args:
            y0:    [B, L, C]   clean signal
            t:     [B, L]      integer timestep per position
            noise: [B, L, C]
        Returns:
            y_t:   [B, L, C]
        """
        noise = default(noise, lambda: torch.randn_like(y0))
        sqrt_ac    = self.sqrt_ac[t]     # [B, L]
        sqrt_1m_ac = self.sqrt_1m_ac[t]  # [B, L]
        return sqrt_ac.unsqueeze(-1) * y0 + sqrt_1m_ac.unsqueeze(-1) * noise

    def _noise_from_y0(self, yt, t, y0):
        return (extract(self.sqrt_recip_ac, t, yt.shape) * yt - y0) / extract(self.sqrt_recipm1_ac, t, yt.shape)

    # ── Training forward ─────────────────────────────────────────────────

    def forward(self, y: Tensor, context: Tensor, tau: Optional[Tensor] = None, delta: Optional[Tensor] = None):
        """
        Training step.
        """
        B, _, C = y.shape
        device = y.device

        cond = self.condition_encoder(context, tau=tau, delta=delta)

        # ── Timestep sampling ───────────────────────
        _use_bernoulli = (self.bernoulli_sampling and self._t_delta is not None)
        if _use_bernoulli:
            t_scalar = self._sample_timestep_bernoulli(B, device)
        else:
            t_scalar = torch.randint(
                0, self.num_diffusion_steps, (B,), device=device)
        y_target = y
        t = t_scalar.unsqueeze(1).expand(B, self.max_prediction_length)

        # ── Add Gaussian noise ────────────────────────────────────────────
        y_t = self.q_sample(y_target, t)
        y_input = y_t
        t_input = t

        # ── Main denoiser forward ─────────────────────────────────────────
        loss_fn = F.l1_loss
        y0_pred = self.denoiser(y_input, t_input, cond, tau=tau, delta=delta)
        
        # Losses
        denoise_loss = loss_fn(y0_pred, y_target, reduction="none")
        denoise_loss_cp = denoise_loss.detach().mean((1,2)).cpu()

        fft1 = torch.fft.fft(y0_pred.transpose(1, 2), norm='forward')
        fft2 = torch.fft.fft(y_target.transpose(1, 2), norm='forward')
        fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
        fourier_loss = loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') \
                        + loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
        denoise_loss +=  1.0 * fourier_loss

        # Filter outliers
        denoise_loss = denoise_loss[denoise_loss < 100].mean()

        return {
            "total": denoise_loss,
            "step_loss": denoise_loss_cp,
            "step": t_scalar.cpu(),
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
        tau: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ):
        """
        Generate forecasts using DDIM sampling.

        Args:
            context:                [B, ctx_len, C]
            eval_prediction_length: forecast horizon to generate; must be
                                    ≤ max_prediction_length (training length).
                                    Defaults to the training prediction_length.
            num_samples:            ensemble size
            stop_step:              if set, stop full-mode sampling at this t
            return_scores:          if True, return (pred, scores, steps)
        Returns:
            predictions [B, N, eval_len, C]  (or tuple when return_scores=True)
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
        y_t = torch.randn(B_eff, eval_len, C, device=device)

        # Uniform sampling timesteps from T-1 down to -1
        times = torch.linspace(self.num_diffusion_steps - 1, -1, self.num_sampling_steps + 1).round().long().tolist()
        pairs = list(zip(times[:-1], times[1:]))

        all_scores: list = []
        all_steps: list = []
        all_cand_scores = []
        for t_now, t_next in pairs:
            t_b = torch.full((B_eff,), t_now, device=device, dtype=torch.long)
            t_nb = torch.full((B_eff,), t_next, device=device, dtype=torch.long)

            t_input = t_b.unsqueeze(1).expand(B_eff, eval_len)
            y0_pred = self.denoiser(y_t, t_input, cond, tau=tau, delta=delta)
            noise_pred = self._noise_from_y0(y_t, t_b, y0_pred)

            if return_scores:
                c_avg = rearrange(y0_pred, "(b n) l c -> b n l c", n=num_samples).mean(1)
                all_cand_scores.append(rearrange(y0_pred, "(b n) l c -> b n l c", n=num_samples).detach().cpu().numpy())
                all_scores.append(c_avg.mean(dim=(-2, -1)))
                all_steps.append(t_now)

            # Early stopping
            if not return_scores and stop_step is not None and t_now <= stop_step:
                break

            if t_next < 0:
                # np.savez('cand_all', np.stack(all_cand_scores, axis=1))
                break

            alpha_t = extract(self.alphas_cumprod, t_b, y_t.shape)
            alpha_s = extract(self.alphas_cumprod, t_nb, y_t.shape)
            sigma = self.eta * ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt()
            c_coef = (1 - alpha_s - sigma ** 2).sqrt()
            y_t = y0_pred * alpha_s.sqrt() + c_coef * noise_pred + sigma * torch.randn_like(y_t)

        if return_scores:
            scores = torch.stack(all_scores, dim=1)
            steps_t = torch.tensor(all_steps, device=y0_pred.device, dtype=torch.float)
            return y0_pred, scores, steps_t
        return y0_pred

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
            eta=cfg.eta,
            bernoulli_sampling=getattr(cfg, "bernoulli_sampling", False),
            bernoulli_p=getattr(cfg, "bernoulli_p", 0.7),
        )
