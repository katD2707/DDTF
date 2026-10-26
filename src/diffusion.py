"""
Diffusion model for time series forecasting.

Implements a standard DDPM/DDIM diffusion process for time series.
The denoiser predicts x_start, noise, or a mix of both (noise below the
early-stopping step, x_start above it).
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


def dft_forward_real_input(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Explicit DFT equivalent to:

        fft = torch.fft.fft(x, dim=-1, norm="forward")
        real = fft.real
        imag = fft.imag

    Args:
        x: Real-valued tensor with shape [..., T].

    Returns:
        real: [..., T]
        imag: [..., T]

    Warning:
        This implementation is O(T^2), whereas FFT is O(T log T).
    """
    if not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")

    length = x.size(-1)

    # Construct angles in float32 for better CUDA compatibility,
    # then convert the matrices to the input dtype.
    index = torch.arange(
        length,
        device=x.device,
        dtype=torch.float32,
    )

    # angle[k, n] = 2*pi*k*n/T
    angle = (
        2.0
        * math.pi
        * index[:, None]
        * index[None, :]
        / length
    )

    cosine = torch.cos(angle).to(dtype=x.dtype)
    sine = torch.sin(angle).to(dtype=x.dtype)

    # DFT:
    # X[k] = (1/T) * sum_n x[n] * (cos(angle) - i*sin(angle))
    real = torch.matmul(x, cosine.transpose(0, 1)) / length
    imag = -torch.matmul(x, sine.transpose(0, 1)) / length

    return real, imag

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
      - Denoiser          (DiT-style, predicts y_0 and/or eps)
      - Standard DDPM/DDIM diffusion schedule
    """

    PRED_X_START = 0
    PRED_NOISE = 1

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
        prediction_type: str = "x_start",
    ):
        super().__init__()
        assert prediction_type in ("x_start", "noise", "mixed"), \
            f"Unknown prediction_type: {prediction_type}"
        self.context_length = context_length
        self.max_prediction_length = max_prediction_length
        self.num_diffusion_steps = num_diffusion_steps
        self.num_sampling_steps = num_sampling_steps
        self.eta = eta
        self.patch_size = patch_size
        self.prediction_type = prediction_type
        self._stop_step: Optional[float] = None  # set dynamically by the training loop (mixed mode)

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

    # ── Prediction-type helpers ──────────────────────────────────────────

    def _pred_type_for(self, t: Tensor, stop_step: Optional[float]) -> Tensor:
        """Per-sample prediction target (PRED_X_START / PRED_NOISE) for timesteps t [B].

        In mixed mode, t < stop_step uses noise prediction and t >= stop_step
        uses x_start prediction; without a stop_step everything is x_start.
        """
        if self.prediction_type == "noise":
            return torch.full_like(t, self.PRED_NOISE)
        if self.prediction_type == "mixed" and stop_step is not None:
            return (t < stop_step).long()
        return torch.full_like(t, self.PRED_X_START)

    def _train_pred_type_for(self, t: Tensor, stop_step: Optional[float]) -> Tensor:
        """Training-time prediction target for timesteps t [B].

        In mixed mode each sample is trained on noise with probability
        r = stop_step / T if t < stop_step, else 1 - r; otherwise x_start.
        Other modes (and mixed before calibration) match ``_pred_type_for``.
        """
        if self.prediction_type != "mixed" or stop_step is None:
            return self._pred_type_for(t, stop_step)
        r = float(stop_step) / self.num_diffusion_steps
        p_noise = torch.where(t < stop_step, r, 1.0 - r)
        return torch.bernoulli(p_noise).long()

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

    def _y0_from_noise(self, yt, t, noise):
        return extract(self.sqrt_recip_ac, t, yt.shape) * yt - extract(self.sqrt_recipm1_ac, t, yt.shape) * noise

    # ── Training forward ─────────────────────────────────────────────────

    def forward(self, y: Tensor, context: Tensor):
        """
        Training step.

        Each sample's target (x_start or noise) is chosen from its timestep
        via ``_train_pred_type_for`` and all samples go through one denoiser pass.
        """
        B, _, C = y.shape
        device = y.device

        cond = self.condition_encoder(context)

        # ── Timestep sampling ───────────────────────
        t_scalar = torch.randint(
            0, self.num_diffusion_steps, (B,), device=device)
        y_target = y
        t = t_scalar.unsqueeze(1).expand(B, self.max_prediction_length)
        pred_type = self._train_pred_type_for(t_scalar, self._stop_step)  # [B]

        # ── Add Gaussian noise ────────────────────────────────────────────
        noise = torch.randn_like(y_target)
        y_t = self.q_sample(y_target, t, noise)
        y_input = y_t
        t_input = t

        # ── Main denoiser forward ─────────────────────────────────────────
        loss_fn = F.l1_loss
        pred = self.denoiser(y_input, t_input, cond, pred_type=pred_type)

        # x_start loss: L1 + Fourier L1 against y_0
        x0_loss = loss_fn(pred, y_target, reduction="none")
        fft1 = torch.fft.fft(pred.transpose(1, 2), norm='forward')
        fft2 = torch.fft.fft(y_target.transpose(1, 2), norm='forward')
        fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
        fourier_loss = loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') \
                        + loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')

        # noise loss: L1 against eps
        eps_loss = loss_fn(pred, noise, reduction="none")

        is_noise = (pred_type == self.PRED_NOISE).view(B, 1, 1)
        denoise_loss_cp = torch.where(is_noise, eps_loss, x0_loss).detach().mean((1, 2)).cpu()
        denoise_loss = torch.where(is_noise, eps_loss, x0_loss + 1.0 * fourier_loss)

        # fft1_real, fft1_imag = dft_forward_real_input(y0_pred.transpose(1, 2))
        # fft2_real, fft2_imag = dft_forward_real_input(y_target.transpose(1, 2))

        # fft1_real = fft1_real.transpose(1, 2)
        # fft1_imag = fft1_imag.transpose(1, 2)

        # fft2_real = fft2_real.transpose(1, 2)
        # fft2_imag = fft2_imag.transpose(1, 2)

        # fourier_loss = loss_fn(fft1_real, fft2_real, reduction='none') \
        #                  + loss_fn(fft1_imag, fft2_imag, reduction='none')
        # denoise_loss +=  1.0 * fourier_loss

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
        tgt: Optional[Tensor] = None,
    ):
        """
        Generate forecasts using DDIM sampling.

        Args:
            context:                [B, ctx_len, C]
            eval_prediction_length: forecast horizon to generate; must be
                                    ≤ max_prediction_length (training length).
                                    Defaults to the training prediction_length.
            num_samples:            ensemble size
            stop_step:              x_start / noise mode: stop sampling at this t.
                                    mixed mode: switch from x_start to noise
                                    prediction below this t and run to the end.
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

        cond = self.condition_encoder(context)
        if num_samples > 1:
            cond = repeat(cond, "b c p d -> (b n) c p d", n=num_samples)

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
            pred_type = self._pred_type_for(t_b, stop_step)
            pred = self.denoiser(y_t, t_input, cond, pred_type=pred_type)
            if pred_type[0].item() == self.PRED_NOISE:
                noise_pred = pred
                y0_pred = self._y0_from_noise(y_t, t_b, noise_pred)
            else:
                y0_pred = pred
                noise_pred = self._noise_from_y0(y_t, t_b, y0_pred)
            # all_cand_scores.append(rearrange(y0_pred, "(b n) l c -> b n l c", n=num_samples).detach().cpu().numpy())
            if return_scores:
                if tgt is None:
                    c_avg = rearrange(y0_pred, "(b n) l c -> b n l c", n=num_samples).mean(1)
                else:
                    c_avg = ((rearrange(y0_pred, "(b n) l c -> b n l c", n=num_samples)-tgt.unsqueeze(1))**2).mean(1)
                all_scores.append(rearrange(c_avg, "b l c -> (b c) l").mean(-1))
                all_steps.append(t_now)

            # Early stopping (mixed mode keeps sampling with the noise head instead)
            if (self.prediction_type != "mixed" and not return_scores
                    and stop_step is not None and t_now <= stop_step):
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
            prediction_type=getattr(cfg, "prediction_type", "x_start"),
        )
