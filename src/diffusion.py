"""
FFT-based diffusion model (f-DM) for time series forecasting.

Implements the staged FFT diffusion process where noise is applied
progressively in the frequency domain via K degradation stages,
decoupled from the T diffusion timesteps.

Reference: f-DM (Gu et al., 2022) — arXiv:2210.04955
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from tqdm.auto import tqdm

from src.conditioner import ConditionEncoder
from src.dit_denoiser import DiTDenoiser
from src.csdi_denoiser import CSDIDenoiser
from src.revin import Projector
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


# ═════════════════════════════════════════════════════════════════════════════
# Main Model: FFTDiffusion
# ═════════════════════════════════════════════════════════════════════════════

class FFTDiffusion(nn.Module):
    """
    FFT-based diffusion model for time series forecasting (f-DM).

    Combines:
      - ConditionEncoder  (encodes historical context)
      - Denoiser          (DiT-style, predicts y_0 + delta_t)
      - StagedFFT         (frequency-domain degradation)
      - Standard DDPM/DDIM diffusion schedule
    """

    def __init__(
        self,
        context_length: int = 96,
        prediction_length: int = 96,
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
        denoiser_type: str = "dit",
        channel_independent: bool = True,
        num_features: int = 1,
        use_destationary: bool = False,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_diffusion_steps = num_diffusion_steps
        self.num_sampling_steps = num_sampling_steps
        self.num_fft_stages = num_fft_stages
        self.eta = eta
        self.channel_independent = channel_independent
        self.use_destationary = use_destationary

        # Sub-modules
        self.condition_encoder = ConditionEncoder(
            context_length, patch_size, d_model_encoder, n_encoder_layers, n_heads, dropout,
            channel_independent=channel_independent, num_features=num_features,
            use_destationary=use_destationary,
        )
        num_ctx_patches = self.condition_encoder.num_patches

        if denoiser_type == "csdi":
            self.denoiser = CSDIDenoiser(
                prediction_length=prediction_length,
                num_context_patches=num_ctx_patches,
                patch_size=patch_size,
                d_model=d_model_denoiser,
                d_model_condition=d_model_encoder,
                n_layers=n_denoiser_layers,
                n_heads=n_heads,
                num_diffusion_steps=num_diffusion_steps,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                channel_independent=channel_independent,
                num_features=num_features,
            )
        else:
            self.denoiser = DiTDenoiser(
                prediction_length=prediction_length,
                num_context_patches=num_ctx_patches,
                patch_size=patch_size,
                d_model=d_model_denoiser,
                d_model_condition=d_model_encoder,
                n_layers=n_denoiser_layers,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                channel_independent=channel_independent,
                num_features=num_features,
                use_destationary=use_destationary,
            )

        # Staged FFT
        self.staged_fft = StagedFFT(K=num_fft_stages, seq_len=prediction_length, schedule=topk_schedule)
        boundaries = make_stage_schedule(num_fft_stages, num_diffusion_steps, stage_schedule)
        self.register_buffer("stage_boundaries", boundaries)

        # Standard DDPM schedule (used for the Gaussian noise component)
        self._setup_schedule(num_diffusion_steps, beta_start, beta_end, noise_schedule)

        # De-stationary attention projectors (Non-stationary Transformers)
        # Shared projectors output per-head scalars [B_eff, H].
        # Each sub-module (conditioner / denoiser) has its own learned
        # delta_expand to project the per-head scalar to its sequence length.
        if use_destationary:
            enc_in = 1 if channel_independent else num_features
            self.tau_learner = Projector(
                enc_in=enc_in, seq_len=context_length,
                hidden_dim=d_model_denoiser, output_dim=n_heads,
            )
            self.delta_learner = Projector(
                enc_in=enc_in, seq_len=context_length,
                hidden_dim=d_model_denoiser, output_dim=n_heads,
            )
            self._n_heads = n_heads

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
        x_k = self.staged_fft.apply_stage(y0, k)
        x_k1 = self.staged_fft.apply_stage(y0, (k + 1).clamp(max=self.num_fft_stages))
        w = w.view(-1, 1, 1)
        x_t = (1 - w) * x_k + w * x_k1
        return x_t, x_k

    # ── q-sample (add Gaussian noise) ────────────────────────────────────

    def q_sample(self, y0: Tensor, t: Tensor, noise: Optional[Tensor] = None):
        noise = default(noise, lambda: torch.randn_like(y0))
        return extract(self.sqrt_ac, t, y0.shape) * y0 + extract(self.sqrt_1m_ac, t, y0.shape) * noise

    def _noise_from_y0(self, yt, t, y0):
        return (extract(self.sqrt_recip_ac, t, yt.shape) * yt - y0) / extract(self.sqrt_recipm1_ac, t, yt.shape)

    # ── Training forward ─────────────────────────────────────────────────

    def _compute_destationary(self, context: Tensor,
                              revin_stats: Optional[Tuple[Tensor, Tensor]]):
        """
        Compute de-stationary attention factors from context + RevIN stats.

        Returns (tau, delta) or (None, None) if disabled / stats unavailable.
        tau:   [B_eff, H, 1, 1]  multiplicative scale (ready for attn scores)
        delta: [B_eff, H]        per-head additive bias (expanded by each module)
        """
        if not self.use_destationary or revin_stats is None:
            return None, None

        mean, std = revin_stats  # each [B, 1, C]

        if self.channel_independent:
            ctx = rearrange(context, "b l c -> (b c) l 1")      # [B*C, L, 1]
            mean_p = rearrange(mean, "b 1 c -> (b c) 1 1")
            std_p = rearrange(std, "b 1 c -> (b c) 1 1")
        else:
            ctx = context                                         # [B, L, C]
            mean_p, std_p = mean, std

        tau = self.tau_learner(ctx, std_p).exp()                  # [B_eff, H]
        delta = self.delta_learner(ctx, mean_p)                   # [B_eff, H]

        tau = tau.unsqueeze(-1).unsqueeze(-1)                     # [B_eff, H, 1, 1]
        return tau, delta

    def forward(self, y: Tensor, context: Tensor,
                revin_stats: Optional[Tuple[Tensor, Tensor]] = None):
        """
        Training step.

        Args:
            y:       target  [B, pred_len, C]
            context: history [B, ctx_len, C]
        Returns:
            loss_dict with keys 'total', 'stage', 'degrad'
        """
        B = y.shape[0]
        device = y.device

        t = torch.randint(0, self.num_diffusion_steps, (B,), device=device)
        tau, delta = self._compute_destationary(context, revin_stats)
        cond = self.condition_encoder(context, tau=tau, delta=delta)

        # FFT degradation targets
        y_target, y_k = self._compute_xt(y, t)

        # Add Gaussian noise on top
        noise = torch.randn_like(y_target)
        y_t = self.q_sample(y_target, t, noise)

        # Predict
        y0_pred, delta_pred = self.denoiser(y_t, t, cond, tau=tau, delta=delta)
        delta_target = y_k - y_target

        # Losses
        stage_loss = F.mse_loss(y0_pred, y_target, reduction="none")
        fft1 = torch.fft.fft(y0_pred.transpose(1, 2), norm='forward')
        fft2 = torch.fft.fft(y_target.transpose(1, 2), norm='forward')
        fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
        fourier_loss = F.mse_loss(torch.real(fft1), torch.real(fft2), reduction='none') \
                        + F.mse_loss(torch.imag(fft1), torch.imag(fft2), reduction='none')
        stage_loss +=  1.0 * fourier_loss
        degrad_loss = F.mse_loss(delta_pred, delta_target, reduction="none")

        # Filter outliers
        stage_loss = stage_loss[stage_loss < 100].mean()
        degrad_loss = degrad_loss[degrad_loss < 100].mean()

        total = stage_loss + degrad_loss
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
        num_samples: int = 1,
        stop_step: Optional[int] = None,
        return_scores: bool = False,
        revin_stats: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """
        Generate forecasts via DDIM sampling.

        Args:
            context: [B, ctx_len, C]
            num_samples: ensemble size
            stop_step: if set, stop early at this diffusion timestep
            return_scores: if True, return (pred, scores, steps) for pilot calibration
            revin_stats: optional (mean [B,1,C], std [B,1,C]) from RevIN
        Returns:
            predictions [B, pred_len, C]  (or tuple when return_scores=True)
        """
        B_orig, _, C = context.shape
        device = context.device

        tau, delta = self._compute_destationary(context, revin_stats)

        cond = self.condition_encoder(context, tau=tau, delta=delta)
        if num_samples > 1:
            cond = repeat(cond, "b c p d -> (b n) c p d", n=num_samples)
            if tau is not None:
                tau = repeat(tau, "b ... -> (b n) ...", n=num_samples)
                delta = repeat(delta, "b ... -> (b n) ...", n=num_samples)

        B_eff = B_orig * num_samples
        y_t = torch.randn(B_eff, self.prediction_length, C, device=device)

        times = select_sampling_timesteps(self.stage_boundaries, self.num_diffusion_steps, self.num_sampling_steps)
        pairs = list(zip(times[:-1], times[1:]))

        all_scores: list = []
        all_steps: list = []

        for t_now, t_next in pairs:
            t_b = torch.full((B_eff,), t_now, device=device, dtype=torch.long)
            t_nb = torch.full((B_eff,), t_next, device=device, dtype=torch.long)

            y0_pred, delta_pred = self.denoiser(y_t, t_b, cond, tau=tau, delta=delta)
            noise_pred = self._noise_from_y0(y_t, t_b, y0_pred)
            candidate = y0_pred + delta_pred

            # Collect scores for pilot calibration
            if return_scores:
                if num_samples > 1:
                    c_avg = rearrange(candidate, "(b n) l c -> b n l c", n=num_samples).mean(1)
                else:
                    c_avg = candidate
                all_scores.append(c_avg.mean(dim=(-2, -1)))
                all_steps.append(t_now)

            # Early stopping
            if not return_scores and stop_step is not None and t_now <= stop_step:
                y_t = candidate
                break

            if t_next < 0:
                y_t = candidate
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
            y_t = rearrange(y_t, "(b n) l c -> b n l c", n=num_samples).mean(1)

        if return_scores:
            scores = torch.stack(all_scores, dim=1)
            steps_t = torch.tensor(all_steps, device=device, dtype=torch.float)
            return y_t, scores, steps_t
        return y_t

    @torch.no_grad()
    def find_optimal_stop_step(
        self,
        pilot_contexts: Tensor,
        num_samples: int = 1,
        smooth_window: Optional[int] = None,
        min_frac: float = 0.1,
        safety_margin: int = 0,
        revin_stats: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> dict:
        """
        Run pilot samples through full diffusion to find the optimal
        global stopping step via piecewise-linear breakpoint detection.

        Args:
            pilot_contexts: [N, ctx_len, C]
            num_samples: ensemble size per pilot
            smooth_window: moving-avg window (None = auto)
            min_frac: min fraction of trajectory on each side
            safety_margin: stop this many indices before breakpoint
            revin_stats: optional (mean [B,1,C], std [B,1,C]) from RevIN

        Returns:
            dict from find_global_early_stop
        """
        _, scores, steps = self.sample(
            pilot_contexts, num_samples=num_samples, return_scores=True,
            revin_stats=revin_stats,
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
            prediction_length=cfg.prediction_length,
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
            denoiser_type=cfg.denoiser_type,
            channel_independent=cfg.channel_independent,
            num_features=cfg.num_features,
            use_destationary=cfg.use_destationary,
        )
