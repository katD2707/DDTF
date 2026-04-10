"""
CSDI-style denoiser for FFT diffusion.

Adapted from the CSDI diff_models architecture (residual blocks with
time + feature transformers, gated convolutions, and side-info conditioning).
Wrapped to match the same interface as the DiT denoiser:
    (yt, t, cond) -> (y0_pred, delta_t_pred)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat


# ── Building blocks ──────────────────────────────────────────────────────────

def _conv1d_init(in_ch: int, out_ch: int, kernel: int = 1) -> nn.Conv1d:
    layer = nn.Conv1d(in_ch, out_ch, kernel)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    """Sinusoidal table + 2-layer MLP (same as CSDI)."""

    def __init__(self, num_steps: int, emb_dim: int = 128, proj_dim: Optional[int] = None):
        super().__init__()
        proj_dim = proj_dim or emb_dim
        self.register_buffer(
            "table", self._build(num_steps, emb_dim // 2), persistent=False,
        )
        self.proj1 = nn.Linear(emb_dim, proj_dim)
        self.proj2 = nn.Linear(proj_dim, proj_dim)

    @staticmethod
    def _build(T: int, half: int) -> Tensor:
        steps = torch.arange(T).unsqueeze(1).float()
        freqs = (10.0 ** (torch.arange(half).float() / (half - 1) * 4.0)).unsqueeze(0)
        table = steps * freqs
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)

    def forward(self, t: Tensor) -> Tensor:
        x = self.table[t]
        return F.silu(self.proj2(F.silu(self.proj1(x))))


class ResidualBlock(nn.Module):
    """One CSDI residual block: time-attention → feature-attention → gate."""

    def __init__(self, channels: int, side_dim: int, diff_emb_dim: int, n_heads: int):
        super().__init__()
        self.diff_proj = nn.Linear(diff_emb_dim, channels)
        self.cond_proj = _conv1d_init(side_dim, 2 * channels)
        self.mid_proj = _conv1d_init(channels, 2 * channels)
        self.out_proj = _conv1d_init(channels, 2 * channels)

        enc = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads, dim_feedforward=channels,
            activation="gelu", batch_first=True,
        )
        self.time_layer = nn.TransformerEncoder(enc, num_layers=1)

        enc2 = nn.TransformerEncoderLayer(
            d_model=channels, nhead=n_heads, dim_feedforward=channels,
            activation="gelu", batch_first=True,
        )
        self.feat_layer = nn.TransformerEncoder(enc2, num_layers=1)

    # -- helpers for time / feature axis attention --
    def _attn_time(self, y: Tensor, shape: Tuple) -> Tensor:
        B, ch, K, L = shape
        if L == 1:
            return y
        y = y.reshape(B, ch, K, L).permute(0, 2, 1, 3).reshape(B * K, ch, L)
        y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # batch-first
        return y.reshape(B, K, ch, L).permute(0, 2, 1, 3).reshape(B, ch, K * L)

    def _attn_feat(self, y: Tensor, shape: Tuple) -> Tensor:
        B, ch, K, L = shape
        if K == 1:
            return y
        y = y.reshape(B, ch, K, L).permute(0, 3, 1, 2).reshape(B * L, ch, K)
        y = self.feat_layer(y.permute(0, 2, 1)).permute(0, 2, 1)  # batch-first
        return y.reshape(B, L, ch, K).permute(0, 2, 3, 1).reshape(B, ch, K * L)

    def forward(self, x: Tensor, cond: Tensor, diff_emb: Tensor) -> Tuple[Tensor, Tensor]:
        B, ch, K, L = x.shape
        base = x.shape
        x_flat = x.reshape(B, ch, K * L)

        y = x_flat + self.diff_proj(diff_emb).unsqueeze(-1)
        y = self._attn_time(y, base)
        y = self._attn_feat(y, base)
        y = self.mid_proj(y)

        cond_flat = cond.reshape(B, cond.shape[1], K * L)
        y = y + self.cond_proj(cond_flat)

        gate, filt = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.out_proj(y)
        res, skip = y.chunk(2, dim=1)
        return (x_flat.reshape(base) + res.reshape(base)) / math.sqrt(2.0), skip.reshape(base)


# ── CSDI backbone ────────────────────────────────────────────────────────────

class CSDIBackbone(nn.Module):
    """
    Full CSDI backbone with input/output projections.

    Input shape:  (B, input_ch, K, L)
    Side info:    (B, side_dim, K, L)
    Output shape: (B, out_ch, K, L)
    """

    def __init__(
        self,
        input_ch: int,
        channels: int,
        side_dim: int,
        num_steps: int,
        diff_emb_dim: int,
        n_heads: int,
        n_layers: int,
        out_ch: int = 1,
    ):
        super().__init__()
        self.out_ch = out_ch
        self.diff_emb = DiffusionEmbedding(num_steps, diff_emb_dim, diff_emb_dim)
        self.in_proj = _conv1d_init(input_ch, channels, 1)
        self.out_proj1 = _conv1d_init(channels, channels, 1)
        self.out_proj2 = _conv1d_init(channels, out_ch, 1)
        nn.init.zeros_(self.out_proj2.weight)

        self.layers = nn.ModuleList([
            ResidualBlock(channels, side_dim, diff_emb_dim, n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, side: Tensor, t: Tensor) -> Tensor:
        B, inp, K, L = x.shape
        x = self.in_proj(x.reshape(B, inp, K * L))
        x = F.relu(x).reshape(B, -1, K, L)

        d = self.diff_emb(t)
        skips = []
        for layer in self.layers:
            x, s = layer(x, side, d)
            skips.append(s)

        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(skips))
        x = x.reshape(B, -1, K * L)
        x = F.relu(self.out_proj1(x))
        x = self.out_proj2(x).reshape(B, self.out_ch, K, L)
        return x


# ── Denoiser wrapper (same interface as DiT Denoiser) ────────────────────────

class CSDIDenoiser(nn.Module):
    """
    CSDI-style denoiser wrapped to match the DiT denoiser interface.

    forward(yt, t, cond) -> (y0_pred, delta_t_pred)

    Internally it constructs side-info from the condition encoder output
    (time embeddings + feature embeddings) and runs a single CSDIBackbone
    with 2× output channels, split into y0 and delta predictions.
    """

    def __init__(
        self,
        prediction_length: int,
        num_context_patches: int,
        patch_size: int,        # unused, kept for interface compat
        d_model: int,           # channels for CSDI residual blocks
        d_model_condition: int,
        n_layers: int,
        n_heads: int,
        num_diffusion_steps: int = 50,
        diff_emb_dim: int = 128,
        mlp_ratio: int = 4,     # unused, kept for interface compat
        dropout: float = 0.1,   # unused, kept for interface compat
        channel_independent: bool = True,
        num_features: int = 1,
    ):
        super().__init__()
        self.prediction_length = prediction_length
        self.d_cond = d_model_condition
        self.num_patches = num_context_patches
        self.channel_independent = channel_independent

        # Side-info dimension:
        #   time_emb (d_model_condition) + feat_emb (d_model_condition) + 1 (cond_mask)
        time_emb_dim = d_model_condition
        feat_emb_dim = d_model_condition
        side_dim = time_emb_dim + feat_emb_dim + 1

        self.time_emb_dim = time_emb_dim
        self.feat_emb_dim = feat_emb_dim

        # Learnable feature embedding (indexed by channel id)
        # Max 512 channels should be enough for any benchmark
        self.feat_embed = nn.Embedding(512, feat_emb_dim)

        # Time (position) embedding via sinusoidal
        # Precomputed in forward from prediction_length

        # Condition projection: flatten encoder patches -> per-timestep
        self.cond_proj = nn.Linear(num_context_patches * d_model_condition, prediction_length)

        # Single backbone with 2x output for y0 and delta
        self.backbone = CSDIBackbone(
            input_ch=2, channels=d_model, side_dim=side_dim,
            num_steps=num_diffusion_steps, diff_emb_dim=diff_emb_dim,
            n_heads=n_heads, n_layers=n_layers, out_ch=2,
        )

    @staticmethod
    def _sinusoidal_time_emb(length: int, dim: int, device: torch.device) -> Tensor:
        """[L, dim] sinusoidal positional embedding."""
        pos = torch.arange(length, device=device).unsqueeze(1).float()
        div = 1.0 / torch.pow(
            10000.0, torch.arange(0, dim, 2, device=device).float() / dim,
        )
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(
        self,
        yt: Tensor,
        t: Tensor,
        cond: Tensor,
        tau: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            yt:    [B, L, C]  noisy target
            t:     [B]        timesteps
            cond:  [B, C, P, D_cond]  from ConditionEncoder
            tau:   ignored (CSDI uses nn.TransformerEncoder internally)
            delta: ignored
        Returns:
            (y0_pred, delta_pred)  each [B, L, C]
        """
        B, L, C = yt.shape
        device = yt.device

        if self.channel_independent:
            return self._forward_ci(yt, t, cond, B, L, C, device)
        return self._forward_cd(yt, t, cond, B, L, C, device)

    # ── channel-independent forward (original) ───────────────────────────

    def _forward_ci(self, yt, t, cond, B, L, C, device):
        # ── Build side-info  (B, side_dim, C, L) ─────────────────────────
        # Time (position) embedding  -> (L, time_emb_dim) -> (B, L, C, D)
        t_pe = self._sinusoidal_time_emb(L, self.time_emb_dim, device)
        t_pe = t_pe.unsqueeze(0).unsqueeze(2).expand(B, L, C, -1)

        f_ids = torch.arange(C, device=device)
        f_pe = self.feat_embed(f_ids).unsqueeze(0).unsqueeze(0).expand(B, L, C, -1)

        side = torch.cat([t_pe, f_pe], dim=-1).permute(0, 3, 2, 1)
        cond_mask = torch.ones(B, 1, C, L, device=device)
        side = torch.cat([side, cond_mask], dim=1)

        # ── Build 2-channel input  (B, 2, C, L) ─────────────────────────
        noisy = yt.permute(0, 2, 1).unsqueeze(1)

        cond_flat = rearrange(cond, "b c p d -> (b c) (p d)")
        cond_seq = self.cond_proj(cond_flat)
        cond_seq = rearrange(cond_seq, "(b c) l -> b 1 c l", b=B, c=C)
        inp = torch.cat([cond_seq, noisy], dim=1)

        # ── Run backbone ─────────────────────────────────────────────────
        out = self.backbone(inp, side, t)
        y0_out = out[:, 0].permute(0, 2, 1)
        delta_out = out[:, 1].permute(0, 2, 1)
        return y0_out, delta_out

    # ── channel-dependent forward ─────────────────────────────────────────

    def _forward_cd(self, yt, t, cond, B, L, C, device):
        # ── Build side-info  (B, side_dim, C, L) ─────────────────────────
        t_pe = self._sinusoidal_time_emb(L, self.time_emb_dim, device)
        t_pe = t_pe.unsqueeze(0).unsqueeze(2).expand(B, L, C, -1)

        f_ids = torch.arange(C, device=device)
        f_pe = self.feat_embed(f_ids).unsqueeze(0).unsqueeze(0).expand(B, L, C, -1)

        side = torch.cat([t_pe, f_pe], dim=-1).permute(0, 3, 2, 1)
        cond_mask = torch.ones(B, 1, C, L, device=device)
        side = torch.cat([side, cond_mask], dim=1)

        # ── Build 2-channel input  (B, 2, C, L) ─────────────────────────
        noisy = yt.permute(0, 2, 1).unsqueeze(1)

        # Condition: average over C, project to L, broadcast across C
        cond_avg = cond.mean(dim=1)              # [B, P, D_cond]
        cond_flat = cond_avg.reshape(B, -1)      # [B, P * D_cond]
        cond_seq = self.cond_proj(cond_flat)     # [B, L]
        cond_seq = cond_seq.reshape(B, 1, 1, L).expand(B, 1, C, L)

        inp = torch.cat([cond_seq, noisy], dim=1)

        # ── Run backbone ─────────────────────────────────────────────────
        out = self.backbone(inp, side, t)
        y0_out = out[:, 0].permute(0, 2, 1)
        delta_out = out[:, 1].permute(0, 2, 1)
        return y0_out, delta_out
