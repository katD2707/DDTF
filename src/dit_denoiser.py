"""
Simplified DiT-style denoiser for FFT diffusion.

Predicts clean signal y_0 and degradation delta_t from noisy input,
conditioned on encoded context and diffusion timestep.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from src.embeddings import PatchEmbedding, TimestepEmbedding


# ── Helpers ───────────────────────────────────────────────────────────────────

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ── adaLN-Zero Transformer Block ─────────────────────────────────────────────

class AdaLNBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        mlp_hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, d_model), nn.Dropout(dropout),
        )
        # 6 modulation outputs: shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))

    def forward(self, x: Tensor, c: Tensor, ctx: Optional[Tensor] = None) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)

        # Cross-attention: Q from x, K/V from ctx
        B, L, D = x.shape
        h = modulate(self.norm1(x), shift_attn, scale_attn)
        q = self.q_proj(h).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        kv_in = ctx if ctx is not None else h
        S = kv_in.shape[1]
        kv = self.kv_proj(kv_in).reshape(B, S, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        attn_out = self.proj(attn_out)
        x = x + gate_attn.unsqueeze(1) * attn_out

        # MLP
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


# ── Final projection with adaLN ──────────────────────────────────────────────

class FinalLayer(nn.Module):
    def __init__(self, d_model: int, output_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ── Denoiser ─────────────────────────────────────────────────────────────────

class DiTDenoiser(nn.Module):
    """
    DiT-style denoiser.  Predicts (y_0_pred, delta_t_pred) for the FFT
    diffusion framework (f-DM).

    The condition from the encoder is projected per-patch and fed to the
    transformer blocks via cross-attention.
    """

    def __init__(
        self,
        prediction_length: int,
        num_context_patches: int,
        patch_size: int,
        d_model: int,
        d_model_condition: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        channel_independent: bool = True,
        num_features: int = 1,
        use_sc: bool = False,
    ):
        super().__init__()
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self.d_model = d_model
        self.channel_independent = channel_independent
        self.num_features = num_features
        self.n_heads = n_heads
        self.use_sc = use_sc

        self.timestep_emb = TimestepEmbedding(d_model)
        self.context_proj = nn.Linear(d_model_condition, d_model) if d_model_condition != d_model else nn.Identity()

        nf = 1 if channel_independent else num_features
        self.patch_embedding = PatchEmbedding(patch_size, d_model,
                                              use_positional=True, num_features=nf)
        out_patch = patch_size * nf
        self.num_target_patches = self.patch_embedding.compute_num_patches(prediction_length)

        # Self-conditioning (optional)
        if use_sc:
            self.sc_patch_embedding = PatchEmbedding(patch_size, d_model,
                                                     use_positional=True, num_features=nf)
            self.input_proj = nn.Linear(d_model * 2, d_model)

        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.final_layer = FinalLayer(d_model, out_patch)
        self.delta_head = FinalLayer(d_model, out_patch)

        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        nn.init.normal_(self.patch_embedding.projection.weight, std=0.02)
        if self.use_sc:
            nn.init.normal_(self.sc_patch_embedding.projection.weight, std=0.02)
            nn.init.normal_(self.input_proj.weight, std=0.02)
        if isinstance(self.context_proj, nn.Linear):
            nn.init.normal_(self.context_proj.weight, std=0.02)
        nn.init.normal_(self.timestep_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_emb.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        for head in [self.final_layer, self.delta_head]:
            nn.init.zeros_(head.adaLN[-1].weight)
            nn.init.zeros_(head.adaLN[-1].bias)
            nn.init.zeros_(head.linear.weight)
            nn.init.zeros_(head.linear.bias)

    def forward(
        self,
        yt: Tensor,
        t: Tensor,
        cond: Tensor,
        sc_signal: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            yt:        noisy target [B, L, C]
            t:         timesteps [B]
            cond:      condition [B, C, P, D_cond] (CI) or [B, 1, P, D_cond] (CD)
            sc_signal: self-conditioning signal [B, L, C] or None
        Returns:
            (y0_pred, delta_pred)  each [B, L, C]
        """
        B, L, C = yt.shape

        if self.channel_independent:
            return self._forward_ci(yt, t, cond, B, L, C, sc_signal)
        return self._forward_cd(yt, t, cond, B, L, C, sc_signal)

    def _forward_ci(self, yt, t, cond, B, L, C, sc_signal=None):
        yt_ci = rearrange(yt, "b l c -> b c l")
        t_rep = repeat(t, "b -> (b c)", c=C)

        h = self.patch_embedding(yt_ci)             # [B*C, P, d_embed]

        cond_seq = rearrange(cond, "b c p d -> (b c) p d")
        cond_emb = self.context_proj(cond_seq)      # [B*C, P_ctx, d_model]

        # Self-conditioning embedding (optional)
        if self.use_sc:
            if sc_signal is not None:
                sc_ci = rearrange(sc_signal, "b l c -> b c l")
                sc_emb = self.sc_patch_embedding(sc_ci)  # [B*C, P, d_model]
            else:
                sc_emb = torch.zeros_like(h)             # [B*C, P, d_model]
            h = self.input_proj(torch.cat([h, sc_emb], dim=-1))

        c = self.timestep_emb(t_rep)                 # [B*C, d_model]
        for block in self.blocks:
            h = block(h, c, cond_emb)

        y0_out = self.final_layer(h, c).flatten(-2)[:, :L]
        delta_out = self.delta_head(h, c).flatten(-2)[:, :L]

        y0_out = rearrange(y0_out, "(b c) l -> b l c", b=B, c=C)
        delta_out = rearrange(delta_out, "(b c) l -> b l c", b=B, c=C)
        return y0_out, delta_out

    # ── channel-dependent forward ─────────────────────────────────────────

    def _forward_cd(self, yt, t, cond, B, L, C, sc_signal=None):
        P = self.patch_size
        yt = rearrange(yt, "b l c -> b c l")

        h = self.patch_embedding(yt)               # [B, nP, d_embed]

        cond_avg = cond.mean(dim=1)                        # [B, P_cond, D_cond]
        cond_emb = self.context_proj(cond_avg)             # [B, P_ctx, d_model]

        # Self-conditioning embedding (optional)
        if self.use_sc:
            if sc_signal is not None:
                sc_cd = rearrange(sc_signal, "b l c -> b c l")
                sc_emb = self.sc_patch_embedding(sc_cd)  # [B, nP, d_embed]
            else:
                sc_emb = torch.zeros_like(h)             # [B, nP, d_embed]
            h = self.input_proj(torch.cat([h, sc_emb], dim=-1))

        c = self.timestep_emb(t)
        for block in self.blocks:
            h = block(h, c, cond_emb)

        # [B, nP, P*C] → [B, L, C]
        y0_out = self.final_layer(h, c).reshape(B, -1, P, C)
        y0_out = y0_out.reshape(B, -1, C)[:, :L, :]

        delta_out = self.delta_head(h, c).reshape(B, -1, P, C)
        delta_out = delta_out.reshape(B, -1, C)[:, :L, :]

        return y0_out, delta_out
