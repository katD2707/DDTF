"""
Simplified DiT-style denoiser for FFT diffusion.

Predicts clean signal y_0 or noise eps from noisy input, conditioned on
encoded context, diffusion timestep, and a prediction-type indicator.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from src.embeddings import PatchEmbedding, TimestepEmbedding, RotaryPositionalEmbedding, PositionalEmbedding


# ── Helpers ───────────────────────────────────────────────────────────────────

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


# ── adaLN-Zero Transformer Block ─────────────────────────────────────────────

class AdaLNBlock(nn.Module):
    """
    adaLN-Zero transformer block.

    Uses cross-attention (CA) to attend to encoder context,
    followed by an MLP sublayer.  Both CA and MLP are gated
    through adaLN-Zero modulation (6 outputs).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout

        # ── Cross-attention sublayer (history encoder context) ─────────────
        self.norm1   = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2   = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ca_q_proj  = nn.Linear(d_model, d_model)
        self.ca_kv_proj  = nn.Linear(d_model, 2*d_model)
        self.proj    = nn.Linear(d_model, d_model)
        mlp_hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, d_model), nn.Dropout(dropout),
        )

        # adaLN: 6 outputs (shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        ctx: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:         [B, L, D]   input patches
            c:         [B, L, D]   per-patch adaLN condition (timestep embedding)
            ctx:       [B, S, D]   cross-attention context (history encoder)
            attn_mask: unused, kept for interface compatibility
        """
        B, L, D = x.shape
        shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=-1)

        # ── Cross-attention to history encoder (SDPA) ──────────────────────
        h     = modulate(self.norm1(x), shift_ca, scale_ca)
        q     = self.ca_q_proj(h).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        kv_in = ctx if ctx is not None else h
        S     = kv_in.shape[1]
        kv    = self.ca_kv_proj(kv_in).reshape(B, S, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v  = kv[0], kv[1]  # [B, H, S, D_h]
        
        # Compute attention scores: [B, H, L, S]
        scores = torch.einsum("...ld,...sd->...ls", q, k) / math.sqrt(self.head_dim)

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
        
        attn_out = torch.einsum("...ij,...jd->...id", attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        attn_out = self.proj(attn_out)
        x = x + gate_ca * attn_out

        # MLP
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(h)
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
    DiT-style denoiser.  Predicts y_0_pred for the FFT
    diffusion framework (f-DM).

    The condition from the encoder is projected per-patch and fed to the
    transformer blocks via cross-attention.
    """

    def __init__(
        self,
        max_prediction_length: int,
        patch_size: int,
        d_model: int,
        d_model_condition: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_prediction_length = max_prediction_length
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_heads = n_heads

        self.timestep_emb = TimestepEmbedding(d_model, patching=True, patch_size=patch_size)
        # Prediction-target indicator: 0 = x_start, 1 = noise
        self.pred_type_emb = nn.Embedding(2, d_model)
        self.context_proj = nn.Linear(d_model_condition, d_model) if d_model_condition != d_model else nn.Identity()

        self.patch_embedding = PatchEmbedding(patch_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model)
        self.num_target_patches = self.patch_embedding.compute_num_patches(max_prediction_length)

        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.final_layer = FinalLayer(d_model, patch_size)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        nn.init.normal_(self.patch_embedding.projection.weight, std=0.02)
        if isinstance(self.context_proj, nn.Linear):
            nn.init.normal_(self.context_proj.weight, std=0.02)
        nn.init.normal_(self.timestep_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_emb.mlp[2].weight, std=0.02)
        nn.init.normal_(self.pred_type_emb.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        for head in [self.final_layer]:
            nn.init.zeros_(head.adaLN[-1].weight)
            nn.init.zeros_(head.adaLN[-1].bias)
            nn.init.zeros_(head.linear.weight)
            nn.init.zeros_(head.linear.bias)

    def forward(
        self,
        input: Tensor,
        t: Tensor,
        cond: Tensor,
        pred_type: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            input:       [B, L, C]  noisy prediction sequence
            t:           [B, L]  per-position timesteps
            cond:        [B, C, P_ctx, D_cond]  encoder output
            pred_type:   [B] long, 0 = predict x_start, 1 = predict noise
                         (None → all x_start)
        Returns:
            pred: [B, L, C]  y_0 or eps depending on pred_type
        """
        B, L, C = input.shape

        if pred_type is None:
            pred_type = torch.zeros(B, dtype=torch.long, device=input.device)
        t_rep = repeat(t, "b l -> (b c) l", c=C)
        pred_type_rep = repeat(pred_type, "b -> (b c)", c=C)

        h = self.patch_embedding(input)  # [B*C, P_in, D]
        h = h + self.positional_embedding(h)
        c = self.timestep_emb(t_rep)    # [B*C, P_in, D]
        c = c + self.pred_type_emb(pred_type_rep).unsqueeze(1)

        cond_emb = self.context_proj(rearrange(cond, "b c p d -> (b c) p d"))
        for block in self.blocks:
            h = block(h, c, cond_emb)

        y0_out = self.final_layer(h, c).flatten(-2)

        y0_out = rearrange(y0_out, "(b c) l -> b l c", b=B, c=C)
        return y0_out
