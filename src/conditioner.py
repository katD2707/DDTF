import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from src.embeddings import PatchEmbedding, PositionalEmbedding


# ── Transformer block ────────────────────────────────────────────────────────

class ConditionBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        head_dim = d_model // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.res_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        mlp_hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, tau = None, delta = None) -> Tensor:
        B, L, D = x.shape
        # self-attention
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        # apply de-stationary modulation if provided
        tau = 1.0 if tau is None else tau.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        delta = 0.0 if delta is None else delta.view(B, 1, 1, L)  # [B, 1, 1, L]
        attn = attn * tau + delta

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        sa_out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        sa_out = self.proj(sa_out)

        x = self.norm1(x + self.res_drop(sa_out))
        x = self.norm2(x + self.mlp(x))
        return x


# ── Condition Encoder (lightweight transformer) ──────────────────────────────

class ConditionEncoder(nn.Module):
    """Encode historical context into patch-level embeddings."""

    def __init__(
        self,
        context_length: int,
        patch_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_emb = PatchEmbedding(patch_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.num_patches = self.patch_emb.compute_num_patches(context_length)
        self.blocks = nn.ModuleList([
            ConditionBlock(d_model, n_heads, mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, tau: Tensor = None, delta: Tensor = None) -> Tensor:
        """
        Args:
            x:     [B, context_len, C]
            tau:   [B, 1, C] optional pre-computed tau (if None, computed from x)
            delta: [B, L, C] optional pre-computed delta (if None, computed from x)
        Returns:
            [B, C, num_patches, d_model], tau, delta
        """
        B, L, C = x.shape
        x = rearrange(x, "b l c -> (b c) l")       # [B*C, L]
        tau = rearrange(tau, "b 1 c -> (b c) 1") if tau is not None else None  # [B*C, 1]
        delta = rearrange(delta, "b l c -> (b c) l") if delta is not None else None  # [B*C, L]
        h = self.patch_emb(x)                      # [B*C, P, D]
        h = h + self.pos_emb(h)

        for block in self.blocks:
            h = block(h, tau=tau, delta=delta)
        h = rearrange(h, "(b c) p d -> b c p d", b=B, c=C)
        return h