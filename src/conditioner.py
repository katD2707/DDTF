import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from src.embeddings import PatchEmbedding, RotaryPositionalEmbedding


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
        self.proj = nn.Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(head_dim)
        mlp_hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(mlp_hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.rope(q, k)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        sa_out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        sa_out = self.proj(sa_out)
        x = x + sa_out

        x = x + self.mlp(self.norm2(x))
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
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_emb = PatchEmbedding(patch_size, d_model, use_positional=True)
        self.num_patches = self.patch_emb.compute_num_patches(context_length)
        self.blocks = nn.ModuleList([
            ConditionBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x:     [B, context_len, C]
        Returns:
            CI: [B, C, num_patches, d_model]
            CD: [B, 1, num_patches, d_model]
        """
        B, L, C = x.shape
        x = rearrange(x, "b l c -> (b c) l")       # [B*C, L]
        h = self.patch_emb(x)                      # [B*C, P, D]

        for block in self.blocks:
            h = block(h)
        h = rearrange(h, "(b c) p d -> b c p d", b=B, c=C)
        return h