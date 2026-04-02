import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from src.embeddings import PatchEmbedding

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, context_len, C]
        Returns:
            cond: [B, C, num_patches, d_model]
        """
        B, L, C = x.shape
        x = rearrange(x, "b l c -> b c l")          # [B, C, L]
        h = self.patch_emb(x)                         # [B*C, P, D]
        h = self.encoder(h)                            # [B*C, P, D]
        h = rearrange(h, "(b c) p d -> b c p d", b=B, c=C)
        return h