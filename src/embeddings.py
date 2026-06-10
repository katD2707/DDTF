"""Embedding modules: TimestepEmbedding, PatchEmbedding, RoPE."""

import math
from typing import Optional, Tuple
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── Positional Embedding (sinusoidal, fixed) ─────────────────────────────────

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1), :]


# ── Rotary Position Embedding ────────────────────────────────────────────────

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._cos: Optional[Tensor] = None
        self._sin: Optional[Tensor] = None
        self._cached_len = 0

    def _update(self, seq_len: int, device, dtype):
        if seq_len > self._cached_len:
            self._cached_len = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos = emb.cos()[None, None, :, :]
            self._sin = emb.sin()[None, None, :, :]

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        seq_len = q.shape[2]
        self._update(seq_len, q.device, q.dtype)
        cos = self._cos[:, :, :seq_len, :].to(q.dtype)
        sin = self._sin[:, :, :seq_len, :].to(q.dtype)
        return (q * cos) + (self._rotate_half(q) * sin), (k * cos) + (self._rotate_half(k) * sin)


# ── Patch Embedding ──────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 24, d_model: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.stride = patch_size
        self.projection = nn.Linear(patch_size, d_model, bias=False)
        self.padding_layer = nn.ReplicationPad1d((0, patch_size))

    def forward(self, x: Tensor) -> Tensor:
        """
            x: [B*C, L] or [B, L, C]  →  [B*C, num_patches, d_model]
        """
        if x.dim() == 3:
            B, L, C = x.shape
            x = rearrange(x, "b l c -> (b c) l")
        else:
            L = x.shape[-1]
        if (L - self.patch_size) % self.stride != 0:
            x = self.padding_layer(x.unsqueeze(1)).squeeze(1)
        x = x.unfold(-1, self.patch_size, self.stride)  # [B*C, N, P]
        x = self.projection(x)
        return x

    def compute_num_patches(self, seq_len: int) -> int:
        return (seq_len - self.patch_size) // self.stride + 1


# ── Timestep Embedding ───────────────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256, patching: bool = False, patch_size: int = 24):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim
        self.patching = patching
        self.patch_size = self.stride = patch_size

    @staticmethod
    def sinusoidal(t: Tensor, dim: int) -> Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[..., None].float() * freqs[None, ...]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: Tensor) -> Tensor:
        if self.patching:
            if t.dim() == 3:
                t = rearrange(t, "b l c -> (b c) l")
            else:
                L = t.shape[-1]
            if (L - self.patch_size) % self.stride != 0:
                raise ValueError(f"Input length {L} is not compatible with patch size {self.patch_size}, select other patch size.") 
            t = t.unfold(-1, self.patch_size, self.stride)  # [B*C, N, P]
            t = t.float().mean(dim=-1)  # [B*C, N]

        return self.mlp(self.sinusoidal(t, self.freq_dim))
