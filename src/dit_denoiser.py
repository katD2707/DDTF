"""
Simplified DiT-style denoiser for FFT diffusion.

Predicts clean signal y_0 and degradation delta_t from noisy input,
conditioned on encoded context and diffusion timestep.

Block-diffusion mode follows BD3LMS (Arriaga-Carrasco et al., 2024)
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

    When ``use_causal_block_attn=True`` a block-diffusion self-attention (SA)
    sublayer is prepended before the cross-attention (CA) sublayer.
    The SA operates on the full concatenated [y_t | y_kv] sequence (2P tokens)
    using the pre-built BD3LMS-style ``block_diff_mask`` passed as ``attn_mask``.
    Both SA and CA use torch SDPA.

    adaLN outputs:
      use_causal_block_attn=False : 6  (shift_ca, scale_ca, gate_ca,
                                        shift_mlp, scale_mlp, gate_mlp)
      use_causal_block_attn=True  : 9  (shift_sa, scale_sa, gate_sa + the 6 above)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_causal_block_attn: bool = False,
    ):
        super().__init__()
        self.use_causal_block_attn = use_causal_block_attn
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout

        # ── BD3LMS-style block self-attention sublayer ─────────────────────
        if use_causal_block_attn:
            self.norm_sa     = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.sa_qkv_proj = nn.Linear(d_model, 3 * d_model)
            self.sa_out_proj = nn.Linear(d_model, d_model)

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

        # adaLN: 9 outputs when SA sublayer is present, else 6
        n_mods = 9 if use_causal_block_attn else 6
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, n_mods * d_model))

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        ctx: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x:         [B, L, D]   input patches
                       (L = P in full mode; L = 2P in block mode)
            c:         [B, L, D]   per-patch adaLN condition (timestep embedding)
            ctx:       [B, S, D]   cross-attention context (history encoder)
            attn_mask: [L, L] bool, True = allowed to attend (BD3LMS mask);
                       None → unrestricted (full mode, SA sublayer absent)
            tau:       [B, 1]   optional de-stationary parameter
            delta:     [B, L]   optional de-stationary parameter
        """
        B, L, D = x.shape
        mods = self.adaLN(c).chunk(9 if self.use_causal_block_attn else 6, dim=-1)

        if self.use_causal_block_attn:
            shift_sa, scale_sa, gate_sa, \
            shift_ca, scale_ca, gate_ca, \
            shift_mlp, scale_mlp, gate_mlp = mods

            # ── Block SA: SDPA over full [y_t | y_kv] sequence ───────────
            h   = modulate(self.norm_sa(x), shift_sa, scale_sa)
            qkv = self.sa_qkv_proj(h).reshape(B, L, 3, self.n_heads, self.head_dim).transpose(1, 3) # each [B, H, L, D_h]
            queries, keys, values = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
            
            # Compute attention scores: [B, H, L, L]
            scores = torch.einsum("...ld,...sd->...ls", queries, keys) / math.sqrt(self.head_dim)
            
            # Apply de-stationary factors if provided (pre-softmax)
            tau = 1.0 if tau is None else tau.reshape(B, 1, 1, 1)  # reshape for broadcasting
            delta = 0.0 if delta is None else delta.reshape(B, 1, 1, -1)  # reshape for broadcasting
            scores = scores * tau
            
            # Apply attention mask
            if attn_mask is not None:
                scores.masked_fill_(~attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Softmax and dropout
            attn_weights = torch.softmax(scores, dim=-1)
            if self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
            
            sa_out = torch.einsum("...ij,...jd->...id", attn_weights, values)
            sa_out = sa_out.transpose(1, 2).reshape(B, L, D)
            sa_out = self.sa_out_proj(sa_out)
            x = x + gate_sa * sa_out
        else:
            shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = mods

        # ── Cross-attention to history encoder (SDPA) ──────────────────────
        h     = modulate(self.norm1(x), shift_ca, scale_ca)
        q     = self.ca_q_proj(h).reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        kv_in = ctx if ctx is not None else h
        S     = kv_in.shape[1]
        kv    = self.ca_kv_proj(kv_in).reshape(B, S, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v  = kv[0], kv[1]  # [B, H, S, D_h]
        
        # Compute attention scores: [B, H, L, S]
        scores = torch.einsum("...ld,...sd->...ls", q, k) / math.sqrt(self.head_dim)
        
        # Apply de-stationary factors if provided (pre-softmax)
        scores = scores * tau + delta
        
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
    DiT-style denoiser.  Predicts (y_0_pred, delta_t_pred) for the FFT
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
        use_causal_block_attn: bool = False,
        block_size: Optional[int] = 24,
    ):
        super().__init__()
        self.max_prediction_length = max_prediction_length
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_causal_block_attn = use_causal_block_attn

        self.timestep_emb = TimestepEmbedding(d_model, patching=True, patch_size=patch_size)
        self.context_proj = nn.Linear(d_model_condition, d_model) if d_model_condition != d_model else nn.Identity()

        self.patch_embedding = PatchEmbedding(patch_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model)
        self.num_target_patches = self.patch_embedding.compute_num_patches(max_prediction_length)

        if use_causal_block_attn:
            # assert block_size is not None and block_size % patch_size == 0, (
            #     "block_size must be provided and divisible by patch_size for block diffusion mode")
            self.block_size = block_size
            # BD3LMS block_diff_mask: [2P, 2P] bool, True = attend
            self.register_buffer(
                "block_diff_mask",
                self._build_block_diff_mask(self.num_target_patches, block_size // patch_size),
                persistent=False,
            )

        self.blocks = nn.ModuleList([
            AdaLNBlock(d_model, n_heads, mlp_ratio, dropout,
                       use_causal_block_attn=use_causal_block_attn)
            for _ in range(n_layers)
        ])
        self.final_layer = FinalLayer(d_model, patch_size)
        self.delta_head  = FinalLayer(d_model, patch_size)

        self._init_weights()

    # ── BD3LMS-style attention mask ───────────────────────────────────────

    @staticmethod
    def _build_block_diff_mask(num_patches: int, block_size: int) -> Tensor:
        """
        Build the BD3LMS-style block-diffusion attention mask.

        Combined sequence: 2*num_patches tokens.
          positions   0 .. P-1   → y_t  (noisy query half)
          positions   P .. 2P-1  → y_kv (clean/noisy context half)
        block_size: number of patches per block (ppb = block_size // patch_size).

        Each patch index is mapped to a block via ``patch_idx // block_size``.

        Three component masks (BD3LMS notation):
          M_BD  – block diagonal: patches in the same block and same half
                  attend to each other
          M_OBC – offset block-causal: y_t patches in block b attend to
                  y_kv patches in blocks < b
          M_BC  – block-causal: y_kv patches in block b attend to y_kv
                  patches in blocks ≤ b

        Returns a bool tensor [2P, 2P], True = allowed to attend.
        """
        n     = num_patches
        total = 2 * n
        q_idx  = torch.arange(total).unsqueeze(1)   # [2P, 1]
        kv_idx = torch.arange(total).unsqueeze(0)   # [1, 2P]

        x0_flag_q  = q_idx  >= n   # True → y_kv half
        x0_flag_kv = kv_idx >= n

        # Patch index within its own half (0..P-1), then floor-divide by block_size
        patch_q  = torch.where(x0_flag_q,  q_idx  - n, q_idx)
        patch_kv = torch.where(x0_flag_kv, kv_idx - n, kv_idx)
        block_q  = patch_q  // block_size
        block_kv = patch_kv // block_size

        m_bd  = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
        m_obc = (block_q >  block_kv) & x0_flag_kv & ~x0_flag_q
        m_bc  = (block_q >= block_kv) & x0_flag_kv &  x0_flag_q

        return m_bd | m_obc | m_bc   # True = can attend

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
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
            if block.use_causal_block_attn:
                nn.init.xavier_uniform_(block.sa_qkv_proj.weight)
                nn.init.zeros_(block.sa_out_proj.weight)
                nn.init.zeros_(block.sa_out_proj.bias)
        for head in [self.final_layer, self.delta_head]:
            nn.init.zeros_(head.adaLN[-1].weight)
            nn.init.zeros_(head.adaLN[-1].bias)
            nn.init.zeros_(head.linear.weight)
            nn.init.zeros_(head.linear.bias)

    def forward(
        self,
        input: Tensor,
        t: Tensor,
        cond: Tensor,
        tau: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
        sample_mode: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input:       [B, L, C]  where
                           block training  → L = 2 * max_prediction_length
                                              (cat of y_t | y_kv halves)
                           sample_mode     → L = b_end  (growing window)
            t:           [B, L]  per-position timesteps; encoding mirrors L
            cond:        [B, C, P_ctx, D_cond]  encoder output
            tau, delta:  optional pre-computed de-stationary parameters for the input
            sample_mode: if True, use growing-window inference.
                         Both training halves are patchified separately from
                         position 0, so past clean blocks in the growing window
                         naturally share the same positional indices as y_kv
                         blocks in training.
        Returns:
            (y0_pred, delta_pred) each [B, L_out, C]
              training: L_out = 2 * max_prediction_length
              sample_mode: L_out = b_end (caller slices [b_start:b_end])
        """
        B, L, C = input.shape

        t_rep = repeat(t, "b l -> (b c) l", c=C)
        tau = rearrange(tau, "b 1 c -> (b c) 1") if tau is not None else None
        delta = rearrange(delta, "b l c -> (b c) l") if delta is not None else None
        if self.use_causal_block_attn:
            # Training: split [y_t | y_kv] and patchify each half from position 0.
            # y_kv block b gets the same positional indices as block b in the
            # growing-window inference, keeping train/inference consistent.
            attn_mask = self.block_diff_mask
            if sample_mode:
                h = self.patch_embedding(input)
                attn_mask = attn_mask[self.num_target_patches:self.num_target_patches+h.shape[1],
                                      self.num_target_patches:self.num_target_patches+h.shape[1]]
                h = h + self.positional_embedding(h)
            else:
                h_yt  = self.patch_embedding(input[:, :L // 2])
                h_ykv = self.patch_embedding(input[:, L // 2:])
                pos_emb = self.positional_embedding(h_yt)
                h_yt = h_yt + pos_emb
                h_ykv = h_ykv + pos_emb
                h = torch.cat([h_yt, h_ykv], dim=1)
        else:
            # Inference growing window: patchify [0..b_end] from position 0.
            attn_mask = None
            h = self.patch_embedding(input)  # [B*C, P_in, D]
            h = h + self.positional_embedding(h)

        cond_emb = self.context_proj(rearrange(cond, "b c p d -> (b c) p d"))
        c = self.timestep_emb(t_rep)    # [B*C, P_in, D]
        for block in self.blocks:
            h = block(h, c, cond_emb, attn_mask=attn_mask, tau=tau, delta=delta)

        y0_out = self.final_layer(h, c).flatten(-2)
        delta_out = self.delta_head(h, c).flatten(-2)

        y0_out = rearrange(y0_out, "(b c) l -> b l c", b=B, c=C)
        delta_out = rearrange(delta_out, "(b c) l -> b l c", b=B, c=C)
        y0_out = y0_out * std + mean if (mean is not None and std is not None) else y0_out
        delta_out = delta_out * std + mean if (mean is not None and std is not None) else delta_out
        return y0_out, delta_out
