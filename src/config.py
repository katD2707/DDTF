"""Configuration dataclasses."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    # Input/Output
    context_length: int = 96
    prediction_length: int = 96
    num_features: int = 1  # determined at runtime

    # Patch embedding
    patch_size: int = 24

    # Condition encoder (simple transformer)
    d_model_encoder: int = 128
    n_encoder_layers: int = 2
    n_heads: int = 4

    # Denoiser (DiT-style)
    d_model_denoiser: int = 128
    n_denoiser_layers: int = 2
    mlp_ratio: int = 4

    # Denoiser type
    denoiser_type: Literal["dit", "csdi"] = "dit"

    # Common
    dropout: float = 0.1

    # Diffusion (FFT-based f-DM)
    num_diffusion_steps: int = 50
    num_sampling_steps: int = 10
    beta_start: float = 1e-4
    beta_end: float = 0.1
    noise_schedule: Literal["linear", "cosine", "quad", "sigmoid"] = "linear"
    num_fft_stages: int = 8
    stage_schedule: Literal["linear", "cosine", "quadratic", "sqrt"] = "linear"
    topk_schedule: Literal["linear", "cosine", "quadratic", "sqrt"] = "linear"
    eta: float = 1.0  # DDIM eta


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    num_epochs: int = 50
    gradient_clip: float = 1.0

    # LR schedule
    lr_scheduler: Literal["step", "cosine", "none"] = "cosine"
    lr_step_size: int = 10
    lr_decay_factor: float = 0.5
    warmup_epochs: int = 5

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.99

    # Data
    num_workers: int = 2
    pin_memory: bool = True

    # Early stopping during diffusion sampling
    use_early_stop: bool = False
    num_pilot_samples: int = 100
    early_stop_safety_margin: int = 0

    # Logging
    log_interval: int = 50
    eval_interval: int = 5
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    device: str = "auto"
