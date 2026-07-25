"""Configuration dataclasses with YAML serialization."""

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _from_dict(cls, d: Dict[str, Any]):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    csv: str = "data/ETTh1.csv"
    target_col: Optional[str] = None




# ═════════════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    # Input/Output
    context_length: int = 96
    prediction_length: int = 1056
    num_features: int = 1  # determined at runtimes

    # RevIN
    use_revin: bool = True
    revin_affine: bool = False

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

    # Common
    dropout: float = 0.1

    # De-stationary (learned projector for tau/delta)
    use_de_stationary: bool = False
    p_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    p_hidden_layers: int = 2

    # Diffusion (FFT-based f-DM)
    num_diffusion_steps: int = 50
    num_sampling_steps: int = 10
    beta_start: float = 1e-4
    beta_end: float = 0.1
    noise_schedule: Literal["linear", "cosine", "quad", "sigmoid", "exp"] = "linear"
    eta: float = 1.0  # DDIM eta

    # Bernoulli timestep sampling (DDPM-BS, Zheng et al. 2024)
    # When enabled, training timesteps are sampled non-uniformly via a
    # Bernoulli gate: with prob p sample from [t_delta, T-1] (high-noise),
    # with prob 1-p sample from [1, t_delta] (low-noise).
    # t_delta is derived dynamically from the early-stopping stop_step.
    bernoulli_sampling: bool = False
    bernoulli_p: float = 0.7

# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    test_batch_size: int = 1
    num_epochs: int = 50
    gradient_clip: float = 1.0

    # LR schedule
    lr_scheduler: Literal["step", "cosine", "none"] = "cosine"
    lr_step_size: int = 10
    lr_decay_factor: float = 0.5
    warmup_epochs: int = 5

    # AMP (automatic mixed precision)
    use_amp: bool = False
    amp_dtype: str = "float16"  # float16 or bfloat16

    # Training early stopping (patience-based on val loss)
    patience: int = 0  # 0 = disabled

    # Resume from checkpoint
    resume: str = ""  # path to checkpoint, empty = start fresh

    # Data
    num_workers: int = 2
    pin_memory: bool = True

    # Early stopping during diffusion sampling
    use_early_stop: bool = False
    num_pilot_samples: int = 100
    early_stop_safety_margin: int = 0

    # Sampling
    num_samples: int = 1              # ensemble size for evaluation

    # Logging
    log_interval: int = 50
    eval_interval: int = 5
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    seed: int = 42
    device: str = "auto"


# ═════════════════════════════════════════════════════════════════════════════
# Top-level container
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    data: DataConfig = None          # type: ignore[assignment]
    model: ModelConfig = None        # type: ignore[assignment]
    training: TrainingConfig = None  # type: ignore[assignment]


    def __post_init__(self):
        self.data = self.data or DataConfig()
        self.model = self.model or ModelConfig()
        self.training = self.training or TrainingConfig()


    # ── I/O ───────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
        }

    def save_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        """Build from a (possibly nested) dict.

        Supports two formats:
          1. Nested:   {data: {csv: ...}, model: {context_length: ...}, training: {batch_size: ...}}
          2. Flat:     {csv: ..., context_length: ..., batch_size: ...}
        """
        if any(k in raw for k in ("data", "model", "training")):
            data_cfg = _from_dict(DataConfig, raw.get("data", {}))
            model_cfg = _from_dict(ModelConfig, raw.get("model", {}))
            train_cfg = _from_dict(TrainingConfig, raw.get("training", {}))
        else:
            data_cfg = _from_dict(DataConfig, raw)
            model_cfg = _from_dict(ModelConfig, raw)
            train_cfg = _from_dict(TrainingConfig, raw)
        return cls(data=data_cfg, model=model_cfg, training=train_cfg)

    def override(self, overrides: Dict[str, Any]):
        """Apply flat key-value overrides (from CLI args) to matching fields."""
        for key, val in overrides.items():
            if val is None:
                continue
            for sub_cfg in (self.data, self.model, self.training):
                if hasattr(sub_cfg, key):
                    setattr(sub_cfg, key, val)
                    break
