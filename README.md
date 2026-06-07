# FFT-Diff

A frequency-domain diffusion model (f-DM) for multivariate time series forecasting.  
Noise is applied progressively in the FFT domain across **K staged degradation steps**, decoupled from the **T diffusion timesteps**, enabling structured, interpretable denoising.

---

## Features

- **FFT staged diffusion** — frequency-domain noise schedule with configurable stage/topk schedules
- **DiT denoiser** — transformer with adaLN-Zero and RoPE
- **DDIM sampling** — fast inference with stage-boundary-aware timestep selection
- **Early stopping inference** — piecewise-linear breakpoint calibration on pilot samples to reduce sampling steps
- **Standard benchmark splits** — fixed train/val/test indices for ETTh1/h2, ETTm1/m2, Weather, ECL, Traffic, Exchange, ILI

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## Quick Start

```bash
# Train with DiT denoiser (default)
python train.py --csv data/ETTh1.csv --epochs 50

# Longer horizon, bigger model
python train.py \
    --csv data/weather.csv \
    --context_length 336 \
    --prediction_length 192 \
    --d_model_encoder 256 \
    --d_model_denoiser 256 \
    --n_encoder_layers 3 \
    --n_denoiser_layers 4 \
    --epochs 100

# Univariate — target a single column
python train.py --csv data/ETTh1.csv --target_col OT

# Enable early stopping at test time
python train.py --csv data/ETTh1.csv --use_early_stop --num_pilot_samples 64
```

---

## All CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--csv` | *(required)* | Path to input CSV file |
| `--target_col` | `None` | Single target column (univariate mode) |
| **Model** | | |
| `--context_length` | `96` | Length of input context window |
| `--prediction_length` | `96` | Length of forecast horizon |
| `--patch_size` | `24` | Patch size for encoder |
| `--d_model_encoder` | `128` | Encoder hidden dimension |
| `--d_model_denoiser` | `128` | Denoiser hidden dimension |
| `--n_encoder_layers` | `2` | Number of encoder layers |
| `--n_denoiser_layers` | `2` | Number of denoiser layers |
| `--n_heads` | `4` | Attention heads |
| `--mlp_ratio` | `4` | MLP expansion ratio (DiT) |
| `--dropout` | `0.1` | Dropout rate |
| **Diffusion** | | |
| `--num_diffusion_steps` | `50` | Total diffusion timesteps T |
| `--num_sampling_steps` | `10` | DDIM sampling steps |
| `--num_fft_stages` | `8` | Number of FFT degradation stages K |
| `--stage_schedule` | `linear` | Stage boundary schedule |
| `--topk_schedule` | `linear` | Top-k frequency schedule |
| `--noise_schedule` | `linear` | Beta schedule: `linear`, `cosine`, `quad`, `sigmoid` |
| `--beta_start` | `1e-4` | Starting beta value |
| `--beta_end` | `0.1` | Ending beta value |
| `--eta` | `1.0` | DDIM eta (0 = deterministic) |
| **Training** | | |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `5e-4` | Learning rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--gradient_clip` | `1.0` | Gradient clipping norm |
| `--lr_scheduler` | `cosine` | LR schedule: `cosine`, `step`, `none` |
| `--warmup_epochs` | `5` | Linear warmup epochs |
| `--num_workers` | `2` | DataLoader workers |
| `--eval_interval` | `5` | Validation every N epochs |
| `--save_interval` | `10` | Checkpoint every N epochs |
| `--checkpoint_dir` | `checkpoints` | Directory for saving checkpoints |
| `--seed` | `42` | Random seed |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| **Early Stopping** | | |
| `--use_early_stop` | `False` | Calibrate early stopping before test evaluation |
| `--num_pilot_samples` | `100` | Pilot batch size for calibration |
| `--early_stop_safety_margin` | `0` | Steps to subtract from detected breakpoint |

---

## Supported Datasets

Datasets are identified by the CSV filename stem (case-insensitive). Fixed standard splits are applied automatically:

| Dataset | Train | Val | Test |
|---|---|---|---|
| ETTh1, ETTh2 | 0 – 8640 | 8640 – 11520 | 11520 – 14400 |
| ETTm1, ETTm2 | 0 – 34560 | 34560 – 46080 | 46080 – 57600 |
| Weather | 0 – 36887 | 36887 – 42157 | 42157 – 52696 |
| Electricity / ECL | 0 – 18412 | 18412 – 21044 | 21044 – 26304 |
| Traffic | 0 – 12280 | 12280 – 14036 | 14036 – 17544 |
| Exchange | 0 – 5311 | 5311 – 6071 | 6071 – 7588 |
| ILI | 0 – 676 | 676 – 773 | 773 – 966 |

Unknown dataset names raise a `ValueError`.

---

## Project Structure

```
fft_diff/
├── train.py                  # Training entry point
├── requirements.txt
└── src/
    ├── config.py             # ModelConfig / TrainingConfig dataclasses
    ├── dataset.py            # TimeSeriesDataset + load_csv_dataset()
    ├── diffusion.py          # FFTDiffusion — main model
    ├── dit_denoiser.py       # DiT denoiser (adaLN-Zero, RoPE)
    ├── embeddings.py         # TimestepEmbedding, PatchEmbedding, RoPE
    ├── early_stopping.py     # find_global_early_stop()
    └── utils.py              # extract(), default()
```

---

## Reference

> **f-DM: A Multi-stage Diffusion Model via Progressive Signal Transformation**  
> Gu et al., 2022 — [arXiv:2210.04955](https://arxiv.org/abs/2210.04955)
