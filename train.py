"""
Training script for FFT-Diffusion time series forecasting.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --batch_size 32 --denoiser_type csdi
    python train.py --config config.yaml --csv data/weather.csv --num_epochs 100
"""

import argparse
import copy
import math
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Ensure the repo root is on the path so `src.*` imports work when
# running from inside the fft_diff/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.dataset import load_csv_dataset
from src.diffusion import FFTDiffusion


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    for ep, mp in zip(ema_model.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(mp.data, alpha=1 - decay)


@torch.no_grad()
def evaluate(model, loader, device, stop_step=None):
    model.eval()
    total_mae, total_mse, n = 0.0, 0.0, 0
    for ctx, tgt in loader:
        ctx, tgt = ctx.to(device), tgt.to(device)
        pred = model.sample(ctx, num_samples=1, stop_step=stop_step)
        total_mae += (pred - tgt).abs().sum().item()
        total_mse += ((pred - tgt) ** 2).sum().item()
        n += tgt.numel()
    return {"mae": total_mae / n, "mse": total_mse / n}


# ── CLI parsing ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FFT-Diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # Data overrides
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--target_col", type=str, default=None)

    # Model overrides
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--prediction_length", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--d_model_encoder", type=int, default=None)
    parser.add_argument("--d_model_denoiser", type=int, default=None)
    parser.add_argument("--n_encoder_layers", type=int, default=None)
    parser.add_argument("--n_denoiser_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--mlp_ratio", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--denoiser_type", type=str, default=None,
                        choices=["dit", "csdi"])
    parser.add_argument("--num_diffusion_steps", type=int, default=None)
    parser.add_argument("--num_sampling_steps", type=int, default=None)
    parser.add_argument("--num_fft_stages", type=int, default=None)
    parser.add_argument("--stage_schedule", type=str, default=None)
    parser.add_argument("--topk_schedule", type=str, default=None)
    parser.add_argument("--noise_schedule", type=str, default=None)
    parser.add_argument("--beta_start", type=float, default=None)
    parser.add_argument("--beta_end", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)

    # Training overrides
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--gradient_clip", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        choices=["step", "cosine", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--use_ema", action="store_true", default=None)
    parser.add_argument("--no_ema", dest="use_ema", action="store_false")
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--use_early_stop", action="store_true", default=None)
    parser.add_argument("--no_early_stop", dest="use_early_stop",
                        action="store_false")
    parser.add_argument("--num_pilot_samples", type=int, default=None)
    parser.add_argument("--early_stop_safety_margin", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def build_config(args) -> Config:
    """Build Config from YAML (if given) then apply CLI overrides."""
    if args.config is not None:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()
    # Apply any non-None CLI values as overrides
    cli_overrides = {k: v for k, v in vars(args).items()
                     if k != "config" and v is not None}
    cfg.override(cli_overrides)
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = build_config(args)

    dcfg = cfg.data
    mcfg = cfg.model
    tcfg = cfg.training

    set_seed(tcfg.seed)

    # Device
    if tcfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(tcfg.device)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_csv_dataset(
        dcfg.csv,
        context_length=mcfg.context_length,
        prediction_length=mcfg.prediction_length,
        target_col=dcfg.target_col,
    )
    mcfg.num_features = train_ds.num_channels
    print(f"Data: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test  "
          f"| channels={mcfg.num_features}")

    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                              num_workers=tcfg.num_workers, pin_memory=tcfg.pin_memory,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False,
                            num_workers=tcfg.num_workers, pin_memory=tcfg.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=tcfg.batch_size, shuffle=False,
                             num_workers=tcfg.num_workers, pin_memory=tcfg.pin_memory)

    # ── Model ─────────────────────────────────────────────────────────────
    model = FFTDiffusion.from_config(mcfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # EMA
    if tcfg.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.eval()
    else:
        ema_model = model

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate,
                                  weight_decay=tcfg.weight_decay)

    total_steps = tcfg.num_epochs * len(train_loader)
    warmup_steps = tcfg.warmup_epochs * len(train_loader)

    if tcfg.lr_scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif tcfg.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=tcfg.lr_step_size, gamma=tcfg.lr_decay_factor)
    else:
        scheduler = None

    # ── Save effective config ─────────────────────────────────────────────
    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
    cfg.save_yaml(os.path.join(tcfg.checkpoint_dir, "config.yaml"))
    print(f"Config saved to {tcfg.checkpoint_dir}/config.yaml")

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_mse = float("inf")
    global_step = 0

    for epoch in range(1, tcfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tcfg.num_epochs}", leave=False)
        for ctx, tgt in pbar:
            ctx, tgt = ctx.to(device), tgt.to(device)
            loss_dict = model(tgt, ctx)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.gradient_clip)
            optimizer.step()

            if scheduler is not None and tcfg.lr_scheduler == "cosine":
                scheduler.step()

            if tcfg.use_ema:
                update_ema(ema_model, model, tcfg.ema_decay)

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if scheduler is not None and tcfg.lr_scheduler == "step":
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:>3d} | loss={avg_loss:.4f} | lr={lr_now:.2e}")

        # Validation
        if epoch % tcfg.eval_interval == 0:
            stop_step = None
            if tcfg.use_early_stop:
                pilot_ctx, _ = next(iter(val_loader))
                pilot_ctx = pilot_ctx[:tcfg.num_pilot_samples].to(device)
                es_result = ema_model.find_optimal_stop_step(
                    pilot_ctx, num_samples=1,
                    safety_margin=tcfg.early_stop_safety_margin,
                )
                stop_step = es_result["stop_step"]
                print(f"  ↳ Early stop calibrated: stop_step={stop_step}")

            metrics = evaluate(ema_model, val_loader, device, stop_step=stop_step)
            print(f"  ↳ Val  MAE={metrics['mae']:.4f}  MSE={metrics['mse']:.4f}")
            if metrics["mse"] < best_val_mse:
                best_val_mse = metrics["mse"]
                torch.save(ema_model.state_dict(),
                           os.path.join(tcfg.checkpoint_dir, "best.pt"))
                print("  ↳ Saved best checkpoint")

        # Periodic save
        if epoch % tcfg.save_interval == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(),
            }
            if tcfg.use_ema:
                ckpt["ema"] = ema_model.state_dict()
            torch.save(ckpt, os.path.join(tcfg.checkpoint_dir, f"epoch_{epoch}.pt"))

    # ── Test ──────────────────────────────────────────────────────────────
    best_path = os.path.join(tcfg.checkpoint_dir, "best.pt")
    if os.path.exists(best_path):
        ema_model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True))

    stop_step = None
    if tcfg.use_early_stop:
        pilot_ctx, _ = next(iter(val_loader))
        pilot_ctx = pilot_ctx[:tcfg.num_pilot_samples].to(device)
        es_result = ema_model.find_optimal_stop_step(
            pilot_ctx, num_samples=1,
            safety_margin=tcfg.early_stop_safety_margin,
        )
        stop_step = es_result["stop_step"]
        print(f"Early stop calibrated: stop_step={stop_step}")

    test_metrics = evaluate(ema_model, test_loader, device, stop_step=stop_step)
    print(f"\nTest  MAE={test_metrics['mae']:.4f}  MSE={test_metrics['mse']:.4f}")


if __name__ == "__main__":
    main()
