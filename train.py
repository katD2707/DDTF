"""
Training script for FFT-Diffusion time series forecasting.

Usage:
    python train.py --csv data/ETTh1.csv --epochs 50 --batch_size 64
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

from src.config import ModelConfig, TrainingConfig
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train FFT-Diffusion model")
    # Data
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--target_col", type=str, default=None, help="Target column (univariate)")
    # Model
    parser.add_argument("--context_length", type=int, default=96)
    parser.add_argument("--prediction_length", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=24)
    parser.add_argument("--d_model_encoder", type=int, default=128)
    parser.add_argument("--d_model_denoiser", type=int, default=128)
    parser.add_argument("--n_encoder_layers", type=int, default=2)
    parser.add_argument("--n_denoiser_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--num_sampling_steps", type=int, default=10)
    parser.add_argument("--num_fft_stages", type=int, default=8)
    parser.add_argument("--stage_schedule", type=str, default="linear")
    parser.add_argument("--topk_schedule", type=str, default="linear")
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--denoiser_type", type=str, default="dit", choices=["dit", "csdi"])
    # Early stopping (inference)
    parser.add_argument("--use_early_stop", action="store_true", help="Calibrate early stopping before test")
    parser.add_argument("--num_pilot_samples", type=int, default=100)
    parser.add_argument("--early_stop_safety_margin", type=int, default=0)
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["step", "cosine", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_csv_dataset(
        args.csv,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        target_col=args.target_col,
    )
    num_channels = train_ds.num_channels
    print(f"Data: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test  "
          f"| channels={num_channels}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = FFTDiffusion(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        patch_size=args.patch_size,
        d_model_encoder=args.d_model_encoder,
        d_model_denoiser=args.d_model_denoiser,
        n_encoder_layers=args.n_encoder_layers,
        n_denoiser_layers=args.n_denoiser_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        num_diffusion_steps=args.num_diffusion_steps,
        num_sampling_steps=args.num_sampling_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        noise_schedule=args.noise_schedule,
        num_fft_stages=args.num_fft_stages,
        stage_schedule=args.stage_schedule,
        topk_schedule=args.topk_schedule,
        eta=args.eta,
        denoiser_type=args.denoiser_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # EMA copy
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    if args.lr_scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.save_interval, gamma=0.5)
    else:
        scheduler = None

    # ── Training loop ─────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_mse = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for ctx, tgt in pbar:
            ctx, tgt = ctx.to(device), tgt.to(device)
            loss_dict = model(tgt, ctx)
            loss = loss_dict["total"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            if scheduler is not None and args.lr_scheduler == "cosine":
                scheduler.step()

            update_ema(ema_model, model, args.ema_decay)
            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if scheduler is not None and args.lr_scheduler == "step":
            scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:>3d} | loss={avg_loss:.4f} | lr={lr_now:.2e}")

        # Validation
        if epoch % args.eval_interval == 0:
            stop_step = None
            if args.use_early_stop:
                # Calibrate on a pilot batch from validation set
                pilot_ctx, _ = next(iter(val_loader))
                pilot_ctx = pilot_ctx[:args.num_pilot_samples].to(device)
                es_result = ema_model.find_optimal_stop_step(
                    pilot_ctx,
                    num_samples=1,
                    safety_margin=args.early_stop_safety_margin,
                )
                stop_step = es_result["stop_step"]
                print(f"Early stopping calibrated: stop_step={stop_step}")
            
            metrics = evaluate(ema_model, val_loader, device, stop_step=stop_step)
            print(f"  ↳ Val  MAE={metrics['mae']:.4f}  MSE={metrics['mse']:.4f}")
            if metrics["mse"] < best_val_mse:
                best_val_mse = metrics["mse"]
                torch.save(ema_model.state_dict(), os.path.join(args.checkpoint_dir, "best.pt"))
                print("  ↳ Saved best checkpoint")

        # Periodic save
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt"))

    # ── Test ──────────────────────────────────────────────────────────────
    ema_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "best.pt"),
                                         map_location=device, weights_only=True))

    stop_step = None
    if args.use_early_stop:
        # Calibrate on a pilot batch from validation set
        pilot_ctx, _ = next(iter(val_loader))
        pilot_ctx = pilot_ctx[:args.num_pilot_samples].to(device)
        es_result = ema_model.find_optimal_stop_step(
            pilot_ctx,
            num_samples=1,
            safety_margin=args.early_stop_safety_margin,
        )
        stop_step = es_result["stop_step"]
        print(f"Early stopping calibrated: stop_step={stop_step}")

    test_metrics = evaluate(ema_model, test_loader, device, stop_step=stop_step)
    print(f"\nTest  MAE={test_metrics['mae']:.4f}  MSE={test_metrics['mse']:.4f}")


if __name__ == "__main__":
    main()
