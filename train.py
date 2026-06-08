"""
Training script for FFT-Diffusion time series forecasting.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --batch_size 32
    python train.py --config config.yaml --csv data/weather.csv --num_epochs 100
    python train.py --config config.yaml --resume checkpoints/epoch_20.pt
"""

import argparse
import math
import os
import sys
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Ensure the repo root is on the path so `src.*` imports work when
# running from inside the fft_diff/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.dataset import load_csv_dataset
from src.diffusion import FFTDiffusion
from src.metrics import compute_metrics, MetricTracker
from src.revin import RevIN


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    revin: Optional[RevIN] = None,
    num_samples: int = 1,
    stop_step: Optional[int] = None,
    eval_length: Optional[int] = None,
    use_early_stop: bool = False,
    early_stop_safety_margin: int = 0,
) -> dict:
    """
    Evaluate model on a dataloader.  When RevIN is provided the context
    is normalized before sampling and predictions are de-normalized before
    computing metrics so that scores are in original scale.
    """
    model.eval()
    all_preds, all_targets = [], []
    for ctx, tgt in loader:
        ctx, tgt = ctx.to(device), tgt.to(device)

        tau, delta = None, None
        if revin is not None:
            _, tau, delta = revin(ctx, mode="norm")

        pred = model.sample(ctx, num_samples=num_samples, stop_step=stop_step,
                             eval_prediction_length=eval_length,
                             use_early_stop=use_early_stop,
                             early_stop_safety_margin=early_stop_safety_margin,
                             tau=tau, delta=delta)

        # if revin is not None:
        #     pred = revin(pred, mode="denorm")
            # tgt is already in original scale (not normalized)

        all_preds.append(pred.cpu())
        all_targets.append(tgt.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return compute_metrics(preds, targets)


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
    parser.add_argument("--use_revin", action="store_true", default=None)
    parser.add_argument("--use_de_stationary", action="store_true", default=None,
                        help="Enable learned de-stationary projector (tau/delta)")
    parser.add_argument("--num_diffusion_steps", type=int, default=None)
    parser.add_argument("--num_sampling_steps", type=int, default=None)
    parser.add_argument("--num_fft_stages", type=int, default=None)
    parser.add_argument("--stage_schedule", type=str, default=None)
    parser.add_argument("--topk_schedule", type=str, default=None)
    parser.add_argument("--noise_schedule", type=str, default=None)
    parser.add_argument("--beta_start", type=float, default=None)
    parser.add_argument("--beta_end", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--transform_type", type=str, default=None,
                        choices=["fft", "ma"])
    parser.add_argument("--ma_kernel_size", type=int, default=None)
    # Block diffusion
    parser.add_argument("--block_diffusion_mode", type=str, default=None,
                        choices=["full", "fixed"])
    parser.add_argument("--block_size", type=int, default=None)

    # Training overrides
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--gradient_clip", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        choices=["step", "cosine", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--use_amp", action="store_true", default=None)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_early_stop", action="store_true", default=None)
    parser.add_argument("--num_pilot_samples", type=int, default=None)
    parser.add_argument("--early_stop_safety_margin", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
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
        eval_prediction_length=tcfg.eval_prediction_length,
        target_col=dcfg.target_col,
        perturbation_cfg=cfg.perturbation,
    )
    mcfg.num_features = train_ds.num_channels
    print(f"Data: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test  "
          f"| channels={mcfg.num_features}")

    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                              num_workers=tcfg.num_workers, pin_memory=tcfg.pin_memory,
                              drop_last=True)
    val_loader = DataLoader(test_ds, batch_size=tcfg.batch_size, shuffle=False,
                            num_workers=tcfg.num_workers, pin_memory=tcfg.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=tcfg.batch_size, shuffle=False,
                             num_workers=tcfg.num_workers, pin_memory=tcfg.pin_memory)

    # ── Model ─────────────────────────────────────────────────────────────
    model = FFTDiffusion.from_config(mcfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── RevIN ─────────────────────────────────────────────────────────────
    revin = None
    if mcfg.use_revin:
        revin = RevIN(num_features=mcfg.num_features, in_len=mcfg.context_length, out_len=model.condition_encoder.num_patches,
                      affine=mcfg.revin_affine, use_de_stationary=mcfg.use_de_stationary,
                      p_hidden_dims=mcfg.p_hidden_dims, p_hidden_layers=mcfg.p_hidden_layers).to(device)
        print("RevIN enabled")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    params = list(model.parameters())
    if revin is not None and mcfg.revin_affine:
        params += list(revin.parameters())
    optimizer = torch.optim.AdamW(params, lr=tcfg.learning_rate,
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

    # ── AMP ───────────────────────────────────────────────────────────────
    amp_enabled = tcfg.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if tcfg.amp_dtype == "bfloat16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled)
    if amp_enabled:
        print(f"AMP enabled (dtype={tcfg.amp_dtype})")

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 1
    best_val_mse = float("inf")
    global_step = 0

    if tcfg.resume:
        if os.path.isfile(tcfg.resume):
            print(f"Resuming from {tcfg.resume}")
            ckpt = torch.load(tcfg.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if amp_enabled and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            if revin is not None and "revin" in ckpt:
                revin.load_state_dict(ckpt["revin"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_mse = ckpt.get("best_val_mse", float("inf"))
            global_step = ckpt.get("global_step", 0)
            print(f"  Resumed at epoch {start_epoch}, best_val_mse={best_val_mse:.6f}")
        else:
            print(f"Warning: resume path '{tcfg.resume}' not found, starting fresh")

    # ── Save effective config ─────────────────────────────────────────────
    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
    cfg.save_yaml(os.path.join(tcfg.checkpoint_dir, "config.yaml"))
    print(f"Config saved to {tcfg.checkpoint_dir}/config.yaml")

    # ── Training early stopping (patience) ────────────────────────────────
    patience_counter = 0

    # ── Training loop ─────────────────────────────────────────────────────
    tracker = MetricTracker()

    for epoch in range(start_epoch, tcfg.num_epochs + 1):
        model.train()
        tracker.reset()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tcfg.num_epochs}", leave=False)
        for ctx, tgt in pbar:
            ctx, tgt = ctx.to(device), tgt.to(device)

            # RevIN: normalize context and target for training
            if revin is not None:
                _, tau, delta = revin(ctx, mode="norm")

            optimizer.zero_grad()

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss_dict = model(tgt, ctx, tau, delta) if revin is not None else model(tgt, ctx)
                loss = loss_dict["total"]

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.gradient_clip)
                optimizer.step()

            if scheduler is not None and tcfg.lr_scheduler == "cosine":
                scheduler.step()

            tracker.update({"loss": loss.item()})
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if scheduler is not None and tcfg.lr_scheduler == "step":
            scheduler.step()

        avg_loss = tracker.get_average("loss")
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:>3d} | loss={avg_loss:.4f} | lr={lr_now:.2e}")

        # ── Validation ────────────────────────────────────────────────────
        if epoch % tcfg.eval_interval == 0:
            stop_step = None
            if tcfg.use_early_stop and mcfg.block_diffusion_mode == "full":
                pilot_ctx, _ = next(iter(val_loader))
                pilot_ctx = pilot_ctx[:tcfg.num_pilot_samples].to(device)
                if revin is not None:
                    pilot_ctx, tau, delta = revin(pilot_ctx, mode="norm")
                es_result = model.find_optimal_stop_step(
                    pilot_ctx, num_samples=tcfg.num_samples,
                    safety_margin=tcfg.early_stop_safety_margin,
                    tau=tau, delta=delta,
                )
                stop_step = es_result["stop_step"]
                print(f"  ↳ Early stop calibrated: stop_step={stop_step}")

            metrics = evaluate(model, val_loader, device, revin=revin,
                               num_samples=tcfg.num_samples, stop_step=stop_step,
                               eval_length=tcfg.eval_prediction_length,
                               use_early_stop=tcfg.use_early_stop,
                               early_stop_safety_margin=tcfg.early_stop_safety_margin)
            print(f"  ↳ Val  MSE={metrics['mse']:.4f}  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}")

            if metrics["mse"] < best_val_mse:
                best_val_mse = metrics["mse"]
                patience_counter = 0
                # Save best checkpoint (full state for resumability)
                best_ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_mse": best_val_mse,
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                }
                if revin is not None:
                    best_ckpt["revin"] = revin.state_dict()
                torch.save(best_ckpt, os.path.join(tcfg.checkpoint_dir, "best.pt"))
                print("  ↳ Saved best checkpoint")
            else:
                patience_counter += 1

            # Patience-based early stopping
            if tcfg.patience > 0 and patience_counter >= tcfg.patience:
                print(f"  ↳ Early stopping: no improvement for {tcfg.patience} eval cycles")
                break

        # ── Periodic save (full resumable checkpoint) ─────────────────────
        if epoch % tcfg.save_interval == 0:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_mse": best_val_mse,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(),
            }
            if scheduler is not None:
                ckpt["scheduler"] = scheduler.state_dict()
            if amp_enabled:
                ckpt["scaler"] = scaler.state_dict()
            if revin is not None:
                ckpt["revin"] = revin.state_dict()
            torch.save(ckpt, os.path.join(tcfg.checkpoint_dir, f"epoch_{epoch}.pt"))

    # ── Test ──────────────────────────────────────────────────────────────
    best_path = os.path.join(tcfg.checkpoint_dir, "best.pt")
    if os.path.exists(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model"])
        if revin is not None and "revin" in best_ckpt:
            revin.load_state_dict(best_ckpt["revin"])
        print("Loaded best checkpoint for testing")

    stop_step = None
    if tcfg.use_early_stop and mcfg.block_diffusion_mode == "full":
        pilot_ctx, _ = next(iter(val_loader))
        pilot_ctx = pilot_ctx[:tcfg.num_pilot_samples].to(device)
        if revin is not None:
            pilot_ctx, tau, delta = revin(pilot_ctx, mode="norm")
        es_result = model.find_optimal_stop_step(
            pilot_ctx, num_samples=tcfg.num_samples,
            safety_margin=tcfg.early_stop_safety_margin,
            tau=tau, delta=delta,
        )
        stop_step = es_result["stop_step"]
        print(f"Early stop calibrated: stop_step={stop_step}")

    test_metrics = evaluate(model, test_loader, device, revin=revin,
                            num_samples=tcfg.num_samples, stop_step=stop_step,
                            eval_length=tcfg.eval_prediction_length,
                            use_early_stop=tcfg.use_early_stop,
                            early_stop_safety_margin=tcfg.early_stop_safety_margin)
    print(f"\nTest  MSE={test_metrics['mse']:.4f}  MAE={test_metrics['mae']:.4f}  "
          f"RMSE={test_metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
