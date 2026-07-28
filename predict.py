"""
Inference / evaluation script for a trained FFT-Diffusion model.

Usage:
    python predict.py --checkpoint checkpoints/best.pt --csv data/ETTh1.csv
    python predict.py --checkpoint checkpoints/best.pt --csv data/ETTh1.csv --num_samples 5
    python predict.py --checkpoint checkpoints/best.pt --csv data/ETTh1.csv --use_auto_stop
    python predict.py --checkpoint checkpoints/best.pt --csv data/ETTh1.csv --save_predictions --output_dir results/
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.dataset import load_csv_dataset
from src.diffusion import FFTDiffusion
from src.metrics import compute_metrics
from src.revin import RevIN
from src.early_stopping import find_global_early_stop


def parse_args():
    parser = argparse.ArgumentParser(description="FFT-Diffusion inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (best.pt or epoch_N.pt)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV (overrides config)")
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--prediction_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_sampling_steps", type=int, default=None,
                        help="Override sampling steps (fewer = faster)")

    # Early stopping during sampling
    parser.add_argument("--use_auto_stop", action="store_true", default=False)
    parser.add_argument("--num_pilot_samples", type=int, default=128)
    parser.add_argument("--auto_stop_safety_margin", type=int, default=0)

    # Output
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def load_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model, config, and optional RevIN from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config
    if "config" in ckpt:
        cfg = Config.from_dict(ckpt["config"])
    else:
        raise ValueError("Checkpoint missing 'config' key. "
                         "Use a checkpoint saved by train.py.")

    mcfg = cfg.model

    # Build model
    model = FFTDiffusion.from_config(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load RevIN if present
    revin = None
    if mcfg.use_revin and "revin" in ckpt:
        revin = RevIN(
            num_features=mcfg.num_features,
            in_len=mcfg.context_length,
            out_len=model.condition_encoder.num_patches,
            affine=mcfg.revin_affine,
            use_de_stationary=mcfg.use_de_stationary,
            p_hidden_dims=mcfg.p_hidden_dims,
            p_hidden_layers=mcfg.p_hidden_layers,
        ).to(device)
        revin.load_state_dict(ckpt["revin"], strict=False)

    return model, revin, cfg


@torch.no_grad()
def run_evaluation(
    model,
    loader: DataLoader,
    device: torch.device,
    revin: Optional[RevIN] = None,
    num_samples: int = 1,
    stop_step: Optional[int] = None,
    eval_length: Optional[int] = None,

):
    """Run evaluation, returning metrics + raw arrays."""
    model.eval()
    all_preds, all_targets, all_samples = [], [], []

    for ctx, tgt in loader:
        ctx, tgt = ctx.to(device), tgt.to(device)

        tau, delta = None, None
        if revin is not None:
            ctx, tau, delta = revin(ctx, mode="norm")

        pred = model.sample(ctx, num_samples=num_samples, stop_step=stop_step,
                            eval_prediction_length=eval_length,
                            tau=tau, delta=delta)

        if revin is not None:
            # RevIN stats are [B, 1, C]; repeat to [B*N, 1, C] for all ensemble members
            revin.mean = revin.mean.repeat_interleave(num_samples, dim=0)
            revin.stdev = revin.stdev.repeat_interleave(num_samples, dim=0)
            pred = revin(pred, mode="denorm")
        pred = rearrange(pred, "(b n) l c -> b n l c", n=num_samples)

        all_preds.append(pred.mean(1).cpu())
        all_targets.append(tgt.cpu())
        all_samples.append(pred.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    candidates = torch.cat(all_samples, dim=0)
    metrics = compute_metrics(preds, targets, samples=candidates)
    return metrics, preds.numpy(), targets.numpy()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model, revin, cfg = load_from_checkpoint(args.checkpoint, device)
    mcfg = cfg.model

    # Override from CLI
    if args.csv is not None:
        cfg.data.csv = args.csv
    if args.target_col is not None:
        cfg.data.target_col = args.target_col
    if args.context_length is not None:
        mcfg.context_length = args.context_length
    if args.prediction_length is not None:
        mcfg.prediction_length = args.prediction_length
    if args.num_sampling_steps is not None:
        model.num_sampling_steps = args.num_sampling_steps

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_csv_dataset(
        cfg.data.csv,
        context_length=mcfg.context_length,
        prediction_length=mcfg.prediction_length,
        target_col=cfg.data.target_col,
    )
    print(f"Train samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=device.type == "cuda",
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=device.type == "cuda")

    # ── Auto-stop calibration ─────────────────────────────────────────────
    stop_step = None
    if args.use_auto_stop:
        print("Calibrating auto-stop from pilot samples...")
        pilot_eval_count = 0
        all_scores = []
        test_dtl = iter(test_loader)
        while pilot_eval_count < args.num_pilot_samples:
            pilot_ctx, _ = next(test_dtl)
            pilot_ctx = pilot_ctx[:min(args.batch_size, args.num_pilot_samples-pilot_eval_count)].to(device)
            tau, delta = None, None
            if revin is not None:
                pilot_ctx, tau, delta = revin(pilot_ctx, mode="norm")
            _, scores, steps = model.sample(
                pilot_ctx, num_samples=args.num_samples, return_scores=True, tau=tau, delta=delta,
            )
            pilot_eval_count += args.batch_size
            all_scores.append(scores)
        all_scores = torch.cat(all_scores, dim=0)
        es = find_global_early_stop(
            scores=all_scores, steps=steps,
            smooth_window=None,
            min_frac=0.1,
            safety_margin=args.auto_stop_safety_margin,
        )
        es = find_global_early_stop(
            scores=all_scores, steps=steps,
            smooth_window=None,
            min_frac=0.1,
            safety_margin=args.auto_stop_safety_margin,
        )
        stop_step = es["stop_step"]
        print("Validation auto-stop from pilot ground-truth...")
        pilot_eval_count = 0
        all_scores = []
        test_dtl = iter(val_loader)
        while pilot_eval_count < args.num_pilot_samples:
            pilot_ctx, pilot_tgt = next(test_dtl)
            pilot_ctx = pilot_ctx[:min(args.batch_size, args.num_pilot_samples-pilot_eval_count)].to(device)
            pilot_tgt = pilot_tgt[:min(args.batch_size, args.num_pilot_samples-pilot_eval_count)].to(device)
            tau, delta = None, None
            if revin is not None:
                pilot_ctx, tau, delta = revin(pilot_ctx, mode="norm")
                pilot_tgt, _, _ = revin(tgt, mode="norm", reuse_stats=True)
            _, scores, steps = model.sample(
                pilot_ctx, num_samples=args.num_samples, return_scores=True, tau=tau, delta=delta, tgt=pilot_tgt,
            )
            pilot_eval_count += args.batch_size
            all_scores.append(scores)
        all_scores = torch.cat(all_scores, dim=0)
        es_val = find_global_early_stop(
            scores=all_scores, steps=steps,
            smooth_window=None,
            min_frac=0.1,
            safety_margin=args.auto_stop_safety_margin,
        )
        stop_step_val = es_val["stop_step"]
        print(f"  Auto-stop: step={stop_step} (breakpoint={es['breakpoint_step']:.0f})")
        print(f"  Validation Auto-stop: step={stop_step_val} (breakpoint={es_val['breakpoint_step']:.0f})")

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("Running evaluation...")
    t0 = time.time()
    metrics, preds, targets = run_evaluation(
        model, test_loader, device, revin=revin,
        num_samples=args.num_samples, stop_step=stop_step,
    )
    elapsed = time.time() - t0

    print(f"\nResults ({len(test_ds)} samples, {elapsed:.1f}s):")
    print(f"  MSE  = {metrics['mse']:.6f}")
    print(f"  MAE  = {metrics['mae']:.6f}")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  CRPS = {metrics['crps']:.6f}")

    # ── Save ──────────────────────────────────────────────────────────────
    if args.save_predictions:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "predictions.npy", preds)
        np.save(out_dir / "targets.npy", targets)
        print(f"\nPredictions saved to {out_dir}/")


if __name__ == "__main__":
    main()
