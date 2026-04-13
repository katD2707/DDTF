import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


def build_repeated_ma_kernel(x, K, N):
    # x: [B, L]
    if N == 0:
        return x

    device, dtype = x.device, x.dtype
    kernel = torch.ones(1, 1, K, device=device, dtype=dtype) / K

    eff_kernel = kernel
    for _ in range(N - 1):
        eff_kernel = F.conv1d(eff_kernel, kernel.flip(-1), padding=K - 1)

    x = x.unsqueeze(1)
    pad = eff_kernel.shape[-1] // 2
    y = F.conv1d(x, eff_kernel, padding=pad)
    return y.squeeze(1)


def apply_precomputed_kernel(x, eff_kernel):
    x = x.unsqueeze(1)
    pad = eff_kernel.shape[-1] // 2
    y = F.conv1d(x, eff_kernel, padding=pad)
    return y.squeeze(1)


# --- Example synthetic time series: linear trend + sine waves + noise ---
torch.manual_seed(0)
n = 200
bs = 2000
t = torch.arange(n).unsqueeze(0)
# trend = 0.01 * torch.randn(bs).unsqueeze(-1) * t
# signal = torch.sin(2 * np.pi * t / 50) + 0.7*torch.sin(2 * np.pi * t / 120)
# x = trend + signal + 0.3 * np.random.randn(n)
trend = 0.7*t + 1.2*np.sin(1.7*t) + 0.35*np.cos(0.5*t**2) + 0.15*(t**2)*np.exp(-0.25*t**2)
sigma = 0.15 + 0.10*np.abs(np.sin(0.8*t)) + 0.05*np.abs(t)
x = trend + torch.randn(n) * sigma
eff_kernel = build_repeated_ma_kernel(K=2, N=1)
y = apply_precomputed_kernel(x, eff_kernel)
