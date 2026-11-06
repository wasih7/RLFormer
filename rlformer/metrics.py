"""Image-quality metrics: MSE, PSNR, SSIM (similarity to a reference) and CNR, gCNR (contrast)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def mse(img: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean((np.asarray(img, dtype=np.float64) - np.asarray(ref, dtype=np.float64)) ** 2))


def psnr(img: np.ndarray, ref: np.ndarray, eps: float = 1e-6) -> float:
    """PSNR in dB for images in [0, 1]; ``eps`` in the MSE caps identical images at 60 dB."""
    return float(10 * np.log10(1.0 / (mse(img, ref) + eps)))


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim(img: np.ndarray, ref: np.ndarray, window: int = 11, sigma: float = 1.5, data_range: float = 1.0) -> float:
    """Mean structural similarity (Wang et al., 2004) with a Gaussian window."""
    x = torch.tensor(img, dtype=torch.float64)[None, None]
    y = torch.tensor(ref, dtype=torch.float64)[None, None]
    w = _gaussian_window(window, sigma)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mu_x ** 2
    syy = F.conv2d(y * y, w) - mu_y ** 2
    sxy = F.conv2d(x * y, w) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return float(ssim_map.mean())


def region_mask(grid: np.ndarray, x_range, z_range) -> np.ndarray:
    """Boolean (nz, nx) mask of the pixels inside ``x_range`` x ``z_range`` (metres)."""
    x, z = grid[..., 0], grid[..., 2]
    return (x >= x_range[0]) & (x <= x_range[1]) & (z >= z_range[0]) & (z <= z_range[1])


def cnr(envelope: np.ndarray, fg: np.ndarray, bg: np.ndarray) -> float:
    """Contrast-to-noise ratio |mu_fg - mu_bg| / sqrt((var_fg + var_bg) / 2) on the envelope."""
    a, b = np.asarray(envelope)[fg], np.asarray(envelope)[bg]
    return float(abs(a.mean() - b.mean()) / np.sqrt(0.5 * (a.var() + b.var()) + 1e-12))


def gcnr(envelope: np.ndarray, fg: np.ndarray, bg: np.ndarray, bins: int = 256) -> float:
    """Generalized CNR (Rodriguez-Molares et al., 2020): one minus the overlap of the two histograms."""
    a, b = np.asarray(envelope)[fg], np.asarray(envelope)[bg]
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    return float(1.0 - np.sum(np.minimum(ha, hb)) * (edges[1] - edges[0]))
