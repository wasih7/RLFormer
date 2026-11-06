"""Label-free, differentiable rewards (Eqs. 7-10): feature sharpness, local texture entropy and
coherence factor. Higher is better for all three."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_LAPLACIAN = ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))


def normalize_image(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale an image to a maximum absolute value of 1 (the scale is treated as a constant)."""
    return image / (image.detach().abs().max() + eps)


def feature_sharpness_reward(image: torch.Tensor) -> torch.Tensor:
    """FSR: variance of the Laplacian divided by its mean absolute value (Eq. 7)."""
    kernel = torch.tensor(_LAPLACIAN, dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    lap = F.conv2d(image[None, None], kernel, padding=1)
    return lap.var(unbiased=False) / (lap.abs().mean() + 1e-6)


def local_texture_entropy_reward(image: torch.Tensor, kernel_size: int = 5, num_bins: int = 16,
                                 sigma: float = 0.1, eps: float = 1e-6) -> torch.Tensor:
    """LTE: negative mean local entropy of soft intensity histograms over k x k neighbourhoods (Eqs. 8-9)."""
    h, w = image.shape
    pad = kernel_size // 2
    padded = F.pad(image[None, None], (pad,) * 4, mode="reflect")
    patches = F.unfold(padded, kernel_size, stride=1).view(kernel_size * kernel_size, h, w)
    lo, hi = image.detach().min(), image.detach().max()
    centers = lo + (hi - lo) * torch.arange(num_bins, dtype=image.dtype, device=image.device) / (num_bins - 1)
    dist = patches.unsqueeze(0) - centers.view(num_bins, 1, 1, 1)  # (bins, k*k, H, W)
    hist = torch.exp(-(dist / sigma) ** 2).sum(dim=1)
    prob = hist / (hist.sum(dim=0, keepdim=True) + eps)
    entropy = -(prob * torch.log(prob + eps)).sum(dim=0)
    return -entropy.mean()


def coherence_factor_reward(channels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """CF: mean over pixels of |sum_j s_j|^2 / (m * sum_j |s_j|^2) for delayed channel data (N, m) (Eq. 10)."""
    m = channels.shape[-1]
    return (channels.sum(dim=-1) ** 2 / (m * (channels ** 2).sum(dim=-1) + eps)).mean()


def unsupervised_reward(image: torch.Tensor, channels: torch.Tensor, weights=(0.2, 0.4, 0.4)):
    """Weighted FSR / LTE / CF reward. ``image`` is the (nz, nx) beamformed image, ``channels`` the
    delayed channel data (N, m) of the final step. Returns (reward, components)."""
    fsr = feature_sharpness_reward(normalize_image(image))
    lte = local_texture_entropy_reward(normalize_image(image))
    cf = coherence_factor_reward(channels)
    total = weights[0] * fsr + weights[1] * lte + weights[2] * cf
    return total, {"fsr": fsr.detach(), "lte": lte.detach(), "cf": cf.detach()}
