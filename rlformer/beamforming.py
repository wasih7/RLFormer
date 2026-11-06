"""Reference beamformers (constant speed of sound) and image formation helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import PlaneWaveData


def delay_plane(grid: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Transmit path length of a plane wave to every pixel: x sin(theta) + z cos(theta). grid: (P, 3)."""
    return grid[:, 0] * torch.sin(angle) + grid[:, 2] * torch.cos(angle)


@torch.no_grad()
def das_image(data: PlaneWaveData, grid: np.ndarray, angle_idxs=None, fnum: float | None = None,
              element_chunk: int = 32, device=None, dtype=torch.float32) -> torch.Tensor:
    """Delay-and-sum with the nominal speed of sound; coherent compounding for several angles.

    Returns the complex image I + jQ of shape ``grid.shape[:2]``. No apodization is applied unless a
    receive f-number is given (rectangular receive aperture with |dz / dx| >= fnum).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    angle_idxs = list(range(data.n_angles)) if angle_idxs is None else list(angle_idxs)
    g = torch.tensor(grid, dtype=dtype, device=device).reshape(-1, 3)
    ele = torch.tensor(data.ele_pos, dtype=dtype, device=device)
    angles = torch.tensor(data.angles, dtype=dtype, device=device)
    t0 = torch.tensor(data.time_zero, dtype=dtype, device=device)
    idata = torch.tensor(data.idata, dtype=dtype, device=device)
    qdata = torch.tensor(data.qdata, dtype=dtype, device=device)
    n_samples = idata.shape[-1]
    img_i = torch.zeros(g.shape[0], dtype=dtype, device=device)
    img_q = torch.zeros_like(img_i)
    for t in angle_idxs:
        tx = delay_plane(g, angles[t]) + t0[t] * data.c
        for s in range(0, data.n_elements, element_chunk):
            sl = slice(s, min(s + element_chunk, data.n_elements))
            diff = g.unsqueeze(0) - ele[sl].unsqueeze(1)  # (m, P, 3)
            rx = torch.norm(diff, dim=-1)
            coord = ((tx.unsqueeze(0) + rx) * data.fs / data.c * 2 + 1) / n_samples - 1
            sample_grid = torch.stack((coord, torch.zeros_like(coord)), dim=-1).unsqueeze(1)
            iq = torch.stack((idata[t, sl], qdata[t, sl]), dim=1).unsqueeze(2)  # (m, 2, 1, Ns)
            out = F.grid_sample(iq, sample_grid, align_corners=False)[:, :, 0, :]  # (m, 2, P)
            if fnum is not None:
                mask = (diff[..., 2].abs() / diff[..., 0].abs().clamp_min(1e-12) >= fnum).to(dtype)
                out = out * mask.unsqueeze(1)
            img_i += out[:, 0].sum(0)
            img_q += out[:, 1].sum(0)
    return torch.complex(img_i, img_q).reshape(grid.shape[:2])


def analytic_envelope(rf_image: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Envelope of a real beamformed RF image via the analytic signal along the axial dimension."""
    n = rf_image.shape[dim]
    spectrum = torch.fft.fft(rf_image, dim=dim)
    h = torch.zeros(n, dtype=spectrum.real.dtype, device=rf_image.device)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    shape = [1] * rf_image.dim()
    shape[dim] = n
    return torch.fft.ifft(spectrum * h.view(shape), dim=dim).abs()


def bmode(envelope, dynamic_range: float = 60.0) -> np.ndarray:
    """Log-compressed B-mode in dB, normalized to its maximum and clipped to ``-dynamic_range``."""
    env = np.abs(envelope.detach().cpu().numpy() if torch.is_tensor(envelope) else np.asarray(envelope))
    db = 20 * np.log10(env / (env.max() + 1e-30) + 1e-12)
    return np.maximum(db, -dynamic_range)


def to_unit_range(bmode_db: np.ndarray, dynamic_range: float = 60.0) -> np.ndarray:
    """Map a B-mode image in [-dynamic_range, 0] dB to [0, 1] (used for the similarity metrics)."""
    return (np.clip(bmode_db, -dynamic_range, 0.0) + dynamic_range) / dynamic_range
