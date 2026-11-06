"""Running a trained policy on an acquisition."""

from __future__ import annotations

import numpy as np
import torch

from .beamforming import analytic_envelope
from .data import PlaneWaveData
from .model import RLFormer


def rf_tensor(data: PlaneWaveData, angle_idx: int, device, dtype=torch.float32) -> torch.Tensor:
    """Single plane-wave RF data as an (n_samples, n_elements) tensor."""
    return torch.tensor(data.idata[angle_idx].T, dtype=dtype, device=device)


@torch.no_grad()
def reconstruct(model: RLFormer, rf_data: torch.Tensor, angle_idx: int | None = None, steps: int = 5) -> dict:
    """Roll out the policy for ``steps`` decisions starting from the homogeneous state.

    Returns the beamformed RF image, its envelope and the learned effective speed-of-sound field,
    all as (nz, nx) tensors.
    """
    c_inv, image = model.initial_state()
    for _ in range(steps):
        c_inv, image = model(c_inv, image, rf_data, angle_idx)
    rf = model.fold_patches(image)
    return {"rf": rf, "envelope": analytic_envelope(rf, dim=0), "sos": model.sos_map(c_inv)}


def load_policy(checkpoint: str, data: PlaneWaveData, grid: np.ndarray, device=None):
    """Build a policy for ``data`` / ``grid`` and load a checkpoint written by ``save_checkpoint``.

    Works across grids (see :meth:`RLFormer.load_policy`). Returns ``(model, mode)`` where mode is
    "exact" or "resampled".
    """
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = RLFormer(data, grid, device=device, **ckpt["config"])
    mode = model.load_policy(ckpt["state_dict"], source_patches=tuple(ckpt["n_patches"]))
    model.eval()
    return model, mode
