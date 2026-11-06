"""Policy network of RLFormer: a patch-wise MLP-mixer that refines the slowness field (D2DNN)
followed by an adaptive delay-and-sum beamformer with learnable per-pixel channel weights.

State   s_t     = (1/c_t, image_t), both stored as flat (N, 1) tensors in patch order.
Action  a_t     = (1/c_{t+1}, image_{t+1}).

For every pixel (x, z) the receive/transmit delays follow the hyperbola of Eq. (1),
    t(x_i) = k + b * sqrt(1 + (x_i - h)^2 / a^2) + t0,     h = x, a = z,
with b = z / c(x, z) and k = d_TX / c(x, z), d_TX = x sin(theta) + z cos(theta), so that
changing the local sound speed c(x, z) reshapes the delay curve of that pixel.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PlaneWaveData


class D2DNN(nn.Module):
    """Delay-to-delay network: one MLP-mixer layer over the pixels of a patch.

    Input  : pixel features (x, z, 1/c_t, d_TX/c_t), shape (..., n_points, 4)
    Output : updated slowness 1/c_{t+1}, shape (..., n_points, 1), with
             c_{t+1} = c_t + max_step * tanh(residual)   [m/s].
    """

    def __init__(self, num_points: int, num_features: int = 4, hidden_dim: int = 16, mixer_dim: int = 16,
                 num_layers: int = 1, max_step: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.max_step = max_step
        self.token_mixers = nn.ModuleList()
        self.feature_mixers = nn.ModuleList()
        for _ in range(num_layers):
            self.token_mixers.append(nn.Sequential(
                nn.LayerNorm(num_points), nn.Linear(num_points, mixer_dim), nn.ReLU(), nn.Linear(mixer_dim, num_points)))
            self.feature_mixers.append(nn.Sequential(
                nn.LayerNorm(num_features), nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_features)))
        self.output_layer = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_c = x[..., 2:3]
        for token_mixer, feature_mixer in zip(self.token_mixers, self.feature_mixers):
            x = x + token_mixer(x.transpose(-1, -2)).transpose(-1, -2)
            x = x + feature_mixer(x)
        residual = self.max_step * torch.tanh(self.output_layer(x))
        return 1.0 / (1.0 / (inv_c + self.eps) + residual + self.eps)


class RLFormer(nn.Module):
    """Single plane-wave beamforming policy.

    Parameters
    ----------
    data      : PlaneWaveData describing the acquisition (geometry, sampling, time zero, angles)
    grid      : (nz, nx, 3) reconstruction grid; rows/columns beyond a multiple of ``patch`` are cropped
    angle_idx : index of the transmit angle used by default (default: the central angle)
    patch     : patch size (pixels) processed jointly by the D2DNN
    """

    def __init__(self, data: PlaneWaveData, grid: np.ndarray, angle_idx: int | None = None, patch: int = 50,
                 hidden_dim: int = 16, mixer_dim: int = 16, num_layers: int = 1, max_step: float = 10.0,
                 w_new: float = 0.8, w_prev: float = 0.2, beta_max: float = 1.5, device=None, dtype=torch.float32):
        super().__init__()
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.patch = patch
        self.w_new, self.w_prev, self.beta_max = w_new, w_prev, beta_max
        self.dtype = dtype
        self.config = dict(patch=patch, hidden_dim=hidden_dim, mixer_dim=mixer_dim, num_layers=num_layers,
                           max_step=max_step, w_new=w_new, w_prev=w_prev, beta_max=beta_max)

        nz, nx = grid.shape[0] - grid.shape[0] % patch, grid.shape[1] - grid.shape[1] % patch
        if nz == 0 or nx == 0:
            raise ValueError(f"grid {grid.shape[:2]} is smaller than the patch size {patch}")
        self.out_shape = (nz, nx)
        self.n_patches = (nz // patch, nx // patch)
        self.points_per_patch = patch * patch
        self.n_points = nz * nx
        grid_t = torch.tensor(np.ascontiguousarray(grid[:nz, :nx]), dtype=dtype)
        self.register_buffer("grid_block", torch.stack([self.unfold_patches(grid_t[..., k]) for k in range(3)], dim=1))

        self.n_elements = data.n_elements
        for name in ("ele_x", "angles", "time_zero"):
            self.register_buffer(name, torch.zeros(1))
        self.fs, self.c0 = 1.0, 1540.0
        self.default_angle_idx = data.center_angle_index if angle_idx is None else angle_idx
        self.set_acquisition(data)

        self.d2dnn = D2DNN(self.points_per_patch, 4, hidden_dim, mixer_dim, num_layers, max_step)
        beta = torch.empty(self.n_points, self.n_elements)
        nn.init.normal_(beta, mean=1.0, std=0.1)
        self.summation_weights = nn.Parameter(beta.clamp_(0.0, beta_max))
        self.to(device)

    # ------------------------------------------------------------------ geometry helpers
    @property
    def device(self) -> torch.device:
        return self.grid_block.device

    def set_acquisition(self, data: PlaneWaveData) -> None:
        """Point the policy at another acquisition on the same grid (geometry, sampling, time zero)."""
        if data.n_elements != self.n_elements:
            raise ValueError(f"acquisition has {data.n_elements} elements, the policy was built for {self.n_elements}")
        dev = self.grid_block.device
        self.ele_x = torch.tensor(data.ele_pos[:, 0], dtype=self.dtype, device=dev)
        self.angles = torch.tensor(data.angles, dtype=self.dtype, device=dev)
        self.time_zero = torch.tensor(data.time_zero, dtype=self.dtype, device=dev)
        self.fs, self.c0 = float(data.fs), float(data.c)

    def unfold_patches(self, img: torch.Tensor) -> torch.Tensor:
        """(nz, nx) image -> flat (N,) vector ordered patch by patch (row-major inside each patch)."""
        p = self.patch
        return img.unfold(0, p, p).unfold(1, p, p).reshape(-1)

    def fold_patches(self, flat: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`unfold_patches`: flat (N,) or (N, 1) -> (nz, nx)."""
        p = self.patch
        pz, px = self.n_patches
        return flat.reshape(pz, px, p, p).permute(0, 2, 1, 3).reshape(self.out_shape)

    def initial_state(self):
        """s_0: homogeneous slowness 1/c0 and an empty image."""
        c_inv = torch.full((self.n_points, 1), 1.0 / self.c0, dtype=self.dtype, device=self.device)
        return c_inv, torch.zeros_like(c_inv)

    def sos_map(self, c_inv: torch.Tensor) -> torch.Tensor:
        """Effective speed-of-sound field [m/s] as an (nz, nx) image."""
        return self.fold_patches(1.0 / c_inv)

    # ------------------------------------------------------------------ policy
    def hyperbola_samples(self, c_inv: torch.Tensor, angle_idx: int) -> torch.Tensor:
        """Delays (in samples) of every pixel at every element, from the hyperbola of Eq. (1): (N, M)."""
        x, z = self.grid_block[:, 0:1], self.grid_block[:, 2:3]
        tdx = x * torch.sin(self.angles[angle_idx]) + z * torch.cos(self.angles[angle_idx])
        k, b = c_inv * tdx, c_inv * z
        term = (self.ele_x.unsqueeze(0) - x) ** 2 / z ** 2
        return (k + b * torch.sqrt(1.0 + term) + self.time_zero[angle_idx]) * self.fs

    def sample_channels(self, rf_data: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of the RF channels at fractional sample positions: (N, M)."""
        n_samples = rf_data.shape[0]
        coord = (samples * 2 + 1) / n_samples - 1  # grid_sample coordinates for align_corners=False
        grid = torch.stack((coord.t(), torch.zeros_like(coord.t())), dim=-1).unsqueeze(1)  # (M, 1, N, 2)
        signal = rf_data.t().reshape(self.n_elements, 1, 1, n_samples)
        out = F.grid_sample(signal, grid, align_corners=False)  # (M, 1, 1, N)
        return out.reshape(self.n_elements, -1).t()

    def forward(self, c_inv: torch.Tensor, image: torch.Tensor, rf_data: torch.Tensor,
                angle_idx: int | None = None, return_channels: bool = False):
        """One decision step.

        c_inv : (N, 1) current slowness 1/c_t        image : (N, 1) current beamformed image
        rf_data : (n_samples, n_elements) single plane-wave RF data
        Returns (1/c_{t+1}, image_{t+1}) and, if requested, the delayed channel data (N, M)
        that was summed to form the new image (used by the coherence-factor reward).
        """
        idx = self.default_angle_idx if angle_idx is None else angle_idx
        x, z = self.grid_block[:, 0:1], self.grid_block[:, 2:3]
        tdx = x * torch.sin(self.angles[idx]) + z * torch.cos(self.angles[idx])
        features = torch.cat((x, z, c_inv, c_inv * tdx), dim=1)
        c_inv_next = self.d2dnn(features.view(-1, self.points_per_patch, 4)).reshape(-1, 1)

        channels = self.sample_channels(rf_data, self.hyperbola_samples(c_inv_next, idx))
        weights = torch.clamp(F.relu(self.summation_weights), 0.0, self.beta_max)
        image_next = self.w_new * (channels * weights).sum(dim=-1, keepdim=True) + self.w_prev * image
        if return_channels:
            return c_inv_next, image_next, channels
        return c_inv_next, image_next

    # ------------------------------------------------------------------ weight transfer
    def _beta_maps(self, beta: torch.Tensor, pz: int, px: int) -> torch.Tensor:
        p = self.patch
        return beta.t().reshape(-1, pz, px, p, p).permute(0, 1, 3, 2, 4).reshape(beta.shape[1], pz * p, px * p)

    def _maps_beta(self, maps: torch.Tensor, pz: int, px: int) -> torch.Tensor:
        p = self.patch
        return maps.reshape(maps.shape[0], pz, p, px, p).permute(0, 1, 3, 2, 4).reshape(maps.shape[0], -1).t()

    def load_policy(self, state_dict: dict, source_patches: tuple[int, int] | None = None) -> str:
        """Load a trained policy, possibly obtained on a different grid size.

        The D2DNN is grid independent and is always copied. The per-pixel channel weights beta are
        copied if the grid matches; otherwise ``source_patches`` (number of patches along z and x of
        the training grid) must be given and every channel's weight map is resampled onto this grid
        with bilinear interpolation. Returns "exact" or "resampled".
        """
        self.d2dnn.load_state_dict({k[len("d2dnn."):]: v for k, v in state_dict.items() if k.startswith("d2dnn.")})
        beta = state_dict["summation_weights"].to(self.device, self.dtype)
        if beta.shape == self.summation_weights.shape:
            with torch.no_grad():
                self.summation_weights.copy_(beta)
            return "exact"
        if beta.shape[1] != self.n_elements:
            raise ValueError("number of elements of the checkpoint and of the acquisition differ")
        if source_patches is None:
            raise ValueError("the checkpoint grid differs from this grid: pass source_patches to resample beta")
        maps = self._beta_maps(beta, *source_patches).unsqueeze(0)
        maps = F.interpolate(maps, size=self.out_shape, mode="bilinear", align_corners=False)[0]
        with torch.no_grad():
            self.summation_weights.copy_(self._maps_beta(maps, *self.n_patches).clamp(0.0, self.beta_max))
        return "resampled"
