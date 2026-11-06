import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rlformer.data import make_pixel_grid  # noqa: E402
from rlformer.model import RLFormer  # noqa: E402
from rlformer.synthetic import simulate_plane_wave  # noqa: E402


@pytest.fixture(scope="session")
def scene():
    """Small acquisition (c_true = c_nominal), its grid (cropped by the model) and a policy."""
    scatterers = [(x * 1e-3, z * 1e-3, 1.0) for x in (-3, 3) for z in (12, 18)]
    data = simulate_plane_wave(scatterers, c_true=1540.0, c_nominal=1540.0, n_elements=24)
    dx = data.c / data.fc / 3
    grid = make_pixel_grid([-5e-3, 5e-3], [9e-3, 21e-3], dx, dx)
    model = RLFormer(data, grid, patch=10, device="cpu")
    return data, grid, model
