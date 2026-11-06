import numpy as np
import torch

from rlformer.beamforming import analytic_envelope, bmode, das_image, to_unit_range
from rlformer.metrics import cnr, gcnr, mse, psnr, ssim


def test_identical_images():
    img = np.random.default_rng(0).random((40, 40))
    assert mse(img, img) == 0 and abs(psnr(img, img) - 60.0) < 1e-6 and abs(ssim(img, img) - 1.0) < 1e-9


def test_contrast_metrics():
    rng = np.random.default_rng(1)
    env = rng.random((40, 40))
    env[:, :20] += 3.0
    fg = np.zeros((40, 40), bool); fg[:, :20] = True
    bg = ~fg
    assert gcnr(env, fg, bg) > 0.99 and cnr(env, fg, bg) > 3
    same = np.zeros((40, 40), bool); same[:, :10] = True
    assert gcnr(env, same, same) < 1e-6


def test_envelope_of_a_carrier_is_flat():
    t = torch.arange(512).float()
    carrier = torch.cos(2 * np.pi * 0.1 * t).unsqueeze(1).repeat(1, 4)
    env = analytic_envelope(carrier, dim=0)
    assert torch.allclose(env[50:-50], torch.ones_like(env[50:-50]), atol=0.05)


def test_das_focuses_point_scatterers_at_their_positions(scene):
    data, grid, _ = scene
    img = das_image(data, grid, [0], device="cpu").abs().numpy()
    peak = np.unravel_index(np.argmax(img), img.shape)
    xs = np.array([-3e-3, 3e-3]); zs = np.array([12e-3, 18e-3])
    px, pz = grid[peak[0], peak[1], 0], grid[peak[0], peak[1], 2]
    assert min(abs(px - xs)) < 0.5e-3 and min(abs(pz - zs)) < 0.5e-3


def test_bmode_range():
    img = np.random.default_rng(2).random((20, 20)) + 0.01
    db = bmode(img, 60.0)
    assert db.max() <= 1e-9 and db.min() >= -60.0 - 1e-9
    unit = to_unit_range(db)
    assert unit.min() >= 0 and unit.max() <= 1
