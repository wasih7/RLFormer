import torch

from rlformer.model import RLFormer


def test_fold_unfold_roundtrip(scene):
    _, _, model = scene
    img = torch.randn(*model.out_shape)
    assert torch.equal(model.fold_patches(model.unfold_patches(img)), img)


def test_grid_is_cropped_to_patch_multiples(scene):
    _, grid, model = scene
    assert all(s % model.patch == 0 for s in model.out_shape)
    assert model.out_shape[0] <= grid.shape[0] and model.out_shape[1] <= grid.shape[1]


def test_homogeneous_hyperbola_equals_geometric_delay(scene):
    data, _, model = scene
    c_inv, _ = model.initial_state()
    samples = model.hyperbola_samples(c_inv, 0)  # (N, M)
    x, z = model.grid_block[:, 0:1], model.grid_block[:, 2:3]
    expected = (z + torch.sqrt(z ** 2 + (model.ele_x.unsqueeze(0) - x) ** 2)) / model.c0 * model.fs
    assert torch.allclose(samples, expected, rtol=1e-4)


def test_step_changes_speed_of_sound_by_at_most_max_step(scene):
    data, _, model = scene
    rf = torch.tensor(data.idata[0].T)
    c_inv, image = model.initial_state()
    c_next, image_next = model(c_inv, image, rf)
    assert image_next.shape == image.shape and torch.isfinite(image_next).all()
    delta = (1.0 / c_next - 1.0 / c_inv).abs().max()
    assert delta <= model.config["max_step"] + 1e-2


def test_patch_batching_matches_patchwise_application(scene):
    _, _, model = scene
    feats = torch.randn(3, model.points_per_patch, 4).abs() + 0.5
    batched = model.d2dnn(feats)
    looped = torch.stack([model.d2dnn(f) for f in feats])
    assert torch.allclose(batched, looped, atol=1e-6)


def test_channel_sampling_matches_per_element_interpolation(scene):
    data, _, model = scene
    rf = torch.tensor(data.idata[0].T)
    samples = model.hyperbola_samples(model.initial_state()[0], 0)
    fast = model.sample_channels(rf, samples)
    # explicit linear interpolation (zeros outside the record)
    n = rf.shape[0]
    pos = samples - 0.0
    lo = torch.floor(pos).long()
    frac = pos - lo
    def at(idx):
        valid = ((idx >= 0) & (idx < n)).to(rf.dtype)
        return rf[idx.clamp(0, n - 1), torch.arange(rf.shape[1]).unsqueeze(0).expand_as(idx)] * valid
    slow = at(lo) * (1 - frac) + at(lo + 1) * frac
    assert torch.allclose(fast, slow, atol=1e-4)


def test_policy_transfer_across_grids(scene):
    data, grid, model = scene
    same = RLFormer(data, grid, patch=10, device="cpu")
    assert same.load_policy(model.state_dict(), source_patches=tuple(model.n_patches)) == "exact"
    assert torch.equal(same.summation_weights, model.summation_weights)
    smaller = RLFormer(data, grid[:20, :20], patch=10, device="cpu")
    assert smaller.load_policy(model.state_dict(), source_patches=tuple(model.n_patches)) == "resampled"
    assert smaller.summation_weights.shape == (smaller.n_points, model.n_elements)
    assert torch.isfinite(smaller.summation_weights).all()
