import torch

from rlformer.model import RLFormer
from rlformer.train import TrainConfig, run_episode, train_policy


def test_training_updates_parameters_and_runs_end_to_end(scene, tmp_path):
    data, grid, _ = scene
    model = RLFormer(data, grid, patch=10, device="cpu")
    before = [p.detach().clone() for p in model.parameters()]
    cfg = TrainConfig(epochs=2, steps=3, lr=0.05, center_only=True)
    history = train_policy(model, [data], cfg, checkpoint=str(tmp_path / "m.pt"), log_file=str(tmp_path / "log.csv"), progress=False)
    assert len(history) == 2 and all(torch.isfinite(torch.tensor(history)))
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert (tmp_path / "m.pt").exists() and (tmp_path / "log.csv").exists()


def test_checkpoint_roundtrip(scene, tmp_path):
    from rlformer.inference import load_policy, reconstruct, rf_tensor
    from rlformer.train import save_checkpoint
    data, grid, _ = scene
    model = RLFormer(data, grid, patch=10, device="cpu")
    save_checkpoint(model, str(tmp_path / "m.pt"))
    loaded, mode = load_policy(str(tmp_path / "m.pt"), data, grid, device="cpu")
    assert mode == "exact"
    rf = rf_tensor(data, 0, torch.device("cpu"))
    a, b = reconstruct(model, rf, 0, 3), reconstruct(loaded, rf, 0, 3)
    assert torch.allclose(a["rf"], b["rf"], atol=1e-5)


def test_training_increases_reward_on_defocused_data():
    from rlformer.data import make_pixel_grid
    from rlformer.inference import rf_tensor
    from rlformer.synthetic import simulate_plane_wave

    scatterers = [(x * 1e-3, z * 1e-3, 1.0) for x in (-2, 2) for z in (12, 16)]
    data = simulate_plane_wave(scatterers, c_true=1480.0, n_elements=16)  # nominal 1540 m/s: defocused
    dx = data.c / data.fc / 3
    grid = make_pixel_grid([-4e-3, 4e-3], [9e-3, 19e-3], dx, dx)
    for seed in (0, 1):
        torch.manual_seed(seed)
        model = RLFormer(data, grid, patch=10, device="cpu")
        rf = rf_tensor(data, 0, model.device)
        with torch.no_grad():
            before = float(run_episode(model, rf, 0, 5, (0.2, 0.4, 0.4))[0])
        train_policy(model, [data], TrainConfig(epochs=40, lr=0.05, center_only=True, seed=seed), progress=False)
        with torch.no_grad():
            after = float(run_episode(model, rf, 0, 5, (0.2, 0.4, 0.4))[0])
        assert after > before
