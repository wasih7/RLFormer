"""Training with deterministic policy gradients and a terminal, label-free reward.

An episode starts from the homogeneous state (c = c0, empty image), applies the policy ``steps``
times and receives a single reward from the final state (Eqs. 4-5). Because the reward is
differentiable with respect to the actions, the gradient of Eq. (5) is obtained by back-propagating
through the unrolled episode with autograd and is ascended with Adam (Eq. 11).
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import asdict, dataclass

import torch
from tqdm import tqdm

from .data import PlaneWaveData
from .inference import rf_tensor
from .model import RLFormer
from .rewards import unsupervised_reward


@dataclass
class TrainConfig:
    epochs: int = 100
    steps: int = 5                      # episode length T
    lr: float = 1e-2
    reward_weights: tuple = (0.2, 0.4, 0.4)  # FSR, LTE, CF
    angle_stride: int = 1               # use every n-th steered plane wave of each acquisition
    center_only: bool = False           # train on the central (0 deg) plane wave only
    seed: int = 0


def run_episode(model: RLFormer, rf_data: torch.Tensor, angle_idx: int, steps: int, reward_weights):
    """Roll out one episode; returns the terminal reward (differentiable) and its components."""
    c_inv, image = model.initial_state()
    channels = None
    for t in range(steps):
        if t == steps - 1:
            c_inv, image, channels = model(c_inv, image, rf_data, angle_idx, return_channels=True)
        else:
            c_inv, image = model(c_inv, image, rf_data, angle_idx)
    return unsupervised_reward(model.fold_patches(image), channels, reward_weights)


def save_checkpoint(model: RLFormer, path: str, train_config: TrainConfig | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config, "n_patches": tuple(model.n_patches),
                "out_shape": tuple(model.out_shape), "n_elements": model.n_elements,
                "train_config": asdict(train_config) if train_config else None}, path)


def train_policy(model: RLFormer, datasets: list[PlaneWaveData], cfg: TrainConfig | None = None,
                 checkpoint: str | None = None, log_file: str | None = None, progress: bool = True):
    """Train ``model`` on one or several acquisitions that share the model's grid and probe layout.

    Every (acquisition, steering angle) pair is a single plane-wave sample and is visited once per
    epoch in random order. Returns the list of logged episode rewards.
    """
    cfg = cfg or TrainConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    samples = []
    for d_idx, d in enumerate(datasets):
        angles = [d.center_angle_index] if cfg.center_only else list(range(0, d.n_angles, cfg.angle_stride))
        samples += [(d_idx, a) for a in angles]

    writer, handle = None, None
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handle = open(log_file, "w", newline="")
        writer = csv.writer(handle)
        writer.writerow(["epoch", "dataset", "angle", "reward", "fsr", "lte", "cf"])

    history, active = [], None
    model.train()
    try:
        for epoch in range(cfg.epochs):
            rng.shuffle(samples)
            bar = tqdm(samples, desc=f"epoch {epoch + 1}/{cfg.epochs}", disable=not progress, leave=False)
            epoch_reward = 0.0
            for d_idx, a in bar:
                if active != d_idx:
                    model.set_acquisition(datasets[d_idx])
                    active = d_idx
                rf = rf_tensor(datasets[d_idx], a, model.device, model.dtype)
                reward, parts = run_episode(model, rf, a, cfg.steps, cfg.reward_weights)
                optimizer.zero_grad()
                (-reward).backward()
                optimizer.step()
                value = float(reward.detach())
                epoch_reward += value
                history.append(value)
                bar.set_postfix(reward=f"{value:.4f}")
                if writer:
                    writer.writerow([epoch + 1, d_idx, a, value] + [float(parts[k]) for k in ("fsr", "lte", "cf")])
            if progress:
                print(f"epoch {epoch + 1}/{cfg.epochs}  mean reward {epoch_reward / max(len(samples), 1):.4f}")
            if checkpoint:
                save_checkpoint(model, checkpoint, cfg)
    finally:
        if handle:
            handle.close()
    return history
