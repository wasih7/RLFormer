"""End-to-end check on simulated data (no dataset download needed).

Point scatterers are simulated in a medium with c = 1480 m/s while the nominal speed of sound is
1540 m/s, so delay-and-sum with the nominal value is defocused. A small policy is trained for a few
episodes and the terminal reward (and its components) before and after training is printed; the
total reward should increase. This is a functional check of the pipeline, not a benchmark: the
policy's speed-of-sound update per step is bounded, so the learned field is an effective delay field.

    python scripts/demo_synthetic.py [--episodes 60] [--out demo.png]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from rlformer.beamforming import bmode, das_image
from rlformer.data import make_pixel_grid
from rlformer.inference import reconstruct, rf_tensor
from rlformer.model import RLFormer
from rlformer.synthetic import simulate_plane_wave
from rlformer.train import TrainConfig, run_episode, train_policy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--out", default=None, help="optional figure path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    scatterers = [(x * 1e-3, z * 1e-3, 1.0) for x in (-4, 0, 4) for z in (12, 18, 24)]
    data = simulate_plane_wave(scatterers, c_true=1480.0)
    dx = data.c / data.fc / 3
    grid = make_pixel_grid([-6e-3, 6e-3], [9e-3, 29e-3], dx, dx)
    model = RLFormer(data, grid, patch=20, device=args.device)
    rf = rf_tensor(data, 0, model.device)

    with torch.no_grad():
        before, parts_before = run_episode(model, rf, 0, 5, (0.2, 0.4, 0.4))
    rec_before = reconstruct(model, rf, 0, 5)
    cfg = TrainConfig(epochs=args.episodes, lr=args.lr, center_only=True, seed=args.seed)
    train_policy(model, [data], cfg, progress=False)
    with torch.no_grad():
        after, parts_after = run_episode(model, rf, 0, 5, (0.2, 0.4, 0.4))
    rec_after = reconstruct(model, rf, 0, 5)

    print(f"reward  {float(before):.4f} -> {float(after):.4f}")
    for k in ("fsr", "lte", "cf"):
        print(f"  {k}: {float(parts_before[k]):.4f} -> {float(parts_after[k]):.4f}")
    sos = rec_after["sos"]
    print(f"effective speed of sound after training: mean {sos.mean():.1f} m/s, range [{sos.min():.1f}, {sos.max():.1f}] "
          f"(nominal {data.c:.0f}; at most {model.config['max_step']:.0f} m/s change per step)")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nz, nx = model.out_shape
        reference = das_image(data, grid[:nz, :nx], device=args.device)
        panels = [("DAS, nominal c", bmode(reference.abs())), ("policy, untrained", bmode(rec_before["envelope"])),
                  ("policy, trained", bmode(rec_after["envelope"]))]
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), constrained_layout=True)
        for ax, (name, img) in zip(axes, panels):
            ax.imshow(img, vmin=-40, vmax=0, cmap="gray", aspect="auto")
            ax.set_title(name)
        shown = axes[3].imshow(sos.cpu().numpy(), cmap="inferno", aspect="auto")
        axes[3].set_title("learned SoS (m/s)")
        fig.colorbar(shown, ax=axes[3])
        fig.savefig(args.out, dpi=120)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
