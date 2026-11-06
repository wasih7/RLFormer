"""Train an RLFormer policy on plane-wave acquisitions.

Example:
    python scripts/train.py --data-root data --checkpoint checkpoints/rlformer.pt \
        --train picmus:simulation:contrast_speckle picmus:simulation:resolution_distortion \
                picmus:experiments:contrast_speckle picmus:experiments:resolution_distortion
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from rlformer.data import default_grid, load_acquisition
from rlformer.model import RLFormer
from rlformer.train import TrainConfig, train_policy

DEFAULT_TRAIN = [
    "picmus:simulation:contrast_speckle", "picmus:simulation:resolution_distortion",
    "picmus:experiments:contrast_speckle", "picmus:experiments:resolution_distortion",
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--train", nargs="+", default=DEFAULT_TRAIN, help="acquisition specs used for training")
    p.add_argument("--checkpoint", default="checkpoints/rlformer.pt")
    p.add_argument("--log", default="checkpoints/train_log.csv")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps", type=int, default=5, help="episode length T")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--reward-weights", type=float, nargs=3, default=(0.2, 0.4, 0.4), metavar=("FSR", "LTE", "CF"))
    p.add_argument("--patch", type=int, default=50)
    p.add_argument("--angle-stride", type=int, default=1, help="use every n-th plane wave of each acquisition")
    p.add_argument("--center-only", action="store_true", help="train on the central plane wave only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    datasets, grid = [], None
    for spec in args.train:
        data, zlims = load_acquisition(spec, args.data_root)
        g = default_grid(data, zlims)
        if grid is None:
            grid = g
        elif g.shape != grid.shape:
            raise SystemExit(f"{spec} yields a {g.shape[:2]} grid but the first acquisition yields {grid.shape[:2]}; "
                             "train on acquisitions that share a grid or run separate trainings")
        datasets.append(data)
        print(f"loaded {spec}: {data.n_angles} angles, {data.n_elements} elements, {data.n_samples} samples")

    model = RLFormer(datasets[0], grid, patch=args.patch, device=args.device)
    print(f"grid {model.out_shape}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters, device {model.device}")
    cfg = TrainConfig(epochs=args.epochs, steps=args.steps, lr=args.lr, reward_weights=tuple(args.reward_weights),
                      angle_stride=args.angle_stride, center_only=args.center_only, seed=args.seed)
    train_policy(model, datasets, cfg, checkpoint=args.checkpoint, log_file=args.log)
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
