"""Evaluate a trained policy against single plane-wave DAS and coherent compounding.

Example:
    python scripts/evaluate.py --checkpoint checkpoints/rlformer.pt --data-root data \
        --test picmus:in-vivo:carotid_cross picmus:in-vivo:carotid_long cubdl:OSL:10 --out-dir results

Methods compared (all cropped to the policy's grid): DAS (0 deg), CC (every third plane wave),
RLFormer (0 deg) and the reference CC (all plane waves). Similarity metrics (MSE, PSNR, SSIM) are
computed on B-mode images mapped to [0, 1]; contrast metrics (CNR, gCNR) on the envelope and only
if regions of interest are given with --rois (JSON, millimetres):
    {"picmus:in-vivo:carotid_long": {"fg": {"x": [-3, 3], "z": [15, 22]}, "bg": {"x": [5, 12], "z": [25, 32]}}}
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from rlformer.beamforming import bmode, das_image, to_unit_range
from rlformer.data import default_grid, load_acquisition
from rlformer.inference import load_policy, reconstruct, rf_tensor
from rlformer.metrics import cnr, gcnr, mse, psnr, region_mask, ssim


def evaluate(spec, args, rois):
    data, zlims = load_acquisition(spec, args.data_root)
    grid = default_grid(data, zlims)
    model, mode = load_policy(args.checkpoint, data, grid, device=args.device)
    nz, nx = model.out_shape
    grid = grid[:nz, :nx]
    center = data.center_angle_index

    reference = das_image(data, grid, device=args.device)
    envelopes = {
        "DAS (0 deg)": das_image(data, grid, [center], device=args.device).abs(),
        f"CC ({len(range(0, data.n_angles, 3))} PWs)": das_image(data, grid, range(0, data.n_angles, 3), device=args.device).abs(),
    }
    out = reconstruct(model, rf_tensor(data, center, model.device), center, args.steps)
    envelopes["RLFormer (0 deg)"] = out["envelope"]
    envelopes[f"CC ({data.n_angles} PWs)"] = reference.abs()

    images = {name: bmode(env) for name, env in envelopes.items()}
    ref_unit = to_unit_range(images[f"CC ({data.n_angles} PWs)"])
    results = {}
    for name, img in images.items():
        unit = to_unit_range(img)
        results[name] = {"mse": mse(unit, ref_unit), "psnr": psnr(unit, ref_unit), "ssim": ssim(unit, ref_unit)}
        if spec in rois:
            r = rois[spec]
            fg = region_mask(grid, [v * 1e-3 for v in r["fg"]["x"]], [v * 1e-3 for v in r["fg"]["z"]])
            bg = region_mask(grid, [v * 1e-3 for v in r["bg"]["x"]], [v * 1e-3 for v in r["bg"]["z"]])
            env = envelopes[name].detach().cpu().numpy()
            env = env / env.max()
            results[name].update(cnr=cnr(env, fg, bg), gcnr=gcnr(env, fg, bg))

    tag = spec.replace(":", "_")
    np.savez_compressed(os.path.join(args.out_dir, f"{tag}.npz"), sos=out["sos"].cpu().numpy(), **{k: v for k, v in images.items()})
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        extent = [grid[0, 0, 0] * 1e3, grid[0, -1, 0] * 1e3, grid[-1, 0, 2] * 1e3, grid[0, 0, 2] * 1e3]
        fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
        for ax, (name, img) in zip(axes.flat, list(images.items()) + [("learned SoS (m/s)", None)]):
            if img is None:
                shown = ax.imshow(out["sos"].cpu().numpy(), extent=extent, cmap="inferno")
                fig.colorbar(shown, ax=ax)
            else:
                ax.imshow(img, vmin=-60, vmax=0, cmap="gray", extent=extent)
            ax.set_title(name)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("z (mm)")
        axes.flat[-1].axis("on")
        fig.savefig(os.path.join(args.out_dir, f"{tag}.png"), dpi=150)
        plt.close(fig)
    print(f"[{spec}] policy weights: {mode}")
    for name, m in results.items():
        print("  %-18s " % name + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--test", nargs="+", required=True, help="acquisition specs to evaluate")
    p.add_argument("--rois", default=None, help="JSON file with foreground/background regions (mm)")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rois = json.load(open(args.rois)) if args.rois else {}
    summary = {spec: evaluate(spec, args, rois) for spec in args.test}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
