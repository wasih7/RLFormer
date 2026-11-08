# RLFormer

Illustrative code for **"Reinforcement Learning of Adaptive Speed-of-Sound Fields for Self-Supervised Single
Plane-Wave Ultrasound Beamforming"** (M. Wasih and M. K. Almekkawy, *IEEE Transactions on
Computational Imaging*, vol. 12, pp. 1454-1467, 2026).

Single plane-wave (PW) delay-and-sum (DAS) images have low contrast, while coherent compounding (CC)
of many angles costs frame rate. RLFormer treats single-PW beamforming as a sequential decision
problem: an agent refines a spatially varying *effective* speed-of-sound (SoS) field, which reshapes
the per-pixel delay curves, and learns per-pixel channel weights for the summation. It is trained
without reference images, SoS labels or multi-angle data, using label-free rewards computed from the
reconstruction itself.

## How it works

* **State** `s_t = (1/c_t, image_t)`: the slowness at every pixel and the current beamformed image.
  The initial state is the homogeneous nominal speed of sound and an empty image.
* **Policy** (`rlformer/model.py`)
  * `D2DNN` - an MLP-mixer layer applied patch-wise to the pixel features
    `(x, z, 1/c_t, d_TX/c_t)`. It outputs a bounded residual added to the local
    sound speed.
  * *Adaptive DAS* - the delay curve of every pixel is the hyperbola `t(x_i) = k + b sqrt(1 + (x_i - h)^2 / a^2)`
    with `h = x`, `a = z`, `b = z / c(x, z)`, `k = d_TX / c(x, z)`; the RF channels are interpolated on it
    and summed with learnable per-pixel weights `beta`. The new image is `w_1 * DAS + (1-w_1) * previous image`.
* **Reward** (`rlformer/rewards.py`), received only after the last of `T` steps and computed from the
  reconstruction alone: `w_2 * FSR + w_3 * LTE + (1-w_2-w_3) * CF`
  (feature sharpness = normalized variance of the Laplacian, local texture entropy, coherence factor of the
  delayed channel data).
* **Optimisation** (`rlformer/train.py`) - deterministic policy gradient. The reward is differentiable with
  respect to the actions, so the gradient is obtained by back-propagating through the unrolled episode
  with autograd; Adam.

The learned SoS field is an *effective* delay field under straight-ray, plane-wave DAS assumptions, not
a measurement of tissue sound speed.

## Installation

```bash
git clone https://github.com/wasih7/RLFormer.git
cd RLFormer
pip install -r requirements.txt     # or: pip install -e .
pytest                              # unit tests on simulated data, a few seconds on CPU
```

A GPU is recommended for training and for full-resolution grids (500 x 350 pixels, 128 elements).

## Quick check without data

```bash
python scripts/demo_synthetic.py --out demo.png
```

simulates point scatterers in a 1480 m/s medium (nominal speed 1540 m/s), trains a small policy for a
few episodes and prints the terminal reward and its components before and after training (the total
reward should increase). It is a functional check of the pipeline, not a benchmark.

## Data

Download the [PICMUS](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/) and
[CUBDL](https://ieee-dataport.org/competitions/challenge-ultrasound-beamforming-deep-learning-cubdl)
datasets and arrange them as

```
data/
  picmus/{simulation,experiments,in-vivo}/<target>/<target>_<simu|expe|in-v>_dataset_rf.hdf5
  cubdl/**/INS004*.hdf5  OSL007.hdf5  OSL010.hdf5  TSH007*.hdf5  UFL002*.hdf5 ...
```

Acquisitions are addressed by spec strings: `picmus:<acquisition>:<target>` (e.g.
`picmus:in-vivo:carotid_long`) and `cubdl:<INS|OSL|TSH|UFL>:<number>` (e.g. `cubdl:OSL:10`).

## Training

```bash
python scripts/train.py --data-root data --checkpoint checkpoints/rlformer.pt
```

By default this trains on the four PICMUS phantom sequences (simulation and experiment,
`contrast_speckle` and `resolution_distortion`); every steered plane wave of every sequence is one
single-PW training sample. Useful options: `--epochs`, `--lr`, `--steps`, `--reward-weights`,
`--patch`, `--angle-stride N` (use every N-th angle) and `--center-only` (0 degree plane wave only).
Training cost grows with the number of samples per epoch; `--angle-stride` and `--center-only` give
much shorter runs. Episode rewards are logged to `checkpoints/train_log.csv`.

## Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/rlformer.pt --data-root data \
    --test picmus:in-vivo:carotid_cross picmus:in-vivo:carotid_long cubdl:OSL:10 --out-dir results
```

For every test acquisition this reconstructs the 0 degree plane wave with the policy. MSE, PSNR and SSIM are computed on B-mode
images mapped to [0, 1]; CNR and gCNR are computed on the envelope if regions of interest are supplied
with `--rois` (see `python scripts/evaluate.py --help`).

A policy trained on PICMUS can be applied to a different grid: the D2DNN is copied unchanged and the
per-pixel weights `beta` are resampled onto the new grid.

## Repository layout

```
rlformer/
  data.py          PICMUS / CUBDL loaders, plane-wave container, pixel grid
  model.py         D2DNN and the RLFormer policy (adaptive DAS)
  rewards.py       FSR, LTE, CF and the combined reward
  train.py         episodes, deterministic policy gradient training, checkpoints
  inference.py     policy roll-out and checkpoint loading
  beamforming.py   constant-SoS DAS / coherent compounding, envelope and B-mode helpers
  metrics.py       MSE, PSNR, SSIM, CNR, gCNR
  synthetic.py     small simulated acquisitions for tests and demos
scripts/           train.py, evaluate.py, demo_synthetic.py
tests/             unit tests
```

## Implementation notes

* The reconstruction grid has pixels of one third of a wavelength and covers the aperture and the depth
  range of each dataset; rows and columns beyond a multiple of the patch size are cropped.
* Rewards are computed on the beamformed RF-domain image scaled to a maximum absolute value of 1; the
  coherence factor uses the delayed channel data of the last step. The envelope used for B-mode images and
  contrast metrics is the analytic signal of the RF image along depth.
* DAS and CC baselines in `beamforming.py` use the nominal speed of sound and no apodization.
* PSNR is computed as `10 log10(1 / (MSE + 1e-6))`, so identical images give 60 dB.
* CNR is `|mu_fg - mu_bg| / sqrt((var_fg + var_bg) / 2)`.

## Citation

```bibtex
@article{wasih2026rlformer,
  title   = {Reinforcement Learning of Adaptive Speed-of-Sound Fields for Self-Supervised
             Single Plane-Wave Ultrasound Beamforming},
  author  = {Wasih, Mohammad and Almekkawy, Mohamed Khaled},
  journal = {IEEE Transactions on Computational Imaging},
  volume  = {12},
  pages   = {1454--1467},
  year    = {2026},
  doi     = {10.1109/TCI.2026.3716387}
}
```

## Acknowledgements

The plane-wave data containers and dataset conventions follow the CUBDL evaluation framework
(D. Hyun et al., *IEEE TUFFC*, 2021). Experiments use the PICMUS (H. Liebgott et al., 2016) and
CUBDL (M. Bell et al., 2020) datasets.

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
