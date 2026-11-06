"""Small simulated plane-wave acquisitions, for tests and quick experiments."""

from __future__ import annotations

import numpy as np

from .data import PlaneWaveData


def simulate_plane_wave(scatterers, c_true=1480.0, c_nominal=1540.0, n_elements=32, pitch=0.3e-3,
                        fc=5e6, fs=20e6, n_samples=1024, angle=0.0, bandwidth_cycles=1.5,
                        noise=0.0, seed=0) -> PlaneWaveData:
    """RF data of point scatterers insonified by a single plane wave.

    The medium has a homogeneous speed of sound ``c_true`` while the returned acquisition
    advertises ``c_nominal``: beamforming with the nominal value then yields defocused images,
    which is the situation the policy is trained to correct.

    scatterers : iterable of (x, z, amplitude) [m, m, a.u.]
    """
    rng = np.random.default_rng(seed)
    xe = (np.arange(n_elements) - (n_elements - 1) / 2) * pitch
    t = np.arange(n_samples) / fs
    rf = np.zeros((1, n_elements, n_samples), dtype="float64")
    sigma = bandwidth_cycles / (2 * np.pi * fc)
    for x, z, amp in scatterers:
        tx = (x * np.sin(angle) + z * np.cos(angle)) / c_true
        rx = np.sqrt((x - xe) ** 2 + z ** 2) / c_true
        tau = (tx + rx)[:, None]
        dt = t[None, :] - tau
        rf[0] += amp * np.exp(-0.5 * (dt / sigma) ** 2) * np.cos(2 * np.pi * fc * dt)
    rf += noise * rng.standard_normal(rf.shape)
    from scipy.signal import hilbert
    ele_pos = np.stack([xe, 0 * xe, 0 * xe], axis=1)
    return PlaneWaveData(
        idata=rf.astype("float32"), qdata=np.imag(hilbert(rf, axis=-1)).astype("float32"),
        angles=np.array([angle], dtype="float64"), ele_pos=ele_pos, fc=fc, fs=fs, fdemod=0,
        c=c_nominal, time_zero=np.zeros(1))
