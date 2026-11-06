"""Plane-wave acquisitions (PICMUS, CUBDL) and the beamforming grid.

The container and the dataset conventions (element positions, time zero,
phantom sound speeds) follow the CUBDL evaluation framework (Hyun et al.,
IEEE TUFFC, 2021).

Data layout expected under ``--data-root``::

    data/
      picmus/<simulation|experiments|in-vivo>/<target>/<target>_<acq>_dataset_rf.hdf5
      cubdl/**/INS004*.hdf5, OSL007.hdf5, OSL010.hdf5, TSH007*.hdf5, UFL002*.hdf5, ...

Acquisitions are addressed by spec strings: ``picmus:<acq>:<target>`` or
``cubdl:<SOURCE>:<number>``, e.g. ``picmus:in-vivo:carotid_long``, ``cubdl:OSL:10``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob

import h5py
import numpy as np
from scipy.signal import convolve, hilbert

PICMUS_ACQUISITIONS = ("simulation", "experiments", "in-vivo")
PICMUS_TARGETS = ("contrast_speckle", "resolution_distortion", "carotid_long", "carotid_cross")

# Phantom sound speeds [m/s] used for the CUBDL acquisitions.
INS_SOUND_SPEED = {
    1: 1521, 2: 1517, 3: 1506, 4: 1501, 5: 1506, 6: 1509, 7: 1490, 8: 1504, 9: 1473,
    10: 1502, 11: 1511, 12: 1535, 13: 1453, 14: 1542, 15: 1539, 16: 1466, 17: 1462,
    18: 1479, 19: 1469, 20: 1464, 21: 1508, 22: 1558, 23: 1463, 24: 1547, 25: 1477, 26: 1497,
}
OSL_SOUND_SPEED = {2: 1536, 3: 1543, 4: 1538, 5: 1539, 6: 1541, 7: 1540}
UFL_SOUND_SPEED = {1: 1526, 2: 1523, 4: 1523, 5: 1523}

# Depth range [m] of the reconstruction grid per dataset family.
DEPTH_LIMITS = {"PICMUS": (5e-3, 55e-3), "OSL": (10e-3, 65e-3), "UFL": (10e-3, 50e-3), "TSH": (10e-3, 45e-3)}


@dataclass
class PlaneWaveData:
    """One plane-wave acquisition.

    idata, qdata : (n_angles, n_elements, n_samples) in-phase / quadrature data
    angles       : (n_angles,) transmit angles [rad]
    ele_pos      : (n_elements, 3) element positions [m]
    fc, fs       : centre / sampling frequency [Hz]
    fdemod       : demodulation frequency [Hz] (0 for RF data)
    c            : nominal speed of sound [m/s]
    time_zero    : (n_angles,) time of the first sample for every transmit [s]
    """

    idata: np.ndarray
    qdata: np.ndarray
    angles: np.ndarray
    ele_pos: np.ndarray
    fc: float
    fs: float
    fdemod: float
    c: float
    time_zero: np.ndarray

    def __post_init__(self):
        self.validate()

    @property
    def n_angles(self) -> int:
        return self.idata.shape[0]

    @property
    def n_elements(self) -> int:
        return self.idata.shape[1]

    @property
    def n_samples(self) -> int:
        return self.idata.shape[2]

    @property
    def center_angle_index(self) -> int:
        return self.n_angles // 2

    def validate(self) -> None:
        assert self.idata.shape == self.qdata.shape and self.idata.ndim == 3
        n_angles, n_chans, _ = self.idata.shape
        assert self.angles.ndim == 1 and self.angles.size == n_angles
        assert self.ele_pos.shape == (n_chans, 3)
        assert self.fc > 1e5 and self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        assert 1000 <= self.c <= 2000
        assert self.time_zero.ndim == 1 and self.time_zero.size == n_angles


def make_pixel_grid(xlims, zlims, dx, dz) -> np.ndarray:
    """Cartesian pixel grid of shape (nz, nx, 3) with y = 0."""
    eps = 1e-10
    x = np.arange(xlims[0], xlims[1] + eps, dx)
    z = np.arange(zlims[0], zlims[1] + eps, dz)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    return np.stack((xx, 0 * xx, zz), axis=-1)


def default_grid(data: PlaneWaveData, zlims, pixels_per_wavelength: float = 3.0):
    """Grid spanning the aperture and ``zlims`` with square pixels of lambda / ``pixels_per_wavelength``."""
    xlims = [data.ele_pos[0, 0], data.ele_pos[-1, 0]]
    dx = data.c / data.fc / pixels_per_wavelength
    return make_pixel_grid(xlims, zlims, dx, dx)


# --------------------------------------------------------------------------- PICMUS

def load_picmus(root: str, acq: str, target: str, dtype: str = "rf") -> PlaneWaveData:
    assert acq in PICMUS_ACQUISITIONS, f"acq must be one of {PICMUS_ACQUISITIONS}"
    assert target in PICMUS_TARGETS, f"target must be one of {PICMUS_TARGETS}"
    assert dtype in ("rf", "iq")
    fname = os.path.join(root, "picmus", acq, target, f"{target}_{acq[:4]}_dataset_{dtype}.hdf5")
    with h5py.File(fname, "r") as h5:
        f = h5["US"]["US_DATASET0000"]
        idata = np.array(f["data"]["real"], dtype="float32")
        qdata = np.array(f["data"]["imag"], dtype="float32")
        angles = np.array(f["angles"])
        fs = np.array(f["sampling_frequency"]).item()
        c = np.array(f["sound_speed"]).item()
        time_zero = np.array(f["initial_time"])
        ele_pos = np.array(f["probe_geometry"]).T
    fc = 5208000.0  # centre frequency of the PICMUS probe
    if dtype == "rf":
        qdata = np.imag(hilbert(idata, axis=-1))
    if time_zero.size == 1:
        time_zero = np.ones_like(angles) * time_zero
    return PlaneWaveData(idata, qdata, angles, ele_pos, fc, fs, fc if dtype == "iq" else 0, c, time_zero)


# --------------------------------------------------------------------------- CUBDL

def _find(root: str, pattern: str) -> str:
    hits = [y for x in os.walk(root) for y in glob(os.path.join(x[0], pattern))]
    assert hits, f"no file matching {pattern} under {root}"
    return sorted(hits)[0]


def _center_elements(ele_pos: np.ndarray) -> np.ndarray:
    ele_pos = ele_pos.copy()
    ele_pos[:, 0] -= np.mean(ele_pos[:, 0])
    return ele_pos


def _linear_array(n_elements: int, pitch: float) -> np.ndarray:
    xpos = np.arange(n_elements) * pitch
    xpos -= np.mean(xpos)
    return np.stack([xpos, 0 * xpos, 0 * xpos], axis=1)


def load_ins(root: str, acq: int) -> PlaneWaveData:
    with h5py.File(_find(root, "INS{:03d}*.hdf5".format(acq)), "r") as f:
        idata = np.array(f["channel_data"], dtype="float32")
        angles = np.linspace(-16, 16, idata.shape[0]) * np.pi / 180
        fc = np.array(f["modulation_frequency"]).item()
        fs = np.array(f["sampling_frequency"]).item()
        time_zero = -1 * np.array(f["start_time"], dtype="float32")[0]
        ele_pos = _center_elements(np.array(f["element_positions"], dtype="float32").T)
    c = INS_SOUND_SPEED.get(acq, 1540)
    for i, a in enumerate(angles):  # time zero refers to the array centre
        time_zero[i] += ele_pos[-1, 0] * np.abs(np.sin(a)) / c
    return PlaneWaveData(idata, np.imag(hilbert(idata, axis=-1)), angles, ele_pos, fc, fs, 0, c, time_zero)


def load_osl(root: str, acq: int) -> PlaneWaveData:
    assert acq in (2, 3, 4, 5, 6, 7, 10), "OSL acquisitions 2-7 and 10 contain plane-wave data"
    with h5py.File(_find(root, "OSL{:03d}.hdf5".format(acq)), "r") as f:
        idata = np.array(f["channel_data"], dtype="float32")
        angles = np.array(f["transmit_direction"][0], dtype="float32")
        fc = np.array(f["modulation_frequency"]).item()
        fs = np.array(f["sampling_frequency"]).item()
        time_zero = -1 * np.array(f["start_time"], dtype="float32")[0]
        ele_pos = _center_elements(np.array(f["element_positions"], dtype="float32").T)
    c = OSL_SOUND_SPEED.get(acq, 1540)
    return PlaneWaveData(idata, np.imag(hilbert(idata, axis=-1)), angles, ele_pos, fc, fs, 0, c, time_zero)


def load_tsh(root: str, acq: int) -> PlaneWaveData:
    with h5py.File(_find(root, "TSH{:03d}*.hdf5".format(acq)), "r") as f:
        angles = np.array(f["angles"])
        idata = np.array(f["channel_data"], dtype="float32")
        idata = np.transpose(np.reshape(idata, (128, len(angles), -1)), (1, 0, 2))
        fc = np.array(f["modulation_frequency"]).item()
        fs = np.array(f["sampling_frequency"]).item()
    c = 1540.0
    ele_pos = _linear_array(idata.shape[1], 0.3e-3)  # L11-4v
    time_zero = np.array([ele_pos[-1, 0] * np.abs(np.sin(a)) / c for a in angles], dtype="float32")
    return PlaneWaveData(idata, np.imag(hilbert(idata, axis=-1)), angles, ele_pos, fc, fs, 0, c, time_zero)


def load_ufl(root: str, acq: int) -> PlaneWaveData:
    with h5py.File(_find(root, "UFL{:03d}*.hdf5".format(acq)), "r") as f:
        idata = np.array(f["channel_data"], dtype="float32")
        angles = np.array(f["angles"]) * np.pi / 180
        fc = np.array(f["modulation_frequency"]).item()
        fs = np.array(f["channel_data_sampling_frequency"]).item()
        time_zero = -1 * np.array(f["channel_data_t0"], dtype="float32")
    c = UFL_SOUND_SPEED.get(acq, 1525)
    if time_zero.size == 1:
        time_zero = np.ones_like(angles) * time_zero
    ele_pos = _linear_array(idata.shape[1], 0.245e-3)  # LA533
    # The UFL data are stored modulated: demodulate, low-pass filter and decimate.
    data = idata + 1j * np.imag(hilbert(idata, axis=-1))
    phase = np.reshape(np.arange(idata.shape[2], dtype="float"), (1, 1, -1)) * fc / fs
    data = data * np.exp(-2j * np.pi * phase)
    ds = int(np.floor(fs / fc))
    data = convolve(data, np.ones((1, 1, ds)) / ds, "same")[:, :, ::ds]
    return PlaneWaveData(np.real(data).astype("float32"), np.imag(data).astype("float32"), angles, ele_pos, fc, fs / ds, fc, c, time_zero)


_CUBDL_LOADERS = {"INS": load_ins, "OSL": load_osl, "TSH": load_tsh, "UFL": load_ufl}


def load_cubdl(root: str, source: str, acq: int) -> PlaneWaveData:
    source = source.upper()
    if source not in _CUBDL_LOADERS:
        raise ValueError(f"unknown CUBDL source {source!r}; choose from {sorted(_CUBDL_LOADERS)}")
    return _CUBDL_LOADERS[source](os.path.join(root, "cubdl"), int(acq))


# --------------------------------------------------------------------------- specs

def load_acquisition(spec: str, root: str):
    """Load an acquisition from a spec string. Returns ``(data, zlims)``."""
    kind, *rest = spec.split(":")
    if kind == "picmus" and len(rest) == 2:
        return load_picmus(root, rest[0], rest[1]), DEPTH_LIMITS["PICMUS"]
    if kind == "cubdl" and len(rest) == 2:
        source, acq = rest[0].upper(), int(rest[1])
        zlims = DEPTH_LIMITS[source]
        if source == "INS":
            zlims = (10e-3, 50e-3) if acq >= 13 else (10e-3, 60e-3)
        elif source == "OSL" and acq == 10:
            zlims = (5e-3, 50e-3)
        return load_cubdl(root, source, acq), zlims
    raise ValueError(f"cannot parse acquisition spec {spec!r}; expected 'picmus:<acq>:<target>' or 'cubdl:<SOURCE>:<number>'")
