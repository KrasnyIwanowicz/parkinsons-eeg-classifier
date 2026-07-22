"""
Hand-crafted feature extraction from EEG epochs: spectral band power
(via Welch's method) and Hjorth parameters. These features feed the
classical ML baseline and the PCA/SVD dimensionality-reduction step —
the deep models (models.py) can also consume them as a per-epoch
sequence input instead of raw voltage.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def band_power(epoch_data: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    """Average power per frequency band, per channel.

    Parameters
    ----------
    epoch_data : array, shape (n_channels, n_times)
    sfreq : sampling frequency in Hz

    Returns
    -------
    dict mapping band name -> array of shape (n_channels,)
    """
    freqs, psd = welch(epoch_data, fs=sfreq, nperseg=min(256, epoch_data.shape[-1]))
    return {
        band: psd[:, (freqs >= lo) & (freqs <= hi)].mean(axis=-1)
        for band, (lo, hi) in BANDS.items()
    }


def hjorth_parameters(epoch_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hjorth activity, mobility, and complexity per channel.

    epoch_data : array, shape (n_channels, n_times)
    """
    first_deriv = np.diff(epoch_data, axis=-1)
    second_deriv = np.diff(first_deriv, axis=-1)

    activity = np.var(epoch_data, axis=-1)
    mobility = np.sqrt(np.var(first_deriv, axis=-1) / (activity + 1e-12))
    complexity = np.sqrt(
        np.var(second_deriv, axis=-1) / (np.var(first_deriv, axis=-1) + 1e-12)
    ) / (mobility + 1e-12)
    return activity, mobility, complexity


def extract_features(epoch_data: np.ndarray, sfreq: float) -> np.ndarray:
    """Build one flat feature vector for a single epoch.

    Concatenates 5 band powers + 3 Hjorth parameters, each per channel,
    into one 1D vector of length 8 * n_channels.

    Band power and Hjorth activity are both raw power-like quantities
    (variance, or variance-derived) with heavily right-skewed
    distributions — a handful of high-power/artifact epochs dominate the
    raw scale, which wrecks StandardScaler's mean/std and crushes normal
    epoch-to-epoch variation into a tiny sliver of the standardized
    range. Log-transforming them is standard EEG feature-engineering
    practice and is what fixed a real bug here: an earlier SHAP run
    showed exactly 0.0 importance for every band and for hjorth_activity,
    while mobility/complexity (already scale-invariant ratios) survived
    -- a strong signal that raw-power skew, not "these bands don't
    matter", explained the zeros.
    """
    powers = band_power(epoch_data, sfreq)
    activity, mobility, complexity = hjorth_parameters(epoch_data)

    log_powers = [np.log10(p + 1e-20) for p in powers.values()]
    log_activity = np.log10(activity + 1e-20)

    return np.concatenate(log_powers + [log_activity, mobility, complexity])
