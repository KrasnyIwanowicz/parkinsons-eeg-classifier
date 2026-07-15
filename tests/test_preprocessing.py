"""
Tests run on a synthetic MNE Raw object (random data, fake channel names)
so CI never needs to download the real ~2 GB clinical EEG dataset just to
check that the pipeline doesn't crash.
"""
import mne
import numpy as np

from src.preprocessing import make_epochs, preprocess_raw


def _make_synthetic_raw(n_channels=8, sfreq=256.0, duration_sec=10.0):
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    rng = np.random.default_rng(42)
    data = rng.normal(size=(n_channels, int(sfreq * duration_sec))) * 1e-6
    return mne.io.RawArray(data, info, verbose=False)


def test_preprocess_raw_runs_and_preserves_shape():
    raw = _make_synthetic_raw()
    cleaned = preprocess_raw(raw, n_ica_components=4)
    assert cleaned.get_data().shape == raw.get_data().shape


def test_make_epochs_shapes():
    raw = _make_synthetic_raw(duration_sec=20.0)
    epochs = make_epochs(raw, duration=2.0)
    data = epochs.get_data()
    assert data.shape[0] == len(epochs)
    assert data.shape[1] == 8  # n_channels
