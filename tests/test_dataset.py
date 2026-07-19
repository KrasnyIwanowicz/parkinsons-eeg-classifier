"""
Test for src/dataset.py's build_subject_sequences, using synthetic MNE
raw data wrapped in the same Subject dataclass real loading produces —
no real EEG download needed.
"""
import mne
import numpy as np

from src.data_loading import Subject
from src.dataset import build_subject_sequences

_CFG = {
    "l_freq": 1.0,
    "h_freq": 45.0,
    "notch_freq": 50.0,
    "n_ica_components": 4,
    "epoch_duration": 2.0,
    "epoch_overlap": 0.0,
}


def _make_synthetic_subject(subject_id, label, n_channels=6, sfreq=256.0, duration_sec=20.0):
    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    rng = np.random.default_rng(hash(subject_id) % (2**31))
    data = rng.normal(size=(n_channels, int(sfreq * duration_sec))) * 1e-6
    raw = mne.io.RawArray(data, info, verbose=False)
    return Subject(subject_id=subject_id, label=label, raw=raw)


def test_build_subject_sequences_shapes():
    subjects = [
        _make_synthetic_subject("hc1", 0, duration_sec=20.0),
        _make_synthetic_subject("pd1", 1, duration_sec=14.0),
    ]
    sequences = build_subject_sequences(subjects, _CFG)

    assert len(sequences) == 2
    for seq in sequences:
        # 8 hand-crafted features (5 bands + 3 Hjorth) per channel, 6 channels
        assert seq["features"].shape[1] == 8 * 6
        assert seq["features"].dtype == np.float32

    # shorter recording -> fewer 2-second epochs -> shorter sequence
    seq_lens = {s["subject_id"]: s["features"].shape[0] for s in sequences}
    assert seq_lens["pd1"] < seq_lens["hc1"]
