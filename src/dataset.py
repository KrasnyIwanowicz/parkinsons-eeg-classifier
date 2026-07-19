"""
Per-subject sequence construction for the deep models.

The SVM baseline (run_experiment.py) treats every epoch as an
independent sample. The LSTM/Attention-LSTM instead need each subject's
whole recording as one sequence of per-epoch feature vectors — that's
the only way the "sequence" in LSTM means anything.
"""
from __future__ import annotations

import numpy as np

from src.data_loading import Subject
from src.features import extract_features
from src.preprocessing import make_epochs, preprocess_raw


def build_subject_sequences(subjects: list[Subject], preprocessing_cfg: dict) -> list[dict]:
    """Preprocess every subject once and return per-subject sequences.

    Returns a list of dicts: {"subject_id", "label", "features"} where
    features has shape (seq_len, feature_dim) — seq_len is that
    subject's epoch count (recordings are 3-5 min, so this varies
    subject to subject; the training loop in scripts/train_deep_model.py
    handles one subject at a time specifically so this variable length
    is never a problem).

    Computed once and reused across every LOSO fold — preprocessing
    doesn't depend on the train/test split, so redoing it inside each
    fold would just be wasted computation.
    """
    sequences = []
    for subject in subjects:
        raw_clean = preprocess_raw(
            subject.raw,
            l_freq=preprocessing_cfg["l_freq"],
            h_freq=preprocessing_cfg["h_freq"],
            notch_freq=preprocessing_cfg["notch_freq"],
            n_ica_components=preprocessing_cfg["n_ica_components"],
        )
        epochs = make_epochs(
            raw_clean,
            duration=preprocessing_cfg["epoch_duration"],
            overlap=preprocessing_cfg["epoch_overlap"],
        )
        sfreq = raw_clean.info["sfreq"]
        feature_seq = np.stack(
            [extract_features(epoch_data, sfreq) for epoch_data in epochs.get_data()]
        )
        sequences.append(
            {
                "subject_id": subject.subject_id,
                "label": subject.label,
                "features": feature_seq.astype(np.float32),
            }
        )
    return sequences
