"""
Data loading utilities for the UC San Diego Resting-State EEG Parkinson's
Disease dataset (OpenNeuro ds002778).

Dataset citation:
    Swann, N.C. (2021). UC San Diego Resting State EEG Data from Patients
    with Parkinson's Disease. OpenNeuro. [Dataset]
    https://doi.org/10.18112/openneuro.ds002778.v1.0.5

The dataset is NOT bundled with this repository — it's clinical EEG data
and doesn't belong in git. Download it yourself before running anything:

    pip install openneuro-py
    openneuro-py download --dataset ds002778 --target data/raw

or via DataLad:

    datalad install https://github.com/OpenNeuroDatasets/ds002778.git data/raw
    cd data/raw && datalad get .

This module then reads the resulting BIDS-formatted directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import mne
    from mne_bids import BIDSPath, read_raw_bids
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "mne and mne-bids are required. Install with `pip install mne mne-bids`."
    ) from exc


@dataclass
class Subject:
    subject_id: str
    label: int  # 1 = PD, 0 = control
    raw: "mne.io.Raw"


def load_participants_table(bids_root: Path) -> pd.DataFrame:
    """Load participants.tsv and derive a binary PD/control label column."""
    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    # ds002778 encodes group as e.g. "PD" / "CTL" — the exact column name
    # has varied slightly across dataset versions, so check both.
    group_col = "Group" if "Group" in participants.columns else "group"
    participants["label"] = (participants[group_col].str.upper() == "PD").astype(int)
    return participants


def load_subject_raw(bids_root: Path, subject_id: str, task: str = "rest") -> "mne.io.Raw":
    """Load a single subject's raw EEG recording via MNE-BIDS."""
    bids_path = BIDSPath(subject=subject_id, task=task, root=bids_root)
    return read_raw_bids(bids_path, verbose=False)


def load_dataset(bids_root: str | Path, task: str = "rest") -> list[Subject]:
    """Load every subject's raw recording + label.

    Returns a list of Subject(subject_id, label, raw) rather than one big
    concatenated array — keeping subject identity attached here is what
    makes the leave-one-subject-out cross-validation in evaluation.py
    possible later.
    """
    bids_root = Path(bids_root)
    participants = load_participants_table(bids_root)

    subjects = []
    for _, row in participants.iterrows():
        sub_id = str(row["participant_id"]).replace("sub-", "")
        try:
            raw = load_subject_raw(bids_root, sub_id, task=task)
        except FileNotFoundError:
            continue  # some public releases have partial coverage
        subjects.append(Subject(subject_id=sub_id, label=int(row["label"]), raw=raw))
    return subjects
