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
    """Load participants.tsv and derive a binary PD/control label column.

    NOTE: ds002778's participants.tsv has no "Group"/"diagnosis" column at
    all — the label lives only in the participant_id prefix itself
    ("sub-hc*" = healthy control, "sub-pd*" = Parkinson's). Verified
    directly against the dataset's participants.tsv on GitHub.
    """
    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    participants["label"] = participants["participant_id"].str.contains("pd").astype(int)
    return participants


def _infer_session(subject_id: str) -> str:
    """ds002778 session naming: healthy controls were recorded once
    (session "hc"); PD patients were recorded twice — off medication
    ("off") and on medication ("on"). We default to "off" so the
    PD-vs-control comparison isn't confounded by medication effects on
    the EEG itself. Pass session= explicitly to load_subject_raw if you
    want the "on" condition instead (e.g. for a separate on/off analysis)."""
    if subject_id.startswith("hc"):
        return "hc"
    if subject_id.startswith("pd"):
        return "off"
    raise ValueError(f"Unrecognized subject id prefix: {subject_id!r}")


def load_subject_raw(
    bids_root: Path, subject_id: str, task: str = "rest", session: str | None = None
) -> "mne.io.Raw":
    """Load a single subject's raw EEG recording via MNE-BIDS, restricted
    to the 32 scalp EEG channels.

    ds002778's raw files carry 41 channels, not 32: 32 named scalp
    electrodes (Fp1...Cz) plus 8 BioSemi "EXG" auxiliary channels
    (references/EOG-adjacent, not part of the scalp montage) and one
    "Status" trigger/event-code channel — not physiological signal at
    all. Verified directly against a loaded subject's raw.ch_names.
    Without this, feature extraction was silently running on all 41
    "channels" instead of the intended 32, for every model so far.
    """
    if session is None:
        session = _infer_session(subject_id)
    bids_path = BIDSPath(subject=subject_id, session=session, task=task, root=bids_root)
    raw = read_raw_bids(bids_path, verbose=False)

    non_scalp = ["Status"] + [f"EXG{i}" for i in range(1, 9)]
    to_drop = [ch for ch in non_scalp if ch in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)
    return raw


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
        except Exception as exc:  # noqa: BLE001 — deliberately broad
            # A couple of subjects in this dataset ship as preprocessed
            # .mat files instead of raw BIDS EEG for one condition; skip
            # anything that fails to load rather than crashing the whole
            # run, but say so loudly so it isn't silently swept away.
            print(f"[load_dataset] skipping sub-{sub_id}: {exc}")
            continue
        subjects.append(Subject(subject_id=sub_id, label=int(row["label"]), raw=raw))
    return subjects
