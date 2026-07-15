"""
End-to-end experiment runner:
  load data -> preprocess -> extract features -> LOSO cross-validate the
  baseline model -> print results.

Usage:
    python scripts/run_experiment.py --bids-root data/raw --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_dataset
from src.evaluation import loso_cross_validate
from src.features import extract_features
from src.models import build_baseline_pipeline
from src.preprocessing import make_epochs, preprocess_raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bids-root", required=True, help="Path to the downloaded BIDS dataset"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    subjects = load_dataset(args.bids_root)
    print(f"Loaded {len(subjects)} subjects")

    all_features, all_labels, all_groups = [], [], []
    for subject in subjects:
        raw_clean = preprocess_raw(
            subject.raw,
            l_freq=cfg["preprocessing"]["l_freq"],
            h_freq=cfg["preprocessing"]["h_freq"],
            notch_freq=cfg["preprocessing"]["notch_freq"],
            n_ica_components=cfg["preprocessing"]["n_ica_components"],
        )
        epochs = make_epochs(
            raw_clean,
            duration=cfg["preprocessing"]["epoch_duration"],
            overlap=cfg["preprocessing"]["epoch_overlap"],
        )
        sfreq = raw_clean.info["sfreq"]
        for epoch_data in epochs.get_data():
            all_features.append(extract_features(epoch_data, sfreq))
            all_labels.append(subject.label)
            all_groups.append(subject.subject_id)

    X = np.array(all_features)
    y = np.array(all_labels)
    groups = np.array(all_groups)

    results = loso_cross_validate(
        lambda: build_baseline_pipeline(cfg["model"]["pca_components"]), X, y, groups
    )
    print(f"Epoch-level mean accuracy: {results['mean_epoch_accuracy']:.3f}")
    if results["epoch_auc"] is not None:
        print(f"Epoch-level ROC-AUC (pooled out-of-fold): {results['epoch_auc']:.3f}")
    print("Epoch-level confusion matrix:\n", results["epoch_confusion_matrix"])
    print(
        f"\nSubject-level accuracy (majority vote, n={results['n_subjects']}): "
        f"{results['subject_accuracy']:.3f}"
    )
    print("Subject-level confusion matrix:\n", results["subject_confusion_matrix"])


if __name__ == "__main__":
    main()
