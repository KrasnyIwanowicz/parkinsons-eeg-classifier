"""
SHAP explainability for the SVM baseline: which frequency bands and
channels actually drove its predictions?

Important design note: the model here is fit on the FULL dataset (all
31 subjects), not per-LOSO-fold. That's deliberate, not an oversight —
Phase 2's LOSO numbers answer "how well does this generalize to a new
subject"; this script answers a different question, "what does the
model, with everything it can see, actually key off of?" Those call for
different models and that's normal practice, not a contradiction.

Usage:
    python scripts/explain_baseline.py --bids-root data/raw

    # quick sanity check with a smaller background/explain set first —
    # KernelExplainer is the slow part, this cuts runtime a lot:
    python scripts/explain_baseline.py --bids-root data/raw --n-background 15 --n-explain 10 --nsamples 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_dataset
from src.explainability import (
    aggregate_by_band,
    aggregate_by_channel,
    build_feature_labels,
    explain_baseline,
)
from src.features import extract_features
from src.models import build_baseline_pipeline
from src.preprocessing import make_epochs, preprocess_raw


def build_dataset(subjects, cfg):
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
    return np.array(all_features), np.array(all_labels), np.array(all_groups)


def plot_band_importance(band_importance: dict, out_path: Path):
    names = list(band_importance.keys())
    values = [band_importance[n] for n in names]
    order = np.argsort(values)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[i] for i in order][::-1], [values[i] for i in order][::-1], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| (averaged across channels)")
    ax.set_title("Feature-type importance — SVM baseline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_channel_topomap(channel_importance: dict, ch_names: list[str], out_path: Path):
    values = np.array([channel_importance[ch] for ch in ch_names])

    info = mne.create_info(ch_names, sfreq=256.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, match_case=False, on_missing="raise")

    fig, ax = plt.subplots(figsize=(6, 6))
    im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap="Reds")
    ax.set_title("Channel importance — SVM baseline\n(mean |SHAP|, averaged across feature types)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-background", type=int, default=50)
    parser.add_argument("--n-explain", type=int, default=62)
    parser.add_argument(
        "--nsamples", type=int, default=300,
        help="SHAP KernelExplainer samples per explained instance — lower is faster, noisier.",
    )
    parser.add_argument(
        "--pca-components", type=int, default=None,
        help="Override configs/default.yaml's model.pca_components. Use 0 to skip PCA "
             "entirely — recommended once, to check whether PCA is discarding feature "
             "types before SHAP ever sees them.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    subjects = load_dataset(args.bids_root)
    print(f"Loaded {len(subjects)} subjects")
    ch_names = list(subjects[0].raw.ch_names)  # same montage/order across subjects

    X, y, groups = build_dataset(subjects, cfg)
    print(f"Built {X.shape[0]} epochs x {X.shape[1]} features")

    pca_components = (
        args.pca_components if args.pca_components is not None else cfg["model"]["pca_components"]
    )
    pca_components = None if pca_components == 0 else pca_components
    print(f"PCA components: {pca_components if pca_components is not None else 'skipped'}")

    model = build_baseline_pipeline(pca_components)
    model.fit(X, y)
    print(f"Final model training accuracy (not a generalization estimate): {model.score(X, y):.3f}")

    rng = np.random.default_rng(args.seed)
    background_idx = rng.choice(len(X), size=min(args.n_background, len(X)), replace=False)
    explain_idx = rng.choice(len(X), size=min(args.n_explain, len(X)), replace=False)

    print(
        f"Running SHAP KernelExplainer: {len(background_idx)} background, "
        f"{len(explain_idx)} to explain, nsamples={args.nsamples} "
        "(slow step — can take several minutes)"
    )
    shap_values = explain_baseline(model, X[background_idx], X[explain_idx], nsamples=args.nsamples)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_labels = build_feature_labels(ch_names)
    band_importance = aggregate_by_band(mean_abs_shap, len(ch_names))
    channel_importance = aggregate_by_channel(mean_abs_shap, ch_names)

    print("\n=== Importance by feature type (mean |SHAP|, averaged over channels) ===")
    for name, val in sorted(band_importance.items(), key=lambda kv: -kv[1]):
        print(f"  {name:20s} {val:.5f}")

    print("\n=== Top 10 channels (mean |SHAP|, averaged over feature types) ===")
    for name, val in sorted(channel_importance.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {name:6s} {val:.5f}")

    top_n = 15
    top_idx = np.argsort(-mean_abs_shap)[:top_n]
    print(f"\n=== Top {top_n} individual features ===")
    for idx in top_idx:
        print(f"  {feature_labels[idx]:25s} {mean_abs_shap[idx]:.5f}")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    np.savez(
        out_dir / "shap_baseline.npz",
        mean_abs_shap=mean_abs_shap,
        feature_labels=np.array(feature_labels),
        ch_names=np.array(ch_names),
    )
    plot_band_importance(band_importance, out_dir / "shap_band_importance.png")
    plot_channel_topomap(channel_importance, ch_names, out_dir / "shap_channel_topomap.png")

    print(f"\nSaved: {out_dir / 'shap_baseline.npz'}")
    print(f"Saved: {out_dir / 'shap_band_importance.png'}")
    print(f"Saved: {out_dir / 'shap_channel_topomap.png'}")


if __name__ == "__main__":
    main()
