"""
Tests for src/explainability.py. Uses a synthetic dataset where exactly
one feature is predictive and the rest are noise, so SHAP importance
has a known ground truth to check against — not just "did it run
without crashing".
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.explainability import (
    FEATURE_GROUPS,
    aggregate_by_band,
    aggregate_by_channel,
    build_feature_labels,
    explain_baseline,
)


def test_build_feature_labels_length_and_order():
    ch_names = ["Fp1", "Cz", "O1"]
    labels = build_feature_labels(ch_names)
    assert len(labels) == len(ch_names) * len(FEATURE_GROUPS)
    # first block is delta, ordered by channel
    assert labels[:3] == ["Fp1_delta", "Cz_delta", "O1_delta"]


def test_shap_and_aggregation_recover_known_signal():
    """One feature (Cz's beta-band slot) is made predictive; everything
    else is noise. SHAP + both aggregations should surface exactly that
    feature/band/channel as most important."""
    rng = np.random.default_rng(0)
    ch_names = ["Fp1", "Cz", "O1"]
    n_ch = len(ch_names)
    n_features = n_ch * len(FEATURE_GROUPS)

    beta_cz_idx = FEATURE_GROUPS.index("beta") * n_ch + ch_names.index("Cz")
    X = rng.normal(size=(60, n_features))
    y = (X[:, beta_cz_idx] > 0).astype(int)

    model = Pipeline([("scale", StandardScaler()), ("svm", SVC(probability=True, kernel="rbf"))])
    model.fit(X, y)

    shap_values = explain_baseline(model, X[:20], X[20:35], nsamples=100)
    assert shap_values.shape == (15, n_features)

    mean_abs = np.abs(shap_values).mean(axis=0)
    assert np.argmax(mean_abs) == beta_cz_idx

    band_importance = aggregate_by_band(mean_abs, n_ch)
    assert max(band_importance, key=band_importance.get) == "beta"

    channel_importance = aggregate_by_channel(mean_abs, ch_names)
    assert max(channel_importance, key=channel_importance.get) == "Cz"
