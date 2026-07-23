"""
Explainability layer.

Primary result: SHAP values on the baseline SVM, since Phase 2 showed
it's the only model that beats chance. Attention-weight extraction for
the Attention-LSTM is kept here for completeness, but Phase 2 found
that model performs at chance (~0.51-0.54 accuracy across seeds) — so
its attention weights aren't expected to reflect a meaningful,
class-relevant signal, and shouldn't be reported as if they do.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

# Order must exactly match extract_features() in features.py: 5 band
# powers then 3 Hjorth parameters, each block ordered by channel.
BANDS_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
HJORTH_ORDER = ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
FEATURE_GROUPS = BANDS_ORDER + HJORTH_ORDER


def build_feature_labels(ch_names: list[str]) -> list[str]:
    """One label per entry in the 256-length feature vector, e.g. 'Cz_beta'."""
    return [f"{ch}_{group}" for group in FEATURE_GROUPS for ch in ch_names]


def explain_baseline(
    model, X_background: np.ndarray, X_explain: np.ndarray, nsamples: int = 300
) -> np.ndarray:
    """SHAP KernelExplainer over the baseline sklearn pipeline.

    Returns SHAP values for the positive (PD) class only, shape
    (n_explain, n_features). Handles both the list-of-arrays convention
    (older SHAP versions) and the stacked (n_samples, n_features,
    n_classes) array (SHAP >= ~0.45) — verified against the installed
    version, but written defensively since this differs across versions.

    X_background should be a small representative sample (SHAP recommends
    roughly 50-100 rows) used to estimate the model's expected output.
    """
    if shap is None:
        raise ImportError("Install shap: pip install shap")

    explainer = shap.KernelExplainer(model.predict_proba, X_background)
    raw_values = explainer.shap_values(X_explain, nsamples=nsamples)

    if isinstance(raw_values, list):
        return np.asarray(raw_values[1])  # class 1 = PD
    raw_values = np.asarray(raw_values)
    if raw_values.ndim == 3:
        return raw_values[:, :, 1]
    return raw_values


def aggregate_by_band(mean_abs_shap: np.ndarray, n_channels: int) -> dict[str, float]:
    """Average |SHAP| across channels, grouped by feature type (5 bands
    + 3 Hjorth parameters). Uses index arithmetic rather than string
    parsing of labels, since channel names could in principle collide
    as string prefixes of each other."""
    result = {}
    for i, group in enumerate(FEATURE_GROUPS):
        start, end = i * n_channels, (i + 1) * n_channels
        result[group] = float(mean_abs_shap[start:end].mean())
    return result


def aggregate_by_channel(mean_abs_shap: np.ndarray, ch_names: list[str]) -> dict[str, float]:
    """Average |SHAP| across feature types, grouped by channel."""
    n_channels = len(ch_names)
    n_groups = len(FEATURE_GROUPS)
    result = {}
    for j, ch in enumerate(ch_names):
        idx = [j + k * n_channels for k in range(n_groups)]
        result[ch] = float(np.mean(mean_abs_shap[idx]))
    return result


def attention_weights_for_batch(model: "torch.nn.Module", x_batch: "torch.Tensor") -> np.ndarray:
    """Run the attention-LSTM in explain mode and return per-timestep
    attention weights, shape (batch, seq_len). See module docstring —
    not expected to be meaningful given Phase 2's chance-level result."""
    import torch

    model.eval()
    with torch.no_grad():
        _, weights = model(x_batch, return_attention=True)
    return weights.cpu().numpy()
