"""
Explainability layer.

Two different models, two different explanation mechanisms, same
underlying question: which frequency bands / channels / time windows
drove this prediction? That question matters far more for a
clinical-adjacent model than raw accuracy does.
"""
from __future__ import annotations

import numpy as np

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def explain_baseline(model, X_background: np.ndarray, X_explain: np.ndarray):
    """SHAP KernelExplainer over the baseline sklearn pipeline.

    X_background should be a small representative sample (SHAP recommends
    roughly 50-100 rows) used to estimate the model's expected output.
    """
    if shap is None:
        raise ImportError("Install shap: pip install shap")
    explainer = shap.KernelExplainer(model.predict_proba, X_background)
    return explainer.shap_values(X_explain)


def attention_weights_for_batch(model, x_batch) -> np.ndarray:
    """Run the attention-LSTM in explain mode and return per-timestep
    attention weights, shape (batch, seq_len)."""
    import torch

    model.eval()
    with torch.no_grad():
        _, weights = model(x_batch, return_attention=True)
    return weights.cpu().numpy()
