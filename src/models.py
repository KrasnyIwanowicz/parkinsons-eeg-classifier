"""
Model definitions:
  - a classical ML baseline (StandardScaler -> PCA -> SVM) — the floor
    every deep model here has to beat
  - a plain LSTM classifier over per-epoch feature sequences
  - an attention-augmented LSTM, where the attention weights double as
    an explainability signal (see explainability.py)
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_baseline_pipeline(n_pca_components: int | None = 10) -> Pipeline:
    """StandardScaler -> [PCA] -> SVM baseline.

    n_pca_components=None skips PCA entirely, so the SVM sees all 256
    standardized features directly. Useful for checking whether PCA is
    bottlenecking information flow — with only 10 components out of 256
    features, PCA finds directions of maximum *variance*, which has no
    guarantee of aligning with what's actually predictive. If a feature
    type shows exactly zero SHAP importance with PCA on but nonzero with
    it off, that's PCA discarding it, not genuine unimportance.
    """
    steps = [("scale", StandardScaler())]
    if n_pca_components is not None:
        steps.append(("pca", PCA(n_components=n_pca_components, random_state=42)))
    steps.append(("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")))
    return Pipeline(steps)


try:
    import torch
    import torch.nn as nn

    class LSTMClassifier(nn.Module):
        """LSTM over a sequence of per-epoch feature vectors; final
        hidden state feeds a linear classifier. Dropout before the
        classifier head matters more than usual here — with ~30 training
        subjects per LOSO fold, overfitting is the default outcome
        without it."""

        def __init__(
            self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.3
        ):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x):  # x: (batch, seq_len, input_dim)
            _, (h_n, _) = self.lstm(x)
            return self.classifier(self.dropout(h_n[-1])).squeeze(-1)

    class AttentionLSTMClassifier(nn.Module):
        """LSTM with additive (Bahdanau-style) attention over hidden
        states instead of just the final one. The attention weights are
        a built-in explainability signal: they show which time windows
        in the sequence drove the prediction."""

        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.attn_score = nn.Linear(hidden_dim, 1)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x, return_attention: bool = False):
            outputs, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
            scores = self.attn_score(outputs).squeeze(-1)  # (batch, seq_len)
            weights = torch.softmax(scores, dim=-1)
            context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
            logits = self.classifier(self.dropout(context)).squeeze(-1)
            return (logits, weights) if return_attention else logits

except ImportError:  # pragma: no cover
    LSTMClassifier = None
    AttentionLSTMClassifier = None
