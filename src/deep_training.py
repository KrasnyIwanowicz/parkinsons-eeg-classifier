"""
Training logic for the LSTM / Attention-LSTM models, factored out of
scripts/train_deep_model.py so it's unit-testable like everything else
in src/ — the script itself is just CLI plumbing around this.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def standardize_sequences(train_sequences: list[dict], test_sequence: dict) -> tuple[list[dict], dict]:
    """Z-score features using statistics from the training subjects only
    — fitting on the test subject too would leak information across the
    fold boundary, the exact mistake LOSO exists to avoid."""
    all_train_features = np.concatenate([s["features"] for s in train_sequences], axis=0)
    mean = all_train_features.mean(axis=0, keepdims=True)
    std = all_train_features.std(axis=0, keepdims=True) + 1e-8

    train_scaled = [{**s, "features": (s["features"] - mean) / std} for s in train_sequences]
    test_scaled = {**test_sequence, "features": (test_sequence["features"] - mean) / std}
    return train_scaled, test_scaled


def train_one_fold(
    model: nn.Module,
    train_sequences: list[dict],
    lr: float,
    weight_decay: float,
    n_train_epochs: int,
    device: "torch.device",
) -> nn.Module:
    """Train one model on one fold's training subjects.

    n_train_epochs = passes through the training subjects — unrelated to
    EEG epochs (2-second windows). "Epoch" is unfortunately overloaded
    terminology here, as it is throughout ML generally.

    Trains one subject-sequence at a time (batch size 1) rather than
    padding variable-length sequences into batches — simpler and fast
    enough at this dataset scale (~30 training subjects per fold).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(n_train_epochs):
        for idx in np.random.permutation(len(train_sequences)):
            seq = train_sequences[idx]
            x = torch.tensor(seq["features"], device=device).unsqueeze(0)
            y = torch.tensor([float(seq["label"])], device=device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_one(model: nn.Module, sequence: dict, device: "torch.device") -> float:
    """Predicted probability of the positive (PD) class for one subject."""
    model.eval()
    x = torch.tensor(sequence["features"], device=device).unsqueeze(0)
    return torch.sigmoid(model(x)).item()
