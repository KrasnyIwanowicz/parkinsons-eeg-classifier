"""
Tests for src/deep_training.py using small synthetic, separable
per-subject sequences — fast, deterministic, no real EEG needed.
"""
import numpy as np
import torch

from src.deep_training import predict_one, standardize_sequences, train_one_fold
from src.models import AttentionLSTMClassifier, LSTMClassifier


def _make_synthetic_sequences(n_subjects=6, feature_dim=10, seed=0):
    rng = np.random.default_rng(seed)
    sequences = []
    for i in range(n_subjects):
        label = i % 2
        seq_len = int(rng.integers(15, 30))
        base = 0.0 if label == 0 else 2.5
        features = (base + rng.normal(scale=0.4, size=(seq_len, feature_dim))).astype(np.float32)
        sequences.append({"subject_id": f"s{i}", "label": label, "features": features})
    return sequences


def test_standardize_sequences_zero_means_train_features():
    sequences = _make_synthetic_sequences()
    train_seqs, test_seq = standardize_sequences(sequences[:-1], sequences[-1])
    all_train = np.concatenate([s["features"] for s in train_seqs], axis=0)
    assert np.allclose(all_train.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(all_train.std(axis=0), 1, atol=1e-3)
    # test sequence uses train statistics, not its own — shape preserved, not necessarily zero-mean
    assert test_seq["features"].shape == sequences[-1]["features"].shape


def test_train_one_fold_and_predict_recover_separable_signal():
    """Not a claim about real EEG performance — a check that the
    training loop, forward pass, and prediction plumbing are wired up
    correctly by confirming they can learn an easy synthetic case."""
    sequences = _make_synthetic_sequences(n_subjects=8, seed=1)
    device = torch.device("cpu")

    for model_cls in (LSTMClassifier, AttentionLSTMClassifier):
        train_seqs, test_seq = standardize_sequences(sequences[:-1], sequences[-1])
        model = model_cls(input_dim=10, hidden_dim=8)
        model = train_one_fold(
            model, train_seqs, lr=1e-2, weight_decay=1e-4, n_train_epochs=8, device=device
        )
        prob = predict_one(model, test_seq, device)
        assert 0.0 <= prob <= 1.0

        # model should assign higher probability to the correct class on
        # a couple of held-out training-distribution sanity checks
        pos_seq = {**test_seq, "features": np.full((10, 10), 2.5, dtype=np.float32)}
        neg_seq = {**test_seq, "features": np.full((10, 10), 0.0, dtype=np.float32)}
        pos_prob = predict_one(model, pos_seq, device)
        neg_prob = predict_one(model, neg_seq, device)
        assert pos_prob > neg_prob
