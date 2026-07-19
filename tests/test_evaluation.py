"""
Tests for loso_cross_validate using a small synthetic, separable dataset
— fast, deterministic, and no real EEG data needed.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.evaluation import loso_cross_validate


def _make_synthetic_epochs():
    """4 subjects (2 control, 2 PD), 6 epochs each, with a feature that
    cleanly separates the two classes so a simple model can actually
    learn something — this is about exercising the aggregation logic,
    not testing classifier quality."""
    rng = np.random.default_rng(0)
    X, y, groups = [], [], []
    subjects = [("s1", 0), ("s2", 0), ("s3", 1), ("s4", 1)]
    for subj, label in subjects:
        base = 0.0 if label == 0 else 3.0
        for _ in range(6):
            X.append([base + rng.normal(scale=0.3), rng.normal(scale=0.3)])
            y.append(label)
            groups.append(subj)
    return np.array(X), np.array(y), np.array(groups)


def test_loso_cross_validate_returns_expected_keys():
    X, y, groups = _make_synthetic_epochs()
    results = loso_cross_validate(lambda: LogisticRegression(), X, y, groups)

    expected_keys = {
        "fold_accuracy", "mean_epoch_accuracy", "epoch_auc",
        "epoch_confusion_matrix", "subject_accuracy",
        "subject_confusion_matrix", "n_subjects",
    }
    assert expected_keys.issubset(results.keys())
    assert results["n_subjects"] == 4


def test_loso_cross_validate_auc_is_not_none():
    """Regression test: AUC must be computed from probabilities pooled
    across all folds, not per-fold — per-fold AUC is undefined in
    subject-level LOSO because a single held-out subject is always
    entirely one class, so a per-fold implementation silently returns
    None every single time."""
    X, y, groups = _make_synthetic_epochs()
    results = loso_cross_validate(lambda: LogisticRegression(), X, y, groups)
    assert results["epoch_auc"] is not None
    assert 0.0 <= results["epoch_auc"] <= 1.0


def test_loso_cross_validate_recovers_clean_separation():
    """With a clearly separable synthetic feature, both epoch- and
    subject-level accuracy should be high — a sanity check that the
    aggregation logic isn't scrambling labels somewhere."""
    X, y, groups = _make_synthetic_epochs()
    results = loso_cross_validate(lambda: LogisticRegression(), X, y, groups)
    assert results["mean_epoch_accuracy"] > 0.9
    assert results["subject_accuracy"] == 1.0
