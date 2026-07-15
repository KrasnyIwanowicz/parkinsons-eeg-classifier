"""
Leave-one-subject-out (LOSO) cross-validation.

This is the validation strategy that matters for EEG: random or
epoch-level splits let epochs from the same subject appear in both train
and test, which leaks subject identity and inflates accuracy. LOSO holds
out one entire subject per fold, so the reported score actually reflects
generalization to a new person — not memorized subject-specific noise.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut


def loso_cross_validate(model_builder, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Run LOSO CV.

    Parameters
    ----------
    model_builder : callable, () -> sklearn-compatible estimator
        Called fresh every fold so no fitted state leaks between folds.
    X : array, shape (n_epochs, n_features)
    y : array, shape (n_epochs,) — binary labels (1 = PD, 0 = control)
    groups : array, shape (n_epochs,) — subject id per epoch

    Returns
    -------
    dict with per-fold and aggregate accuracy / ROC-AUC, plus the summed
    confusion matrix across all folds.
    """
    logo = LeaveOneGroupOut()
    fold_acc, fold_auc = [], []
    total_cm = np.zeros((2, 2), dtype=int)

    for train_idx, test_idx in logo.split(X, y, groups):
        model = model_builder()
        model.fit(X[train_idx], y[train_idx])

        preds = model.predict(X[test_idx])
        probs = (
            model.predict_proba(X[test_idx])[:, 1]
            if hasattr(model, "predict_proba")
            else preds
        )

        fold_acc.append(accuracy_score(y[test_idx], preds))
        # ROC-AUC is undefined when the held-out subject's epochs are all
        # one class (common with few epochs per subject) — skip those folds.
        if len(np.unique(y[test_idx])) > 1:
            fold_auc.append(roc_auc_score(y[test_idx], probs))
        total_cm += confusion_matrix(y[test_idx], preds, labels=[0, 1])

    return {
        "fold_accuracy": fold_acc,
        "mean_accuracy": float(np.mean(fold_acc)),
        "fold_auc": fold_auc,
        "mean_auc": float(np.mean(fold_auc)) if fold_auc else None,
        "confusion_matrix": total_cm,
    }
