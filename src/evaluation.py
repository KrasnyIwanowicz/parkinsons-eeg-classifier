"""
Leave-one-subject-out (LOSO) cross-validation.

This is the validation strategy that matters for EEG: random or
epoch-level splits let epochs from the same subject appear in both train
and test, which leaks subject identity and inflates accuracy. LOSO holds
out one entire subject per fold, so the reported score actually reflects
generalization to a new person — not memorized subject-specific noise.

Two important details that are easy to get wrong here (both fixed after
hitting them on real data):

1. ROC-AUC cannot be computed per fold. Each fold's test set is exactly
   one subject, and a subject is entirely one class — so y_test never
   has both classes present, ever, by construction. AUC has to be
   computed once, at the end, over probabilities pooled across all folds.
2. Per-epoch accuracy treats every 2-second window as an independent
   case, but epochs from the same subject aren't independent of each
   other. Per-subject accuracy (majority vote across that subject's
   epochs) is the more honest, more clinically meaningful number.
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
    dict with epoch-level accuracy/AUC/confusion-matrix, and
    subject-level accuracy/confusion-matrix (majority vote per subject
    over that subject's out-of-fold epoch predictions).
    """
    logo = LeaveOneGroupOut()
    fold_acc = []
    epoch_cm = np.zeros((2, 2), dtype=int)

    pooled_y_true, pooled_y_prob, pooled_groups = [], [], []

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
        epoch_cm += confusion_matrix(y[test_idx], preds, labels=[0, 1])

        pooled_y_true.extend(y[test_idx])
        pooled_y_prob.extend(probs)
        pooled_groups.extend(groups[test_idx])

    pooled_y_true_arr = np.asarray(pooled_y_true)
    pooled_y_prob_arr = np.asarray(pooled_y_prob)
    pooled_groups_arr = np.asarray(pooled_groups)

    epoch_auc = (
        roc_auc_score(pooled_y_true_arr, pooled_y_prob_arr)
        if len(np.unique(pooled_y_true_arr)) > 1
        else None
    )

    # Subject-level: majority vote across each subject's out-of-fold epochs.
    subject_true, subject_pred = [], []
    for subj in np.unique(pooled_groups_arr):
        mask = pooled_groups_arr == subj
        subject_true.append(int(pooled_y_true_arr[mask][0]))  # one label per subject
        subject_pred.append(int(np.round((pooled_y_prob_arr[mask] > 0.5).mean())))

    return {
        "fold_accuracy": fold_acc,
        "mean_epoch_accuracy": float(np.mean(fold_acc)),
        "epoch_auc": epoch_auc,
        "epoch_confusion_matrix": epoch_cm,
        "subject_accuracy": accuracy_score(subject_true, subject_pred),
        "subject_confusion_matrix": confusion_matrix(subject_true, subject_pred, labels=[0, 1]),
        "n_subjects": len(subject_true),
    }
