"""
LOSO-evaluate the LSTM or Attention-LSTM classifier.

Each subject's whole recording is one sequence, trained one subject at a
time. With ~30 training subjects per fold, that's simpler and just as
fast as padding/packing variable-length sequences into real batches, and
it sidesteps the bug surface that comes with getting
pack_padded_sequence wrong.

Usage:
    python scripts/train_deep_model.py --bids-root data/raw --model lstm
    python scripts/train_deep_model.py --bids-root data/raw --model attention

    # quick sanity check on a handful of folds before committing to all 31:
    python scripts/train_deep_model.py --bids-root data/raw --model lstm --max-folds 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_dataset
from src.dataset import build_subject_sequences
from src.deep_training import predict_one, standardize_sequences, train_one_fold
from src.models import AttentionLSTMClassifier, LSTMClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["lstm", "attention"], default="lstm")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-train-epochs", type=int, default=30)
    parser.add_argument(
        "--hidden-dim", type=int, default=None,
        help="Overrides configs/default.yaml's model.lstm_hidden_dim if set.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout before the final classifier layer.",
    )
    parser.add_argument(
        "--max-folds", type=int, default=None,
        help="Only run this many LOSO folds — for a quick sanity check before the full run.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for weight init and training-order shuffling. Fixed by default "
             "so results are reproducible — without this, results can swing substantially "
             "run to run on a dataset this small.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    subjects = load_dataset(args.bids_root)
    print(f"Loaded {len(subjects)} subjects")

    sequences = build_subject_sequences(subjects, cfg["preprocessing"])
    input_dim = sequences[0]["features"].shape[1]
    print(f"Built {len(sequences)} subject sequences, feature dim = {input_dim}")

    model_cls = LSTMClassifier if args.model == "lstm" else AttentionLSTMClassifier
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else cfg["model"]["lstm_hidden_dim"]

    subject_ids = np.array([s["subject_id"] for s in sequences])
    labels = np.array([s["label"] for s in sequences])
    logo = LeaveOneGroupOut()

    pooled_true, pooled_prob = [], []
    start = time.time()
    for fold_i, (train_idx, test_idx) in enumerate(
        logo.split(np.zeros(len(sequences)), labels, subject_ids)
    ):
        if args.max_folds is not None and fold_i >= args.max_folds:
            print(f"Stopping after {args.max_folds} folds (--max-folds)")
            break

        train_sequences = [sequences[i] for i in train_idx]
        test_sequence = sequences[test_idx[0]]
        train_sequences, test_sequence = standardize_sequences(train_sequences, test_sequence)

        model = model_cls(input_dim=input_dim, hidden_dim=hidden_dim, dropout=args.dropout).to(device)
        model = train_one_fold(
            model, train_sequences, args.lr, args.weight_decay, args.n_train_epochs, device
        )

        prob = predict_one(model, test_sequence, device)
        pooled_true.append(test_sequence["label"])
        pooled_prob.append(prob)
        elapsed = time.time() - start
        print(
            f"[{fold_i + 1}/{len(sequences)}, {elapsed:.0f}s elapsed] "
            f"sub-{test_sequence['subject_id']}: true={test_sequence['label']}, "
            f"pred_prob={prob:.3f}"
        )

    pooled_true = np.array(pooled_true)
    pooled_prob = np.array(pooled_prob)
    pooled_pred = (pooled_prob > 0.5).astype(int)

    acc = accuracy_score(pooled_true, pooled_pred)
    auc = roc_auc_score(pooled_true, pooled_prob) if len(np.unique(pooled_true)) > 1 else None
    cm = confusion_matrix(pooled_true, pooled_pred, labels=[0, 1])

    print(f"\n{args.model} — subject-level LOSO accuracy: {acc:.3f}")
    if auc is not None:
        print(f"{args.model} — subject-level LOSO ROC-AUC: {auc:.3f}")
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
