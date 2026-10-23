"""Binary discrimination metrics; conditional label scores are not calibrated risks."""

import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels, probabilities, threshold=0.5):
    y = torch.as_tensor(labels)
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    if y.ndim != 1 or p.shape != y.shape or y.numel() == 0:
        raise ValueError("Nonempty aligned vectors required")
    if not ((y == 0) | (y == 1)).all() or not torch.isfinite(p).all():
        raise ValueError("Invalid labels or scores")
    if (p < 0).any() or (p > 1).any() or not 0 <= threshold <= 1:
        raise ValueError("Scores and threshold must be in [0, 1]")
    pred, positive = p >= threshold, y == 1
    tp = int((pred & positive).sum())
    fp = int((pred & ~positive).sum())
    fn = int((~pred & positive).sum())
    tn = int((~pred & ~positive).sum())
    both_classes = len(torch.unique(y)) == 2
    return {
        "n": int(y.numel()),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / y.numel(),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "auroc": float(roc_auc_score(y.numpy(), p.numpy())) if both_classes else None,
        "auprc": (
            float(average_precision_score(y.numpy(), p.numpy()))
            if both_classes
            else None
        ),
    }
