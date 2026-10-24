"""Audit paired predictions and choose development-only recall-constrained thresholds."""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha
from memeguard.metrics import binary_metrics


def select_threshold(labels, scores, target=0.90):
    candidates = sorted(set([0.0, *map(float, scores)]))
    feasible = [binary_metrics(labels, scores, t) for t in candidates]
    feasible = [
        m for m in feasible if m["recall"] is not None and m["recall"] >= target
    ]
    if not feasible:
        raise ValueError("Recall target is infeasible")
    return max(feasible, key=lambda m: (m["tn"], m["threshold"]))


def main():
    base = Path(
        "D:/career-projects/runs/MemeGuard/20261023-first-phase-v1/zero-shot/predictions.json"
    )
    adapted = Path(
        "D:/career-projects/runs/MemeGuard/20261024-sft-v1/evaluation/predictions.json"
    )
    first, second = read(base), read(adapted)
    if len(first) != 128 or len(second) != 128:
        raise ValueError("Development size changed")
    if len({r["id"] for r in first}) != 128:
        raise ValueError("Duplicate prediction ID")
    for a, b in zip(first, second):
        if any(a[k] != b[k] for k in ("id", "label", "image_sha256")):
            raise ValueError("Unpaired predictions")
    y = np.array([r["label"] for r in first])
    p = np.array([r["p_hateful_conditional"] for r in first])
    q = np.array([r["p_hateful_conditional"] for r in second])
    rng = np.random.default_rng(17)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    deltas = []
    for _ in range(1000):
        sample = np.concatenate(
            [rng.choice(negative, len(negative)), rng.choice(positive, len(positive))]
        )
        deltas.append(
            roc_auc_score(y[sample], q[sample]) - roc_auc_score(y[sample], p[sample])
        )
    config = {
        "scope": "Thresholds chosen on original 128 development records only; freeze before extra validation",
        "target_recall": 0.90,
        "baseline": select_threshold(y, p),
        "sft": select_threshold(y, q),
        "prediction_sha256": {"baseline": sha(base), "sft": sha(adapted)},
    }
    save(ROOT / "configs/thresholds-v1.json", config)
    report = {
        "n": 128,
        "baseline_fixed": binary_metrics(y, p),
        "sft_fixed": binary_metrics(y, q),
        "threshold_selection": config,
        "auroc_delta_bootstrap_95_percentile": np.quantile(
            deltas, [0.025, 0.975]
        ).tolist(),
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 17,
        "limitation": "Paired stratified sample bootstrap on development data; not independent-test or multi-training-seed evidence",
    }
    save(ROOT / "reports/sft-v1/diagnostics.json", report)
    print("THRESHOLD_DIAGNOSTICS", report, flush=True)


if __name__ == "__main__":
    main()
