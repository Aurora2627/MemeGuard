"""Audit the completed scaling run against paired development predictions."""

from pathlib import Path
import json, shutil, hashlib
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("D:/career-projects/runs/MemeGuard/20261026-scale-resume112-v1")
OLD = Path("D:/career-projects/runs/MemeGuard/20261024-validation-v1")
OUT = ROOT / "reports/scaling-results-v1"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert read(RUN / "status.json")["status"] == "complete"
    assert read(RUN / "training/status.json")["optimizer_steps"] == 512
    new = read(RUN / "development-predictions.json")
    assert len(new) == 372 and len({r["id"] for r in new}) == 372
    labels = np.array([r["label"] for r in new])
    score = np.array([r["p_hateful_conditional"] for r in new])
    result = {
        "scope": "Reused 372-row development set; no independent test claim",
        "bootstrap_samples": 2000,
        "bootstrap_seed": 17,
        "comparisons": {},
    }
    for name in ("baseline", "sft"):
        p = OLD / name / "predictions.json"
        old = read(p)
        by_id = {r["id"]: r for r in old}
        assert len(by_id) == len(old) == 372
        aligned = [by_id[r["id"]] for r in new]
        assert all(
            (a["label"], a["image_sha256"]) == (b["label"], b["image_sha256"])
            for a, b in zip(new, aligned)
        )
        prev = np.array([r["p_hateful_conditional"] for r in aligned])
        rng = np.random.default_rng(17)
        deltas = []
        for _ in range(2000):
            idx = rng.integers(0, len(labels), len(labels))
            if len(np.unique(labels[idx])) == 2:
                deltas.append(
                    roc_auc_score(labels[idx], score[idx])
                    - roc_auc_score(labels[idx], prev[idx])
                )
        result["comparisons"][name] = {
            "old_auroc": float(roc_auc_score(labels, prev)),
            "new_auroc": float(roc_auc_score(labels, score)),
            "delta_auroc": float(
                roc_auc_score(labels, score) - roc_auc_score(labels, prev)
            ),
            "paired_percentile_ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
            "old_predictions_sha256": sha(p),
            "new_predictions_sha256": sha(RUN / "development-predictions.json"),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (
        "development-metrics.json",
        "frozen-threshold.json",
        "status.json",
        "recovery-receipt.json",
    ):
        shutil.copy2(RUN / name, OUT / name)
    shutil.copy2(RUN / "training/status.json", OUT / "training-status.json")
    (OUT / "paired-comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
