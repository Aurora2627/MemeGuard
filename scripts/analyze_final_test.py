"""Verify frozen final predictions and export aggregate test evidence; never tune on test."""

from pathlib import Path
import hashlib
import json
import math
import shutil
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from final_test import verify_freeze, validate_prefix
from memeguard.metrics import binary_metrics

RUN = ROOT.parent / "runs/MemeGuard/20261027-final-test-v1"
OUT = ROOT / "reports/final-test-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verify_freeze()
    status = read(RUN / "status.json")
    assert status["status"] == "complete"
    assert not (RUN / "failure.json").exists()
    audit = read(RUN / "test-audit.json")
    manifest = read(RUN / "test-manifest.json")
    assert audit["test_records"] == len(manifest) == 1000
    assert audit["cross_split_exact_test_pairs"] == 0
    assert audit["manifest_sha256"] == sha(RUN / "test-manifest.json")
    records, metrics = {}, {}
    for variant in ("baseline", "selected"):
        pred_path = RUN / (variant + "-predictions.json")
        records[variant] = read(pred_path)
        validate_prefix(records[variant], manifest)
        assert len(records[variant]) == 1000
        metrics[variant] = read(RUN / (variant + "-metrics.json"))
        assert metrics[variant]["predictions_sha256"] == sha(pred_path)
        assert metrics[variant]["freeze_sha256"] == status["freeze_sha256"]
        y = [r["label"] for r in records[variant]]
        scores = [r["p_hateful_conditional"] for r in records[variant]]
        for operating_point in ("frozen_threshold", "fixed_0_5"):
            saved = metrics[variant][operating_point]
            actual = binary_metrics(y, scores, saved["threshold"])
            for key, value in actual.items():
                assert key in saved and math.isclose(
                    value, saved[key], rel_tol=1e-10, abs_tol=1e-12
                ), (variant, key)
    old, new = records["baseline"], records["selected"]
    assert all(
        (a["id"], a["label"], a["image_sha256"])
        == (b["id"], b["label"], b["image_sha256"])
        for a, b in zip(old, new)
    )
    labels = np.array([r["label"] for r in old])
    before = np.array([r["p_hateful_conditional"] for r in old])
    after = np.array([r["p_hateful_conditional"] for r in new])
    rng = np.random.default_rng(17)
    deltas = {"auroc": [], "auprc": []}
    for _ in range(2000):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) != 2:
            continue
        for key, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
            deltas[key].append(
                float(fn(labels[idx], after[idx]) - fn(labels[idx], before[idx]))
            )
    result = {
        "scope": "Official test_seen, 1000 paired examples; frozen models and development-selected thresholds",
        "bootstrap_samples": 2000,
        "bootstrap_seed": 17,
        "bootstrap_limitations": "Example resampling only; does not include training-seed variation, semantic duplicate clustering, model-selection uncertainty or pretraining exposure.",
        "model_selection": "Highest AUROC on reused 372-row development set among already specified candidates; no test tuning",
        "freeze_sha256": status["freeze_sha256"],
        "analysis_source_sha256": sha(Path(__file__)),
        "comparisons": {},
        "frozen_operating_point_deltas": {},
        "evidence_sha256": {},
    }
    for key in deltas:
        result["comparisons"][key] = {
            "baseline": metrics["baseline"]["frozen_threshold"][key],
            "selected": metrics["selected"]["frozen_threshold"][key],
            "delta": metrics["selected"]["frozen_threshold"][key]
            - metrics["baseline"]["frozen_threshold"][key],
            "paired_percentile_ci95": np.quantile(deltas[key], [0.025, 0.975]).tolist(),
            "valid_resamples": len(deltas[key]),
        }
    for key in ("tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1"):
        result["frozen_operating_point_deltas"][key] = (
            metrics["selected"]["frozen_threshold"][key]
            - metrics["baseline"]["frozen_threshold"][key]
        )
    for name in (
        "baseline-metrics.json",
        "selected-metrics.json",
        "test-audit.json",
        "status.json",
        "freeze-receipt.json",
    ):
        result["evidence_sha256"][name] = sha(RUN / name)
        shutil.copyfile(RUN / name, OUT / name)
    (OUT / "test-comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    print("FINAL_REPORT_EVIDENCE_VERIFIED")


if __name__ == "__main__":
    main()
