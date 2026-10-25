"""Verify paired ablations and estimate development-sample uncertainty."""

from pathlib import Path
import json, shutil
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("D:/career-projects/runs/MemeGuard/20261025-ablations-v1")
OUT = ROOT / "reports/ablations-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    state = read(RUN / "status.json")
    assert state["status"] == "complete"
    modes = ("joint", "text_only", "image_only", "shuffled_image")
    records = {m: read(RUN / (m + "-development-predictions.json")) for m in modes}
    base = records["joint"]
    assert len(base) == 372
    keys = [(r["id"], r["label"], r["image_sha256"]) for r in base]
    assert len({r["id"] for r in base}) == 372
    for rows in records.values():
        assert [(r["id"], r["label"], r["image_sha256"]) for r in rows] == keys
    old = read(
        Path(
            "D:/career-projects/runs/MemeGuard/20261024-validation-v1/sft/predictions.json"
        )
    )
    oldmap = {r["id"]: r for r in old}
    delta = max(
        abs(r["p_hateful_conditional"] - oldmap[r["id"]]["p_hateful_conditional"])
        for r in base
    )
    assert delta < 1e-6, "Joint rerun differs from previous adapter evaluation"
    y = np.array([r["label"] for r in base])
    scores = {
        m: np.array([r["p_hateful_conditional"] for r in records[m]]) for m in modes
    }
    rng = np.random.default_rng(17)
    negatives = np.flatnonzero(y == 0)
    positives = np.flatnonzero(y == 1)
    differences = {m: [] for m in modes[1:]}
    for _ in range(2000):
        ix = np.concatenate(
            [
                rng.choice(negatives, len(negatives)),
                rng.choice(positives, len(positives)),
            ]
        )
        joint = roc_auc_score(y[ix], scores["joint"][ix])
        for m in differences:
            differences[m].append(joint - roc_auc_score(y[ix], scores[m][ix]))
    result = dict(
        scope="Paired stratified bootstrap on reused development examples; not training-seed or population uncertainty",
        resamples=2000,
        seed=17,
        joint_rerun_max_absolute_difference=delta,
        comparisons={},
    )
    for m, values in differences.items():
        result["comparisons"][m] = dict(
            auroc_difference=float(
                roc_auc_score(y, scores["joint"]) - roc_auc_score(y, scores[m])
            ),
            percentile_95_interval=np.quantile(values, [0.025, 0.975]).tolist(),
        )
    OUT.mkdir(exist_ok=True, parents=True)
    for name in ["summary.json", "status.json", "config.json", "data-audit.json"]:
        shutil.copy2(RUN / name, OUT / name)
    (OUT / "paired-analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
