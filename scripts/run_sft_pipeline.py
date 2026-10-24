"""Acquire, audit, train, and evaluate in sequential child processes."""

import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, now


def main():
    out = Path("D:/career-projects/runs/MemeGuard/20261024-sft-v1")
    out.mkdir(parents=True, exist_ok=False)
    state = {"status": "starting", "started": now()}
    save(out / "status.json", state)
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        PYTHONIOENCODING="utf-8",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    jobs = [
        ("testing", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
        ("acquiring", ["scripts/acquire_sft.py"]),
        ("training", ["scripts/train_sft.py", "--output", str(out / "training")]),
        (
            "evaluating",
            [
                "scripts/evaluate_adapter.py",
                "--adapter",
                str(out / "training/adapter"),
                "--output",
                str(out / "evaluation"),
            ],
        ),
    ]
    try:
        for stage, args in jobs:
            state.update(status=stage)
            save(out / "status.json", state)
            with (out / (stage + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run(
                    [sys.executable, "-u", "-X", "utf8", *args],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        baseline = read(
            "D:/career-projects/runs/MemeGuard/20261023-first-phase-v1/zero-shot/metrics.json"
        )
        adapter = read(out / "evaluation/metrics.json")
        comparison = {
            "scope": "Same 128 development examples; no held-out test claim",
            "baseline": baseline,
            "sft": adapter,
            "deltas": {
                key: adapter[key] - baseline[key]
                for key in ("auroc", "auprc", "f1", "accuracy", "precision", "recall")
            },
            "training": read(out / "training/status.json"),
        }
        save(out / "comparison.json", comparison)
        state.update(status="complete", finished=now())
        save(out / "status.json", state)
        print("SFT_PIPELINE_COMPLETE", comparison, flush=True)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
