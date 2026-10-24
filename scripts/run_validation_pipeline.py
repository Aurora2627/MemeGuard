"""Freeze development thresholds, acquire disjoint validation, evaluate sequentially."""

import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, now


def main():
    out = Path("D:/career-projects/runs/MemeGuard/20261024-validation-v1")
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
        ("diagnostics", ["scripts/analyze_sft.py"]),
        ("acquiring", ["scripts/acquire_validation.py"]),
        (
            "baseline",
            [
                "scripts/evaluate_validation.py",
                "--variant",
                "baseline",
                "--output",
                str(out / "baseline"),
            ],
        ),
        (
            "sft",
            [
                "scripts/evaluate_validation.py",
                "--variant",
                "sft",
                "--output",
                str(out / "sft"),
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
        base = read(out / "baseline/predictions.json")
        sft = read(out / "sft/predictions.json")
        if (
            len(base) != 372
            or len(sft) != 372
            or any(
                any(a[k] != b[k] for k in ("id", "label", "image_sha256"))
                for a, b in zip(base, sft)
            )
        ):
            raise ValueError("Predictions are not paired")
        save(
            out / "comparison.json",
            {
                "scope": "372 additional validation records; no official test claims",
                "baseline": read(out / "baseline/metrics.json"),
                "sft": read(out / "sft/metrics.json"),
                "threshold_config": read(ROOT / "configs/thresholds-v1.json"),
            },
        )
        state.update(status="complete", finished=now())
        save(out / "status.json", state)
        print("VALIDATION_PIPELINE_COMPLETE", flush=True)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
