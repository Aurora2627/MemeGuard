"""Run the pre-specified data scaling experiment and sequential development evaluation."""

import argparse, subprocess, sys, traceback, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, now, sha


def evaluate(out):
    import torch
    from peft import PeftModel
    from memeguard.runtime import load_model, encode, label_ids
    from memeguard.metrics import binary_metrics
    from memeguard.data import audit_manifest

    save(
        out / "evaluation-data-audit.json",
        audit_manifest(
            Path("D:/career-projects/data/MemeGuard/scale-and-all-dev-v1.json")
        ),
    )
    model, processor, _ = load_model()
    adapter = out / "training/adapter"
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    ids = label_ids(processor)
    calibration = [
        r
        for r in read("D:/career-projects/data/MemeGuard/pilot-v1.json")
        if r["role"] == "development_pilot"
    ]
    comparison = read("D:/career-projects/data/MemeGuard/validation-v1.json")
    assert len(calibration) == 128 and len(comparison) == 372
    assert not {r["id"] for r in calibration} & {r["id"] for r in comparison}
    threshold = None
    for name, rows in [("calibration", calibration), ("development", comparison)]:
        predictions = []
        with torch.inference_mode():
            for row in rows:
                inputs = encode(processor, row)
                result = model(**inputs, use_cache=False)
                score = float(
                    torch.softmax(result.logits[0, -1, ids].float(), dim=0)[1]
                )
                predictions.append(
                    dict(
                        id=row["id"],
                        label=row["label"],
                        p_hateful_conditional=score,
                        image_sha256=row["image_sha256"],
                    )
                )
                del inputs, result
                save(out / (name + "-predictions.json"), predictions)
                save(
                    out / "evaluation-status.json",
                    dict(
                        status="evaluating",
                        split=name,
                        completed=len(predictions),
                        total=len(rows),
                    ),
                )
        y = [r["label"] for r in predictions]
        p = [r["p_hateful_conditional"] for r in predictions]
        if name == "calibration":
            candidates = [binary_metrics(y, p, t) for t in sorted({0.0, *p})]
            selected = max(
                (m for m in candidates if m["recall"] >= 0.9),
                key=lambda m: (m["tn"], m["threshold"]),
            )
            threshold = selected["threshold"]
            save(
                out / "frozen-threshold.json",
                dict(
                    selection=selected,
                    scope="Original 128 only",
                    predictions_sha256=sha(out / "calibration-predictions.json"),
                ),
            )
        save(
            out / (name + "-metrics.json"),
            dict(
                fixed_0_5=binary_metrics(y, p),
                frozen_threshold=binary_metrics(y, p, threshold),
            ),
        )
    save(
        out / "evaluation-status.json",
        dict(
            status="complete",
            finished=now(),
            adapter_sha256=sha(adapter / "adapter_model.safetensors"),
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    out = Path("D:/career-projects/runs/MemeGuard/20261026-scale-v1")
    if args.evaluate:
        evaluate(out)
        return
    out.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="training",
        started=now(),
        scope="Reused development comparison, not untouched holdout",
    )
    save(out / "status.json", state)
    shutil.copy2(__file__, out / Path(__file__).name)
    try:
        for name, cmd in [
            ("acquiring", [sys.executable, "-u", "scripts/acquire_scale.py"]),
            (
                "training",
                [
                    sys.executable,
                    "-u",
                    "scripts/train_sft.py",
                    "--config",
                    "configs/sft-v3.json",
                    "--output",
                    str(out / "training"),
                ],
            ),
            ("evaluation", [sys.executable, "-u", __file__, "--evaluate"]),
        ]:
            state["status"] = name
            save(out / "status.json", state)
            with (out / (name + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run(
                    cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        state.update(status="complete", finished=now())
    except Exception:
        state.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        save(out / "status.json", state)


if __name__ == "__main__":
    main()
