"""Evaluate frozen variants on 372 additional dev rows using frozen thresholds."""

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path
import torch
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now
from memeguard.data import audit_manifest
from memeguard.metrics import binary_metrics
from memeguard.runtime import load_model, encode, label_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "sft"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "preparing",
        "variant": args.variant,
        "started": now(),
        "scope": "Additional dev_seen validation; not official test_seen",
    }
    save(out / "status.json", state)
    try:
        manifest = Path("D:/career-projects/data/MemeGuard/validation-v1.json")
        rows = read(manifest)
        save(out / "data-audit.json", audit_manifest(manifest))
        calibration = read(ROOT / "configs/thresholds-v1.json")
        threshold = calibration[args.variant]["threshold"]
        excluded = {
            r["id"]
            for r in read("D:/career-projects/data/MemeGuard/pilot-v1.json")
            if r["role"] == "development_pilot"
        }
        if len(rows) != 372 or any(r["id"] in excluded for r in rows):
            raise ValueError("Validation overlaps threshold-selection records")
        state.update(
            manifest_sha256=sha(manifest),
            threshold=threshold,
            threshold_config_sha256=sha(ROOT / "configs/thresholds-v1.json"),
        )
        for path in [
            *ROOT.glob("src/memeguard/*.py"),
            *ROOT.glob("configs/*.json"),
            Path(__file__),
        ]:
            target = out / "sources" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        model, processor, identity = load_model()
        if args.variant == "sft":
            adapter = Path(
                "D:/career-projects/runs/MemeGuard/20261024-sft-v1/training/adapter"
            )
            state["adapter_sha256"] = sha(adapter / "adapter_model.safetensors")
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        ids = label_ids(processor)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        predictions = []
        with torch.inference_mode():
            for row in rows:
                inputs = encode(processor, row)
                result = model(**inputs, use_cache=False)
                score = float(
                    torch.softmax(result.logits[0, -1, ids].float(), dim=0)[1]
                )
                predictions.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "p_hateful_conditional": score,
                        "image_sha256": row["image_sha256"],
                    }
                )
                del inputs, result
                save(out / "predictions.json", predictions)
                state.update(
                    status="evaluating", completed=len(predictions), total=len(rows)
                )
                save(out / "status.json", state)
                if len(predictions) % 32 == 0:
                    print(
                        "VALIDATION_PROGRESS",
                        args.variant,
                        len(predictions),
                        "/372",
                        flush=True,
                    )
        y = [r["label"] for r in predictions]
        p = [r["p_hateful_conditional"] for r in predictions]
        metrics = {
            "fixed_0_5": binary_metrics(y, p),
            "frozen_recall_target_threshold": binary_metrics(y, p, threshold),
        }
        save(out / "metrics.json", metrics)
        state.update(
            status="complete",
            finished=now(),
            seconds=time.perf_counter() - start,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            predictions_sha256=sha(out / "predictions.json"),
        )
        save(out / "status.json", state)
        print("VALIDATION_COMPLETE", args.variant, metrics, flush=True)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
