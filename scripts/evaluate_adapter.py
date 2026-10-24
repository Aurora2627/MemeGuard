"""Evaluate a trained adapter on the unchanged development-only pilot, not a final test-set result."""

import argparse
import importlib.metadata
import shutil
import sys
import time
import traceback
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now
from memeguard.data import audit_manifest
from memeguard.metrics import binary_metrics
from memeguard.runtime import load_model, encode, label_ids
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--adapter", required=True, type=Path)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "preparing",
        "started": now(),
        "scope": "128-record development pilot only; not held-out test evidence",
    }
    save(out / "status.json", state)
    try:
        manifest = Path("D:/career-projects/data/MemeGuard/pilot-v1.json")
        audit = audit_manifest(manifest)
        save(out / "data-audit.json", audit)
        rows = [row for row in read(manifest) if row["role"] == "development_pilot"]
        if len(rows) != 128:
            raise ValueError("Frozen development pilot size changed")
        source_hashes = {}
        for path in [
            *ROOT.glob("src/memeguard/*.py"),
            *ROOT.glob("configs/*.json"),
            Path(__file__),
        ]:
            relative = path.relative_to(ROOT)
            target = out / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            source_hashes[relative.as_posix()] = sha(path)
        save(out / "source-sha256.json", source_hashes)
        state.update(
            status="loading",
            manifest_sha256=sha(manifest),
            torch=torch.__version__,
            transformers=importlib.metadata.version("transformers"),
            gpu=torch.cuda.get_device_name(0),
        )
        save(out / "status.json", state)
        model, processor, identity = load_model()
        state["adapter_sha256"] = sha(args.adapter / "adapter_model.safetensors")
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.eval()
        ids = label_ids(processor)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        predictions = []
        with torch.inference_mode():
            for row in rows:
                inputs = encode(processor, row)
                output = model(**inputs, use_cache=False)
                logits = output.logits[0, -1, ids].float()
                probability = float(torch.softmax(logits, dim=0)[1])
                predictions.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "p_hateful_conditional": probability,
                        "prediction": int(probability >= 0.5),
                        "image_sha256": row["image_sha256"],
                        "tokens": inputs["input_ids"].shape[-1],
                    }
                )
                del output, inputs
                save(out / "predictions.json", predictions)
                state.update(
                    status="evaluating", completed=len(predictions), total=len(rows)
                )
                save(out / "status.json", state)
                if len(predictions) % 16 == 0:
                    print(
                        "ADAPTER_PROGRESS", len(predictions), "/", len(rows), flush=True
                    )
        metrics = binary_metrics(
            [r["label"] for r in predictions],
            [r["p_hateful_conditional"] for r in predictions],
        )
        save(out / "metrics.json", metrics)
        state.update(
            status="complete",
            finished=now(),
            seconds=time.perf_counter() - started,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            score_definition="P(label token 1 | label token in {0,1}); uncalibrated",
            predictions_sha256=sha(out / "predictions.json"),
        )
        save(out / "status.json", state)
        print("ADAPTER_COMPLETE", metrics, flush=True)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
