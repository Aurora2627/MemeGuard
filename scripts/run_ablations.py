"""Fixed-adapter modality ablation; no official test data or model selection."""

import argparse, json, sys, shutil, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now
from memeguard.data import audit_manifest
from memeguard.metrics import binary_metrics
from memeguard.ablations import MODES, encode_ablation, rotated_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="preparing",
        started=now(),
        scope="Reused development diagnostics; not independent test",
    )
    save(out / "status.json", state)
    try:
        cfg = read(ROOT / "configs/ablation-v1.json")
        save(out / "config.json", cfg)
        save(out / "data-audit.json", audit_manifest(Path(cfg["audit_manifest"])))
        cal = [
            r
            for r in read(cfg["calibration_manifest"])
            if r["role"] == "development_pilot"
        ]
        dev = read(cfg["development_manifest"])
        assert len(cal) == 128 and len(dev) == 372
        assert not {r["id"] for r in cal} & {r["id"] for r in dev}
        for p in [*ROOT.glob("src/memeguard/*.py"), Path(__file__)]:
            target = out / "sources" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        import torch
        from peft import PeftModel
        from memeguard.runtime import load_model, label_ids

        model, processor, _ = load_model()
        adapter = Path(cfg["adapter"])
        state["adapter_sha256"] = sha(adapter / "adapter_model.safetensors")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model.eval()
        ids = label_ids(processor)
        summary = {}
        for mode in MODES:
            threshold = None
            summary[mode] = {}
            for split, rows in [("calibration", cal), ("development", dev)]:
                mapping = rotated_images(rows)
                predictions = []
                for row in rows:
                    with torch.inference_mode():
                        inputs = encode_ablation(
                            processor, row, mode, mapping[row["id"]]
                        )
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
                    state.update(
                        status="evaluating",
                        mode=mode,
                        split=split,
                        completed=len(predictions),
                        total=len(rows),
                        updated=now(),
                    )
                    save(out / "status.json", state)
                    if len(predictions) % 32 == 0:
                        print("PROGRESS", mode, split, len(predictions), flush=True)
                save(out / (mode + "-" + split + "-predictions.json"), predictions)
                y = [r["label"] for r in predictions]
                p = [r["p_hateful_conditional"] for r in predictions]
                if split == "calibration":
                    candidates = [binary_metrics(y, p, t) for t in sorted({0.0, *p})]
                    selected = max(
                        (m for m in candidates if m["recall"] >= 0.9),
                        key=lambda m: (m["tn"], m["threshold"]),
                    )
                    threshold = selected["threshold"]
                    save(
                        out / (mode + "-threshold.json"),
                        dict(selection=selected, scope="Original 128 only"),
                    )
                summary[mode][split] = dict(
                    fixed_0_5=binary_metrics(y, p),
                    frozen_threshold=binary_metrics(y, p, threshold),
                )
                save(out / "summary.json", summary)
        state.update(status="complete", finished=now())
    except Exception:
        state.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        save(out / "status.json", state)


if __name__ == "__main__":
    main()
