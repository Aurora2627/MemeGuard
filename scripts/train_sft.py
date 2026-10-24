"""One pre-specified epoch of answer-only QLoRA on 1024 training examples."""

import argparse
import math
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now
from memeguard.data import audit_manifest
from memeguard.runtime import load_model, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    cfg = read(ROOT / "configs/sft-v1.json")
    state = {"status": "preparing", "started": now(), "config": cfg}
    save(out / "status.json", state)
    try:
        manifest = Path(cfg["manifest"])
        save(out / "data-audit.json", audit_manifest(manifest))
        rows = [r for r in read(manifest) if r["role"] == "sft_train"]
        if len(rows) != cfg["training_examples"]:
            raise ValueError("Training size differs from fixed protocol")
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        random.Random(cfg["seed"]).shuffle(rows)
        hashes = {}
        for path in [
            *ROOT.glob("src/memeguard/*.py"),
            *ROOT.glob("configs/*.json"),
            Path(__file__),
        ]:
            relative = path.relative_to(ROOT)
            target = out / "sources" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[relative.as_posix()] = sha(path)
        save(out / "source-sha256.json", hashes)
        state["manifest_sha256"] = sha(manifest)
        model, processor, identity = load_model()
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This reviewed training profile requires BF16 CUDA")
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg["rank"],
                lora_alpha=cfg["alpha"],
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            ),
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("No trainable adapter parameters")
        if any(
            "visual" in name and p.requires_grad for name, p in model.named_parameters()
        ):
            raise ValueError("Vision encoder must remain frozen")
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg["learning_rate"], weight_decay=0.01
        )
        steps = math.ceil(len(rows) / cfg["gradient_accumulation"])
        warmup = max(1, math.ceil(steps * 0.05))

        def scale(step):
            if step < warmup:
                return (step + 1) / warmup
            return max(0.0, (steps - step) / max(1, steps - warmup))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        updates = 0
        with (out / "training.jsonl").open("w", encoding="utf-8") as log:
            for index, row in enumerate(rows):
                inputs = encode(processor, row, answer=row["label"])
                result = model(**inputs, use_cache=False)
                loss = result.loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                losses.append(float(loss.detach()))
                group_start = (index // cfg["gradient_accumulation"]) * cfg[
                    "gradient_accumulation"
                ]
                group_size = min(cfg["gradient_accumulation"], len(rows) - group_start)
                (loss / group_size).backward()
                del result, inputs, loss
                if (index + 1) % cfg["gradient_accumulation"] == 0 or index + 1 == len(
                    rows
                ):
                    norm = torch.nn.utils.clip_grad_norm_(
                        trainable, 1.0, error_if_nonfinite=True
                    )
                    if float(norm) == 0:
                        raise ValueError("Zero adapter gradient")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    record = {
                        "step": updates,
                        "examples": index + 1,
                        "loss": sum(losses[-group_size:]) / group_size,
                        "gradient_norm": float(norm),
                        "next_lr": scheduler.get_last_lr()[0],
                        "seconds": time.perf_counter() - start,
                    }
                    log.write(__import__("json").dumps(record) + "\n")
                    log.flush()
                    state.update(
                        status="training",
                        completed_examples=index + 1,
                        optimizer_steps=updates,
                        last_loss=record["loss"],
                    )
                    save(out / "status.json", state)
                    if updates % 8 == 0:
                        print("SFT_PROGRESS", record, flush=True)
                    if updates % 64 == 0:
                        model.save_pretrained(
                            str(out / ("checkpoint-" + str(updates))),
                            safe_serialization=True,
                        )
        model.save_pretrained(str(out / "adapter"), safe_serialization=True)
        processor.save_pretrained(str(out / "adapter"))
        state.update(
            status="complete",
            finished=now(),
            optimizer_steps=updates,
            examples=len(rows),
            epochs=1,
            mean_loss=sum(losses) / len(losses),
            trainable_parameters=sum(p.numel() for p in trainable),
            seconds=time.perf_counter() - start,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        )
        save(out / "status.json", state)
        print("SFT_COMPLETE", state, flush=True)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
