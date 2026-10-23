"""Two real QLoRA updates on training examples; capacity evidence, not model quality."""

import argparse
import sys
import traceback
from pathlib import Path
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now
from memeguard.runtime import load_model, encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    state = {
        "status": "loading",
        "started": now(),
        "scope": "Capacity probe only: two training examples, two optimizer updates; no quality claim",
    }
    save(out / "status.json", state)
    try:
        rows = [
            r
            for r in read("D:/career-projects/data/MemeGuard/pilot-v1.json")
            if r["role"] == "capacity_train"
        ][:2]
        if len(rows) != 2:
            raise ValueError("Two training-only capacity examples required")
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        state["seed"] = 17
        model, processor, identity = load_model()
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            ),
        )
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters")
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        before = sum(
            float(p.detach().float().abs().sum())
            for name, p in model.named_parameters()
            if p.requires_grad and "lora_B" in name
        )
        model.train()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        for row in rows:
            inputs = encode(processor, row, answer=row["label"])
            output = model(**inputs)
            if not torch.isfinite(output.loss):
                raise ValueError("Non-finite training loss")
            output.loss.backward()
            gradients = [p.grad for p in trainable if p.grad is not None]
            if (
                not gradients
                or not all(torch.isfinite(g).all() for g in gradients)
                or not any(g.abs().sum() > 0 for g in gradients)
            ):
                raise ValueError("Missing, zero or non-finite training gradients")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(output.loss.detach()))
            del output, inputs
            save(out / "losses.json", losses)
        after = sum(
            float(p.detach().float().abs().sum())
            for name, p in model.named_parameters()
            if p.requires_grad and "lora_B" in name
        )
        if after == before:
            raise ValueError("Adapter weights did not change")
        model.save_pretrained(out / "adapter", safe_serialization=True)
        state.update(
            status="complete",
            finished=now(),
            steps=len(losses),
            losses=losses,
            trainable_parameters=sum(p.numel() for p in trainable),
            adapter_updated=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            source_sha256=sha(Path(__file__)),
        )
        save(out / "status.json", state)
        print("QLORA_CAPACITY_COMPLETE", state, flush=True)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
