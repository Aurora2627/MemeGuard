"""Load the frozen CUDA classifier once; never replace inference with demo scores."""

import contextlib
import json
import time
from pathlib import Path
from .settings import ROOT, digest


class FrozenClassifier:
    def __init__(self, registry):
        import torch
        from peft import PeftModel
        from memeguard.runtime import load_model, label_ids

        frozen = json.loads(
            (ROOT / "reports/final-test-v1/freeze.json").read_text(encoding="utf-8-sig")
        )
        for name, expected in frozen["hashes"].items():
            p = Path(name)
            if (
                p.name in ("sources.lock.json", "identity.json")
                and digest(p) != expected
            ):
                raise ValueError("Frozen base model identity changed")
        chosen = next(r for r in registry.values() if r["variant"] == "selected")
        for name, expected in chosen["expected_hashes"].items():
            if digest(name) != expected:
                raise ValueError("Frozen adapter checksum mismatch")
        self.model, self.processor, _ = load_model()
        self.model = PeftModel.from_pretrained(
            self.model, chosen["adapter"], is_trainable=False
        )
        self.model.eval()
        self.ids = label_ids(self.processor)
        self.registry = registry
        self.torch = torch

    def score(self, job, image):
        from memeguard.runtime import encode

        release = self.registry[job["release"]]
        if release["fingerprint"] != job["fingerprint"]:
            raise ValueError("Queued model identity differs from registry")
        torch = self.torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        row = {"image": str(image), "text": job["submission"]["text"]}
        inputs = encode(self.processor, row)
        adapter_context = (
            self.model.disable_adapter()
            if release["variant"] == "baseline"
            else contextlib.nullcontext()
        )
        with adapter_context, torch.inference_mode():
            result = self.model(**inputs, use_cache=False)
            score = float(
                result.logits[0, -1, self.ids].float().softmax(dim=0)[1].cpu()
            )
        del inputs, result
        torch.cuda.synchronize()
        return score, time.perf_counter() - start
