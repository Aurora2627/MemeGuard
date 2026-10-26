"""Atomic local checkpoints for adapter, optimizer, scheduler and RNG state."""

import json, random, hashlib
from pathlib import Path
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    config,
    manifest_sha256,
    next_index,
    updates,
    losses,
):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(temporary), safe_serialization=True)
    state = dict(
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python_rng=random.getstate(),
        next_index=next_index,
        updates=updates,
        losses=losses,
    )
    torch.save(state, temporary / "training-state.pt")
    meta = dict(
        format=1,
        config=config,
        manifest_sha256=manifest_sha256,
        training_state_sha256=digest(temporary / "training-state.pt"),
        complete=True,
    )
    (temporary / "resume.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    temporary.rename(path)


def read_checkpoint(path, config, manifest_sha256):
    path = Path(path)
    if not (path / "resume.json").exists():
        raise ValueError(
            "Adapter-only checkpoint: optimizer/scheduler/RNG cannot be recovered exactly"
        )
    meta = json.loads((path / "resume.json").read_text(encoding="utf-8"))
    if (
        meta.get("format") != 1
        or not meta.get("complete")
        or meta["config"] != config
        or meta["manifest_sha256"] != manifest_sha256
    ):
        raise ValueError("Checkpoint protocol or manifest mismatch")
    if digest(path / "training-state.pt") != meta["training_state_sha256"]:
        raise ValueError("Checkpoint state checksum mismatch")
    return torch.load(path / "training-state.pt", map_location="cpu", weights_only=True)


def restore_state(state, optimizer, scheduler):
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA required to restore CUDA RNG")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
