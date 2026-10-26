"""Resume a verified full checkpoint and evaluate in separate GPU processes."""

import argparse, ctypes, os, shutil, subprocess, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    if a.evaluate:
        from run_scaling_restart import evaluate

        evaluate(a.output)
        return
    if a.checkpoint is None:
        raise ValueError("A complete checkpoint is required")
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="validating_checkpoint",
        started=now(),
        checkpoint=str(a.checkpoint),
        scope="Resumed 4096-row training; reused development evaluation",
    )
    save(a.output / "status.json", state)
    shutil.copy2(__file__, a.output / Path(__file__).name)
    # Prevent automatic system sleep only while this process runs; do not force screen on.
    keep_awake = False
    try:
        if os.name == "nt":
            keep_awake = bool(
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            )
        state["system_sleep_inhibition_active"] = keep_awake
        from memeguard.checkpoints import read_checkpoint

        config = read(ROOT / "configs/sft-v3-restart.json")
        recovered = read_checkpoint(a.checkpoint, config, sha(Path(config["manifest"])))
        if recovered["updates"] != 112 or recovered["next_index"] != 896:
            raise ValueError("This registered run expects checkpoint 112 / example 896")
        state.update(
            restored_optimizer_steps=recovered["updates"],
            restored_examples=recovered["next_index"],
            training_state_sha256=sha(a.checkpoint / "training-state.pt"),
            adapter_sha256=sha(a.checkpoint / "adapter_model.safetensors"),
        )
        save(a.output / "recovery-receipt.json", state)
        del recovered
        commands = [
            (
                "training",
                [
                    sys.executable,
                    "-u",
                    "scripts/train_sft.py",
                    "--config",
                    "configs/sft-v3-restart.json",
                    "--resume",
                    str(a.checkpoint),
                    "--output",
                    str(a.output / "training"),
                ],
            ),
            (
                "evaluation",
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "--evaluate",
                    "--output",
                    str(a.output),
                ],
            ),
        ]
        for stage, command in commands:
            state.update(status=stage, updated=now())
            save(a.output / "status.json", state)
            with (a.output / (stage + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        state.update(status="complete", finished=now())
    except Exception:
        state.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        save(a.output / "status.json", state)


if __name__ == "__main__":
    main()
