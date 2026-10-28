"""Validate an exported feedback package before invoking the existing PyTorch trainer."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.service.settings import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()
    audit = json.loads((args.export / "audit.json").read_text(encoding="utf-8"))
    if digest(args.export / "manifest.json") != audit["manifest_sha256"]:
        raise ValueError("Export manifest changed after holdout audit")
    if digest(args.export / "training-config.json") != audit["training_config_sha256"]:
        raise ValueError("Training config changed after audit")
    if not audit["train_ready"]:
        raise ValueError("Not enough eligible human labels for a feedback training run")
    if args.output.exists():
        raise ValueError("Use a new output directory")
    from memeguard.data import audit_manifest

    print(json.dumps(audit_manifest(args.export / "manifest.json"), ensure_ascii=False))
    if not args.validate_only:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/train_sft.py"),
                "--config",
                str(args.export / "training-config.json"),
                "--output",
                str(args.output),
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
