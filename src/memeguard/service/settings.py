"""Explicit runtime locations and immutable, frozen model release identities."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Settings:
    state: Path
    registry: Path = ROOT / "configs/service-models.json"
    max_batch: int = 10
    max_upload_bytes: int = 5 * 1024 * 1024
    max_body_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 16_000_000
    max_jobs: int = 1000
    lease_seconds: int = 300

    @classmethod
    def default(cls):
        return cls(
            Path(
                os.environ.get(
                    "MEMEGUARD_STATE", "D:/career-projects/services/MemeGuard"
                )
            )
        )

    @property
    def database(self):
        return self.state / "service.sqlite3"

    @property
    def media(self):
        return self.state / "media"

    def initialize(self):
        self.media.mkdir(parents=True, exist_ok=True)


def releases(settings):
    config = json.loads(settings.registry.read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported release registry")
    result = {}
    for name, spec in config["releases"].items():
        if spec["variant"] not in ("baseline", "selected"):
            raise ValueError("Unknown model variant")
        freeze_path = ROOT / spec["freeze"]
        frozen = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
        variant = spec["variant"]
        threshold = frozen["thresholds"][variant]
        if not 0 < threshold < 1:
            raise ValueError("Invalid frozen threshold")
        adapter = frozen["adapter"] if variant == "selected" else None
        expected = {}
        if adapter:
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                path = Path(adapter) / filename
                match = next(
                    (v for k, v in frozen["hashes"].items() if Path(k) == path), None
                )
                if not match:
                    raise ValueError("Adapter missing from frozen evidence")
                expected[str(path)] = match
        record = dict(
            id=name,
            variant=variant,
            threshold=threshold,
            freeze_sha256=digest(freeze_path),
            adapter=adapter,
            expected_hashes=expected,
            scope="English hateful memes; uncalibrated conditional score",
        )
        record["fingerprint"] = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest()
        result[name] = record
    if config["default"] not in result:
        raise ValueError("Default release is not registered")
    return config["default"], result


def public_release(record):
    return {
        key: record[key]
        for key in ("id", "variant", "threshold", "fingerprint", "scope")
    }
