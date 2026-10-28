"""Export consented human labels with fixed-holdout leakage checks."""

import hashlib
import json
import time
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from .settings import ROOT, digest


def pixel_sha(path):
    with Image.open(path) as image:
        pixels = np.array(image.convert("RGB"))
    return hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def export_feedback(store, protected_manifests=None):
    settings = store.settings
    if protected_manifests is None:
        config = json.loads(
            (ROOT / "configs/service-feedback.json").read_text(encoding="utf-8")
        )
        protected_manifests = [Path(p) for p in config["protected_manifests"]]
    protected_images = set()
    protected_files = {}
    protected_count = 0
    for manifest in protected_manifests:
        manifest = Path(manifest)
        if not manifest.is_file():
            raise ValueError("Protected manifest unavailable; export halted")
        protected_files[str(manifest)] = digest(manifest)
        for row in json.loads(manifest.read_text(encoding="utf-8-sig")):
            if row.get("split") not in (
                "dev_seen",
                "test_seen",
                "dev_unseen",
                "test_unseen",
            ):
                continue
            path = Path(row["image"])
            if digest(path) != row["image_sha256"]:
                raise ValueError("Protected image changed; export halted")
            protected_images.add(pixel_sha(path))
            protected_count += 1
    accepted = []
    excluded = []
    for row in store.reviewed():
        reason = None
        if row["source"] != "user":
            reason = "non_user_demo_or_test"
        elif not (row["consent"] and row["review_consent"]):
            reason = "missing_explicit_consent"
        image = settings.media / (row["image_sha"] + ".png")
        if not reason and pixel_sha(image) in protected_images:
            reason = "protected_image_reuse"
        if reason:
            excluded.append({"submission_id": row["id"], "reason": reason})
            continue
        accepted.append(
            dict(
                id="feedback_" + row["id"],
                text=row["text"],
                label=row["label"],
                image=str(image),
                image_sha256=digest(image),
                review_id=row["review_id"],
                source_model=row["release"],
                reviewed_at=row["reviewed_at"],
            )
        )
    groups = sorted(
        {r["image_sha256"] for r in accepted},
        key=lambda x: hashlib.sha256(("feedback-v1:" + x).encode()).hexdigest(),
    )
    validation = set(groups[: max(1, len(groups) // 5)]) if len(groups) > 1 else set()
    rows = []
    for row in accepted:
        is_dev = row["image_sha256"] in validation
        rows.append(
            dict(
                row,
                split="dev_seen" if is_dev else "train",
                role="development_pilot" if is_dev else "sft_train",
            )
        )
    count_train = sum(r["role"] == "sft_train" for r in rows)
    counts = {
        str(label): sum(r["role"] == "sft_train" and r["label"] == label for r in rows)
        for label in (0, 1)
    }
    ready = count_train >= 32 and min(counts.values()) >= 5
    output = (
        settings.state
        / "exports"
        / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    )
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save("manifest.json", rows)
    config = json.loads(
        (ROOT / "configs/sft-v3-restart.json").read_text(encoding="utf-8-sig")
    )
    config.update(
        manifest=str(output / "manifest.json"),
        training_examples=count_train,
        evaluation="New feedback-development split; old final test is observed and cannot select a new model",
    )
    save("training-config.json", config)
    report = dict(
        export_id=output.name,
        created=time.time(),
        accepted=len(rows),
        train=count_train,
        validation=len(rows) - count_train,
        train_class_counts=counts,
        train_ready=ready,
        readiness_rule="32 training examples, 5 per class: operational floor, not statistical guarantee",
        excluded=excluded,
        protected_records=protected_count,
        protected_manifests=protected_files,
        manifest_sha256=digest(output / "manifest.json"),
        scope="Explicit-consent latest human labels; exact decoded-image holdout audit; near duplicates not excluded",
    )
    report["training_config_sha256"] = hashlib.sha256(
        (output / "training-config.json").read_bytes()
    ).hexdigest()
    save("audit.json", report)
    with store.connect() as db:
        store.event(
            db,
            "feedback_export",
            output.name,
            {"accepted": len(rows), "train_ready": ready},
        )
    return report
