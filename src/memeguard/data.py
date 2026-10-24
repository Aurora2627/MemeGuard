"""Validate the actual pilot images and distinguish pair duplicates from image reuse."""

import hashlib
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read, sha


def validate_row(row):
    if type(row.get("label")) is not int or row["label"] not in (0, 1):
        raise ValueError("Expected binary integer label")
    if not str(row.get("id", "")) or not isinstance(row.get("text"), str):
        raise ValueError("Missing ID or caption")
    if row.get("role") not in ("capacity_train", "sft_train", "development_pilot"):
        raise ValueError("Unreviewed pilot role")
    expected = "train" if row["role"] in ("capacity_train", "sft_train") else "dev_seen"
    if row.get("split") != expected:
        raise ValueError("Pilot role conflicts with upstream split")


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def pair_digest(image_rgb, text):
    digest = hashlib.sha256()
    digest.update(str(image_rgb.shape).encode())
    digest.update(image_rgb.tobytes())
    digest.update(normalized_text(text).encode("utf-8"))
    return digest.hexdigest()


def audit_manifest(path):
    rows = read(path)
    ids = set()
    pairs = defaultdict(list)
    images = defaultdict(set)
    counts = Counter()
    for row in rows:
        validate_row(row)
        key = (row["split"], row["id"])
        if key in ids:
            raise ValueError("Repeated pilot ID")
        ids.add(key)
        image = Path(row["image"])
        if sha(image) != row["image_sha256"]:
            raise ValueError("Pilot image changed")
        with Image.open(image) as source:
            source.verify()
        with Image.open(image) as source:
            rgb = np.array(source.convert("RGB"))
        if min(rgb.shape[:2]) < 8:
            raise ValueError("Degenerate image")
        digest = pair_digest(rgb, row["text"])
        pairs[digest].append(row["split"])
        images[hashlib.sha256(rgb.tobytes()).hexdigest()].add(row["split"])
        counts[row["role"] + ":label" + str(row["label"])] += 1
    cross_split_pairs = [
        digest for digest, splits in pairs.items() if len(set(splits)) > 1
    ]
    if cross_split_pairs:
        raise ValueError("Exact image-caption leakage across pilot splits")
    return {
        "records": len(rows),
        "counts": dict(counts),
        "decoded_images": len(rows),
        "cross_split_exact_pairs": len(cross_split_pairs),
        "reused_images_across_splits": sum(
            len(splits) > 1 for splits in images.values()
        ),
        "within_split_exact_pair_duplicates": sum(
            len(values) - 1 for values in pairs.values()
        ),
        "manifest_sha256": sha(path),
        "scope": "Pilot images only. Full-corpus image/near-duplicate audit remains pending. Image reuse alone is reported, not equated with identical image-caption leakage.",
    }
