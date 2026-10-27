"""One frozen final-test run with cached acquisition and exact-prefix recovery."""

from pathlib import Path
from collections import Counter, defaultdict
import argparse, concurrent.futures, ctypes, hashlib, json, sys, traceback, urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now
from prepare_final import DATA, OUT, REPORT

FREEZE = REPORT / "freeze.json"


def verify_freeze(path=FREEZE):
    frozen = read(path)
    for file, digest in frozen["hashes"].items():
        if sha(file) != digest:
            raise ValueError("Frozen asset changed: " + file)
    receipt = read(OUT / "freeze-receipt.json")
    if sha(path) != receipt["sha256"]:
        raise ValueError("Freeze receipt mismatch")
    return frozen


def validate_prefix(predictions, rows):
    if len(predictions) > len(rows):
        raise ValueError("Unexpected extra predictions")
    for a, b in zip(predictions, rows):
        if (a["id"], a["label"], a["image_sha256"]) != (
            b["id"],
            b["label"],
            b["image_sha256"],
        ):
            raise ValueError("Saved prediction prefix mismatch")


def acquire(frozen):
    from acquire_pilot import fetch

    lock = read(ROOT / "configs/sources.lock.json")["dataset"]
    raw = Path(lock["local_root"])
    annotation = raw / "data/FB/test_seen.jsonl"
    rows = [
        json.loads(line)
        for line in annotation.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(rows) != frozen["expected_test_rows"] or len(
        {str(r["id"]) for r in rows}
    ) != len(rows):
        raise ValueError("Unexpected test membership")
    jobs = []
    for r in rows:
        if type(r["label"]) is not int or r["label"] not in (0, 1):
            raise ValueError("Invalid test label")
        name = Path(r["img"].replace(chr(92), "/")).name
        if not name or name in (".", ".."):
            raise ValueError("Unsafe image name")
        relative = "data/FB/Image/test_seen/" + name
        jobs.append(
            dict(
                id=str(r["id"]),
                label=r["label"],
                text=r["text"],
                split="test_seen",
                role="final_test",
                source_path=relative,
                image=str(raw / relative),
            )
        )

    def download(row):
        p = Path(row["image"])
        if not p.exists():
            url = (
                "https://huggingface.co/datasets/"
                + lock["repository"]
                + "/resolve/"
                + lock["revision"]
                + "/"
                + urllib.parse.quote(row["source_path"], safe="/")
            )
            body = fetch(url)
            if len(body) < 100:
                raise ValueError("Invalid image payload")
            p.parent.mkdir(parents=True, exist_ok=True)
            temp = p.with_suffix(p.suffix + ".part")
            temp.write_bytes(body)
            temp.replace(p)
        return dict(row, image_sha256=sha(p))

    complete = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(download, jobs):
            complete.append(r)
            if len(complete) % 25 == 0:
                save(
                    OUT / "status.json",
                    dict(
                        status="acquiring",
                        completed=len(complete),
                        total=len(jobs),
                        updated=now(),
                    ),
                )
                print("TEST_IMAGES_READY", len(complete), flush=True)
    save(OUT / "test-manifest.json", complete)
    return complete, sha(annotation)


def audit(test, annotation_hash):
    import numpy as np
    from PIL import Image
    from memeguard.data import pair_digest

    previous = read(DATA / "scale-and-all-dev-v1.json")
    if {r["id"] for r in previous} & {r["id"] for r in test}:
        raise ValueError("Test ID overlaps train/dev")
    pairs = defaultdict(set)
    images = defaultdict(set)
    ids = set()
    for index, r in enumerate(previous + test):
        key = (r["split"], r["id"])
        if key in ids:
            raise ValueError("Repeated manifest ID")
        ids.add(key)
        if sha(r["image"]) != r["image_sha256"]:
            raise ValueError("Image checksum mismatch")
        with Image.open(r["image"]) as im:
            im.verify()
        with Image.open(r["image"]) as im:
            rgb = np.array(im.convert("RGB"))
        pairs[pair_digest(rgb, r["text"])].add(r["split"])
        images[hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()].add(
            r["split"]
        )
        if index % 100 == 0:
            save(
                OUT / "status.json",
                dict(
                    status="auditing",
                    completed=index,
                    total=len(previous) + len(test),
                    updated=now(),
                ),
            )
    overlap = sum("test_seen" in s and len(s) > 1 for s in pairs.values())
    image_reuse = sum("test_seen" in s and len(s) > 1 for s in images.values())
    summary = dict(
        test_records=len(test),
        comparison_records=len(previous),
        test_label_counts=dict(Counter(r["label"] for r in test)),
        cross_split_exact_test_pairs=overlap,
        test_images_reused_across_splits=image_reuse,
        annotation_sha256=annotation_hash,
        manifest_sha256=sha(OUT / "test-manifest.json"),
        scope="4096 used training + all 500 dev + 1000 test images; exact decoded image-caption duplicates audited. Near duplicates and pretraining exposure are not excluded.",
    )
    save(OUT / "test-audit.json", summary)
    if overlap:
        raise ValueError("Exact test pair overlaps train/dev; evaluation halted")
    return summary


def evaluate(rows, frozen):
    import torch
    from peft import PeftModel
    from memeguard.runtime import load_model, encode, label_ids
    from memeguard.metrics import binary_metrics

    model, processor, _ = load_model()
    ids = label_ids(processor)
    model.eval()
    for variant in ("baseline", "selected"):
        verify_freeze()
        if variant == "selected":
            model = PeftModel.from_pretrained(
                model, frozen["adapter"], is_trainable=False
            )
            model.eval()
        path = OUT / (variant + "-predictions.json")
        predictions = read(path) if path.exists() else []
        validate_prefix(predictions, rows)
        with torch.inference_mode():
            for row in rows[len(predictions) :]:
                inputs = encode(processor, row)
                result = model(**inputs, use_cache=False)
                score = float(
                    torch.softmax(result.logits[0, -1, ids].float(), dim=0)[1]
                )
                predictions.append(
                    dict(
                        id=row["id"],
                        label=row["label"],
                        image_sha256=row["image_sha256"],
                        p_hateful_conditional=score,
                    )
                )
                del inputs, result
                save(path, predictions)
                save(
                    OUT / "status.json",
                    dict(
                        status="evaluating",
                        variant=variant,
                        completed=len(predictions),
                        total=len(rows),
                        updated=now(),
                    ),
                )
                if len(predictions) % 50 == 0:
                    print("TEST_PREDICTIONS", variant, len(predictions), flush=True)
        labels = [r["label"] for r in predictions]
        scores = [r["p_hateful_conditional"] for r in predictions]
        save(
            OUT / (variant + "-metrics.json"),
            dict(
                frozen_threshold=binary_metrics(
                    labels, scores, frozen["thresholds"][variant]
                ),
                fixed_0_5=binary_metrics(labels, scores),
                predictions_sha256=sha(path),
                freeze_sha256=sha(FREEZE),
            ),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("acquire", "evaluate", "all"), default="all"
    )
    args = parser.parse_args()
    frozen = verify_freeze()
    if (OUT / "status.json").exists() and read(OUT / "status.json").get(
        "status"
    ) == "complete":
        print("FINAL_TEST_ALREADY_COMPLETE")
        return
    keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    try:
        if args.stage in ("acquire", "all"):
            rows, annotation_hash = acquire(frozen)
            audit(rows, annotation_hash)
            save(
                OUT / "status.json", dict(status="ready_for_evaluation", updated=now())
            )
        if args.stage in ("evaluate", "all"):
            rows = read(OUT / "test-manifest.json")
            check = read(OUT / "test-audit.json")
            if (
                check["manifest_sha256"] != sha(OUT / "test-manifest.json")
                or check["cross_split_exact_test_pairs"]
            ):
                raise ValueError("Invalid test audit")
            verify_freeze()
            evaluate(rows, frozen)
            save(
                OUT / "status.json",
                dict(status="complete", finished=now(), freeze_sha256=sha(FREEZE)),
            )
        print("FINAL_STAGE_COMPLETE", args.stage, flush=True)
    except Exception:
        save(OUT / "failure.json", dict(failed=now(), traceback=traceback.format_exc()))
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
