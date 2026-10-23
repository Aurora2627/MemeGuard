"""Pin upstream assets and acquire a small, label-independent development pilot."""

import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now

PROXY = "http://127.0.0.1:7890"
DATA = Path("D:/career-projects/data/MemeGuard")
REPORT = ROOT / "reports/bootstrap"


def fetch(url):
    last = None
    for attempt in range(4):
        try:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
            )
            with opener.open(url, timeout=40) as response:
                return response.read()
        except Exception as error:
            last = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError("Download failed: " + url) from last


def main():
    state = {"status": "acquiring", "started": now(), "downloaded_images": 0}
    save(REPORT / "acquisition-status.json", state)
    try:
        lock_path = ROOT / "configs/sources.lock.json"
        lock = read(lock_path)
        revision = read(REPORT / "hf-dataset.json")["sha"]
        raw = DATA / "raw" / revision
        raw.mkdir(parents=True, exist_ok=True)
        lock["dataset"] = {
            "repository": "Jingbiao/RA-HMD",
            "revision": revision,
            "source": "https://huggingface.co/datasets/Jingbiao/RA-HMD",
            "card_license": "cc-by-nc-4.0",
            "local_root": str(raw),
        }
        if "model" not in lock:
            model_id = "Qwen/Qwen2-VL-2B-Instruct"
            info = json.loads(fetch("https://huggingface.co/api/models/" + model_id))
            tree = json.loads(
                fetch(
                    "https://huggingface.co/api/models/"
                    + model_id
                    + "/tree/"
                    + info["sha"]
                    + "?recursive=true&limit=1000"
                )
            )
            save(REPORT / "model-files.json", tree)
            lock["model"] = {
                "repository": model_id,
                "revision": info["sha"],
                "local_root": "D:/career-projects/models/MemeGuard/" + info["sha"],
            }
        save(lock_path, lock)
        annotations = []
        split_rows = {}
        for split in ("train", "dev_seen", "test_seen"):
            relative = "data/FB/" + split + ".jsonl"
            target = raw / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    fetch(
                        "https://huggingface.co/datasets/Jingbiao/RA-HMD/resolve/"
                        + revision
                        + "/"
                        + relative
                    )
                )
            rows = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            if not rows or any(
                set(("id", "img", "text", "label")) - set(row) for row in rows
            ):
                raise ValueError("Missing image/text/label fields in " + split)
            if any(
                type(row["label"]) is not int or row["label"] not in (0, 1)
                for row in rows
            ):
                raise ValueError("Non-binary or missing labels in " + split)
            if len({str(row["id"]) for row in rows}) != len(rows):
                raise ValueError("Duplicate annotation IDs in " + split)
            split_rows[split] = rows
            annotations.append(
                {
                    "split": split,
                    "rows": len(rows),
                    "label_counts": dict(Counter(str(row["label"]) for row in rows)),
                    "sha256": sha(target),
                    "source": relative,
                }
            )
            print("ANNOTATIONS_VALID", split, len(rows), flush=True)
        jobs = []
        for split, limit, role in (
            ("train", 16, "capacity_train"),
            ("dev_seen", 128, "development_pilot"),
        ):
            # Membership uses only IDs and a fixed salt, not labels or model outputs.
            selected = sorted(
                split_rows[split],
                key=lambda row: hashlib.sha256(
                    ("memeguard-pilot-v1:" + str(row["id"])).encode()
                ).hexdigest(),
            )[:limit]
            for row in selected:
                filename = Path(row["img"]).name
                if filename != row["img"].replace("\\", "/").split("/")[-1]:
                    raise ValueError("Unsafe image name")
                relative = "data/FB/Image/" + split + "/" + filename
                jobs.append(
                    {
                        "id": str(row["id"]),
                        "label": row["label"],
                        "text": row["text"],
                        "split": split,
                        "role": role,
                        "source_path": relative,
                        "image": str(raw / relative),
                    }
                )

        def download(job):
            path = Path(job["image"])
            if not path.exists():
                body = fetch(
                    "https://huggingface.co/datasets/Jingbiao/RA-HMD/resolve/"
                    + revision
                    + "/"
                    + urllib.parse.quote(job["source_path"], safe="/")
                )
                if len(body) < 100:
                    raise ValueError("Invalid image payload")
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix + ".part")
                temp.write_bytes(body)
                temp.replace(path)
            return dict(job, image_sha256=sha(path))

        completed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for row in executor.map(download, jobs):
                completed.append(row)
                if len(completed) % 16 == 0:
                    state["downloaded_images"] = len(completed)
                    save(REPORT / "acquisition-status.json", state)
                    print("IMAGES_READY", len(completed), "/", len(jobs), flush=True)
        save(DATA / "pilot-v1.json", completed)
        save(
            REPORT / "annotation-audit.json",
            {
                "annotations": annotations,
                "pilot_records": len(completed),
                "selection": "sha256 of fixed salt plus ID; no label-dependent sampling; test images and predictions untouched",
            },
        )
        state.update(
            status="acquired",
            finished=now(),
            downloaded_images=len(completed),
            manifest=str(DATA / "pilot-v1.json"),
            manifest_sha256=sha(DATA / "pilot-v1.json"),
        )
        save(REPORT / "acquisition-status.json", state)
        print("PILOT_ACQUIRED", len(completed), flush=True)
    except Exception as error:
        state.update(status="failed", error=str(error))
        save(REPORT / "acquisition-status.json", state)
        raise


if __name__ == "__main__":
    main()
