"""Acquire a deterministic 1024-row training subset; keep the frozen pilot dev rows."""

import concurrent.futures
import hashlib
import json
import sys
import traceback
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now
from memeguard.data import audit_manifest
from acquire_pilot import fetch

DATA = Path("D:/career-projects/data/MemeGuard")


def main():
    report = ROOT / "reports/sft-v1"
    report.mkdir(parents=True, exist_ok=True)
    state = {"status": "acquiring", "started": now(), "downloaded": 0}
    save(report / "acquisition-status.json", state)
    try:
        lock = read(ROOT / "configs/sources.lock.json")["dataset"]
        raw = Path(lock["local_root"])
        annotation = raw / "data/FB/train.jsonl"
        rows = [
            json.loads(line)
            for line in annotation.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        selected = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                ("memeguard-pilot-v1:" + str(row["id"])).encode()
            ).hexdigest(),
        )[:1024]
        jobs = []
        for row in selected:
            name = Path(row["img"]).name
            if "/" in name or "\\" in name or name in (".", ".."):
                raise ValueError("Unsafe image filename")
            relative = "data/FB/Image/train/" + name
            jobs.append(
                {
                    "id": str(row["id"]),
                    "label": row["label"],
                    "text": row["text"],
                    "split": "train",
                    "role": "sft_train",
                    "source_path": relative,
                    "image": str(raw / relative),
                }
            )

        def download(row):
            path = Path(row["image"])
            if not path.exists():
                url = (
                    "https://huggingface.co/datasets/Jingbiao/RA-HMD/resolve/"
                    + lock["revision"]
                    + "/"
                    + urllib.parse.quote(row["source_path"], safe="/")
                )
                content = fetch(url)
                if len(content) < 100:
                    raise ValueError("Invalid image payload")
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix + ".part")
                temp.write_bytes(content)
                temp.replace(path)
            return dict(row, image_sha256=sha(path))

        complete = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(download, jobs):
                complete.append(row)
                if len(complete) % 32 == 0:
                    state["downloaded"] = len(complete)
                    save(report / "acquisition-status.json", state)
                    print("TRAIN_IMAGES_READY", len(complete), "/1024", flush=True)
        complete.extend(
            row
            for row in read(DATA / "pilot-v1.json")
            if row["role"] == "development_pilot"
        )
        manifest = DATA / "sft-v1.json"
        save(manifest, complete)
        audit = audit_manifest(manifest)
        save(report / "data-audit.json", audit)
        state.update(
            status="acquired",
            finished=now(),
            downloaded=1024,
            manifest_sha256=sha(manifest),
        )
        save(report / "acquisition-status.json", state)
        print("SFT_DATA_READY", audit, flush=True)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(report / "acquisition-status.json", state)
        raise


if __name__ == "__main__":
    main()
