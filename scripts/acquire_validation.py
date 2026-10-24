"""Acquire the remaining dev_seen rows, excluding the 128 calibration IDs."""

import concurrent.futures
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
    out = ROOT / "reports/validation-v1"
    out.mkdir(parents=True, exist_ok=True)
    state = {"status": "acquiring", "started": now(), "downloaded": 0}
    save(out / "acquisition-status.json", state)
    try:
        lock = read(ROOT / "configs/sources.lock.json")["dataset"]
        raw = Path(lock["local_root"])
        original = [
            json.loads(line)
            for line in (raw / "data/FB/dev_seen.jsonl")
            .read_text(encoding="utf-8-sig")
            .splitlines()
            if line.strip()
        ]
        excluded = {
            r["id"]
            for r in read(DATA / "pilot-v1.json")
            if r["role"] == "development_pilot"
        }
        selected = [r for r in original if str(r["id"]) not in excluded]
        if len(selected) != 372:
            raise ValueError("Unexpected additional validation size")

        def download(row):
            name = Path(row["img"]).name
            relative = "data/FB/Image/dev_seen/" + name
            path = raw / relative
            if not path.exists():
                content = fetch(
                    "https://huggingface.co/datasets/Jingbiao/RA-HMD/resolve/"
                    + lock["revision"]
                    + "/"
                    + urllib.parse.quote(relative, safe="/")
                )
                if len(content) < 100:
                    raise ValueError("Invalid image payload")
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix + ".part")
                temp.write_bytes(content)
                temp.replace(path)
            return {
                "id": str(row["id"]),
                "label": row["label"],
                "text": row["text"],
                "split": "dev_seen",
                "role": "development_pilot",
                "source_path": relative,
                "image": str(path),
                "image_sha256": sha(path),
            }

        complete = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(download, selected):
                complete.append(row)
                if len(complete) % 16 == 0:
                    state["downloaded"] = len(complete)
                    save(out / "acquisition-status.json", state)
                    print("VALIDATION_IMAGES", len(complete), "/372", flush=True)
        combined = DATA / "train-and-all-dev-v1.json"
        save(combined, read(DATA / "sft-v1.json") + complete)
        audit = audit_manifest(combined)
        save(out / "data-audit.json", audit)
        manifest = DATA / "validation-v1.json"
        save(manifest, complete)
        state.update(
            status="acquired",
            downloaded=len(complete),
            finished=now(),
            manifest_sha256=sha(manifest),
        )
        save(out / "acquisition-status.json", state)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(out / "acquisition-status.json", state)
        raise


if __name__ == "__main__":
    main()
