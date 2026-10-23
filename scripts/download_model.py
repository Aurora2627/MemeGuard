"""Resume model downloads and verify immutable Hub object identities."""

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, sha, now


def verified(path, record):
    if not path.is_file() or path.stat().st_size != record["size"]:
        return False
    if record.get("lfs"):
        return sha(path) == record["lfs"]["oid"]
    digest = hashlib.sha1()
    digest.update(("blob " + str(record["size"])).encode() + bytes([0]))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == record["oid"]


def main():
    model = read(ROOT / "configs/sources.lock.json")["model"]
    directory = Path(model["local_root"])
    directory.mkdir(parents=True, exist_ok=True)
    records = [
        row
        for row in read(ROOT / "reports/bootstrap/model-files.json")
        if row.get("type") == "file"
        and (
            row["path"].endswith((".json", ".safetensors", ".txt"))
            or row["path"] == "LICENSE"
        )
    ]
    if not any(row["path"].endswith(".safetensors") for row in records):
        raise ValueError("No safe tensor weights found")
    status_path = ROOT / "reports/bootstrap/model-status.json"
    state = {
        "status": "downloading",
        "started": now(),
        "files_total": len(records),
        "files_complete": 0,
        "bytes_total": sum(row["size"] for row in records),
    }
    save(status_path, state)
    completed = []
    try:
        for row in records:
            path = directory / row["path"]
            if path.parent != directory:
                raise ValueError("Nested model asset not reviewed")
            if not verified(path, row):
                if path.exists():
                    raise ValueError(
                        "Existing final model file failed integrity check: " + path.name
                    )
                temporary = path.with_suffix(path.suffix + ".part")
                url = (
                    "https://huggingface.co/"
                    + model["repository"]
                    + "/resolve/"
                    + model["revision"]
                    + "/"
                    + urllib.parse.quote(row["path"], safe="/")
                )
                for attempt in range(4):
                    try:
                        offset = temporary.stat().st_size if temporary.exists() else 0
                        request = urllib.request.Request(
                            url,
                            headers=(
                                {"Range": "bytes=" + str(offset) + "-"}
                                if offset
                                else {}
                            ),
                        )
                        opener = urllib.request.build_opener(
                            urllib.request.ProxyHandler(
                                {"https": "http://127.0.0.1:7890"}
                            )
                        )
                        with opener.open(request, timeout=60) as response:
                            if offset and response.status == 206:
                                if not response.headers.get(
                                    "Content-Range", ""
                                ).startswith("bytes " + str(offset) + "-"):
                                    raise ValueError("Unexpected resume range")
                            elif offset:
                                offset = 0
                            mode = "ab" if offset else "wb"
                            last_report = time.monotonic()
                            with temporary.open(mode) as stream:
                                while True:
                                    chunk = response.read(1024 * 1024)
                                    if not chunk:
                                        break
                                    stream.write(chunk)
                                    offset += len(chunk)
                                    if time.monotonic() - last_report > 5:
                                        state.update(
                                            current_file=path.name,
                                            current_bytes=offset,
                                            current_total=row["size"],
                                        )
                                        save(status_path, state)
                                        last_report = time.monotonic()
                        if not verified(temporary, row):
                            raise ValueError(
                                "Downloaded model asset identity mismatch: " + path.name
                            )
                        temporary.replace(path)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2**attempt)
            completed.append(
                {"file": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
            )
            state.update(files_complete=len(completed), current_file=path.name)
            save(status_path, state)
            print("MODEL_FILE_VERIFIED", path.name, flush=True)
        identity = dict(model, files=completed)
        save(directory / "identity.json", identity)
        state.update(
            status="verified",
            finished=now(),
            identity_sha256=sha(directory / "identity.json"),
        )
        save(status_path, state)
        print("MODEL_VERIFIED", directory, flush=True)
    except Exception as error:
        state.update(status="failed", error=str(error))
        save(status_path, state)
        raise


if __name__ == "__main__":
    main()
