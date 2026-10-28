"""Start missing local services, with independent logs and persistent queue state."""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.service.settings import Settings
from memeguard.service.store import Store


def main():
    settings = Settings.default()
    settings.initialize()
    web_alive = False
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8787/healthz", timeout=2
        ) as response:
            web_alive = response.status == 200
    except OSError:
        pass
    from memeguard.service.worker import exclusive_worker

    try:
        with exclusive_worker(settings.state / "worker.lock"):
            worker_alive = False
    except (OSError, RuntimeError):
        worker_alive = True
    flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        if os.name == "nt"
        else 0
    )
    children = []
    for name, script, alive in [
        ("web", "serve.py", web_alive),
        ("worker", "service_worker.py", worker_alive),
    ]:
        if alive:
            print(name.upper() + "_ALREADY_RUNNING")
            continue
        if name == "worker":
            (settings.state / "stop-worker.request").unlink(missing_ok=True)
        with (settings.state / (name + ".log")).open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "scripts" / script)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        children.append({"role": name, "pid": process.pid, "script": script})
    if children:
        (settings.state / "last-start.json").write_text(
            json.dumps(children, indent=2), encoding="utf-8"
        )
    print("LOCAL_WORKBENCH http://127.0.0.1:8787", json.dumps(children))


if __name__ == "__main__":
    main()
