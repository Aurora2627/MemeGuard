"""Wait for verified assets, then run the development pilot and training capacity probe."""

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.common import read, save, now


def main():
    out = Path("D:/career-projects/runs/MemeGuard/20261023-first-phase-v1")
    out.mkdir(parents=True, exist_ok=False)
    state = {"status": "waiting_for_assets", "started": now()}
    save(out / "status.json", state)
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        PYTHONIOENCODING="utf-8",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    try:
        deadline = time.monotonic() + 90 * 60
        expected = {
            "environment-status.json": "installed",
            "model-status.json": "verified",
            "acquisition-status.json": "acquired",
        }
        while True:
            ready = True
            for filename, success in expected.items():
                path = ROOT / "reports/bootstrap" / filename
                status = read(path) if path.exists() else {"status": "pending"}
                if status["status"] == "failed":
                    raise RuntimeError(
                        filename + ": " + status.get("error", "unknown failure")
                    )
                ready = ready and status["status"] == success
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Asset preparation exceeded 90 minutes")
            time.sleep(10)
        state.update(status="testing")
        save(out / "status.json", state)
        with (out / "tests.txt").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        state.update(status="baseline_running")
        save(out / "status.json", state)
        with (out / "baseline.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-X",
                    "utf8",
                    "scripts/run_baseline.py",
                    "--output",
                    str(out / "zero-shot"),
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        state.update(status="capacity_probe_running")
        save(out / "status.json", state)
        with (out / "capacity.log").open("w", encoding="utf-8") as log:
            capacity = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-X",
                    "utf8",
                    "scripts/probe_qlora.py",
                    "--output",
                    str(out / "qlora-capacity"),
                ],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        state.update(
            status=(
                "complete"
                if capacity.returncode == 0
                else "baseline_complete_capacity_failed"
            ),
            finished=now(),
            capacity_probe_passed=capacity.returncode == 0,
        )
        save(out / "status.json", state)
        print("FIRST_PHASE_FINISHED", state, flush=True)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
