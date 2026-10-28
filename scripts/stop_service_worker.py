"""Ask the CUDA worker to finish its current job and release GPU memory."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.service.settings import Settings


def main():
    settings = Settings.default()
    settings.initialize()
    (settings.state / "stop-worker.request").write_text(
        "Finish current job, then stop.\n", encoding="utf-8"
    )
    print(
        "STOP_REQUESTED: verify /api/metrics shows stopped before starting training. Queued jobs persist."
    )


if __name__ == "__main__":
    main()
