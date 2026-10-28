"""Start the durable CUDA worker separately from the web API."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if __name__ == "__main__":
    from memeguard.service.worker import run

    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    run(once=p.parse_args().once)
