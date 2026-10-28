"""Start the review service from VS Code; bind only to Windows loopback."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import uvicorn
    from memeguard.service.app import create_app

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8787)
    args = p.parse_args()
    uvicorn.run(
        create_app(), host="127.0.0.1", port=args.port, workers=1, access_log=False
    )


if __name__ == "__main__":
    main()
