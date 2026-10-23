"""Install isolated, pinned CUDA dependencies without changing other projects."""

from pathlib import Path
import datetime
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV_PYTHON = Path("D:/career-projects/envs/memeguard/Scripts/python.exe")
REPORT = ROOT / "reports/bootstrap"
STATUS = REPORT / "environment-status.json"


def save(data):
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(STATUS)


def main():
    state = {
        "status": "installing",
        "started": datetime.datetime.now().astimezone().isoformat(),
        "interpreter": str(ENV_PYTHON),
    }
    save(state)
    environment = dict(
        os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONIOENCODING="utf-8"
    )
    steps = [
        [
            "install",
            "torch==2.6.0",
            "torchvision==0.21.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        [
            "install",
            "-r",
            str(ROOT / "requirements-runtime.txt"),
            "--index-url",
            "https://pypi.org/simple",
        ],
    ]
    try:
        with (REPORT / "environment-install.log").open("w", encoding="utf-8") as log:
            for index, arguments in enumerate(steps):
                state["stage"] = index + 1
                save(state)
                subprocess.run(
                    [
                        str(ENV_PYTHON),
                        "-m",
                        "pip",
                        *arguments,
                        "--proxy",
                        "http://127.0.0.1:7890",
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        result = subprocess.run(
            [str(ENV_PYTHON), "-m", "pip", "check"],
            env=environment,
            capture_output=True,
        )
        (REPORT / "pip-check.txt").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError("Dependency consistency check failed")
        freeze = subprocess.check_output(
            [str(ENV_PYTHON), "-m", "pip", "freeze"], env=environment
        )
        (REPORT / "environment-freeze.txt").write_bytes(freeze)
        state.update(
            status="installed",
            finished=datetime.datetime.now().astimezone().isoformat(),
        )
        save(state)
    except Exception as error:
        state.update(status="failed", error=str(error))
        save(state)
        raise


if __name__ == "__main__":
    main()
