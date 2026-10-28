"""Install only the web-service dependencies into the existing project interpreter."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("D:/career-projects/runs/MemeGuard/20261028-service-v1")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "environment-before.txt").write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    encoding="utf-8",
)
with (OUT / "install.log").open("w", encoding="utf-8") as log:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(ROOT / "requirements-service.txt"),
            "-i",
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "--timeout",
            "30",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
print("INSTALL_EXIT", result.returncode)
if result.returncode:
    raise SystemExit(result.returncode)
(OUT / "environment-after.txt").write_text(
    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    encoding="utf-8",
)
