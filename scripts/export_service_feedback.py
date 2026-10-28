"""Prepare audited feedback data; never auto-train or auto-promote."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    from memeguard.service.settings import Settings
    from memeguard.service.store import Store
    from memeguard.service.feedback import export_feedback

    print(
        json.dumps(
            export_feedback(Store(Settings.default())), ensure_ascii=False, indent=2
        )
    )
