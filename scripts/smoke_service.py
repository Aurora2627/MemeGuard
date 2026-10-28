"""Real CUDA HTTP smoke check; synthetic cases are excluded from feedback training."""

import argparse
import base64
import io
import json
import time
import uuid
from pathlib import Path
import httpx
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with httpx.Client(base_url="http://127.0.0.1:8787", timeout=120) as client:
        token = client.get("/api/session").raise_for_status().json()["token"]
        client.headers["X-MemeGuard-Token"] = token

        def post(path, data):
            response = client.post(path, json=data)
            response.raise_for_status()
            return response.json()

        def score(sid):
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                detail = client.get("/api/submissions/" + sid).raise_for_status().json()
                job = detail["jobs"][0]
                if job["state"] == "failed":
                    raise RuntimeError(job["error"])
                if job["state"] == "succeeded":
                    return job
                time.sleep(1)
            raise TimeoutError("Inference did not complete within 300 seconds")

        releases = client.get("/api/releases").raise_for_status().json()
        print("RELEASES", releases, flush=True)
        initial = releases["active"]
        try:
            for release in ["qlora-v0.9.0", "baseline-v0.9.0"]:
                post(
                    "/api/releases/active",
                    {"release": release, "operator": "integration-test"},
                )
                image = Image.new("RGB", (280, 196), "#e8efe9")
                draw = ImageDraw.Draw(image)
                draw.rectangle((70, 60, 210, 160), fill="#7b9e84")
                draw.text((16, 18), "A peaceful garden", fill="#203a2b")
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                item = {
                    "image_base64": base64.b64encode(buf.getvalue()).decode(),
                    "text": "A peaceful garden. " + release,
                    "consent": False,
                    "source": "integration_test",
                }
                data = {"items": [item], "idempotency_key": "smoke-" + uuid.uuid4().hex}
                submitted = post("/api/submissions", data)
                print("SUBMITTED", submitted, flush=True)
                assert post("/api/submissions", data) == submitted
                sid = submitted["items"][0]["id"]
                job = score(sid)
                assert job["release"] == release and 0 <= job["score"] <= 1
                results.append(
                    {
                        key: job[key]
                        for key in [
                            "id",
                            "release",
                            "state",
                            "score",
                            "threshold",
                            "bucket",
                            "seconds",
                        ]
                    }
                )
                print("SCORED", results[-1], flush=True)
            export = post("/api/feedback/export", {})
            assert (
                export["accepted"] == 0
            ), "Synthetic cases must not become training samples"
            results.append({"feedback_export": export})
        finally:
            post(
                "/api/releases/active",
                {"release": initial, "operator": "integration-test-restore"},
            )
        report = {
            "scope": "Synthetic real-CUDA engineering smoke; not a quality evaluation or human study",
            "results": results,
            "metrics": client.get("/api/metrics").raise_for_status().json(),
        }
        (args.output / "smoke.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("SMOKE_PASSED", flush=True)


if __name__ == "__main__":
    main()
