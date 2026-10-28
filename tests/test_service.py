"""Service contracts: durable transitions, optimistic review and feedback leakage."""

import base64
import io
import json
import sys
import tempfile
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memeguard.service.app import create_app
from memeguard.service.settings import Settings, digest
from memeguard.service.store import Conflict, Store
from memeguard.service.feedback import export_feedback


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(Path(self.temp.name))
        self.app = create_app(self.settings, testing=True)
        self.client = TestClient(self.app)
        self.client.headers["X-MemeGuard-Token"] = self.client.get(
            "/api/session"
        ).json()["token"]
        self.store = self.app.state.store
        self.release = self.app.state.registry["qlora-v0.9.0"]

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def item(
        self, text="A normal photograph", color="white", consent=True, source="user"
    ):
        stream = io.BytesIO()
        Image.new("RGB", (32, 32), color).save(stream, format="PNG")
        return dict(
            text=text,
            image_base64=base64.b64encode(stream.getvalue()).decode(),
            consent=consent,
            source=source,
        )

    def submit(self, item=None, key=None):
        response = self.client.post(
            "/api/submissions",
            json={
                "items": [item or self.item()],
                "idempotency_key": key or uuid.uuid4().hex,
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["items"][0]

    def score(self, sid=None, score=0.3):
        if sid is None:
            sid = self.submit()["id"]
        job = self.store.claim("test-worker")
        self.assertIsNotNone(job)
        self.store.complete(job, "test-worker", score, 0.01)
        return sid, job

    def review(self, sid, job, version=0, verdict=0, consent=True):
        return self.client.post(
            "/api/submissions/" + sid + "/reviews",
            json=dict(
                job_id=job["id"],
                expected_version=version,
                verdict=verdict,
                reviewer="test-reviewer",
                note="synthetic contract test",
                consent=consent,
            ),
        )

    def test_write_requires_session_token(self):
        self.client.headers.pop("X-MemeGuard-Token")
        self.assertEqual(self.client.post("/api/submissions", json={}).status_code, 403)

    def test_cross_origin_write_rejected(self):
        self.assertEqual(
            self.client.post(
                "/api/submissions", json={}, headers={"Origin": "https://other.example"}
            ).status_code,
            403,
        )

    def test_invalid_image_rejected_without_submission(self):
        item = self.item()
        item["image_base64"] = "not-base64"
        self.assertEqual(
            self.client.post(
                "/api/submissions", json={"items": [item], "idempotency_key": "invalid"}
            ).status_code,
            422,
        )
        self.assertEqual(self.store.metrics()["submissions"], 0)

    def test_batch_is_all_or_nothing(self):
        response = self.client.post(
            "/api/submissions",
            json={
                "items": [self.item(), dict(self.item(), text="")],
                "idempotency_key": "batch",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.store.metrics()["submissions"], 0)

    def test_idempotency_and_exact_pair_deduplication(self):
        a = self.submit(key="same")
        b = self.submit(key="same")
        self.assertEqual(a, b)
        c = self.submit(key="other")
        self.assertTrue(c["duplicate"])
        changed = self.client.post(
            "/api/submissions",
            json={"items": [self.item(text="changed")], "idempotency_key": "same"},
        )
        self.assertEqual(changed.status_code, 409)
        self.assertEqual(self.store.metrics()["submissions"], 1)

    def test_same_image_different_caption_is_not_duplicate(self):
        self.submit()
        self.submit(self.item(text="Another caption"))
        self.assertEqual(self.store.metrics()["submissions"], 2)

    def test_claim_is_atomic_under_concurrency(self):
        self.submit()
        with ThreadPoolExecutor(2) as pool:
            jobs = list(pool.map(self.store.claim, ["one", "two"]))
        self.assertEqual(sum(j is not None for j in jobs), 1)

    def test_expired_lease_recovers_and_stale_result_is_rejected(self):
        self.submit()
        old = self.store.claim("old")
        with self.store.connect() as db:
            db.execute(
                "UPDATE jobs SET lease=? WHERE id=?", (time.time() - 1, old["id"])
            )
        new = self.store.claim("new")
        self.assertEqual(old["id"], new["id"])
        with self.assertRaises(Conflict):
            self.store.complete(old, "old", 0.2, 0.1)
        self.store.complete(new, "new", 0.2, 0.1)

    def test_worker_heartbeat_extends_lease(self):
        self.submit()
        job = self.store.claim("worker")
        self.store.heartbeat("worker")
        with self.store.connect() as db:
            lease = db.execute(
                "SELECT lease FROM jobs WHERE id=?", (job["id"],)
            ).fetchone()[0]
        self.assertGreater(lease, time.time() + 200)

    def test_reviews_are_append_only_with_revision_check(self):
        sid, job = self.score()
        self.assertEqual(self.review(sid, job).status_code, 200)
        self.assertEqual(self.review(sid, job).status_code, 409)
        self.assertEqual(self.review(sid, job, version=1, verdict=1).status_code, 200)
        self.assertEqual(len(self.store.detail(sid)["reviews"]), 2)

    def test_review_requires_scored_job(self):
        result = self.submit()
        self.assertEqual(
            self.review(result["id"], {"id": result["job_id"]}).status_code, 409
        )

    def test_retry_preserves_original_result(self):
        sid, job = self.score()
        self.assertEqual(
            self.client.post("/api/submissions/" + sid + "/retry", json={}).status_code,
            200,
        )
        self.assertEqual(
            self.client.post("/api/submissions/" + sid + "/retry", json={}).status_code,
            409,
        )
        self.assertEqual(len(self.store.detail(sid)["jobs"]), 2)

    def test_release_switch_does_not_mutate_existing_jobs(self):
        result = self.submit()
        switched = self.client.post(
            "/api/releases/active",
            json={"release": "baseline-v0.9.0", "operator": "tester"},
        )
        self.assertEqual(switched.status_code, 200)
        self.assertEqual(
            self.store.detail(result["id"])["jobs"][0]["release"], "qlora-v0.9.0"
        )
        new = self.submit(self.item(text="next version"))
        self.assertEqual(
            self.store.detail(new["id"])["jobs"][0]["release"], "baseline-v0.9.0"
        )

    def test_invalid_score_not_persisted(self):
        self.submit()
        job = self.store.claim("worker")
        for value in (float("nan"), float("inf"), -1, 2):
            with self.assertRaises(ValueError):
                self.store.complete(job, "worker", value, 0.1)

    def test_frozen_threshold_used_for_priority(self):
        sid, job = self.score(score=self.release["threshold"])
        self.assertEqual(self.store.detail(sid)["jobs"][0]["bucket"], "priority")

    def test_feedback_excludes_holdout_and_unconsented_content(self):
        sid, job = self.score()
        self.review(sid, job)
        image = self.settings.media / (self.store.detail(sid)["image_sha"] + ".png")
        protected = Path(self.temp.name) / "holdout.json"
        protected.write_text(
            json.dumps(
                [dict(split="test_seen", image=str(image), image_sha256=digest(image))]
            )
        )
        report = export_feedback(self.store, [protected])
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(report["excluded"][0]["reason"], "protected_image_reuse")

    def test_feedback_uses_human_label_and_remains_not_train_ready(self):
        sid, job = self.score(score=0.9)
        self.review(sid, job, verdict=0)
        report = export_feedback(self.store, [])
        rows = json.loads(
            (
                self.settings.state / "exports" / report["export_id"] / "manifest.json"
            ).read_text()
        )
        self.assertEqual(rows[0]["label"], 0)
        self.assertFalse(report["train_ready"])

    def test_demo_feedback_never_enters_training(self):
        result = self.submit(self.item(source="demo"))
        sid, job = self.score(result["id"])
        self.review(sid, job)
        report = export_feedback(self.store, [])
        self.assertEqual(report["accepted"], 0)

    def test_consent_is_required_at_upload_and_review(self):
        result = self.submit(self.item(consent=False))
        sid, job = self.score(result["id"])
        self.review(sid, job)
        self.assertEqual(export_feedback(self.store, [])["accepted"], 0)

    def test_monitoring_and_not_ready_without_worker(self):
        self.assertEqual(self.client.get("/readyz").status_code, 503)
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertIn("memeguard_submissions_total", self.client.get("/metrics").text)

    def test_no_arbitrary_file_download(self):
        self.assertEqual(
            self.client.get("/api/exports/not-an-export/secret.json").status_code, 404
        )

    def test_database_survives_store_recreation(self):
        result = self.submit()
        new = Store(self.settings)
        self.assertEqual(new.detail(result["id"])["text"], "A normal photograph")

    def test_non_object_json_returns_validation_error(self):
        for path in [
            "/api/submissions",
            "/api/releases/active",
            "/api/submissions/missing/reviews",
        ]:
            with self.subTest(path=path):
                response = self.client.post(path, json=[])
                self.assertEqual(response.status_code, 422, response.text)

    def test_retry_respects_queue_capacity(self):
        sid, job = self.score()
        from dataclasses import replace

        limited = Store(replace(self.settings, max_jobs=1))
        self.submit(self.item(text="another image caption"))
        with self.assertRaises(Conflict):
            limited.retry(sid, self.release)
        self.assertEqual(len(self.store.detail(sid)["jobs"]), 1)

    def test_live_worker_readiness_and_stale_worker(self):
        self.store.heartbeat("ready-worker", "idle", "test")
        self.assertEqual(self.client.get("/readyz").status_code, 200)
        with self.store.connect() as db:
            db.execute("UPDATE workers SET updated=0")
        self.assertEqual(self.client.get("/readyz").status_code, 503)

    def test_export_training_config_tampering_is_rejected(self):
        import subprocess

        report = export_feedback(self.store, [])
        output = self.settings.state / "exports" / report["export_id"]
        config = output / "training-config.json"
        config.write_text("{}", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/train_service_feedback.py"),
                "--export",
                str(output),
                "--output",
                str(self.settings.state / "new-run"),
                "--validate-only",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Training config changed after audit", result.stderr)

    def test_priority_queue_precedes_newer_standard_item(self):
        first, job = self.score(score=0.9)
        second = self.submit(self.item(text="newer standard item"))["id"]
        self.score(second, score=0.01)
        rows = self.store.listing()["items"]
        self.assertEqual([r["id"] for r in rows], [first, second])


if __name__ == "__main__":
    unittest.main()
