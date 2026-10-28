"""SQLite transactions own queue state, review history and model-switch audit."""

import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, settings):
        self.settings = settings
        settings.initialize()
        with self.connect() as db:
            db.executescript(
                """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS submissions(
              id TEXT PRIMARY KEY, text TEXT NOT NULL, image_sha TEXT NOT NULL,
              pair_sha TEXT NOT NULL UNIQUE, consent INTEGER NOT NULL, source TEXT NOT NULL,
              created REAL NOT NULL, review_version INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY, submission_id TEXT NOT NULL REFERENCES submissions(id),
              release TEXT NOT NULL, fingerprint TEXT NOT NULL, threshold REAL NOT NULL,
              state TEXT NOT NULL, created REAL NOT NULL, started REAL, finished REAL,
              score REAL, bucket TEXT, seconds REAL, error TEXT, attempt INTEGER NOT NULL DEFAULT 0,
              worker TEXT, lease REAL);
            CREATE INDEX IF NOT EXISTS queue_idx ON jobs(state,created);
            CREATE TABLE IF NOT EXISTS reviews(
              id TEXT PRIMARY KEY, submission_id TEXT NOT NULL REFERENCES submissions(id),
              job_id TEXT NOT NULL REFERENCES jobs(id), version INTEGER NOT NULL,
              verdict INTEGER NOT NULL, reviewer TEXT NOT NULL, note TEXT NOT NULL,
              consent INTEGER NOT NULL, created REAL NOT NULL,
              UNIQUE(submission_id,version));
            CREATE TABLE IF NOT EXISTS events(
              id INTEGER PRIMARY KEY AUTOINCREMENT,type TEXT NOT NULL,
              target TEXT NOT NULL,detail TEXT NOT NULL,created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS requests(
              key TEXT PRIMARY KEY, digest TEXT NOT NULL,response TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS workers(
              id TEXT PRIMARY KEY,state TEXT NOT NULL,detail TEXT NOT NULL,updated REAL NOT NULL);
            """
            )
            db.execute("INSERT OR IGNORE INTO settings VALUES('schema','1')")
            if (
                db.execute("SELECT value FROM settings WHERE key='schema'").fetchone()[
                    0
                ]
                != "1"
            ):
                raise ValueError("Unsupported database schema")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.settings.database, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=20000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def event(self, db, kind, target, detail):
        db.execute(
            "INSERT INTO events(type,target,detail,created) VALUES(?,?,?,?)",
            (kind, target, json.dumps(detail, ensure_ascii=False), time.time()),
        )

    def active(self, default):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO settings VALUES('active',?)", (default,))
            return db.execute(
                "SELECT value FROM settings WHERE key='active'"
            ).fetchone()[0]

    def switch(self, release, reviewer):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            before = db.execute(
                "SELECT value FROM settings WHERE key='active'"
            ).fetchone()
            db.execute("INSERT OR REPLACE INTO settings VALUES('active',?)", (release,))
            self.event(
                db,
                "release_switch",
                release,
                {"before": before[0] if before else None, "operator": reviewer},
            )

    def enqueue(self, db, submission, release):
        jid = uuid.uuid4().hex
        db.execute(
            """INSERT INTO jobs(id,submission_id,release,fingerprint,threshold,state,created)
                      VALUES(?,?,?,?,?,'queued',?)""",
            (
                jid,
                submission,
                release["id"],
                release["fingerprint"],
                release["threshold"],
                time.time(),
            ),
        )
        self.event(db, "queued", jid, {"release": release["id"]})
        return jid

    def submit(self, items, release, key, request_digest):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM requests WHERE key=?", (key,)).fetchone()
            if old:
                if old["digest"] != request_digest:
                    raise Conflict("Idempotency key reused with different content")
                return json.loads(old["response"])
            pending = db.execute(
                "SELECT count(*) FROM jobs WHERE state IN ('queued','running')"
            ).fetchone()[0]
            if pending + len(items) > self.settings.max_jobs:
                raise Conflict("Queue capacity reached; retry later")
            result = []
            for item in items:
                old = db.execute(
                    "SELECT id FROM submissions WHERE pair_sha=?", (item["pair_sha"],)
                ).fetchone()
                if old:
                    result.append({"id": old["id"], "duplicate": True})
                    continue
                sid = uuid.uuid4().hex
                db.execute(
                    """INSERT INTO submissions(id,text,image_sha,pair_sha,consent,source,created)
                              VALUES(?,?,?,?,?,?,?)""",
                    (
                        sid,
                        item["text"],
                        item["image_sha"],
                        item["pair_sha"],
                        int(item["consent"]),
                        item["source"],
                        time.time(),
                    ),
                )
                jid = self.enqueue(db, sid, release)
                result.append({"id": sid, "job_id": jid, "duplicate": False})
            db.execute(
                "INSERT INTO requests VALUES(?,?,?)",
                (key, request_digest, json.dumps(result)),
            )
            return result

    def claim(self, worker):
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            stale = db.execute(
                "SELECT * FROM jobs WHERE state='running' AND lease<?", (now,)
            ).fetchall()
            for row in stale:
                state = "failed" if row["attempt"] >= 3 else "queued"
                db.execute(
                    "UPDATE jobs SET state=?,worker=NULL,lease=NULL,error=? WHERE id=?",
                    (
                        state,
                        "Worker lease expired; inference completion was not committed",
                        row["id"],
                    ),
                )
                self.event(db, "lease_recovery", row["id"], {"state": state})
            row = db.execute(
                "SELECT * FROM jobs WHERE state='queued' ORDER BY created,id LIMIT 1"
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE jobs SET state='running',worker=?,lease=?,attempt=attempt+1,started=?,error=NULL WHERE id=?",
                (worker, now + self.settings.lease_seconds, now, row["id"]),
            )
            self.event(db, "claimed", row["id"], {"worker": worker})
            job = dict(
                db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
            )
            job["submission"] = dict(
                db.execute(
                    "SELECT * FROM submissions WHERE id=?", (job["submission_id"],)
                ).fetchone()
            )
            return job

    def heartbeat(self, worker, state="idle", detail=""):
        now = time.time()
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO workers VALUES(?,?,?,?)",
                (worker, state, detail[:500], now),
            )
            db.execute(
                "UPDATE jobs SET lease=? WHERE worker=? AND state='running'",
                (now + self.settings.lease_seconds, worker),
            )

    def complete(self, job, worker, score, seconds):
        if (
            not math.isfinite(score)
            or not 0 <= score <= 1
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("Invalid model result")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "UPDATE jobs SET state='succeeded',score=?,bucket=?,seconds=?,finished=?,lease=NULL WHERE id=? AND state='running' AND worker=? AND lease>=?",
                (
                    score,
                    "priority" if score >= job["threshold"] else "standard",
                    seconds,
                    time.time(),
                    job["id"],
                    worker,
                    time.time(),
                ),
            ).rowcount
            if count != 1:
                raise Conflict("Lost worker lease; stale output discarded")
            self.event(
                db, "scored", job["id"], {"score": score, "release": job["release"]}
            )

    def fail(self, job, worker, message):
        with self.connect() as db:
            count = db.execute(
                "UPDATE jobs SET state='failed',error=?,finished=?,lease=NULL WHERE id=? AND state='running' AND worker=?",
                (message[:500], time.time(), job["id"], worker),
            ).rowcount
            if count:
                self.event(db, "failed", job["id"], {"error": message[:500]})

    def retry(self, submission, release):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute(
                "SELECT 1 FROM submissions WHERE id=?", (submission,)
            ).fetchone():
                raise KeyError(submission)
            if db.execute(
                "SELECT 1 FROM jobs WHERE submission_id=? AND state IN ('queued','running')",
                (submission,),
            ).fetchone():
                raise Conflict("Submission already has a pending job")
            if (
                db.execute(
                    "SELECT count(*) FROM jobs WHERE state IN ('queued','running')"
                ).fetchone()[0]
                >= self.settings.max_jobs
            ):
                raise Conflict("Queue capacity reached")
            return self.enqueue(db, submission, release)

    def detail(self, sid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
            if not row:
                raise KeyError(sid)
            item = dict(row)
            item["jobs"] = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM jobs WHERE submission_id=? ORDER BY created DESC,id DESC",
                    (sid,),
                )
            ]
            item["reviews"] = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM reviews WHERE submission_id=? ORDER BY version DESC",
                    (sid,),
                )
            ]
            return item

    def listing(self, bucket=None, state=None, offset=0, limit=30):
        # Only the latest job determines the current queue entry; historical scores remain visible.
        with self.connect() as db:
            sql = """SELECT s.*,j.id job_id,j.state,j.score,j.bucket,j.release,j.threshold,j.seconds,j.error,
                   (SELECT verdict FROM reviews r WHERE r.submission_id=s.id ORDER BY version DESC LIMIT 1) verdict
                   FROM submissions s JOIN jobs j ON j.id=(SELECT id FROM jobs WHERE submission_id=s.id ORDER BY created DESC,id DESC LIMIT 1)"""
            where = []
            args = []
            if bucket:
                where.append("j.bucket=?")
                args.append(bucket)
            if state:
                where.append("j.state=?")
                args.append(state)
            if where:
                sql += " WHERE " + " AND ".join(where)
            count = db.execute("SELECT count(*) FROM (" + sql + ")", args).fetchone()[0]
            rows = db.execute(
                sql
                + " ORDER BY CASE WHEN j.bucket='priority' AND s.review_version=0 THEN 0 ELSE 1 END, s.created DESC LIMIT ? OFFSET ?",
                args + [limit, offset],
            ).fetchall()
            return {"total": count, "items": [dict(r) for r in rows]}

    def review(self, sid, job_id, expected_version, verdict, reviewer, note, consent):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
            if not row:
                raise KeyError(sid)
            if row["review_version"] != expected_version:
                raise Conflict("Review changed; reload before saving")
            job = db.execute(
                "SELECT * FROM jobs WHERE id=? AND submission_id=? AND state='succeeded'",
                (job_id, sid),
            ).fetchone()
            if not job:
                raise Conflict("Review must reference a completed inference")
            if verdict not in (0, 1):
                raise ValueError("Expected verdict 0 or 1")
            version = expected_version + 1
            db.execute(
                "INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    uuid.uuid4().hex,
                    sid,
                    job_id,
                    version,
                    verdict,
                    reviewer,
                    note,
                    int(consent),
                    time.time(),
                ),
            )
            db.execute(
                "UPDATE submissions SET review_version=? WHERE id=?", (version, sid)
            )
            self.event(
                db,
                "reviewed",
                sid,
                {"version": version, "reviewer": reviewer, "job_id": job_id},
            )
            return version

    def reviewed(self):
        with self.connect() as db:
            rows = db.execute(
                """SELECT s.*,r.verdict label,r.reviewer,r.note,r.consent review_consent,
                r.id review_id,r.job_id,r.created reviewed_at,j.score,j.threshold,j.release
                FROM submissions s JOIN reviews r ON r.submission_id=s.id AND r.version=s.review_version
                JOIN jobs j ON j.id=r.job_id ORDER BY s.id"""
            ).fetchall()
            return [dict(r) for r in rows]

    def metrics(self):
        with self.connect() as db:
            counts = {
                r["state"]: r["n"]
                for r in db.execute("SELECT state,count(*) n FROM jobs GROUP BY state")
            }
            queue_age = db.execute(
                "SELECT min(created) FROM jobs WHERE state='queued'"
            ).fetchone()[0]
            workers = [dict(r) for r in db.execute("SELECT * FROM workers")]
            samples = [
                dict(r)
                for r in db.execute(
                    "SELECT seconds,score,release FROM jobs WHERE state='succeeded' ORDER BY finished DESC LIMIT 200"
                )
            ]
            total = db.execute("SELECT count(*) FROM submissions").fetchone()[0]
        reviewed = self.reviewed()
        disagreement = sum(
            int(r["label"]) != int(r["score"] >= r["threshold"]) for r in reviewed
        )
        durations = sorted(r["seconds"] for r in samples)
        return {
            "submissions": total,
            "jobs": counts,
            "oldest_queued_seconds": (
                max(0, time.time() - queue_age) if queue_age else 0
            ),
            "workers": workers,
            "recent_successes": len(samples),
            "recent_latency_p50": durations[len(durations) // 2] if durations else None,
            "recent_latency_p95": (
                durations[min(len(durations) - 1, int(len(durations) * 0.95))]
                if durations
                else None
            ),
            "reviewed": len(reviewed),
            "review_disagreements": disagreement,
            "score_histogram": [
                sum(
                    i / 10 <= r["score"] < (i + 1) / 10 or (i == 9 and r["score"] == 1)
                    for r in samples
                )
                for i in range(10)
            ],
            "scope": "Local observed traffic; not unbiased accuracy or business prevalence",
        }
