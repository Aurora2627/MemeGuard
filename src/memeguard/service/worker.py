"""Single CUDA worker with OS lock, heartbeats and recoverable queue leases."""

import contextlib
import json
import os
import threading
import time
import uuid
from .settings import Settings, releases
from .store import Store


@contextlib.contextmanager
def exclusive_worker(path):
    # An OS lock is released by process termination, unlike a stale PID file.
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(settings=None, once=False):
    settings = settings or Settings.default()
    store = Store(settings)
    worker = uuid.uuid4().hex
    stop = threading.Event()
    status = {
        "state": "loading",
        "detail": "Verifying and loading the frozen CUDA model",
    }

    def beat():
        while not stop.is_set():
            store.heartbeat(worker, status["state"], status["detail"])
            stop.wait(10)

    with exclusive_worker(settings.state / "worker.lock"):
        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            from .model import FrozenClassifier

            _, registry = releases(settings)
            model = FrozenClassifier(registry)
            status.update(state="idle", detail="Frozen Qwen2-VL + adapter loaded")
            store.heartbeat(worker, **status)
            print("CUDA_WORKER_READY", worker, flush=True)
            while not (settings.state / "stop-worker.request").exists():
                job = store.claim(worker)
                if job is None:
                    if once:
                        break
                    time.sleep(0.5)
                    continue
                status.update(state="scoring", detail=job["id"])
                store.heartbeat(worker, **status)
                try:
                    score, seconds = model.score(
                        job, settings.media / (job["submission"]["image_sha"] + ".png")
                    )
                    store.complete(job, worker, score, seconds)
                    print(
                        json.dumps(
                            {
                                "event": "scored",
                                "job": job["id"],
                                "release": job["release"],
                                "seconds": seconds,
                            }
                        ),
                        flush=True,
                    )
                except Exception as error:
                    message = type(error).__name__ + ": " + str(error)
                    store.fail(job, worker, message)
                    print(
                        json.dumps(
                            {
                                "event": "failed",
                                "job": job["id"],
                                "error": message[:500],
                            }
                        ),
                        flush=True,
                    )
                status.update(state="idle", detail="Ready")
                if once:
                    break
        finally:
            stop.set()
            thread.join(timeout=15)
            store.heartbeat(worker, "stopped", "Worker stopped; queued jobs persist")
