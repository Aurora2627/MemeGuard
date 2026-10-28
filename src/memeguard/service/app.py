"""Loopback-only FastAPI workbench; model execution lives in a separate process."""

import base64
import binascii
import hashlib
import io
import json
import secrets
import time
import uuid
import warnings
from pathlib import Path
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from .settings import Settings, releases, public_release
from .store import Store, Conflict

STATIC = Path(__file__).parent / "static"


def canonical_item(item, settings):
    if not isinstance(item, dict):
        raise ValueError("Expected an item object")
    text = item.get("text")
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 4000:
        raise ValueError("Caption must contain 1-4000 characters")
    if type(item.get("consent", False)) is not bool:
        raise ValueError("Training consent must be boolean")
    encoded = item.get("image_base64")
    if (
        not isinstance(encoded, str)
        or len(encoded) > settings.max_upload_bytes * 4 // 3 + 16
    ):
        raise ValueError("Image must be at most 5 MiB")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Invalid base64 image") from error
    if len(raw) > settings.max_upload_bytes:
        raise ValueError("Image too large")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if (
                    im.format not in ("PNG", "JPEG", "WEBP")
                    or getattr(im, "n_frames", 1) != 1
                ):
                    raise ValueError("Use a single-frame PNG, JPEG or WebP")
                if im.width * im.height > settings.max_pixels or min(im.size) < 8:
                    raise ValueError("Image dimensions outside supported limits")
                image = ImageOps.exif_transpose(im).convert("RGB")
                image.load()
    except Exception as error:
        raise ValueError("Invalid or unsupported image: " + str(error)[:100]) from error
    clean = io.BytesIO()
    image.save(clean, format="PNG")
    body = clean.getvalue()
    image_sha = hashlib.sha256(body).hexdigest()
    normalized = " ".join(
        __import__("unicodedata").normalize("NFKC", text).casefold().split()
    )
    pair_sha = hashlib.sha256((image_sha + "\n" + normalized).encode()).hexdigest()
    path = settings.media / (image_sha + ".png")
    if not path.exists():
        temp = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temp.write_bytes(body)
        temp.replace(path)
    source = item.get("source", "user")
    if source not in ("user", "demo", "integration_test"):
        raise ValueError("Unknown content source")
    return dict(
        text=text.strip(),
        image_sha=image_sha,
        pair_sha=pair_sha,
        consent=item.get("consent", False),
        source=source,
    )


def create_app(settings=None, testing=False):
    settings = settings or Settings.default()
    store = Store(settings)
    default, registry = releases(settings)
    store.active(default)
    token = secrets.token_urlsafe(32)
    app = FastAPI(
        title="MemeGuard Review Workbench", version="0.10.0", docs_url="/docs"
    )
    app.state.store = store
    app.state.settings = settings
    app.state.registry = registry
    app.state.csrf = token

    @app.middleware("http")
    async def local_guard(request, call_next):
        host = request.url.hostname
        allowed = {"localhost", "127.0.0.1", "::1"} | (
            {"testserver"} if testing else set()
        )
        if host not in allowed:
            return JSONResponse(
                {"detail": "Use the local forwarded address"}, status_code=400
            )
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
                return JSONResponse(
                    {"detail": "Cross-origin write rejected"}, status_code=403
                )
            if not secrets.compare_digest(
                request.headers.get("x-memeguard-token", ""), token
            ):
                return JSONResponse(
                    {"detail": "Refresh the page to obtain a session token"},
                    status_code=403,
                )
            # Bound the body while reading; Content-Length alone is not trusted.
            chunks = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > settings.max_body_bytes:
                    return JSONResponse(
                        {"detail": "Request body too large"}, status_code=413
                    )
                chunks.append(chunk)
            request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({"detail": "Record not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/session")
    def session():
        return {
            "token": token,
            "scope": "Local single-user research workbench; reviewer names are self-reported",
        }

    @app.get("/healthz")
    def health():
        return {"status": "ok", "database": "reachable"}

    @app.get("/readyz")
    def ready():
        workers = store.metrics()["workers"]
        live = any(
            w["state"] in ("idle", "scoring") and time.time() - w["updated"] < 35
            for w in workers
        )
        return JSONResponse(
            {"ready": live, "model_backend": "frozen_cuda"},
            status_code=200 if live else 503,
        )

    @app.get("/api/releases")
    def model_releases():
        return {
            "active": store.active(default),
            "items": [public_release(r) for r in registry.values()],
        }

    @app.post("/api/releases/active")
    async def activate(request: Request):
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        if (
            data.get("release") not in registry
            or not str(data.get("operator", "")).strip()
        ):
            raise ValueError("Choose a registered release and provide operator name")
        store.switch(data["release"], str(data["operator"]).strip()[:80])
        return {
            "active": data["release"],
            "scope": "Applies to new jobs; historical jobs retain their captured release",
        }

    @app.post("/api/submissions")
    async def submit(request: Request):
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        items = data.get("items")
        if not isinstance(items, list) or not 1 <= len(items) <= settings.max_batch:
            raise ValueError("Submit 1-10 images per batch")
        key = data.get("idempotency_key")
        if not isinstance(key, str) or not 1 <= len(key) <= 100:
            raise ValueError("Provide an idempotency key")
        # Decode the entire batch before any DB insertion.
        prepared = [canonical_item(item, settings) for item in items]
        request_digest = hashlib.sha256(
            json.dumps(prepared, sort_keys=True).encode()
        ).hexdigest()
        release = registry[store.active(default)]
        result = store.submit(prepared, release, key, request_digest)
        return JSONResponse({"items": result}, status_code=202)

    @app.get("/api/submissions")
    def listing(
        bucket: str | None = None,
        state: str | None = None,
        offset: int = 0,
        limit: int = 30,
    ):
        if bucket not in (None, "priority", "standard") or state not in (
            None,
            "queued",
            "running",
            "succeeded",
            "failed",
        ):
            raise ValueError("Invalid filter")
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Invalid pagination")
        return store.listing(bucket, state, offset, limit)

    @app.get("/api/submissions/{sid}")
    def detail(sid: str):
        return store.detail(sid)

    @app.get("/api/submissions/{sid}/image")
    def image(sid: str):
        item = store.detail(sid)
        path = settings.media / (item["image_sha"] + ".png")
        if not path.exists():
            raise HTTPException(404, "Stored image unavailable")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/submissions/{sid}/retry")
    def retry(sid: str):
        return {"job_id": store.retry(sid, registry[store.active(default)])}

    @app.post("/api/submissions/{sid}/reviews")
    async def review(sid: str, request: Request):
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        if type(data.get("verdict")) is not int or data["verdict"] not in (0, 1):
            raise ValueError("Choose benign or hateful")
        if (
            type(data.get("expected_version")) is not int
            or type(data.get("consent", False)) is not bool
        ):
            raise ValueError("Invalid review revision or consent")
        reviewer = data.get("reviewer", "")
        if not isinstance(reviewer, str):
            raise ValueError("Reviewer must be text")
        reviewer = reviewer.strip()
        note = data.get("note", "")
        if (
            not reviewer
            or len(reviewer) > 80
            or not isinstance(note, str)
            or len(note) > 2000
        ):
            raise ValueError(
                "Provide a reviewer name and a note shorter than 2000 characters"
            )
        version = store.review(
            sid,
            data.get("job_id"),
            data["expected_version"],
            data["verdict"],
            reviewer,
            note,
            data.get("consent", False),
        )
        return {"version": version}

    @app.get("/api/metrics")
    def metrics():
        return store.metrics()

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus():
        m = store.metrics()
        lines = [
            "# TYPE memeguard_submissions_total gauge",
            "memeguard_submissions_total " + str(m["submissions"]),
        ]
        for state, n in m["jobs"].items():
            lines.append('memeguard_jobs{state="' + state + '"} ' + str(n))
        lines.extend(
            [
                "memeguard_queue_oldest_seconds " + str(m["oldest_queued_seconds"]),
                "memeguard_reviews_total " + str(m["reviewed"]),
                "memeguard_review_disagreements_total "
                + str(m["review_disagreements"]),
            ]
        )
        return "\n".join(lines) + "\n"

    @app.get("/api/feedback")
    def feedback():
        rows = store.reviewed()
        return {
            "eligible": sum(
                bool(r["consent"] and r["review_consent"] and r["source"] == "user")
                for r in rows
            ),
            "reviewed": len(rows),
            "scope": "Only explicit-consent human reviews; immutable holdout checks are required before training",
        }

    @app.post("/api/feedback/export")
    def export():
        from .feedback import export_feedback

        return export_feedback(store)

    @app.get("/api/exports/{export_id}/{filename}")
    def exported_file(export_id: str, filename: str):
        import re

        if not re.fullmatch(
            r"[0-9]{8}-[0-9]{6}-[a-f0-9]{8}", export_id
        ) or filename not in ("audit.json", "manifest.json", "training-config.json"):
            raise HTTPException(404, "Unknown export")
        path = settings.state / "exports" / export_id / filename
        if not path.is_file():
            raise HTTPException(404, "Export not found")
        return FileResponse(path, media_type="application/json", filename=filename)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
