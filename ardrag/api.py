import mimetypes
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ardrag import chat as chat_mod
from ardrag import classify, core, db, mcp_supervisor
from ardrag import classify_job
from ardrag import reindex as reindex_mod
from ardrag.auth import create_session_token, get_current_user, require_auth, verify_credentials
from ardrag.config import (
    ALLOWED_UPLOAD_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    SUPPORTED_EMBEDDING_MODELS,
    UPLOAD_DIR,
)
from ardrag.extract import extract_text
from ardrag.markdown_lite import render_markdown_lite

app = FastAPI(title="Ardrag API")

db.init_db()
core.ensure_collection()


@app.on_event("startup")
def _start_mcp_subprocess():
    mcp_supervisor.start()


@app.on_event("shutdown")
def _stop_mcp_subprocess():
    mcp_supervisor.stop()


WEB_DIR = Path(__file__).parent / "web"
_start_time = time.monotonic()

# Server-rendered HTML fragments for the incremental HTMX migration (Documents tab first — see
# GET /documents/table below). Kept alongside the rest of the vanilla web UI rather than
# replacing it wholesale; other tabs still render entirely client-side for now.
templates = Jinja2Templates(directory=WEB_DIR / "templates")
templates.env.globals["render_markdown_lite"] = render_markdown_lite

AuthDep = Depends(require_auth)

_batch_status = {"running": False, "total": 0, "done": 0, "current": None, "results": []}


def _serialize_doc(doc: db.Document) -> dict:
    return {
        "id": doc.id,
        "name": doc.name,
        "content_hash": doc.content_hash,
        "chunk_count": doc.chunk_count,
        "chunk_size": doc.chunk_size,
        "chunk_overlap": doc.chunk_overlap,
        "vendor": doc.vendor,
        "doc_type": doc.doc_type,
        "tags": doc.tags,
        "uploaded_at": doc.uploaded_at.isoformat(),
    }


def _parse_tags(raw: list[str]) -> list[str]:
    tags: list[str] = []
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if part and part not in tags:
                tags.append(part)
    return tags


# ---- auth routes ----


@app.get("/login")
def login_page():
    return FileResponse(WEB_DIR / "login.html")


@app.post("/login")
def login_submit(username: str = Form(...), password: str = Form(...)):
    if not verify_credentials(username, password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_session_token(username)
    resp = JSONResponse({"status": "ok"})
    resp.set_cookie(
        SESSION_COOKIE_NAME, token, max_age=SESSION_MAX_AGE_SECONDS, httponly=True, samesite="lax"
    )
    return resp


@app.post("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE_NAME)
    return resp


# ---- document ingestion ----


def _ingest_bytes(
    filename: str,
    raw: bytes,
    vendor: Optional[str],
    doc_type: Optional[str],
    tags: list[str],
    ai_classify: bool = False,
) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        return {
            "filename": filename,
            "status": "error",
            "detail": f"File type '{ext or '(none)'}' is not allowed. Allowed: {allowed}",
        }
    if len(raw) > MAX_UPLOAD_SIZE_BYTES:
        return {
            "filename": filename,
            "status": "error",
            "detail": f"File is {len(raw) / 1024 / 1024:.1f}MB, exceeds the {MAX_UPLOAD_SIZE_BYTES / 1024 / 1024:.0f}MB limit",
        }

    text = extract_text(filename, raw)
    if not text.strip():
        return {"filename": filename, "status": "error", "detail": "No extractable text in document"}

    content_hash = core.hash_content(text)
    existing = db.get_document_by_name(filename)
    if existing and existing.content_hash == content_hash:
        return {"filename": filename, "status": "unchanged", "document": _serialize_doc(existing)}

    chunk_count = core.index_document(filename, text)

    dest = Path(UPLOAD_DIR) / filename
    dest.write_bytes(raw)

    ai_error = None
    if ai_classify:
        try:
            result = classify.classify_document(filename, text)
            vendor, doc_type, tags = result["vendor"], result["doc_type"], result["tags"]
        except classify.ClassifyError as e:
            # Fall back to whatever manual values were given (or defaults) rather than failing
            # the whole upload just because AI classification failed.
            ai_error = str(e)

    doc = db.upsert_document(filename, content_hash, chunk_count, vendor=vendor, doc_type=doc_type, tags=tags or None)
    result = {
        "filename": filename,
        "status": "replaced" if existing else "created",
        "document": _serialize_doc(doc),
    }
    if ai_error:
        result["ai_classify_error"] = ai_error
    return result


@app.post("/documents")
async def upload_document(
    file: UploadFile,
    vendor: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
    tags: list[str] = Form([]),
    ai_classify: bool = Form(False),
    _: str = AuthDep,
):
    raw = await file.read()
    result = _ingest_bytes(file.filename, raw, vendor, doc_type, _parse_tags(tags), ai_classify=ai_classify)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["detail"])
    return result


def _run_batch_ingest(
    files_data: list[tuple[str, bytes]],
    vendor: Optional[str],
    doc_type: Optional[str],
    tags: list[str],
    ai_classify: bool,
) -> None:
    _batch_status.update(running=True, total=len(files_data), done=0, current=None, results=[])
    try:
        for filename, raw in files_data:
            _batch_status["current"] = filename
            try:
                result = _ingest_bytes(filename, raw, vendor, doc_type, tags, ai_classify=ai_classify)
            except Exception as e:
                result = {"filename": filename, "status": "error", "detail": str(e)}
            _batch_status["results"].append(result)
            _batch_status["done"] += 1
    finally:
        _batch_status["running"] = False
        _batch_status["current"] = None


@app.post("/documents/batch")
async def upload_documents_batch(
    files: list[UploadFile],
    vendor: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
    tags: list[str] = Form([]),
    ai_classify: bool = Form(False),
    _: str = AuthDep,
):
    if _batch_status["running"]:
        raise HTTPException(status_code=409, detail="A batch upload is already running")
    if reindex_mod.get_status()["running"]:
        raise HTTPException(status_code=409, detail="A reindex is running — wait for it to finish first")

    files_data = [(f.filename, await f.read()) for f in files]
    parsed_tags = _parse_tags(tags)
    thread = threading.Thread(
        target=_run_batch_ingest, args=(files_data, vendor, doc_type, parsed_tags, ai_classify)
    )
    thread.start()
    return {"status": "started", "total": len(files_data)}


@app.get("/documents/batch/status")
def batch_status(_: str = AuthDep):
    return _batch_status


@app.get("/documents")
def get_documents(
    vendor: Optional[str] = None,
    doc_type: Optional[str] = None,
    tag: Optional[str] = None,
    q: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    _: str = AuthDep,
):
    docs = db.list_documents(vendor=vendor, doc_type=doc_type, tag=tag, q=q)
    total = len(docs)
    page = max(1, page)
    page_size = max(1, min(page_size, 200))
    start = (page - 1) * page_size
    items = docs[start : start + page_size]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, -(-total // page_size)),
        "items": [_serialize_doc(d) for d in items],
    }


@app.get("/documents/table")
def get_documents_table(
    request: Request,
    vendor: Optional[str] = None,
    doc_type: Optional[str] = None,
    tag: Optional[str] = None,
    q: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    _: str = AuthDep,
):
    """Server-rendered HTML fragment version of GET /documents, for the Documents tab's HTMX-driven
    table (see ardrag/web/templates/_documents_table.html and the doc-filter-form wiring in
    index.html). Same filtering/pagination semantics as the JSON endpoint above, which stays as-is
    for programmatic use.

    Unlike the JSON endpoint, this one is always called with every form field present (HTMX
    serializes the whole #doc-filter-form on each request, including untouched selects) — an
    unset filter arrives as an empty string, not an absent param, so it has to be normalized to
    None here or e.g. vendor="" would filter down to zero documents instead of meaning "any".
    """
    vendor, doc_type, tag, q = (v or None for v in (vendor, doc_type, tag, q))
    docs = db.list_documents(vendor=vendor, doc_type=doc_type, tag=tag, q=q)
    total = len(docs)
    page_size = max(1, min(page_size, 200))
    total_pages = max(1, -(-total // page_size))
    page = max(1, min(page, total_pages))
    start_idx = (page - 1) * page_size
    items = docs[start_idx : start_idx + page_size]
    return templates.TemplateResponse(
        request,
        "_documents_table.html",
        {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "start": 0 if total == 0 else start_idx + 1,
            "end": min(page * page_size, total),
        },
    )


@app.get("/folders")
def get_folders(_: str = AuthDep):
    return db.list_folders()


@app.get("/tags")
def get_tags(_: str = AuthDep):
    return db.list_all_tags()


@app.patch("/documents/{doc_id}")
def update_document(
    doc_id: int,
    vendor: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
    tags: list[str] = Form(None),
    _: str = AuthDep,
):
    parsed_tags = _parse_tags(tags) if tags is not None else None
    doc = db.update_document_metadata(doc_id, vendor=vendor, doc_type=doc_type, tags=parsed_tags)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "updated", "document": _serialize_doc(doc)}


@app.post("/documents/bulk-update")
def bulk_update_documents(
    doc_ids: list[int] = Form(...),
    vendor: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None),
    tags: list[str] = Form(None),
    _: str = AuthDep,
):
    parsed_tags = _parse_tags(tags) if tags is not None else None
    updated = []
    not_found = []
    for doc_id in doc_ids:
        doc = db.update_document_metadata(doc_id, vendor=vendor, doc_type=doc_type, tags=parsed_tags)
        if doc:
            updated.append(_serialize_doc(doc))
        else:
            not_found.append(doc_id)
    return {"updated_count": len(updated), "not_found": not_found, "documents": updated}


def _resolve_document_file(doc_id: int) -> tuple[db.Document, Path]:
    doc = db.get_document_by_id(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(UPLOAD_DIR) / doc.name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Original file is missing from storage")
    return doc, path


@app.get("/documents/{doc_id}/download")
def download_document(doc_id: int, _: str = AuthDep):
    doc, path = _resolve_document_file(doc_id)
    media_type = mimetypes.guess_type(doc.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=doc.name)


@app.get("/documents/{doc_id}/preview")
def preview_document(doc_id: int, _: str = AuthDep):
    doc, path = _resolve_document_file(doc_id)
    media_type = mimetypes.guess_type(doc.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, content_disposition_type="inline")


@app.post("/documents/{doc_id}/reembed")
def reembed_document(doc_id: int, chunk_size: int = Form(...), chunk_overlap: int = Form(...), _: str = AuthDep):
    if chunk_size < 100:
        raise HTTPException(status_code=400, detail="Chunk size must be at least 100 characters")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="Chunk overlap must be 0 or greater, and smaller than chunk size")
    if reindex_mod.get_status()["running"] or _batch_status["running"]:
        raise HTTPException(status_code=409, detail="Another indexing job is running — wait for it to finish first")

    doc, path = _resolve_document_file(doc_id)
    raw = path.read_bytes()
    text = extract_text(doc.name, raw)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text in this document")

    chunk_count = core.index_document(doc.name, text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    updated = db.update_document_chunk_config(doc_id, chunk_size, chunk_overlap, chunk_count)
    return {"status": "reembedded", "document": _serialize_doc(updated)}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, _: str = AuthDep):
    doc = db.delete_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    core.delete_document_chunks(doc.name)
    upload_path = Path(UPLOAD_DIR) / doc.name
    if upload_path.exists():
        upload_path.unlink()
    return {"status": "deleted", "document": _serialize_doc(doc)}


@app.get("/search")
def search_documents(
    q: str,
    top_k: int = None,
    vendor: Optional[str] = None,
    doc_type: Optional[str] = None,
    tag: Optional[str] = None,
    _: str = AuthDep,
):
    from ardrag.config import DEFAULT_TOP_K

    return core.search(q, top_k=top_k or DEFAULT_TOP_K, vendor=vendor, doc_type=doc_type, tag=tag)


def _compute_health() -> dict:
    docs = db.list_documents()
    total_chunks = sum(d.chunk_count for d in docs)
    upload_dir = Path(UPLOAD_DIR)
    disk_bytes = sum(f.stat().st_size for f in upload_dir.glob("**/*") if f.is_file())
    du = shutil.disk_usage(upload_dir)

    vendor_counts: dict[str, int] = {}
    for f in db.list_folders():
        vendor_counts[f["vendor"]] = vendor_counts.get(f["vendor"], 0) + f["count"]

    return {
        "uptime_seconds": round(time.monotonic() - _start_time),
        "document_count": len(docs),
        "chunk_count": total_chunks,
        "uploads_disk_usage_bytes": disk_bytes,
        "disk_total_bytes": du.total,
        "disk_used_bytes": du.used,
        "disk_free_bytes": du.free,
        "qdrant": core.get_qdrant_health(),
        "settings": _serialize_settings(db.get_settings()),
        "reindex": reindex_mod.get_status(),
        "batch_upload": _batch_status,
        "mcp_usage": db.get_mcp_usage_summary(),
        "classify_usage": db.get_deepseek_usage_summary(),
        "classify": classify_job.get_status(),
        "vendor_counts": sorted(vendor_counts.items(), key=lambda kv: -kv[1]),
        "recent_documents": docs[:6],
    }


@app.get("/health")
def health(_: str = AuthDep):
    return _compute_health()


def _fmt_bytes(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    units = ["KB", "MB", "GB", "TB"]
    i = -1
    while True:
        n /= 1024
        i += 1
        if n < 1024 or i == len(units) - 1:
            break
    return f"{n:.1f} {units[i]}"


def _fmt_uptime(seconds: int) -> str:
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def _bar_series(entries: dict, max_items: int = 10) -> list[dict]:
    items = list(entries.items())[:max_items]
    max_value = max((v for _, v in items), default=1) or 1
    return [{"label": k, "value": v, "pct": round(v / max_value * 100, 1)} for k, v in items]


def _hourly_series(hourly: list[dict]) -> list[dict]:
    from datetime import datetime

    max_value = max((h["count"] for h in hourly), default=1) or 1
    out = []
    for h in hourly:
        hour_dt = datetime.fromisoformat(h["hour"])
        pct = max(h["count"] / max_value * 100, 4 if h["count"] > 0 else 0)
        out.append({"label": hour_dt.strftime("%-I%p").lower(), "count": h["count"], "pct": round(pct, 1)})
    return out


@app.get("/monitoring/html")
def monitoring_html(request: Request, _: str = AuthDep):
    h = _compute_health()
    s = h["settings"]
    embedding_label = (
        f"{s['embedding_api_model'] or '(not set)'} (API, dim {s['embedding_api_dimension']})"
        if s["embedding_provider"] == "api"
        else f"{s['embedding_model']} (local)"
    )
    ai_model_label = s["deepseek_model"] if s["deepseek_api_key_set"] else None

    reindex = h["reindex"]
    batch = h["batch_upload"]
    mcp = h["mcp_usage"]
    ai = h["classify_usage"]

    return templates.TemplateResponse(
        request,
        "_monitoring.html",
        {
            "h": h,
            "s": s,
            "embedding_label": embedding_label,
            "ai_model_label": ai_model_label,
            "reindex": reindex,
            "batch": batch,
            "uptime_label": _fmt_uptime(h["uptime_seconds"]),
            "uploads_disk_label": _fmt_bytes(h["uploads_disk_usage_bytes"]),
            "disk_free_label": _fmt_bytes(h["disk_free_bytes"]),
            "mcp": mcp,
            "mcp_by_tool_series": _bar_series(mcp["by_tool"]),
            "mcp_hourly_series": _hourly_series(mcp["hourly"]),
            "ai": ai,
            "ai_hourly_series": _hourly_series(ai["hourly_tokens"]),
            "vendor_series": _bar_series(dict(h["vendor_counts"])),
            "recent_documents": h["recent_documents"],
        },
    )


def _serialize_settings(settings: db.Settings) -> dict:
    return {
        "id": settings.id,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_api_base_url": settings.embedding_api_base_url,
        "embedding_api_model": settings.embedding_api_model,
        "embedding_api_key_set": bool(settings.embedding_api_key),
        "embedding_api_dimension": settings.embedding_api_dimension,
        "max_embedding_tokens": settings.max_embedding_tokens,
        "embedding_api_batch_token_limit": settings.embedding_api_batch_token_limit,
        "embedding_api_batch_byte_limit": settings.embedding_api_batch_byte_limit,
        "classify_base_url": settings.classify_base_url,
        "deepseek_model": settings.deepseek_model,
        "deepseek_api_key_set": bool(settings.deepseek_api_key),
    }


def _embedding_identity(settings: db.Settings) -> tuple:
    """The effective embedding config — used to decide whether a settings save changed anything
    that would make existing vectors stale (and therefore require a reindex)."""
    if settings.embedding_provider == "api":
        return ("api", settings.embedding_api_base_url, settings.embedding_api_model, settings.embedding_api_dimension)
    return ("local", settings.embedding_model)


@app.get("/settings")
def get_settings(_: str = AuthDep):
    return {
        "current": _serialize_settings(db.get_settings()),
        "available_models": SUPPORTED_EMBEDDING_MODELS,
    }


@app.post("/settings")
def update_settings(
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    embedding_provider: Optional[str] = Form(None),
    embedding_model: Optional[str] = Form(None),
    embedding_api_base_url: Optional[str] = Form(None),
    embedding_api_key: Optional[str] = Form(None),
    embedding_api_model: Optional[str] = Form(None),
    max_embedding_tokens: Optional[int] = Form(None),
    embedding_api_batch_token_limit: Optional[int] = Form(None),
    embedding_api_batch_byte_limit: Optional[int] = Form(None),
    classify_base_url: Optional[str] = Form(None),
    deepseek_api_key: Optional[str] = Form(None),
    deepseek_model: Optional[str] = Form(None),
    _: str = AuthDep,
):
    old_settings = db.get_settings()

    if embedding_provider and embedding_provider not in ("local", "api"):
        raise HTTPException(status_code=400, detail="embedding_provider must be 'local' or 'api'")
    if embedding_model and embedding_model not in SUPPORTED_EMBEDDING_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {embedding_model}")

    # If the *effective* provider (new value, or the existing one if not being changed this call)
    # is "api", validate it with a live test call before persisting anything — measures the real
    # vector dimension (Qdrant needs an exact size up front) and catches a bad key/URL/model
    # immediately rather than silently breaking search/ingestion later. Only actually re-probe
    # when the embedding config itself changed (or has never been validated yet) — otherwise an
    # unrelated save (e.g. just the classification key) would needlessly cost an API call, and
    # would fail outright if the embedding provider happened to be down at that moment even
    # though nothing about it was being touched.
    effective_provider = embedding_provider or old_settings.embedding_provider
    embedding_api_dimension = None
    if effective_provider == "api":
        effective_base_url = embedding_api_base_url or old_settings.embedding_api_base_url
        effective_api_key = embedding_api_key or old_settings.embedding_api_key
        effective_model = embedding_api_model or old_settings.embedding_api_model
        if not (effective_base_url and effective_api_key and effective_model):
            raise HTTPException(
                status_code=400, detail="Base URL, API key, and model are all required for API embedding"
            )
        embedding_config_changed = (
            old_settings.embedding_provider != "api"
            or effective_base_url != old_settings.embedding_api_base_url
            or effective_api_key != old_settings.embedding_api_key
            or effective_model != old_settings.embedding_api_model
            or not old_settings.embedding_api_dimension
        )
        if embedding_config_changed:
            try:
                embedding_api_dimension = core.probe_api_embedding_dimension(
                    effective_base_url, effective_api_key, effective_model
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not validate API embedding config: {e}")

    settings = db.update_settings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_api_base_url=embedding_api_base_url,
        embedding_api_key=embedding_api_key or None,
        embedding_api_model=embedding_api_model,
        embedding_api_dimension=embedding_api_dimension,
        max_embedding_tokens=max_embedding_tokens,
        embedding_api_batch_token_limit=embedding_api_batch_token_limit,
        embedding_api_batch_byte_limit=embedding_api_batch_byte_limit,
        classify_base_url=classify_base_url,
        deepseek_api_key=deepseek_api_key or None,
        deepseek_model=deepseek_model,
    )

    reindex_required = _embedding_identity(old_settings) != _embedding_identity(settings)
    return {
        "settings": _serialize_settings(settings),
        "model_changed": reindex_required,
        "reindex_required": reindex_required,
    }


def _serialize_mcp_settings(settings: db.Settings, request: Request) -> dict:
    base = settings.mcp_public_url or str(request.base_url).rstrip("/")
    return {
        "mcp_sse_enabled": settings.mcp_sse_enabled,
        "mcp_streamable_enabled": settings.mcp_streamable_enabled,
        "mcp_anonymous_enabled": settings.mcp_anonymous_enabled,
        "mcp_oauth_enabled": settings.mcp_oauth_enabled,
        "mcp_public_url": settings.mcp_public_url,
        "sse_url": f"{base}/sse",
        "mcp_url": f"{base}/mcp",
        "mcp_users": [
            {"id": u.id, "username": u.username, "created_at": u.created_at.isoformat()}
            for u in db.mcp_user_list()
        ],
        "mcp_api_tokens": [
            {"id": t.id, "label": t.label, "created_at": t.created_at.isoformat()}
            for t in db.mcp_api_token_list()
        ],
    }


@app.get("/settings/mcp")
def get_mcp_settings(request: Request, _: str = AuthDep):
    return _serialize_mcp_settings(db.get_settings(), request)


@app.get("/settings/mcp/users/html")
def get_mcp_users_html(request: Request, _: str = AuthDep):
    return templates.TemplateResponse(request, "_mcp_users_table.html", {"users": db.mcp_user_list()})


@app.get("/settings/mcp/tokens/html")
def get_mcp_tokens_html(request: Request, _: str = AuthDep):
    return templates.TemplateResponse(request, "_mcp_tokens_table.html", {"tokens": db.mcp_api_token_list()})


@app.post("/settings/mcp")
def update_mcp_settings(
    request: Request,
    mcp_sse_enabled: bool = Form(False),
    mcp_streamable_enabled: bool = Form(False),
    mcp_anonymous_enabled: bool = Form(False),
    mcp_oauth_enabled: bool = Form(False),
    mcp_public_url: str = Form(""),
    _: str = AuthDep,
):
    if mcp_oauth_enabled and not mcp_public_url.strip():
        raise HTTPException(status_code=400, detail="Public URL is required when OAuth is enabled")
    if not mcp_anonymous_enabled and not mcp_oauth_enabled:
        raise HTTPException(
            status_code=400,
            detail="At least one access mode (Anonymous or OAuth) must stay enabled, or no client can connect",
        )
    settings = db.update_settings(
        mcp_sse_enabled=mcp_sse_enabled,
        mcp_streamable_enabled=mcp_streamable_enabled,
        mcp_anonymous_enabled=mcp_anonymous_enabled,
        mcp_oauth_enabled=mcp_oauth_enabled,
        mcp_public_url=mcp_public_url.strip(),
    )
    mcp_supervisor.restart()
    return _serialize_mcp_settings(settings, request)


@app.post("/settings/mcp/users")
def add_mcp_user(username: str = Form(...), password: str = Form(...), _: str = AuthDep):
    username = username.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    try:
        user = db.mcp_user_create(username, password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": user.id, "username": user.username, "created_at": user.created_at.isoformat()}


@app.delete("/settings/mcp/users/{user_id}")
def remove_mcp_user(user_id: int, _: str = AuthDep):
    user = db.mcp_user_delete(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "deleted", "id": user_id}


@app.post("/settings/mcp/tokens")
def add_mcp_api_token(label: str = Form(...), _: str = AuthDep):
    label = label.strip()
    if not label:
        raise HTTPException(status_code=400, detail="Label is required")
    token = db.mcp_api_token_create(label)
    # Returned only here, right after creation — never included in GET /settings/mcp or any
    # other listing, same "reveal once" pattern as a GitHub personal access token.
    return {"id": token.id, "label": token.label, "created_at": token.created_at.isoformat(), "token": token.token}


@app.delete("/settings/mcp/tokens/{token_id}")
def remove_mcp_api_token(token_id: int, _: str = AuthDep):
    token = db.mcp_api_token_delete(token_id)
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    return {"status": "deleted", "id": token_id}


@app.post("/classify/backfill")
def trigger_classify_backfill(scope: str = Form("all"), _: str = AuthDep):
    if not db.get_settings().deepseek_api_key:
        raise HTTPException(status_code=400, detail="Set a DeepSeek API key in Settings first")
    if classify_job.get_status()["running"]:
        raise HTTPException(status_code=409, detail="AI classification already running")
    if reindex_mod.get_status()["running"] or _batch_status["running"]:
        raise HTTPException(status_code=409, detail="Another CPU-heavy job is running — wait for it to finish first")
    thread = threading.Thread(target=classify_job.run_classification, kwargs={"scope": scope})
    thread.start()
    return {"status": "started"}


@app.get("/classify/status")
def classify_status(_: str = AuthDep):
    return classify_job.get_status()


@app.post("/reindex")
def trigger_reindex(recreate_collection: bool = Form(False), _: str = AuthDep):
    if reindex_mod.get_status()["running"]:
        raise HTTPException(status_code=409, detail="Reindex already running")
    if _batch_status["running"]:
        raise HTTPException(status_code=409, detail="A batch upload is running — wait for it to finish first")
    thread = threading.Thread(target=reindex_mod.reindex_all, kwargs={"recreate_collection": recreate_collection})
    thread.start()
    return {"status": "started"}


@app.get("/reindex/status")
def reindex_status(_: str = AuthDep):
    return reindex_mod.get_status()


# ---- AI Chat ----


def _serialize_session(s: db.ChatSession) -> dict:
    return {"id": s.id, "title": s.title, "created_at": s.created_at.isoformat(), "updated_at": s.updated_at.isoformat()}


def _serialize_message(m: db.ChatMessage) -> dict:
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "sources": m.sources,
        "created_at": m.created_at.isoformat(),
        "is_error": False,
    }


@app.get("/chat/sessions")
def list_chat_sessions(_: str = AuthDep):
    return [_serialize_session(s) for s in db.chat_session_list()]


@app.get("/chat/sessions/html")
def list_chat_sessions_html(request: Request, active_id: Optional[int] = None, _: str = AuthDep):
    return templates.TemplateResponse(
        request, "_chat_history.html", {"sessions": db.chat_session_list(), "active_id": active_id}
    )


@app.post("/chat/sessions")
def create_chat_session(_: str = AuthDep):
    return _serialize_session(db.chat_session_create())


@app.get("/chat/sessions/{session_id}/messages")
def get_chat_messages(session_id: int, _: str = AuthDep):
    if not db.chat_session_get(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")
    return [_serialize_message(m) for m in db.chat_message_list(session_id)]


@app.get("/chat/sessions/{session_id}/messages/html")
def get_chat_messages_html(request: Request, session_id: int, _: str = AuthDep):
    if not db.chat_session_get(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")
    messages = [_serialize_message(m) for m in db.chat_message_list(session_id)]
    return templates.TemplateResponse(request, "_chat_messages.html", {"messages": messages})


@app.delete("/chat/sessions/{session_id}")
def delete_chat_session(session_id: int, _: str = AuthDep):
    session = db.chat_session_delete(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"status": "deleted", "id": session_id}


TITLE_MAX_CHARS = 60


def _run_chat_turn(session: db.ChatSession, message: str) -> dict:
    """Persists the user message, runs the RAG answer, and persists+returns the assistant
    message dict — or an error-shaped dict (is_error=True) if the chat call failed, without
    persisting an assistant row for it (a failed turn doesn't leave a phantom assistant message
    in history, matching the original JSON-only endpoint's behavior)."""
    prior = db.chat_message_list(session.id)
    history = [{"role": m.role, "content": m.content} for m in prior]
    db.chat_message_add(session.id, "user", message)

    try:
        result = chat_mod.answer(message, history)
    except chat_mod.ChatError as e:
        return {"role": "assistant", "content": str(e), "sources": [], "is_error": True}

    assistant_msg = db.chat_message_add(session.id, "assistant", result["content"], sources=result["sources"])

    title = session.title
    if title == "New chat":
        title = message[:TITLE_MAX_CHARS] + ("…" if len(message) > TITLE_MAX_CHARS else "")
    db.chat_session_touch(session.id, title=title)

    return _serialize_message(assistant_msg)


@app.post("/chat/sessions/{session_id}/messages")
def post_chat_message(session_id: int, message: str = Form(...), _: str = AuthDep):
    session = db.chat_session_get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    message = message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    result = _run_chat_turn(session, message)
    if result.get("is_error"):
        raise HTTPException(status_code=400, detail=result["content"])
    return result


@app.post("/chat/sessions/{session_id}/messages/html")
def post_chat_message_html(request: Request, session_id: int, message: str = Form(...), _: str = AuthDep):
    session = db.chat_session_get(session_id)
    if not session:
        m = {"role": "assistant", "content": "Chat session not found — start a new chat.", "sources": [], "is_error": True}
        return templates.TemplateResponse(request, "_chat_message_single.html", {"m": m})
    message = message.strip()
    if not message:
        return HTMLResponse("")
    result = _run_chat_turn(session, message)
    return templates.TemplateResponse(request, "_chat_message_single.html", {"m": result})


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def index(request: Request):
        if not get_current_user(request):
            return RedirectResponse("/login")
        return FileResponse(WEB_DIR / "index.html")
