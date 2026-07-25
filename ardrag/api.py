import mimetypes
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

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

AuthDep = Depends(require_auth)

_batch_status = {"running": False, "total": 0, "done": 0, "current": None, "results": []}


def _serialize_doc(doc: db.Document) -> dict:
    return {
        "id": doc.id,
        "name": doc.name,
        "content_hash": doc.content_hash,
        "chunk_count": doc.chunk_count,
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


@app.get("/health")
def health(_: str = AuthDep):
    docs = db.list_documents()
    total_chunks = sum(d.chunk_count for d in docs)
    upload_dir = Path(UPLOAD_DIR)
    disk_bytes = sum(f.stat().st_size for f in upload_dir.glob("**/*") if f.is_file())
    du = shutil.disk_usage(upload_dir)
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
        "deepseek_usage": db.get_deepseek_usage_summary(),
        "classify": classify_job.get_status(),
    }


def _serialize_settings(settings: db.Settings) -> dict:
    return {
        "id": settings.id,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "deepseek_model": settings.deepseek_model,
        "deepseek_api_key_set": bool(settings.deepseek_api_key),
    }


@app.get("/settings")
def get_settings(_: str = AuthDep):
    return {
        "current": _serialize_settings(db.get_settings()),
        "available_models": SUPPORTED_EMBEDDING_MODELS,
    }


@app.post("/settings")
def update_settings(
    embedding_model: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    deepseek_api_key: Optional[str] = Form(None),
    deepseek_model: Optional[str] = Form(None),
    _: str = AuthDep,
):
    if embedding_model and embedding_model not in SUPPORTED_EMBEDDING_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {embedding_model}")
    model_changed = embedding_model and embedding_model != db.get_settings().embedding_model
    settings = db.update_settings(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        deepseek_api_key=deepseek_api_key or None,
        deepseek_model=deepseek_model,
    )
    return {
        "settings": _serialize_settings(settings),
        "model_changed": bool(model_changed),
        "reindex_required": bool(model_changed),
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
    }


@app.get("/settings/mcp")
def get_mcp_settings(request: Request, _: str = AuthDep):
    return _serialize_mcp_settings(db.get_settings(), request)


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


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def index(request: Request):
        if not get_current_user(request):
            return RedirectResponse("/login")
        return FileResponse(WEB_DIR / "index.html")
