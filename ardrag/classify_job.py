"""Background job: run AI classification (DeepSeek) over documents already stored on disk,
updating their vendor/doc_type/tags. Triggered via POST /classify/backfill.
"""
from pathlib import Path

from ardrag import classify, db
from ardrag.config import UPLOAD_DIR
from ardrag.extract import extract_text

_status = {"running": False, "total": 0, "done": 0, "current": None, "errors": []}


def get_status() -> dict:
    return dict(_status)


def run_classification(scope: str = "all") -> dict:
    if _status["running"]:
        return {"status": "already_running"}

    docs = db.list_documents()
    if scope == "uncategorized":
        docs = [d for d in docs if d.vendor == db.UNCATEGORIZED_VENDOR or d.doc_type == db.UNCATEGORIZED_TYPE]

    _status.update(running=True, total=len(docs), done=0, current=None, errors=[])
    try:
        for doc in docs:
            _status["current"] = doc.name
            try:
                path = Path(UPLOAD_DIR) / doc.name
                if not path.exists():
                    raise FileNotFoundError(f"Source file not found on disk: {doc.name}")
                text = extract_text(doc.name, path.read_bytes())
                result = classify.classify_document(doc.name, text)
                db.update_document_metadata(
                    doc.id, vendor=result["vendor"], doc_type=result["doc_type"], tags=result["tags"]
                )
            except Exception as e:
                _status["errors"].append({"filename": doc.name, "detail": str(e)})
            _status["done"] += 1
        return {"status": "completed", "total": _status["total"], "errors": _status["errors"]}
    finally:
        _status["running"] = False
        _status["current"] = None
