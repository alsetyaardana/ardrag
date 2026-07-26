"""Re-index all documents already on disk (UPLOAD_DIR) using the current settings
(embedding model / chunk size / overlap). Used both as a CLI (`python -m ardrag.reindex`)
and by the API's POST /reindex endpoint.
"""
from pathlib import Path

from ardrag import core, db
from ardrag.config import UPLOAD_DIR
from ardrag.extract import extract_text

_status = {"running": False, "total": 0, "done": 0, "current": None, "errors": []}


def get_status() -> dict:
    return dict(_status)


def reindex_all(recreate_collection: bool = False) -> dict:
    if _status["running"]:
        return {"status": "already_running"}

    settings = db.get_settings()
    _status.update(running=True, total=0, done=0, current=None, errors=[])
    try:
        if recreate_collection:
            core.recreate_collection_for_model(settings)
        else:
            core.ensure_collection()

        upload_dir = Path(UPLOAD_DIR)
        files = sorted(p for p in upload_dir.iterdir() if p.is_file())
        _status["total"] = len(files)

        for path in files:
            _status["current"] = path.name
            raw = path.read_bytes()
            text = extract_text(path.name, raw)
            if not text.strip():
                _status["errors"].append({"filename": path.name, "detail": "No extractable text"})
                _status["done"] += 1
                continue
            content_hash = core.hash_content(text)
            chunk_count = core.index_document(path.name, text)
            existing = db.get_document_by_name(path.name)
            db.upsert_document(
                path.name,
                content_hash,
                chunk_count,
                vendor=existing.vendor if existing else None,
                doc_type=existing.doc_type if existing else None,
                tags=existing.tags if existing else None,
            )
            _status["done"] += 1
        return {"status": "completed", "total": _status["total"], "errors": _status["errors"]}
    finally:
        _status["running"] = False
        _status["current"] = None


def main() -> None:
    db.init_db()
    result = reindex_all()
    print(result)


if __name__ == "__main__":
    main()
