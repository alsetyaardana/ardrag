from pathlib import Path

from fastmcp import FastMCP

from ardrag import core, db
from ardrag.config import DEFAULT_TOP_K, MCP_HOST, MCP_PORT, UPLOAD_DIR
from ardrag.extract import extract_text

db.init_db()
core.ensure_collection()

mcp = FastMCP("Ardrag")

DEFAULT_DOCUMENT_MAX_CHARS = 20000


@mcp.tool
def rag_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    vendor: str = None,
    doc_type: str = None,
    tag: str = None,
) -> list[dict]:
    """Semantic search over indexed document chunks. Optionally narrow the search to a specific
    vendor, doc_type folder, and/or tag first — useful once you already know roughly which
    product/category you're after, to avoid near-duplicate documents crowding out the right one."""
    db.log_mcp_call("rag_search")
    return core.search(query, top_k=top_k, vendor=vendor, doc_type=doc_type, tag=tag)


@mcp.tool
def rag_get_document(doc_name: str, max_chars: int = DEFAULT_DOCUMENT_MAX_CHARS) -> dict:
    """Read the full original text of a document by its exact name (as returned by rag_search's
    doc_name field or rag_list_documents' name field) — not just a retrieved chunk. Use this when
    a search result chunk seems incomplete or you need exact figures (part numbers, port counts,
    specs) that might have been split across chunk boundaries. Content is truncated to max_chars
    (default 20000); check the 'truncated' field in the response."""
    db.log_mcp_call("rag_get_document")
    doc = db.get_document_by_name(doc_name)
    if not doc:
        return {"error": f"No document found with name '{doc_name}'"}
    path = Path(UPLOAD_DIR) / doc.name
    if not path.exists():
        return {"error": f"Original file for '{doc_name}' is missing from storage"}
    text = extract_text(doc.name, path.read_bytes())
    truncated = len(text) > max_chars
    return {
        "doc_name": doc.name,
        "vendor": doc.vendor,
        "doc_type": doc.doc_type,
        "tags": doc.tags,
        "text": text[:max_chars],
        "truncated": truncated,
        "total_chars": len(text),
    }


@mcp.tool
def rag_list_documents(vendor: str = None, doc_type: str = None, tag: str = None, q: str = None) -> list[dict]:
    """List documents in the RAG store, optionally filtered by vendor, doc_type folder, tag,
    and/or a substring match on filename (q)."""
    db.log_mcp_call("rag_list_documents")
    return [
        {
            "id": d.id,
            "name": d.name,
            "vendor": d.vendor,
            "doc_type": d.doc_type,
            "tags": d.tags,
            "chunk_count": d.chunk_count,
            "uploaded_at": d.uploaded_at.isoformat(),
        }
        for d in db.list_documents(vendor=vendor, doc_type=doc_type, tag=tag, q=q)
    ]


@mcp.tool
def rag_list_folders() -> list[dict]:
    """List the vendor/doc_type folder structure with document counts."""
    db.log_mcp_call("rag_list_folders")
    return db.list_folders()


@mcp.tool
def rag_list_tags() -> list[str]:
    """List every tag currently in use across all documents."""
    db.log_mcp_call("rag_list_tags")
    return db.list_all_tags()


def _save_note(title: str, content: str) -> dict:
    doc_name = f"note:{title}"
    content_hash = core.hash_content(content)
    chunk_count = core.index_document(doc_name, content)
    doc = db.upsert_document(doc_name, content_hash, chunk_count, vendor="Notes", doc_type="General")
    return {"id": doc.id, "name": doc.name, "chunk_count": doc.chunk_count}


@mcp.tool
def rag_add_note(title: str, content: str) -> dict:
    """Add a freeform note to the RAG store, indexed under the given title as its document name."""
    db.log_mcp_call("rag_add_note")
    return {"status": "saved", "document": _save_note(title, content)}


@mcp.tool
def rag_add_notes(notes: list[dict]) -> dict:
    """Add multiple freeform notes at once. Each item must have 'title' and 'content' keys."""
    db.log_mcp_call("rag_add_notes")
    saved = [_save_note(n["title"], n["content"]) for n in notes]
    return {"status": "saved", "count": len(saved), "documents": saved}


if __name__ == "__main__":
    mcp.run(transport="sse", host=MCP_HOST, port=MCP_PORT)
