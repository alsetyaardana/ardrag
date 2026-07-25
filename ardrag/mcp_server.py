from pathlib import Path

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from ardrag import core, db
from ardrag.auth import verify_credentials
from ardrag.config import DEFAULT_TOP_K, MCP_HOST, MCP_OAUTH_ENABLED, MCP_PORT, MCP_PUBLIC_URL, UPLOAD_DIR
from ardrag.extract import extract_text

db.init_db()
core.ensure_collection()

DEFAULT_DOCUMENT_MAX_CHARS = 20000

_auth_provider = None
if MCP_OAUTH_ENABLED:
    if not MCP_PUBLIC_URL:
        raise RuntimeError("MCP_OAUTH_ENABLED is set but MCP_PUBLIC_URL is empty — set it to the public HTTPS URL of this MCP server.")
    from mcp.server.auth.settings import ClientRegistrationOptions

    from ardrag.oauth_provider import ArdragOAuthProvider

    _auth_provider = ArdragOAuthProvider(
        base_url=MCP_PUBLIC_URL,
        client_registration_options=ClientRegistrationOptions(enabled=True),
    )

mcp = FastMCP("Ardrag", auth=_auth_provider)


_LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>ArdRAG — Authorize</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif; margin: 0; min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    background: #f4f5fb; color: #1a1a1a;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #0f0f12; color: #e8e8e8; }}
    .card {{ background: #1c1c1f !important; border-color: #2a2a2e !important; }}
    input {{ background: #26262a !important; color: #e8e8e8 !important; border-color: #3a3a3e !important; }}
  }}
  .card {{
    width: 360px; background: #ffffff; border: 1px solid #ececf4; border-radius: 16px;
    padding: 32px; box-shadow: 0 10px 30px rgba(0,0,0,0.08);
  }}
  h1 {{ font-size: 1.2rem; margin: 0 0 4px; text-align: center; }}
  .subtitle {{ font-size: 0.82rem; color: #888; margin: 0 0 20px; text-align: center; }}
  label {{ display: block; font-size: 0.82rem; margin-bottom: 6px; margin-top: 14px; }}
  input {{
    width: 100%; padding: 9px 12px; border: 1px solid #dcdcec; border-radius: 8px;
    font-size: 0.9rem; background: #f7f8fd; box-sizing: border-box;
  }}
  button {{
    width: 100%; margin-top: 22px; padding: 10px; border: none; border-radius: 8px;
    background: #2563eb; color: white; font-weight: 600; font-size: 0.92rem; cursor: pointer;
  }}
  .error {{
    margin-top: 14px; padding: 8px 10px; border-radius: 8px; font-size: 0.82rem;
    background: #fee2e2; color: #b91c1c;
  }}
</style></head>
<body>
  <div class="card">
    <h1>Authorize access to ArdRAG</h1>
    <p class="subtitle">{client_name} is requesting access to your documents.</p>
    {error_html}
    <form method="post">
      <input type="hidden" name="client_id" value="{client_id}" />
      <input type="hidden" name="redirect_uri" value="{redirect_uri}" />
      <input type="hidden" name="redirect_uri_provided_explicitly" value="{redirect_uri_provided_explicitly}" />
      <input type="hidden" name="state" value="{state}" />
      <input type="hidden" name="code_challenge" value="{code_challenge}" />
      <input type="hidden" name="scope" value="{scope}" />
      <input type="hidden" name="resource" value="{resource}" />
      <label for="username">Username</label>
      <input type="text" id="username" name="username" autocomplete="username" required autofocus />
      <label for="password">Password</label>
      <input type="password" id="password" name="password" autocomplete="current-password" required />
      <button type="submit">Authorize</button>
    </form>
  </div>
</body></html>
"""


def _oauth_login_html(params: dict, error: str = "") -> str:
    return _LOGIN_PAGE.format(
        client_name=params.get("client_id", "An application"),
        client_id=params.get("client_id", ""),
        redirect_uri=params.get("redirect_uri", ""),
        redirect_uri_provided_explicitly=params.get("redirect_uri_provided_explicitly", "True"),
        state=params.get("state", ""),
        code_challenge=params.get("code_challenge", ""),
        scope=params.get("scope", ""),
        resource=params.get("resource", ""),
        error_html=f'<div class="error">{error}</div>' if error else "",
    )


if MCP_OAUTH_ENABLED:

    @mcp.custom_route("/oauth-login", methods=["GET", "POST"])
    async def oauth_login(request: Request):
        from ardrag.oauth_provider import issue_code_and_redirect

        if request.method == "GET":
            return HTMLResponse(_oauth_login_html(dict(request.query_params)))

        form = await request.form()
        params = {
            "client_id": form.get("client_id", ""),
            "redirect_uri": form.get("redirect_uri", ""),
            "redirect_uri_provided_explicitly": form.get("redirect_uri_provided_explicitly", "True"),
            "state": form.get("state", ""),
            "code_challenge": form.get("code_challenge", ""),
            "scope": form.get("scope", ""),
            "resource": form.get("resource", ""),
        }
        username = form.get("username", "")
        password = form.get("password", "")

        if not verify_credentials(username, password):
            return HTMLResponse(_oauth_login_html(params, error="Invalid username or password."))

        try:
            redirect_url = issue_code_and_redirect(
                client_id=params["client_id"],
                redirect_uri=params["redirect_uri"],
                redirect_uri_provided_explicitly=params["redirect_uri_provided_explicitly"] == "True",
                state=params["state"] or None,
                code_challenge=params["code_challenge"],
                scope=params["scope"],
                subject=username,
                resource=params["resource"] or None,
            )
        except ValueError as e:
            return HTMLResponse(_oauth_login_html(params, error=str(e)), status_code=400)

        return RedirectResponse(url=redirect_url, status_code=302)


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


def _get_document_full(doc_name: str, max_chars: int) -> dict:
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
def rag_get_document(doc_name: str, max_chars: int = DEFAULT_DOCUMENT_MAX_CHARS) -> dict:
    """Read the full original text of a document by its exact name (as returned by rag_search's
    doc_name field or rag_list_documents' name field) — not just a retrieved chunk. Use this when
    a search result chunk seems incomplete or you need exact figures (part numbers, port counts,
    specs) that might have been split across chunk boundaries. Content is truncated to max_chars
    (default 20000); check the 'truncated' field in the response."""
    db.log_mcp_call("rag_get_document")
    return _get_document_full(doc_name, max_chars)


@mcp.tool
def rag_compare_documents(doc_names: list[str], max_chars_per_doc: int = 8000) -> dict:
    """Fetch the full text of 2+ documents in one call, labeled and ready for you (the calling
    model) to draft a side-by-side spec comparison table from — e.g. for presales proposals
    comparing similar products across a product line. Ardrag doesn't build the table itself; it
    just bundles the source material so you don't need N separate rag_get_document calls.
    max_chars_per_doc caps each document's length (lower it if comparing many/long documents to
    stay within your own context budget) — check each entry's 'truncated' field."""
    db.log_mcp_call("rag_compare_documents")
    if len(doc_names) < 2:
        return {"error": "Provide at least 2 doc_names to compare."}
    return {"documents": [_get_document_full(name, max_chars_per_doc) for name in doc_names]}


@mcp.tool
def rag_suggest_bom(items: list[str], top_k_per_item: int = 3) -> dict:
    """Given a list of requirement line-items (e.g. ["access switch 24-port", "core switch",
    "firewall"]), run a search for each and return the top matching documents per item — a
    starting point for a quote/BOM draft. Ardrag only suggests candidate documents per category;
    parsing the user's freeform request into line-items and picking final quantities/models is
    up to you (the calling model). Pair with a BOM-generation skill/tool to turn the picks into
    a final document."""
    db.log_mcp_call("rag_suggest_bom")
    return {
        "results": [
            {
                "item": item,
                "candidates": core.search(item, top_k=top_k_per_item),
            }
            for item in items
        ]
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


def _build_combined_app():
    """Serve both MCP transports on one port so we don't break existing clients.

    Claude Code connects via legacy SSE at /sse (GET stream + POST /messages/?session_id=...).
    Claude Desktop (and other modern clients) use Streamable HTTP, which POSTs JSON-RPC directly
    to a single endpoint — our /sse route only ever accepted GET, so those POSTs were bouncing
    with 405 before Claude Desktop got anywhere near the OAuth flow. Since FastMCP's `run()` only
    stands up one transport at a time, both `http_app()` variants are built separately (each is a
    fully self-contained Starlette app with its own copy of the OAuth routes bound to the same
    `mcp.auth` provider) and merged into a single app: MCP endpoint routes from both, OAuth/
    well-known routes deduplicated by path, and both apps' lifespans (which start/stop each
    transport's session manager) run together via AsyncExitStack.
    """
    import contextlib

    from starlette.applications import Starlette
    from starlette.routing import Route

    sse_app = mcp.http_app(path="/sse", transport="sse")
    streamable_app = mcp.http_app(path="/mcp", transport="streamable-http")

    seen_paths = set()
    combined_routes = []
    for app in (sse_app, streamable_app):
        for route in app.routes:
            path = getattr(route, "path", None)
            if path is not None and path in seen_paths:
                continue
            if path is not None:
                seen_paths.add(path)
            combined_routes.append(route)

    @contextlib.asynccontextmanager
    async def combined_lifespan(app):
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(sse_app.router.lifespan_context(sse_app))
            await stack.enter_async_context(streamable_app.router.lifespan_context(streamable_app))
            yield

    return Starlette(routes=combined_routes, lifespan=combined_lifespan)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(_build_combined_app(), host=MCP_HOST, port=MCP_PORT)
