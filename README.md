# Ardrag

Personal RAG (Retrieval-Augmented Generation) exposed as an MCP server, so any MCP-compatible
AI client (Claude, local models, etc.) can search your personal documents. The generative LLM
is always called by the *client*, not by this server — Ardrag only handles ingestion, embedding,
storage, and retrieval.

## Components

- **Qdrant** — vector database (Docker container). Collection uses **hybrid search**: a named
  dense vector (`BAAI/bge-small-en-v1.5`, semantic) and a named sparse vector (`Qdrant/bm25`,
  exact-term/keyword) per chunk, combined at query time with Reciprocal Rank Fusion. This matters
  for collections of near-identical product datasheets — dense embeddings alone can blur similar
  model codes together (e.g. `IS230-24T` vs `IS230-48T`); BM25 catches the exact-term match dense
  search misses.
- **FastEmbed** — both embedding models run locally on CPU, no external API calls for retrieval.
  Each chunk is embedded together with its source document name (`doc_name + chunk text`) so
  near-identical product datasheets stay distinguishable in vector space. `EMBEDDING_THREADS`
  caps onnxruntime's thread pool (default: all cores minus one) so large batch uploads/reindexes
  can't starve the rest of the container — or the host, on a small VPS.
- **FastAPI backend** — REST API + serves the web UI, at port `8000`. Protected by a custom
  session-cookie login (`/login` page, `ADMIN_USER` / `ADMIN_PASSWORD` in `.env`) — not the
  browser's native Basic Auth popup. The MCP server (port 8001) is a separate process and is
  intentionally **not** covered by this auth.
- **Web UI** (`/`) — file-manager style: upload (single/multiple/AI-classified files, with a live
  progress bar), organize into Vendor / Doc Type folders plus freeform multi-tags, search/filter by
  name/vendor/type/tag with pagination, bulk-edit metadata across a selection, preview/download
  original files, delete documents. Also has:
  - **Dashboard** — system health (doc/chunk counts, Qdrant status incl. hybrid flag, disk usage,
    uptime), reindex/batch-upload progress, **MCP tool usage** (calls by tool, hourly chart) and
    **DeepSeek token usage** (prompt/completion/total tokens, hourly chart) — auto-refreshes every
    5s while this tab is open.
  - **Settings** — embedding model, chunk size/overlap, trigger reindex; DeepSeek API key +
    model, trigger AI-classification backfill over existing documents.
- **DeepSeek (optional)** — if configured with an API key, documents can be auto-classified
  (vendor/doc_type/tags inferred from content) on upload or via a backfill job over existing
  documents. Every call's token usage is logged and shown on the Dashboard. Ardrag works fully
  without this — it's an optional convenience, not required for RAG itself.
- **MCP server** (FastMCP, SSE transport) — at port `8001`, exposes:
  - `rag_search(query, top_k, vendor, doc_type, tag)` — hybrid (dense+sparse) search, optionally
    narrowed to a vendor/doc_type folder and/or tag first (resolved against SQLite, then
    post-filtered against the over-fetched Qdrant results — no reindex needed when metadata changes)
  - `rag_get_document(doc_name, max_chars)` — read a document's full original text (not just a
    chunk), for when a search hit needs verifying against the complete source, e.g. exact port
    counts/specs that might straddle a chunk boundary. Truncates at `max_chars` (default 20000)
    and reports whether it did via the `truncated` field.
  - `rag_compare_documents(doc_names, max_chars_per_doc)` — fetches full text of 2+ documents in
    one call, labeled and ready for the calling model to draft a side-by-side spec comparison
    table from (e.g. presales comparing similar products across a product line). Ardrag bundles
    the source material; the calling model builds the table.
  - `rag_suggest_bom(items, top_k_per_item)` — given a list of requirement line-items (e.g.
    `["access switch 24-port", "core switch", "firewall"]`), searches each and returns top
    candidate documents per item — a starting point for a quote/BOM draft. Pair with a
    BOM-generation skill/tool downstream to turn the picks into a final document.
  - `rag_list_documents(vendor, doc_type, tag, q)` — `q` matches a filename substring
  - `rag_list_folders()`
  - `rag_list_tags()`
  - `rag_add_note(title, content)`
  - `rag_add_notes(notes)` — batch variant, `notes` is a list of `{title, content}` objects
  - Every call is logged (tool name + timestamp) and visible on the Dashboard.

### Authentication

Session-based login, not HTTP Basic Auth. `GET /login` serves a custom login page; `POST /login`
verifies `username`/`password` against `ADMIN_USER`/`ADMIN_PASSWORD` and, on success, sets a
signed `ardrag_session` cookie (`httponly`, 7-day expiry by default — see `SESSION_MAX_AGE_SECONDS`).
`POST /logout` clears it. Every other route (except `/login` itself) requires a valid session —
API endpoints reply `401 {"detail": "Not authenticated"}`, and the web UI's `fetch` wrapper
redirects to `/login` automatically on a 401.

Set `SESSION_SECRET` in `.env` to a random 64-char hex string (e.g.
`python3 -c "import secrets; print(secrets.token_hex(32))"`) so sessions survive container
restarts — without it, a random secret is generated at startup and everyone gets logged out on
every restart. **The MCP server (port 8001) is unauthenticated by design** — it's a separate
process, not covered by this login.

### Folders (Vendor / Doc Type) & tags

Every document has a `vendor` and `doc_type` (both optional, default to `Uncategorized`/`General`)
used for the folder-tree sidebar — plus an independent, freeform list of `tags` (multi-value,
not hierarchical) for cross-cutting metadata that doesn't fit the vendor/type folder structure.

Set them via `vendor`/`doc_type`/`tags` form fields on `POST /documents` or `/documents/batch`
(repeat the `tags` field per tag, or comma-separate within one value — both are accepted), or via
the web UI's Vendor/Doc Type inputs and the tag chip input (type + Enter/comma to add, click ×
to remove). `GET /folders` returns vendor/doc_type combinations with counts for the sidebar tree;
`GET /tags` returns all known tags (for autocomplete); `GET /documents?tag=<value>` filters by tag.

### Settings & reindexing

`GET/POST /settings` controls the active embedding model (`ardrag/config.py:SUPPORTED_EMBEDDING_MODELS`)
and chunk size/overlap, persisted in SQLite. Changing the embedding model changes the vector
dimension, so it requires recreating the Qdrant collection — the UI/API flags `reindex_required`
when that happens. Trigger a reindex with `POST /reindex` (`recreate_collection=true` to also
drop/recreate the collection first) — it runs in a background thread; poll `GET /reindex/status`
for progress. This replaces the old `docker compose exec ... python -m ardrag.reindex` CLI flow
(that still works too, as a fallback with no auth/UI needed).

### Preview & download

`GET /documents/{id}/preview` serves the original stored file inline (browsers render PDFs
natively in a new tab). `GET /documents/{id}/download` serves it as an attachment with the
original filename. Both require an active session, same as everything else in the web UI — handy
for grabbing a document from your phone/another machine without needing the RAG search at all.

### Upload restrictions

Only `.pdf, .txt, .md, .csv, .json, .log` are accepted (`ALLOWED_UPLOAD_EXTENSIONS` in `.env` to
change), and files over 50MB are rejected (`MAX_UPLOAD_SIZE_MB`). `extract_text()` only knows how
to handle these — anything else (`.pptx`, `.docx`, `.xlsx`, images, archives, ...) is binary and
would otherwise get blindly decoded as text, producing huge garbage content that still gets
chunked/embedded, wasting resources and — for large binary files — capable of spiking memory/CPU
hard enough to make the whole host crawl. Both checks happen before any extraction/embedding work
starts, so a rejected file costs nothing.

### Batch upload

`POST /documents/batch` accepts multiple files in one request (`files` form field, repeated) and
processes them independently — one failing file (e.g. empty or unreadable) does not block the
rest. Response includes per-file status plus `succeeded`/`failed` counts. The web UI's file input
supports selecting multiple files and uses this endpoint automatically.

Chunking uses `CHUNK_SIZE=1800` / `CHUNK_OVERLAP=300` characters (see `ardrag/core.py`), sized to
keep spec tables in datasheet-style PDFs mostly intact in one chunk. Default `top_k` for search is
`8`. If you ever change these values, re-index existing documents with:

```bash
docker compose exec ardrag-backend python -m ardrag.reindex
```

This re-reads every file already stored in the upload volume and re-embeds it — no need to
re-upload manually.

Uploading a file with the same name as an existing document replaces it: old chunks are
deleted from Qdrant and the file is re-indexed.

## Local development

```bash
pip install -e .
cp .env.example .env
# run qdrant separately, e.g.:
docker run -p 6333:6333 qdrant/qdrant
# then, in separate terminals:
uvicorn ardrag.api:app --reload
python -m ardrag.mcp_server
```

Web UI: http://localhost:8000 (redirects to `/login` — sign in with `ADMIN_USER`/`ADMIN_PASSWORD` from `.env`)
MCP endpoint: http://localhost:8001/sse (or per FastMCP SSE transport path) — no auth

## VPS Deployment

Prerequisites on the VPS:
- Docker + Docker Compose installed.
- `cloudflared` already running (as a container or host service) and attached to a Docker
  network named `proxy-net`. Ardrag assumes this network already exists — it does **not**
  run its own reverse proxy.

Steps:

```bash
cp .env.example .env
# edit .env if needed (embedding model, etc.)
docker compose up -d --build
```

This starts two services:
- `qdrant` — internal only, not exposed externally.
- `ardrag-backend` — runs both the FastAPI app (port 8000) and the MCP SSE server (port 8001),
  joined to both the default compose network and the external `proxy-net`.

### Exposing via Cloudflare Tunnel

In your `cloudflared` tunnel config, add public hostnames pointing at the `ardrag-backend`
service name (Docker DNS resolves it within `proxy-net`), e.g.:

```yaml
ingress:
  - hostname: ardrag-ui.example.com
    service: http://ardrag-backend:8000
  - hostname: ardrag-mcp.example.com
    service: http://ardrag-backend:8001
  - service: http_status:404
```

### Connecting an MCP client

Point your MCP client at `https://ardrag-mcp.example.com/sse` (adjust path per FastMCP's SSE
transport). The client's LLM (via its own API) will then be able to call `rag_search`,
`rag_list_documents`, and `rag_add_note` against your documents.

### Backing up your data

All persistent state lives in two Docker volumes — `ardrag-data` (SQLite metadata + original
uploaded files) and `qdrant-data` (vector index). Neither is backed up automatically. Before
relying on this in production, set up periodic backups, e.g.:

```bash
docker run --rm -v projectardrag_ardrag-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/ardrag-data-$(date +%F).tar.gz -C /data .
docker run --rm -v projectardrag_qdrant-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/qdrant-data-$(date +%F).tar.gz -C /data .
```

(Volume names are prefixed with the Compose project name — check `docker volume ls` if yours
differ.) Ship the resulting archives off-box (S3, rsync to another host, etc.) on a schedule.

### Pre-production security checklist

- [ ] `ADMIN_USER` / `ADMIN_PASSWORD` changed from any default/example value
- [ ] `SESSION_SECRET` set to a real random value (see Authentication section) — without it,
      everyone is logged out on every container restart
- [ ] DeepSeek API key (if used) has reasonable spend limits set on the DeepSeek side
- [ ] Some form of rate limiting in front of `/login` — Ardrag itself doesn't rate-limit login
      attempts; use Cloudflare's (free) rate limiting rules on the tunnel hostname, or equivalent
- [ ] Backups configured and tested (see above) before you stop treating documents as disposable
- [ ] The MCP endpoint (port 8001) is intentionally unauthenticated — only expose it to clients
      you trust, e.g. via a Cloudflare Access policy or by not exposing it publicly at all if you
      only ever connect to it from the same network

## Resource footprint

Measured on a real 91-document / ~600MB collection (mixed PDF datasheets):

| Resource | Idle | Peak (active embedding/reindex) |
|---|---|---|
| RAM (backend + Qdrant) | ~370 MB | ~1–1.5 GB |
| CPU | <1% | Bounded by `EMBEDDING_THREADS` (cores − 1) |
| Disk (both volumes) | ~600 MB | grows ~6–7 MB per average document |
| Docker image | ~720 MB (fixed, doesn't grow with document count) | — |

**Recommended VPS sizing** (current usage × ~2 for headroom): **4 GB RAM / 2 vCPU / 20 GB disk**.
The 20GB disk figure is generous on purpose — at ~6.5MB/document that's room for several hundred
more documents without needing to resize. No GPU required; both embedding models (dense + sparse)
run on CPU.
