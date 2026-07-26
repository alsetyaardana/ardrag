# 🗂️ ArdRAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Docker Compose](https://img.shields.io/badge/deploy-docker--compose-2496ED.svg?logo=docker&logoColor=white)](docker-compose.yml)
[![MCP](https://img.shields.io/badge/MCP-server-6b46c1.svg)](#-mcp-server)

A self-hosted, personal **RAG (Retrieval-Augmented Generation)** system with a file-manager style
web UI and an **MCP server**, so any MCP-compatible AI client — Claude Code, Claude Desktop,
claude.ai Custom Connectors, or a local model — can search your own documents. The generative LLM
is always called by the *client*, not by ArdRAG — this project only handles ingestion, embedding,
storage, retrieval, and access control on top of them.

Built for personal/small-team document collections (product datasheets, specs, notes) that you
want queryable from any AI client without paying per-document embedding API costs or shipping your
private documents to a third party during indexing.

> 🍴 **Free to reuse.** MIT-licensed — fork it, self-host it, rip out the parts you don't need.

## ✨ Highlights

- 🔍 **Hybrid search** (dense + sparse, fused with RRF) — catches both semantic matches and exact
  model-code/keyword matches that dense-only embeddings tend to blur together.
- 🧠 **Local embeddings** via FastEmbed — no external API calls or per-document cost for indexing.
- 🖥️ **Web UI** — upload, folders (vendor/type), freeform tags, search/filter, bulk edit,
  AI-assisted classification (optional, via DeepSeek), usage dashboard.
- 🔌 **MCP server** with dual transport (legacy SSE **and** modern Streamable HTTP) — connects to
  Claude Code, Claude Desktop, and claude.ai Custom Connectors out of the box.
- 🔐 **Fully GUI-configurable access control** — toggle which transports run, whether access is
  open/anonymous or OAuth-gated (or both at once), and manage MCP-only user accounts — all from
  the Settings tab, no `.env` editing or redeploy required.
- 🐳 **One `docker compose up`** — two containers (Qdrant + backend), sane resource limits baked
  in.

## 🧩 Components

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
  browser's native Basic Auth popup. The MCP server (port 8001) is a separate, supervised
  subprocess with its own independent, GUI-configurable auth story (see below).
- **Web UI** (`/`) — file-manager style: upload (single/multiple/AI-classified files, with a live
  progress bar), organize into Vendor / Doc Type folders plus freeform multi-tags, search/filter by
  name/vendor/type/tag with pagination, bulk-edit metadata across a selection, preview/download
  original files, delete documents. Also has:
  - **Dashboard** — system health (doc/chunk counts, Qdrant status incl. hybrid flag, disk usage,
    uptime), reindex/batch-upload progress, **MCP tool usage** (calls by tool, hourly chart) and
    **DeepSeek token usage** (prompt/completion/total tokens, hourly chart) — auto-refreshes every
    5s while this tab is open.
  - **Settings** — embedding model, chunk size/overlap, trigger reindex; DeepSeek API key +
    model, trigger AI-classification backfill over existing documents; and the **MCP Server**
    panel described below.
- **DeepSeek (optional)** — if configured with an API key, documents can be auto-classified
  (vendor/doc_type/tags inferred from content) on upload or via a backfill job over existing
  documents. Every call's token usage is logged and shown on the Dashboard. ArdRAG works fully
  without this — it's an optional convenience, not required for RAG itself.

## 🔎 MCP Server

At port `8001`, exposes:

- `rag_search(query, top_k, vendor, doc_type, tag)` — hybrid (dense+sparse) search, optionally
  narrowed to a vendor/doc_type folder and/or tag first (resolved against SQLite, then
  post-filtered against the over-fetched Qdrant results — no reindex needed when metadata changes)
- `rag_get_document(doc_name, max_chars)` — read a document's full original text (not just a
  chunk), for when a search hit needs verifying against the complete source, e.g. exact port
  counts/specs that might straddle a chunk boundary. Truncates at `max_chars` (default 20000)
  and reports whether it did via the `truncated` field.
- `rag_compare_documents(doc_names, max_chars_per_doc)` — fetches full text of 2+ documents in
  one call, labeled and ready for the calling model to draft a side-by-side spec comparison
  table from (e.g. presales comparing similar products across a product line). ArdRAG bundles
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

### Transports

Both are served simultaneously (each independently toggleable in the GUI — see below):

- **SSE** (`/sse`) — the legacy MCP transport. Used by Claude Code (`.mcp.json` with
  `"type": "sse"`) and other simpler clients.
- **Streamable HTTP** (`/mcp`) — the modern MCP transport. Used by Claude Desktop and claude.ai
  Custom Connectors (which mandate OAuth — see below).

### 🔐 Access control (fully GUI-configurable)

Everything about who can reach the MCP server, and how, is controlled from **Settings → MCP
Server** in the web UI — no `.env` editing or container rebuild required. Saving any of these
restarts just the MCP subprocess (not the whole container), so changes apply within seconds:

- **Transport toggles** — enable/disable SSE and Streamable HTTP independently.
- **Access mode toggles** — **Anonymous** and **OAuth**, either or both at once:
  - *Anonymous only* (default, matches "open" self-hosting for a private network) — anyone who can
    reach the URL can use the MCP tools, no login involved.
  - *OAuth only* — every request needs a valid Bearer token, obtained by a full OAuth 2.1 flow
    (dynamic client registration, PKCE, refresh tokens) gated behind a login page.
  - *Both* — auth is optional: a request with no `Authorization` header is treated as anonymous,
    but a request that *does* send a token still gets it validated (a bad/expired token is still
    rejected).
  - *Neither* is rejected by the API — there'd be no way for any client to connect.
- **MCP Users** — when OAuth is on, logins are checked against a dedicated user table, completely
  separate from the web UI's own `ADMIN_USER`/`ADMIN_PASSWORD`. Add/remove as many MCP-only
  accounts as you want from the same panel, each independently revocable.
- **API Tokens** — a static, non-expiring Bearer token generated on demand, for clients that can't
  do the interactive OAuth browser redirect at all — e.g.
  [AnythingLLM](https://docs.anythingllm.com/mcp-compatibility/overview), which only supports
  pasting a token into a config file's `headers` field (`{"Authorization": "Bearer <token>"}`), or
  a `curl`/cron script. Shown once at creation (same "reveal once" pattern as a GitHub personal
  access token) and independently revocable — revoking takes effect immediately, no MCP restart
  needed.
- **Public URL** — required only when OAuth is on (used in the OAuth issuer/well-known metadata);
  set it to the real public HTTPS address the MCP server is reachable at.

ArdRAG ships a self-contained OAuth 2.1 authorization server (`ardrag/oauth_provider.py`) — no
third-party IdP (Google/GitHub/etc.) needed. Clients, authorization codes, access tokens, and
refresh tokens (30-day expiry, rotated on refresh) are all persisted in SQLite, so registered
connectors survive container redeploys.

To add ArdRAG to claude.ai: Settings → Connectors → Add custom connector → paste your MCP server's
public URL with the Streamable HTTP path (e.g. `https://ardrag-mcp.example.com/mcp`). claude.ai
handles the registration and redirect dance automatically; you'll see ArdRAG's login page once
during setup (using an MCP user you've created in Settings, not your web admin login).

**Security note:** Anonymous access means anyone with the URL can read your documents through the
MCP tools — same trust level as leaving the web UI's login off. For a public, long-lived
deployment, either turn OAuth on (and turn Anonymous off) or keep the MCP endpoint restricted some
other way (Cloudflare Access on the MCP hostname, IP allowlist, private network only, etc.).

### Web UI authentication

Session-based login, not HTTP Basic Auth. `GET /login` serves a custom login page; `POST /login`
verifies `username`/`password` against `ADMIN_USER`/`ADMIN_PASSWORD` and, on success, sets a
signed `ardrag_session` cookie (`httponly`, 7-day expiry by default — see `SESSION_MAX_AGE_SECONDS`).
`POST /logout` clears it. Every other route (except `/login` itself) requires a valid session —
API endpoints reply `401 {"detail": "Not authenticated"}`, and the web UI's `fetch` wrapper
redirects to `/login` automatically on a 401.

Set `SESSION_SECRET` in `.env` to a random 64-char hex string (e.g.
`python3 -c "import secrets; print(secrets.token_hex(32))"`) so sessions survive container
restarts — without it, a random secret is generated at startup and everyone gets logged out on
every restart.

### 🗂️ Folders (Vendor / Doc Type) & tags

Every document has a `vendor` and `doc_type` (both optional, default to `Uncategorized`/`General`)
used for the folder-tree sidebar — plus an independent, freeform list of `tags` (multi-value,
not hierarchical) for cross-cutting metadata that doesn't fit the vendor/type folder structure.

Set them via `vendor`/`doc_type`/`tags` form fields on `POST /documents` or `/documents/batch`
(repeat the `tags` field per tag, or comma-separate within one value — both are accepted), or via
the web UI's Vendor/Doc Type inputs and the tag chip input (type + Enter/comma to add, click ×
to remove). `GET /folders` returns vendor/doc_type combinations with counts for the sidebar tree;
`GET /tags` returns all known tags (for autocomplete); `GET /documents?tag=<value>` filters by tag.

### ⚙️ Settings & reindexing

`GET/POST /settings` controls the active embedding model (`ardrag/config.py:SUPPORTED_EMBEDDING_MODELS`)
and chunk size/overlap, persisted in SQLite. Changing the embedding model changes the vector
dimension, so it requires recreating the Qdrant collection — the UI/API flags `reindex_required`
when that happens. Trigger a reindex with `POST /reindex` (`recreate_collection=true` to also
drop/recreate the collection first) — it runs in a background thread; poll `GET /reindex/status`
for progress. This replaces the old `docker compose exec ... python -m ardrag.reindex` CLI flow
(that still works too, as a fallback with no auth/UI needed).

### 👀 Preview & download

`GET /documents/{id}/preview` serves the original stored file inline (browsers render PDFs
natively in a new tab). `GET /documents/{id}/download` serves it as an attachment with the
original filename. Both require an active session, same as everything else in the web UI — handy
for grabbing a document from your phone/another machine without needing the RAG search at all.

### 🚫 Upload restrictions

Only `.pdf, .txt, .md, .csv, .json, .log` are accepted (`ALLOWED_UPLOAD_EXTENSIONS` in `.env` to
change), and files over 50MB are rejected (`MAX_UPLOAD_SIZE_MB`). `extract_text()` only knows how
to handle these — anything else (`.pptx`, `.docx`, `.xlsx`, images, archives, ...) is binary and
would otherwise get blindly decoded as text, producing huge garbage content that still gets
chunked/embedded, wasting resources and — for large binary files — capable of spiking memory/CPU
hard enough to make the whole host crawl. Both checks happen before any extraction/embedding work
starts, so a rejected file costs nothing.

### 📦 Batch upload

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

## 🚀 Local development

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
MCP endpoints: http://localhost:8001/sse and http://localhost:8001/mcp — open by default (toggle
access mode from Settings → MCP Server once you're logged into the web UI)

## ☁️ VPS Deployment

Prerequisites on the VPS:
- Docker + Docker Compose installed.
- `cloudflared` already running (as a container or host service) and attached to a Docker
  network named `proxy-net`. ArdRAG assumes this network already exists — it does **not**
  run its own reverse proxy. (Any other reverse-proxy/tunnel setup works too — just point it at
  the `ardrag-backend` service's ports 8000/8001.)

Steps:

```bash
cp .env.example .env
# edit .env if needed (embedding model, admin credentials, etc.)
docker compose up -d --build
```

This starts two services:
- `qdrant` — internal only, not exposed externally.
- `ardrag-backend` — runs the FastAPI app (port 8000) and supervises the MCP server (port 8001)
  as a subprocess it can restart on demand, joined to both the default compose network and the
  external `proxy-net`.

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

- **Claude Code**: add to `.mcp.json` —
  ```json
  { "mcpServers": { "ardrag": { "type": "sse", "url": "https://ardrag-mcp.example.com/sse" } } }
  ```
- **Claude Desktop / claude.ai Custom Connector**: use the Streamable HTTP URL,
  `https://ardrag-mcp.example.com/mcp` — turn on OAuth in Settings → MCP Server first (claude.ai
  requires it) and create at least one MCP user to log in with.

The client's LLM (via its own API) will then be able to call `rag_search`, `rag_get_document`,
`rag_compare_documents`, `rag_suggest_bom`, `rag_list_documents`, `rag_list_folders`,
`rag_list_tags`, `rag_add_note`, and `rag_add_notes` against your documents.

### 💾 Backing up your data

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

### ✅ Pre-production security checklist

- [ ] `ADMIN_USER` / `ADMIN_PASSWORD` changed from any default/example value
- [ ] `SESSION_SECRET` set to a real random value (see Web UI authentication section) — without
      it, everyone is logged out on every container restart
- [ ] Decided on an MCP access mode in Settings → MCP Server (Anonymous is open to anyone who can
      reach the URL — fine on a private network, not for a public one)
- [ ] If using OAuth, at least one MCP user created, and the Public URL field set correctly
- [ ] DeepSeek API key (if used) has reasonable spend limits set on the DeepSeek side
- [ ] Some form of rate limiting in front of `/login` — ArdRAG itself doesn't rate-limit login
      attempts; use Cloudflare's (free) rate limiting rules on the tunnel hostname, or equivalent
- [ ] Backups configured and tested (see above) before you stop treating documents as disposable

## 📊 Resource footprint

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

## 🛠️ Tech stack

FastAPI · FastMCP · Qdrant · FastEmbed (dense + sparse/BM25) · SQLModel/SQLite · Docker Compose ·
vanilla HTML/JS web UI · optional DeepSeek for AI classification.

## ⚠️ Known limitations

Built and tuned for **personal/small-team scale**, not multi-tenant SaaS:

- Single admin account for the web UI (`ADMIN_USER`/`ADMIN_PASSWORD` in `.env`, plain comparison
  — not hashed). Fine for a personal instance; don't reuse this pattern for a public multi-user
  product without hardening it first.
- SQLite as the metadata store — great up to personal/small-team document volumes, not designed
  for high concurrent write load.
- No automated test suite yet (see `TESTING.md` for the manual verification checklist used during
  development) — contributions adding `pytest` coverage are very welcome.

## 🤝 Contributing

Issues and pull requests are welcome — this started as a personal tool, so expect some rough
edges outside the documented scope above. If you're proposing a larger change, opening an issue
first to discuss the approach is appreciated.

## 📄 License

[MIT](LICENSE) — do whatever you want with it, including running your own fork commercially. No
warranty, use at your own risk (see the license text for the legal version of that sentence).
