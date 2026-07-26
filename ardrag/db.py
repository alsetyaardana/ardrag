import hashlib
import json
import os
import secrets
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, text
from sqlmodel import Field, Session, SQLModel, create_engine, select

from ardrag.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    MCP_OAUTH_ENABLED,
    MCP_PUBLIC_URL,
    SQLITE_PATH,
)

_engine = create_engine(f"sqlite:///{SQLITE_PATH}")

UNCATEGORIZED_VENDOR = "Uncategorized"
UNCATEGORIZED_TYPE = "General"


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    content_hash: str
    chunk_count: int
    vendor: str = Field(default=UNCATEGORIZED_VENDOR, index=True)
    doc_type: str = Field(default=UNCATEGORIZED_TYPE, index=True)
    tags_json: str = Field(default="[]")
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def tags(self) -> list[str]:
        try:
            return json.loads(self.tags_json)
        except (TypeError, ValueError):
            return []


class Settings(SQLModel, table=True):
    id: Optional[int] = Field(default=1, primary_key=True)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP)
    deepseek_api_key: str = Field(default="")
    deepseek_model: str = Field(default="deepseek-chat")
    # Embedding provider — "local" (FastEmbed, embedding_model above picks which) or "api"
    # (generic OpenAI-compatible embeddings endpoint). embedding_api_dimension is never
    # user-entered — it's measured server-side from a live test call when the API config is
    # saved (see POST /settings), since Qdrant needs an exact vector size up front.
    embedding_provider: str = Field(default="local")
    embedding_api_base_url: str = Field(default="")
    embedding_api_key: str = Field(default="")
    embedding_api_model: str = Field(default="")
    embedding_api_dimension: int = Field(default=0)
    # Max tokens per embedding input — chunks (with the doc_name prefix core.py adds) that exceed
    # this are split further automatically right before embedding, using the real tokenizer for
    # local FastEmbed models (TextEmbedding.token_count) or a char/4 approximation for API
    # providers (no generic way to know an arbitrary OpenAI-compatible endpoint's tokenizer).
    # 512 matches all 4 curated local models' actual limit; API providers often allow far more
    # (e.g. 8191 for OpenAI) — edit this after switching. 0 disables the check entirely.
    max_embedding_tokens: int = Field(default=512)
    # Separate from max_embedding_tokens above: some OpenAI-compatible gateways (observed with a
    # self-hosted litellm proxy) additionally cap the *combined* token count across every input in
    # a single embeddings.create call, independent of any one input's own size — e.g. "300000
    # tokens per request" even though each individual chunk is well under max_embedding_tokens.
    # core.py batches API embedding calls to stay under this. Only relevant for embedding_provider
    # == "api"; ignored for local (FastEmbed batches in-process with no such limit).
    embedding_api_batch_token_limit: int = Field(default=200000)
    # Some gateways separately cap the raw JSON request body size, independent of token count
    # (observed: "JSON payload (42521863 bytes) is larger than allowed (limit: 33554432 bytes)"
    # even on a request that stayed under the token-count limit above). Batches are flushed on
    # whichever limit — token count or byte size — would be hit first.
    embedding_api_batch_byte_limit: int = Field(default=25_000_000)
    # Classification API base URL — generalizes deepseek_api_key/deepseek_model above (kept as-is
    # to avoid a data migration) into a full generic OpenAI-compatible endpoint. Defaults to
    # DeepSeek's URL for continuity but is fully user-editable.
    classify_base_url: str = Field(default="https://api.deepseek.com")
    # MCP server settings — GUI-configurable, source of truth after first-boot migration seeds
    # them from the legacy .env-only MCP_OAUTH_ENABLED/MCP_PUBLIC_URL values.
    mcp_sse_enabled: bool = Field(default=True)
    mcp_streamable_enabled: bool = Field(default=True)
    mcp_anonymous_enabled: bool = Field(default=True)
    mcp_oauth_enabled: bool = Field(default=False)
    mcp_public_url: str = Field(default="")


class McpUser(SQLModel, table=True):
    """MCP-only login accounts, independent of the web UI's single ADMIN_USER — used for the
    OAuth login form when MCP OAuth access mode is enabled, so MCP access can be granted/revoked
    per-person without touching the shared admin credentials."""

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    password_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class McpApiToken(SQLModel, table=True):
    """Long-lived, non-expiring Bearer tokens generated directly (no interactive OAuth
    authorize/login redirect) for MCP clients that can't do a browser-based OAuth flow — e.g.
    AnythingLLM, which only supports a static token pasted into a config file's `headers` field.
    Validated by the same code path as regular OAuth access tokens (see
    ArdragOAuthProvider.load_access_token in oauth_provider.py)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    token: str = Field(unique=True, index=True)
    label: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class McpCallLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tool_name: str = Field(index=True)
    called_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)


class DeepseekUsageLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    called_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---- OAuth 2.1 authorization server storage (for the MCP server's Claude.ai Custom Connector
# support). Persisted in SQLite rather than in-memory so registered clients/tokens survive
# container redeploys — losing them would silently break every previously-connected client. ----


class OAuthClient(SQLModel, table=True):
    client_id: str = Field(primary_key=True)
    client_secret: Optional[str] = None
    redirect_uris_json: str = "[]"
    client_name: Optional[str] = None
    grant_types_json: str = "[]"
    response_types_json: str = "[]"
    token_endpoint_auth_method: Optional[str] = None
    scope: Optional[str] = None
    client_id_issued_at: Optional[int] = None


class OAuthAuthCode(SQLModel, table=True):
    code: str = Field(primary_key=True)
    client_id: str
    redirect_uri: str
    redirect_uri_provided_explicitly: bool = True
    code_challenge: str
    scopes_json: str = "[]"
    expires_at: float
    subject: str
    resource: Optional[str] = None


class OAuthAccessToken(SQLModel, table=True):
    token: str = Field(primary_key=True)
    client_id: str
    scopes_json: str = "[]"
    expires_at: Optional[float] = None
    subject: str
    resource: Optional[str] = None


class OAuthRefreshToken(SQLModel, table=True):
    token: str = Field(primary_key=True)
    client_id: str
    scopes_json: str = "[]"
    expires_at: Optional[float] = None
    subject: str
    access_token: Optional[str] = None


def _migrate_document_columns() -> None:
    with _engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(document)"))}
        if "vendor" not in cols:
            conn.execute(text(f"ALTER TABLE document ADD COLUMN vendor VARCHAR DEFAULT '{UNCATEGORIZED_VENDOR}'"))
        if "doc_type" not in cols:
            conn.execute(text(f"ALTER TABLE document ADD COLUMN doc_type VARCHAR DEFAULT '{UNCATEGORIZED_TYPE}'"))
        if "tags_json" not in cols:
            conn.execute(text("ALTER TABLE document ADD COLUMN tags_json VARCHAR DEFAULT '[]'"))
        conn.commit()


def _migrate_settings_columns() -> None:
    with _engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(settings)"))}
        if "deepseek_api_key" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN deepseek_api_key VARCHAR DEFAULT ''"))
        if "deepseek_model" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN deepseek_model VARCHAR DEFAULT 'deepseek-chat'"))
        conn.commit()


def _migrate_oauth_columns() -> None:
    with _engine.connect() as conn:
        tables = {row[0] for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        if "oauthauthcode" in tables:
            cols = {row[1] for row in conn.execute(text("PRAGMA table_info(oauthauthcode)"))}
            if "resource" not in cols:
                conn.execute(text("ALTER TABLE oauthauthcode ADD COLUMN resource VARCHAR"))
        if "oauthaccesstoken" in tables:
            cols = {row[1] for row in conn.execute(text("PRAGMA table_info(oauthaccesstoken)"))}
            if "resource" not in cols:
                conn.execute(text("ALTER TABLE oauthaccesstoken ADD COLUMN resource VARCHAR"))
        conn.commit()


def _migrate_mcp_settings_columns() -> None:
    with _engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(settings)"))}
        # Seed from the legacy env-only vars only on first migration (column didn't exist yet),
        # so an existing deployment's current behavior (e.g. MCP_OAUTH_ENABLED=true in .env)
        # carries over instead of silently resetting to the GUI defaults.
        if "mcp_sse_enabled" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN mcp_sse_enabled BOOLEAN DEFAULT 1"))
        if "mcp_streamable_enabled" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN mcp_streamable_enabled BOOLEAN DEFAULT 1"))
        if "mcp_anonymous_enabled" not in cols:
            seed_anonymous = 0 if MCP_OAUTH_ENABLED else 1
            conn.execute(text(f"ALTER TABLE settings ADD COLUMN mcp_anonymous_enabled BOOLEAN DEFAULT {seed_anonymous}"))
        if "mcp_oauth_enabled" not in cols:
            seed_oauth = 1 if MCP_OAUTH_ENABLED else 0
            conn.execute(text(f"ALTER TABLE settings ADD COLUMN mcp_oauth_enabled BOOLEAN DEFAULT {seed_oauth}"))
        if "mcp_public_url" not in cols:
            conn.execute(text(f"ALTER TABLE settings ADD COLUMN mcp_public_url VARCHAR DEFAULT '{MCP_PUBLIC_URL}'"))
        conn.commit()


def _migrate_ai_config_columns() -> None:
    with _engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(settings)"))}
        if "embedding_provider" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN embedding_provider VARCHAR DEFAULT 'local'"))
        if "embedding_api_base_url" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN embedding_api_base_url VARCHAR DEFAULT ''"))
        if "embedding_api_key" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN embedding_api_key VARCHAR DEFAULT ''"))
        if "embedding_api_model" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN embedding_api_model VARCHAR DEFAULT ''"))
        if "embedding_api_dimension" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN embedding_api_dimension INTEGER DEFAULT 0"))
        if "classify_base_url" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN classify_base_url VARCHAR DEFAULT 'https://api.deepseek.com'"))
        if "max_embedding_tokens" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN max_embedding_tokens INTEGER DEFAULT 512"))
        if "embedding_api_batch_token_limit" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN embedding_api_batch_token_limit INTEGER DEFAULT 200000"))
        if "embedding_api_batch_byte_limit" not in cols:
            conn.execute(text("ALTER TABLE settings ADD COLUMN embedding_api_batch_byte_limit INTEGER DEFAULT 25000000"))
        conn.commit()


def init_db() -> None:
    SQLModel.metadata.create_all(_engine)
    _migrate_document_columns()
    _migrate_settings_columns()
    _migrate_oauth_columns()
    _migrate_mcp_settings_columns()
    _migrate_ai_config_columns()
    with Session(_engine) as session:
        if not session.get(Settings, 1):
            session.add(Settings(id=1))
            session.commit()


def get_document_by_name(name: str) -> Optional[Document]:
    with Session(_engine) as session:
        return session.exec(select(Document).where(Document.name == name)).first()


def upsert_document(
    name: str,
    content_hash: str,
    chunk_count: int,
    vendor: Optional[str] = None,
    doc_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Document:
    with Session(_engine) as session:
        existing = session.exec(select(Document).where(Document.name == name)).first()
        if existing:
            existing.content_hash = content_hash
            existing.chunk_count = chunk_count
            existing.uploaded_at = datetime.now(timezone.utc)
            if vendor is not None:
                existing.vendor = vendor
            if doc_type is not None:
                existing.doc_type = doc_type
            if tags is not None:
                existing.tags_json = json.dumps(tags)
            session.add(existing)
            session.commit()
            session.refresh(existing)
            return existing
        doc = Document(
            name=name,
            content_hash=content_hash,
            chunk_count=chunk_count,
            vendor=vendor or UNCATEGORIZED_VENDOR,
            doc_type=doc_type or UNCATEGORIZED_TYPE,
            tags_json=json.dumps(tags or []),
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        return doc


def list_documents(
    vendor: Optional[str] = None,
    doc_type: Optional[str] = None,
    tag: Optional[str] = None,
    q: Optional[str] = None,
) -> list[Document]:
    with Session(_engine) as session:
        stmt = select(Document)
        if vendor is not None:
            stmt = stmt.where(Document.vendor == vendor)
        if doc_type is not None:
            stmt = stmt.where(Document.doc_type == doc_type)
        if q:
            stmt = stmt.where(Document.name.ilike(f"%{q}%"))
        stmt = stmt.order_by(Document.uploaded_at.desc())
        docs = list(session.exec(stmt))
    if tag is not None:
        docs = [d for d in docs if tag in d.tags]
    return docs


def list_folders() -> list[dict]:
    """Return [{vendor, doc_type, count}] distinct combinations for folder-tree UI."""
    with Session(_engine) as session:
        docs = session.exec(select(Document)).all()
    counts: dict[tuple[str, str], int] = {}
    for d in docs:
        key = (d.vendor, d.doc_type)
        counts[key] = counts.get(key, 0) + 1
    return [{"vendor": v, "doc_type": t, "count": c} for (v, t), c in sorted(counts.items())]


def list_all_tags() -> list[str]:
    with Session(_engine) as session:
        docs = session.exec(select(Document)).all()
    tags: set[str] = set()
    for d in docs:
        tags.update(d.tags)
    return sorted(tags)


def get_document_by_id(doc_id: int) -> Optional[Document]:
    with Session(_engine) as session:
        return session.get(Document, doc_id)


def update_document_metadata(
    doc_id: int,
    vendor: Optional[str] = None,
    doc_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> Optional[Document]:
    with Session(_engine) as session:
        doc = session.get(Document, doc_id)
        if not doc:
            return None
        if vendor is not None:
            doc.vendor = vendor
        if doc_type is not None:
            doc.doc_type = doc_type
        if tags is not None:
            doc.tags_json = json.dumps(tags)
        session.add(doc)
        session.commit()
        session.refresh(doc)
        return doc


def delete_document(doc_id: int) -> Optional[Document]:
    with Session(_engine) as session:
        doc = session.get(Document, doc_id)
        if doc:
            session.delete(doc)
            session.commit()
        return doc


def get_settings() -> Settings:
    with Session(_engine) as session:
        settings = session.get(Settings, 1)
        if not settings:
            settings = Settings(id=1)
            session.add(settings)
            session.commit()
            session.refresh(settings)
        return settings


def update_settings(**kwargs) -> Settings:
    with Session(_engine) as session:
        settings = session.get(Settings, 1)
        if not settings:
            settings = Settings(id=1)
            session.add(settings)
        for k, v in kwargs.items():
            if v is not None:
                setattr(settings, k, v)
        session.add(settings)
        session.commit()
        session.refresh(settings)
        return settings


def _hourly_buckets(timestamps_and_values: list[tuple[datetime, int]], hours: int) -> list[dict]:
    """Build a fixed-length (one entry per hour, oldest first) series for the last `hours` hours,
    even for hours with zero activity, so the chart doesn't visually compress/skip gaps.

    SQLite drops tzinfo on round-trip, so timestamps read back from the DB come back as naive
    datetimes even though they were stored as UTC-aware — comparing/keying against an aware `now`
    would silently never match (naive != aware, no error). Everything here is normalized to naive
    UTC before bucketing, and tzinfo is re-added only in the output.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    buckets = {now - timedelta(hours=i): 0 for i in range(hours - 1, -1, -1)}
    for ts, value in timestamps_and_values:
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        bucket = ts.replace(minute=0, second=0, microsecond=0)
        if bucket in buckets:
            buckets[bucket] += value
    return [{"hour": h.replace(tzinfo=timezone.utc).isoformat(), "count": c} for h, c in sorted(buckets.items())]


def log_mcp_call(tool_name: str) -> None:
    with Session(_engine) as session:
        session.add(McpCallLog(tool_name=tool_name))
        session.commit()


def get_mcp_usage_summary(hours: int = 24) -> dict:
    with Session(_engine) as session:
        total_all_time = session.exec(select(func.count()).select_from(McpCallLog)).one()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = list(session.exec(select(McpCallLog).where(McpCallLog.called_at >= cutoff)))

    by_tool: dict[str, int] = defaultdict(int)
    for r in recent:
        by_tool[r.tool_name] += 1

    return {
        "total_calls_all_time": total_all_time,
        "total_calls_window": len(recent),
        "window_hours": hours,
        "by_tool": dict(sorted(by_tool.items(), key=lambda kv: -kv[1])),
        "hourly": _hourly_buckets([(r.called_at, 1) for r in recent], hours),
    }


def log_deepseek_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    with Session(_engine) as session:
        session.add(
            DeepseekUsageLog(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens
            )
        )
        session.commit()


def get_deepseek_usage_summary(hours: int = 24) -> dict:
    with Session(_engine) as session:
        rows = list(session.exec(select(DeepseekUsageLog)))
        # Filtered in SQL (not Python) so the comparison is done consistently by SQLite/SQLAlchemy
        # rather than comparing an aware `cutoff` against naive datetimes read back from SQLite
        # (SQLite drops tzinfo on round-trip — comparing them in Python raises TypeError).
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent = list(session.exec(select(DeepseekUsageLog).where(DeepseekUsageLog.called_at >= cutoff)))

    total_calls = len(rows)
    total_tokens = sum(r.total_tokens for r in rows)
    prompt_tokens = sum(r.prompt_tokens for r in rows)
    completion_tokens = sum(r.completion_tokens for r in rows)

    return {
        "total_calls": total_calls,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "window_hours": hours,
        "hourly_tokens": _hourly_buckets([(r.called_at, r.total_tokens) for r in recent], hours),
    }


# ---- OAuth storage functions ----


def oauth_get_client(client_id: str) -> Optional[dict]:
    with Session(_engine) as session:
        c = session.get(OAuthClient, client_id)
        if not c:
            return None
        return {
            "client_id": c.client_id,
            "client_secret": c.client_secret,
            "redirect_uris": json.loads(c.redirect_uris_json),
            "client_name": c.client_name,
            "grant_types": json.loads(c.grant_types_json),
            "response_types": json.loads(c.response_types_json),
            "token_endpoint_auth_method": c.token_endpoint_auth_method,
            "scope": c.scope,
            "client_id_issued_at": c.client_id_issued_at,
        }


def oauth_save_client(
    client_id: str,
    client_secret: Optional[str],
    redirect_uris: list[str],
    client_name: Optional[str],
    grant_types: list[str],
    response_types: list[str],
    token_endpoint_auth_method: Optional[str],
    scope: Optional[str],
    client_id_issued_at: Optional[int],
) -> None:
    with Session(_engine) as session:
        existing = session.get(OAuthClient, client_id)
        row = existing or OAuthClient(client_id=client_id)
        row.client_secret = client_secret
        row.redirect_uris_json = json.dumps(redirect_uris)
        row.client_name = client_name
        row.grant_types_json = json.dumps(grant_types)
        row.response_types_json = json.dumps(response_types)
        row.token_endpoint_auth_method = token_endpoint_auth_method
        row.scope = scope
        row.client_id_issued_at = client_id_issued_at
        session.add(row)
        session.commit()


def oauth_save_auth_code(
    code: str,
    client_id: str,
    redirect_uri: str,
    redirect_uri_provided_explicitly: bool,
    code_challenge: str,
    scopes: list[str],
    expires_at: float,
    subject: str,
    resource: Optional[str] = None,
) -> None:
    with Session(_engine) as session:
        session.add(
            OAuthAuthCode(
                code=code,
                client_id=client_id,
                redirect_uri=redirect_uri,
                redirect_uri_provided_explicitly=redirect_uri_provided_explicitly,
                code_challenge=code_challenge,
                scopes_json=json.dumps(scopes),
                expires_at=expires_at,
                subject=subject,
                resource=resource,
            )
        )
        session.commit()


def oauth_load_auth_code(code: str) -> Optional[dict]:
    with Session(_engine) as session:
        row = session.get(OAuthAuthCode, code)
        if not row:
            return None
        return {
            "code": row.code,
            "client_id": row.client_id,
            "redirect_uri": row.redirect_uri,
            "redirect_uri_provided_explicitly": row.redirect_uri_provided_explicitly,
            "code_challenge": row.code_challenge,
            "scopes": json.loads(row.scopes_json),
            "expires_at": row.expires_at,
            "subject": row.subject,
            "resource": row.resource,
        }


def oauth_delete_auth_code(code: str) -> None:
    with Session(_engine) as session:
        row = session.get(OAuthAuthCode, code)
        if row:
            session.delete(row)
            session.commit()


def oauth_save_access_token(
    token: str,
    client_id: str,
    scopes: list[str],
    expires_at: Optional[float],
    subject: str,
    resource: Optional[str] = None,
) -> None:
    with Session(_engine) as session:
        session.add(
            OAuthAccessToken(
                token=token,
                client_id=client_id,
                scopes_json=json.dumps(scopes),
                expires_at=expires_at,
                subject=subject,
                resource=resource,
            )
        )
        session.commit()


def oauth_load_access_token(token: str) -> Optional[dict]:
    with Session(_engine) as session:
        row = session.get(OAuthAccessToken, token)
        if not row:
            return None
        return {
            "token": row.token,
            "client_id": row.client_id,
            "scopes": json.loads(row.scopes_json),
            "expires_at": row.expires_at,
            "subject": row.subject,
            "resource": row.resource,
        }


def oauth_delete_access_token(token: str) -> None:
    with Session(_engine) as session:
        row = session.get(OAuthAccessToken, token)
        if row:
            session.delete(row)
            session.commit()


def oauth_save_refresh_token(
    token: str,
    client_id: str,
    scopes: list[str],
    expires_at: Optional[float],
    subject: str,
    access_token: Optional[str],
) -> None:
    with Session(_engine) as session:
        session.add(
            OAuthRefreshToken(
                token=token,
                client_id=client_id,
                scopes_json=json.dumps(scopes),
                expires_at=expires_at,
                subject=subject,
                access_token=access_token,
            )
        )
        session.commit()


def oauth_load_refresh_token(token: str) -> Optional[dict]:
    with Session(_engine) as session:
        row = session.get(OAuthRefreshToken, token)
        if not row:
            return None
        return {
            "token": row.token,
            "client_id": row.client_id,
            "scopes": json.loads(row.scopes_json),
            "expires_at": row.expires_at,
            "subject": row.subject,
            "access_token": row.access_token,
        }


def oauth_delete_refresh_token(token: str) -> None:
    with Session(_engine) as session:
        row = session.get(OAuthRefreshToken, token)
        if row:
            session.delete(row)
            session.commit()


# ---- MCP-specific user accounts (separate from ADMIN_USER) ----


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"{salt.hex()}${digest.hex()}"


def mcp_user_create(username: str, password: str) -> McpUser:
    with Session(_engine) as session:
        if session.exec(select(McpUser).where(McpUser.username == username)).first():
            raise ValueError(f"A user named '{username}' already exists.")
        user = McpUser(username=username, password_hash=_hash_password(password))
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def mcp_user_verify(username: str, password: str) -> bool:
    with Session(_engine) as session:
        user = session.exec(select(McpUser).where(McpUser.username == username)).first()
        if not user:
            return False
        salt_hex, _, expected_hex = user.password_hash.partition("$")
        computed = _hash_password(password, bytes.fromhex(salt_hex))
        return secrets.compare_digest(computed, user.password_hash)


def mcp_user_list() -> list[McpUser]:
    with Session(_engine) as session:
        return list(session.exec(select(McpUser).order_by(McpUser.created_at)))


def mcp_user_delete(user_id: int) -> Optional[McpUser]:
    with Session(_engine) as session:
        user = session.get(McpUser, user_id)
        if user:
            session.delete(user)
            session.commit()
        return user


# ---- MCP static API tokens (for clients that can't do the OAuth browser redirect) ----


def mcp_api_token_create(label: str) -> McpApiToken:
    with Session(_engine) as session:
        token = McpApiToken(token=secrets.token_urlsafe(32), label=label)
        session.add(token)
        session.commit()
        session.refresh(token)
        return token


def mcp_api_token_get(token: str) -> Optional[McpApiToken]:
    with Session(_engine) as session:
        return session.exec(select(McpApiToken).where(McpApiToken.token == token)).first()


def mcp_api_token_list() -> list[McpApiToken]:
    with Session(_engine) as session:
        return list(session.exec(select(McpApiToken).order_by(McpApiToken.created_at)))


def mcp_api_token_delete(token_id: int) -> Optional[McpApiToken]:
    with Session(_engine) as session:
        token = session.get(McpApiToken, token_id)
        if token:
            session.delete(token)
            session.commit()
        return token
