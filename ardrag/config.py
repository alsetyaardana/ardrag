import os
import secrets
from multiprocessing import cpu_count
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ardrag_documents")
SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/ardrag.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# Signs the session cookie. Set SESSION_SECRET in .env for sessions that survive container
# restarts — without it, a random secret is generated at startup and everyone is logged out
# whenever the backend restarts.
SESSION_SECRET = os.getenv("SESSION_SECRET") or secrets.token_hex(32)
SESSION_COOKIE_NAME = "ardrag_session"
SESSION_MAX_AGE_SECONDS = int(os.getenv("SESSION_MAX_AGE_SECONDS", str(7 * 24 * 3600)))

# Registry of embedding models selectable from Settings, mapped to their output vector dimension.
# Keep this curated/small — every entry must be a model FastEmbed can load.
SUPPORTED_EMBEDDING_MODELS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
}

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "8"))

# Caps onnxruntime's intra-op thread pool for the embedding model. Without this, it defaults to
# using every available CPU core, which can starve the rest of the container (API responsiveness,
# Qdrant, and on small VPS instances even the host) during large batch uploads/reindexes. Default
# leaves at least 1 core free.
EMBEDDING_THREADS = int(os.getenv("EMBEDDING_THREADS", str(max(1, cpu_count() - 1))))

# Only file types extract_text() genuinely knows how to handle. Anything else (pptx, docx, xlsx,
# zip, images, ...) is binary and would otherwise get blindly decoded as text ("errors=ignore"),
# producing huge garbage content that still gets chunked/embedded — wasting resources and, for
# large binary files, capable of spiking memory/CPU hard enough to make the whole thing crawl.
ALLOWED_UPLOAD_EXTENSIONS = {
    ext.strip().lower()
    for ext in os.getenv("ALLOWED_UPLOAD_EXTENSIONS", ".pdf,.txt,.md,.csv,.json,.log").split(",")
    if ext.strip()
}

# Hard cap per uploaded file. A single oversized file (accidental wrong upload, huge scanned PDF,
# etc.) shouldn't be able to balloon memory/CPU usage during extraction+embedding.
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")) * 1024 * 1024

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(SQLITE_PATH).parent.mkdir(parents=True, exist_ok=True)
