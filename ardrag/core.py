import hashlib
import uuid

from fastembed import SparseTextEmbedding, TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from tokenizers import Tokenizer

from ardrag import db
from ardrag.config import EMBEDDING_THREADS, QDRANT_COLLECTION, QDRANT_URL, SUPPORTED_EMBEDDING_MODELS

_client = QdrantClient(url=QDRANT_URL)

# BM25 is a pure statistical (non-neural) sparse embedding — no model weights to run inference
# on, so it's essentially free on CPU. It captures exact-term matches (part numbers, model codes)
# that the dense embedding can blur together for near-identical product datasheets.
SPARSE_MODEL_NAME = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Cache of loaded embedder instances, keyed by model name — loading is expensive (downloads +
# initializes an onnxruntime session), so we keep whichever models have been used around instead
# of reloading on every call. Only one is "active" per current settings at a time in practice.
_embedder_cache: dict[str, TextEmbedding] = {}
_sparse_embedder: SparseTextEmbedding = None

# Separate, truncation-disabled clone of each model's tokenizer, used only for measuring the true
# token count of a piece of text (see count_tokens()). FastEmbed's own TextEmbedding.token_count()
# looks like it would do this, but it silently caps at the model's max sequence length (e.g. always
# returns exactly 512 for anything ≥512 real tokens) since it shares the same tokenizer config the
# embedding pipeline uses to truncate — useless for detecting "how much too long is this". Cloning
# the tokenizer (rather than toggling truncation on the live one in-place) avoids any chance of a
# concurrent embed() call on another thread momentarily running without truncation.
_counting_tokenizer_cache: dict[str, Tokenizer] = {}


def _get_counting_tokenizer(model_name: str) -> Tokenizer:
    if model_name not in _counting_tokenizer_cache:
        embedder = _get_embedder(model_name)
        clone = Tokenizer.from_str(embedder.model.tokenizer.to_str())
        clone.no_truncation()
        _counting_tokenizer_cache[model_name] = clone
    return _counting_tokenizer_cache[model_name]


def _get_embedder(model_name: str) -> TextEmbedding:
    if model_name not in _embedder_cache:
        # `threads` caps onnxruntime's intra-op thread pool. Without this, onnxruntime defaults
        # to using every available CPU core, which can starve the rest of the container (and on
        # a small VPS, the host itself) during large batch uploads/reindexes.
        _embedder_cache[model_name] = TextEmbedding(model_name=model_name, threads=EMBEDDING_THREADS)
    return _embedder_cache[model_name]


def _get_sparse_embedder() -> SparseTextEmbedding:
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME, threads=EMBEDDING_THREADS)
    return _sparse_embedder


def vector_size_for(settings: "db.Settings") -> int:
    if settings.embedding_provider == "api":
        if not settings.embedding_api_dimension:
            raise ValueError("API embedding provider has no measured dimension — save it in Settings first.")
        return settings.embedding_api_dimension
    if settings.embedding_model not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(f"Unsupported embedding model: {settings.embedding_model}")
    return SUPPORTED_EMBEDDING_MODELS[settings.embedding_model]


def probe_api_embedding_dimension(base_url: str, api_key: str, model: str) -> int:
    """Makes one live test call to an OpenAI-compatible embeddings endpoint and returns the
    vector length — used to validate a new API embedding config and auto-detect its dimension
    (Qdrant needs an exact size up front; asking the user to know/enter it isn't necessary)."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.embeddings.create(model=model, input=["dimension probe"])
    return len(response.data[0].embedding)


def _vectors_config(vector_size: int):
    return {DENSE_VECTOR_NAME: qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)}


def _sparse_vectors_config():
    return {SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()}


def ensure_collection() -> None:
    settings = db.get_settings()
    vector_size = vector_size_for(settings)
    if _client.collection_exists(QDRANT_COLLECTION):
        return
    try:
        _client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=_vectors_config(vector_size),
            sparse_vectors_config=_sparse_vectors_config(),
        )
    except UnexpectedResponse as e:
        if e.status_code != 409:
            raise


def recreate_collection_for_model(settings: "db.Settings") -> None:
    """Drop and recreate the collection with the vector size required by the current embedding
    config. Destroys all existing vectors — caller is expected to trigger a reindex afterwards."""
    vector_size = vector_size_for(settings)
    if _client.collection_exists(QDRANT_COLLECTION):
        _client.delete_collection(QDRANT_COLLECTION)
    _client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=_vectors_config(vector_size),
        sparse_vectors_config=_sparse_vectors_config(),
    )


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def count_tokens(text: str, settings: "db.Settings") -> int:
    """Real, untruncated token count via a dedicated clone of the active local model's tokenizer,
    or a char/4 approximation for an API provider (no generic way to know an arbitrary
    OpenAI-compatible endpoint's tokenizer)."""
    if settings.embedding_provider == "api":
        return max(1, len(text) // 4)
    return len(_get_counting_tokenizer(settings.embedding_model).encode(text).ids)


def _split_to_fit(doc_name: str, chunk: str, settings: "db.Settings", max_tokens: int) -> list[str]:
    """Repeatedly halves `chunk` until every piece — combined with the doc_name prefix
    index_document() adds before embedding — fits within max_tokens. Pieces at or under 20 chars
    are left alone regardless (guards against looping forever on a max_tokens set absurdly low,
    or on content with no natural split points)."""
    pieces = [chunk]
    while True:
        oversized_idx = [
            i for i, p in enumerate(pieces)
            if len(p) > 20 and count_tokens(f"{doc_name}\n\n{p}", settings) > max_tokens
        ]
        if not oversized_idx:
            return pieces
        new_pieces = []
        for i, p in enumerate(pieces):
            if i in oversized_idx:
                mid = len(p) // 2
                new_pieces.extend([p[:mid], p[mid:]])
            else:
                new_pieces.append(p)
        pieces = new_pieces


def embed_texts(texts: list[str], settings: "db.Settings") -> list[list[float]]:
    if settings.embedding_provider == "api":
        client = OpenAI(base_url=settings.embedding_api_base_url, api_key=settings.embedding_api_key)
        response = client.embeddings.create(model=settings.embedding_api_model, input=texts)
        return [d.embedding for d in response.data]
    embedder = _get_embedder(settings.embedding_model)
    return [vec.tolist() for vec in embedder.embed(texts)]


def embed_sparse(texts: list[str]) -> list[qmodels.SparseVector]:
    embedder = _get_sparse_embedder()
    return [
        qmodels.SparseVector(indices=vec.indices.tolist(), values=vec.values.tolist())
        for vec in embedder.embed(texts)
    ]


def delete_document_chunks(doc_name: str) -> None:
    _client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="doc_name", match=qmodels.MatchValue(value=doc_name))]
            )
        ),
    )


def index_document(doc_name: str, text: str) -> int:
    ensure_collection()
    delete_document_chunks(doc_name)

    settings = db.get_settings()
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return 0

    # chunk_size/overlap (above) are a char-based target, not a guarantee — a chunk whose real
    # token count (with the doc_name prefix added below) exceeds what the active embedding model
    # actually accepts would otherwise get silently truncated by the model's own tokenizer (most
    # embedding backends truncate rather than error), quietly losing the tail of that chunk. Split
    # any offender further right here, pre-emptively, before it's ever sent to the embedder.
    if settings.max_embedding_tokens:
        chunks = [
            piece
            for chunk in chunks
            for piece in _split_to_fit(doc_name, chunk, settings, settings.max_embedding_tokens)
        ]

    # Prepend the document name to what gets embedded (not to the stored/returned text) so the
    # product/file identity contributes to the vector — this matters a lot when many documents
    # describe near-identical products and only differ by model name.
    embed_inputs = [f"{doc_name}\n\n{chunk}" for chunk in chunks]
    dense_vectors = embed_texts(embed_inputs, settings)
    sparse_vectors = embed_sparse(embed_inputs)

    points = [
        qmodels.PointStruct(
            id=str(uuid.uuid4()),
            vector={DENSE_VECTOR_NAME: dense_vec, SPARSE_VECTOR_NAME: sparse_vec},
            payload={"doc_name": doc_name, "text": chunk, "chunk_index": i},
        )
        for i, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense_vectors, sparse_vectors))
    ]
    _client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def search(
    query: str,
    top_k: int,
    vendor: str = None,
    doc_type: str = None,
    tag: str = None,
) -> list[dict]:
    ensure_collection()
    settings = db.get_settings()
    dense_query = embed_texts([query], settings)[0]
    sparse_query = embed_sparse([query])[0]

    # Vendor/doc_type/tag aren't stored in Qdrant's payload (only doc_name/text/chunk_index are),
    # so filtering by them means resolving the allowed doc_names from SQLite first, then
    # over-fetching from Qdrant and filtering down to top_k in Python. This avoids needing a
    # reindex whenever the metadata schema changes.
    allowed_names = None
    if vendor or doc_type or tag:
        allowed_names = {d.name for d in db.list_documents(vendor=vendor, doc_type=doc_type, tag=tag)}
        if not allowed_names:
            return []

    fetch_k = min(top_k * 5, 200) if allowed_names is not None else max(top_k * 2, 20)

    results = _client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            qmodels.Prefetch(query=dense_query, using=DENSE_VECTOR_NAME, limit=fetch_k),
            qmodels.Prefetch(query=sparse_query, using=SPARSE_VECTOR_NAME, limit=fetch_k),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=fetch_k,
    ).points

    out = []
    for r in results:
        doc_name = r.payload.get("doc_name")
        if allowed_names is not None and doc_name not in allowed_names:
            continue
        out.append({"doc_name": doc_name, "text": r.payload.get("text"), "score": r.score})
        if len(out) >= top_k:
            break
    return out


def get_qdrant_health() -> dict:
    try:
        info = _client.get_collection(QDRANT_COLLECTION)
        vectors_cfg = info.config.params.vectors
        dense_cfg = vectors_cfg.get(DENSE_VECTOR_NAME) if isinstance(vectors_cfg, dict) else vectors_cfg
        return {
            "status": "ok",
            "collection_status": str(info.status),
            "points_count": info.points_count,
            "vector_size": dense_cfg.size if dense_cfg else None,
            "distance": str(dense_cfg.distance) if dense_cfg else None,
            "hybrid": bool(info.config.params.sparse_vectors),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
