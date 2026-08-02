import re

from ardrag import core, db
from ardrag.classify import _get_client

_CITATION_MARKER_RE = re.compile(r"\s*\[Source\s+\d+\]")

MAX_HISTORY_MESSAGES = 12
CONTEXT_CHARS_PER_SOURCE = 1500
# Higher than the MCP/document-search default (DEFAULT_TOP_K=8) on purpose: a chat question that
# names two or more products by an abbreviated/informal code (e.g. "iap821 vs 822" instead of the
# indexed "IAP300-821-PE") scores those chunks lower in the hybrid ranking, and an 8-slot budget
# can end up with zero chunks from one of the documents actually being asked about — the model
# then (correctly, per its instructions) says that document isn't in context, even though it's in
# the knowledge base. A wider net makes that starvation far less likely for comparison-style asks.
CHAT_TOP_K = 16

SYSTEM_PROMPT = (
    "You are ArdRAG Intelligence, an assistant that answers questions using only the document "
    "excerpts provided below as context — this is the user's own private document knowledge base. "
    "The source documents are already shown to the user separately below your answer, so do NOT "
    "add inline citation markers like [Source N] in your response text — just answer naturally. "
    "If the context doesn't contain the answer, say so plainly instead of guessing. Keep answers "
    "concise and use markdown (bold, bullet lists, fenced code blocks for commands/config) where "
    "it helps readability."
)


class ChatError(Exception):
    pass


def _build_context(sources: list[dict]) -> str:
    if not sources:
        return "(no matching documents found)"
    blocks = []
    for i, s in enumerate(sources, start=1):
        text = (s["text"] or "")[:CONTEXT_CHARS_PER_SOURCE]
        blocks.append(f"[Source {i}] (from \"{s['doc_name']}\")\n{text}")
    return "\n\n".join(blocks)


def answer(query: str, history: list[dict], top_k: int = None) -> dict:
    """Runs a hybrid RAG search for `query`, then asks the configured classification-API model
    (Settings > Classification, same OpenAI-compatible client used for document classification) to
    answer grounded in the retrieved chunks. `history` is prior turns as
    [{"role": "user"|"assistant", "content": str}, ...], most recent last.

    Raises ChatError if no API key is configured or the call fails.
    """
    settings = db.get_settings()
    if not settings.deepseek_api_key:
        raise ChatError("Classification API key is not configured — set it in Settings > AI Configuration first.")

    sources = core.search(query, top_k=top_k or CHAT_TOP_K)
    context = _build_context(sources)

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"}]
    messages.extend(history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": query})

    client = _get_client(settings.deepseek_api_key, settings.classify_base_url)
    try:
        # No max_tokens cap — let the model run until it finishes naturally or hits the
        # provider's own hard limit. A locally-imposed cap was truncating long comparison-table
        # answers (see finish_reason == "length" handling below, kept as a safety net for
        # whatever ceiling the provider itself enforces).
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            temperature=0.3,
        )
    except Exception as e:
        raise ChatError(f"Chat API call failed: {e}") from e

    if response.usage:
        db.log_deepseek_usage(
            response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens
        )

    choice = response.choices[0]
    content = (choice.message.content or "").strip()
    if not content:
        raise ChatError("Empty response from the chat model.")

    # Safety net for when the model ignores the system prompt's "no inline citations" instruction
    # anyway (observed occasionally) — the source chips rendered below the message are the actual
    # citation mechanism now, so a stray "[Source 6]" in the body is just noise.
    content = _CITATION_MARKER_RE.sub("", content)

    # finish_reason == "length" means the model hit max_tokens mid-answer (e.g. a long comparison
    # table) rather than finishing naturally — surface that instead of silently handing back a
    # truncated response, since the missing tail (a broken markdown table row, a cut-off
    # sentence) reads as a rendering bug otherwise.
    if choice.finish_reason == "length":
        content += "\n\n_(Response was cut short at the length limit — ask a follow-up to continue, or narrow the question.)_"

    return {"content": content, "sources": sources}
