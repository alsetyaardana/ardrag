import json
import re
import time

from openai import OpenAI

from ardrag import db

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
EXCERPT_CHARS = 3000
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2

_client_cache: dict[str, OpenAI] = {}


class ClassifyError(Exception):
    pass


def _get_client(api_key: str) -> OpenAI:
    if api_key not in _client_cache:
        _client_cache[api_key] = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    return _client_cache[api_key]


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ClassifyError(f"No JSON object found in model response: {raw[:200]}")
    return json.loads(match.group(0))


def _call_once(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        raise ClassifyError(f"DeepSeek API call failed: {e}") from e

    if response.usage:
        db.log_deepseek_usage(
            response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens
        )

    choice = response.choices[0]
    raw = choice.message.content or ""
    if not raw.strip():
        raise ClassifyError(f"Empty response from DeepSeek (finish_reason={choice.finish_reason})")
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, ClassifyError) as e:
        raise ClassifyError(
            f"Could not parse DeepSeek response as JSON (finish_reason={choice.finish_reason}): {e}"
        ) from e


def classify_document(filename: str, text: str) -> dict:
    """Ask DeepSeek to infer {vendor, doc_type, tags} for a document.

    Raises ClassifyError if no API key is configured or all attempts fail. DeepSeek occasionally
    returns an empty/truncated response for no discernible reason (observed even for a document
    that succeeded on a prior identical call) — retried a couple of times before giving up.
    """
    settings = db.get_settings()
    if not settings.deepseek_api_key:
        raise ClassifyError("DeepSeek API key is not configured — set it in Settings first.")

    known_vendors = sorted({f["vendor"] for f in db.list_folders()} - {db.UNCATEGORIZED_VENDOR})
    known_types = sorted({f["doc_type"] for f in db.list_folders()} - {db.UNCATEGORIZED_TYPE})
    known_tags = db.list_all_tags()

    excerpt = text.strip()[:EXCERPT_CHARS]

    system_prompt = (
        "You classify business/technical documents for a personal document management system. "
        "Given a filename and an excerpt of its text content, respond with ONLY a JSON object "
        '(no markdown, no explanation) of the form: '
        '{"vendor": "<company/brand name>", "doc_type": "<document category>", "tags": ["tag1", "tag2", ...]}. '
        "Prefer reusing one of the existing vendors/doc_types/tags listed below if it genuinely fits, "
        "to keep the taxonomy consistent — only introduce a new value if none of the existing ones fit. "
        "doc_type should be a short category like 'Datasheet', 'Solution', 'Case Study', 'Company Profile', "
        "'Presentation', 'Whitepaper', etc. tags should be 2-6 short lowercase keywords describing the "
        "product category / topic (e.g. 'switch', 'router', 'access-point', 'server', 'wifi6', 'cloud')."
        f"\n\nExisting vendors: {known_vendors}"
        f"\nExisting doc_types: {known_types}"
        f"\nExisting tags: {known_tags}"
    )
    user_prompt = f"Filename: {filename}\n\nText excerpt:\n{excerpt}"

    client = _get_client(settings.deepseek_api_key)

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            data = _call_once(client, settings.deepseek_model, system_prompt, user_prompt)
            break
        except ClassifyError as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    else:
        raise ClassifyError(f"Failed after {MAX_ATTEMPTS} attempts: {last_error}")

    vendor = str(data.get("vendor") or db.UNCATEGORIZED_VENDOR).strip() or db.UNCATEGORIZED_VENDOR
    doc_type = str(data.get("doc_type") or db.UNCATEGORIZED_TYPE).strip() or db.UNCATEGORIZED_TYPE
    tags = data.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip().lower() for t in tags if str(t).strip()]

    return {"vendor": vendor, "doc_type": doc_type, "tags": tags}
