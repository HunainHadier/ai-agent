import os
from typing import Iterable

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency fallback
    tiktoken = None


def _encoding():
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    encoding = _encoding()
    if encoding is None:
        # Fallback heuristic when tiktoken is unavailable.
        return max(1, len(text) // 4)
    return len(encoding.encode(text))


def count_message_tokens(messages: Iterable[dict[str, str]]) -> int:
    total = 0
    for message in messages:
        total += count_tokens(message.get("content", ""))
    return total
