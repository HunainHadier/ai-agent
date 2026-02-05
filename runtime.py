import importlib.util
import json
import os
import re
from datetime import date
from typing import Any, Iterable

from openai import OpenAI

from app.config import get_openai_api_key, get_openai_model
from app.rag import build_vectorstore, collect_texts, load_vectorstore
from app.storage import (
    agent_file_path,
    agent_vector_dir,
    list_agent_upload_paths,
    normalize_device_id,
)


def _has_vectorstore(device_id: str | int | None, name: str) -> bool:
    vector_dir = agent_vector_dir(device_id, name)
    if not os.path.isdir(vector_dir):
        return False
    return any(
        os.path.isfile(os.path.join(vector_dir, filename))
        for filename in ("index.faiss", "index.pkl")
    )


def _simple_rag_context(files: list[str], query: str, max_chars: int = 4000) -> str:
    if not files:
        return ""
    texts = collect_texts(files)
    if not texts:
        return ""
    query_terms = [term for term in re.split(r"\\W+", query.lower()) if len(term) > 2]
    matches: list[str] = []
    for text in texts:
        text_lower = text.lower()
        if not query_terms or any(term in text_lower for term in query_terms):
            matches.append(text)
    if not matches:
        matches = texts
    combined = "\n\n".join(matches).strip()
    if max_chars > 0 and len(combined) > max_chars:
        combined = combined[:max_chars].rsplit("\n", 1)[0].strip() or combined[:max_chars]
    return combined


_SELECTION_CLIENT: OpenAI | None = None


def _selection_client() -> OpenAI | None:
    global _SELECTION_CLIENT
    if _SELECTION_CLIENT is not None:
        return _SELECTION_CLIENT
    api_key = get_openai_api_key()
    if not api_key:
        return None
    _SELECTION_CLIENT = OpenAI(api_key=api_key)
    return _SELECTION_CLIENT


def _serialize_messages(messages: list[dict[str, str]] | None, limit: int = 6) -> list[dict[str, str]]:
    if not messages:
        return []
    trimmed: list[dict[str, str]] = []
    for item in messages[-limit:]:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        trimmed.append({"role": role or "user", "content": content})
    return trimmed


def _is_arabic_text(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"[\\u0600-\\u06FF]", text))


def _parse_iso_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _iter_body_date_fields(payload: Any) -> Iterable[tuple[str, str]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                yield from _iter_body_date_fields(value)
            elif isinstance(value, str):
                yield str(key), value
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_body_date_fields(item)


def _body_needs_fresh_dates(body: dict[str, Any]) -> bool:
    today = date.today()
    date_keys = ("check", "date", "arrival", "departure")
    for key, value in _iter_body_date_fields(body):
        key_lower = key.lower()
        if not any(token in key_lower for token in date_keys):
            continue
        if not value or "{{" in value:
            return True
        parsed = _parse_iso_date(value)
        if parsed and parsed < today:
            return True
    return False


def _apply_date_values(body: dict[str, Any], values: dict[str, str]) -> None:
    if not body or not values:
        return
    date_start = values.get("date_start")
    date_end = values.get("date_end")
    if not date_start and not date_end:
        return
    for key, value in body.items():
        if not isinstance(key, str):
            continue
        key_lower = key.lower()
        if date_start and any(token in key_lower for token in ("check_in", "checkin", "date_start", "arrival")):
            if value in (None, "", 0) or (isinstance(value, str) and "{{" in value):
                body[key] = date_start
        if date_end and any(token in key_lower for token in ("check_out", "checkout", "date_end", "departure")):
            if value in (None, "", 0) or (isinstance(value, str) and "{{" in value):
                body[key] = date_end


def _llm_select_endpoint(
    requests: list[dict[str, Any]],
    user_input: str,
    messages: list[dict[str, str]] | None = None,
    flow_text: str | None = None,
    allow_methods: set[str] | None = None,
    purpose: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(requests, list) or not requests:
        return None
    client = _selection_client()
    if client is None:
        return None
    endpoints = []
    for req in requests:
        if not isinstance(req, dict):
            continue
        method = str(req.get("method", "GET")).upper()
        if allow_methods and method not in allow_methods:
            continue
        endpoints.append(
            {
                "name": str(req.get("name") or ""),
                "method": method,
                "url": str(req.get("url") or ""),
                "description": str(req.get("description") or ""),
            }
        )
    if not endpoints:
        return None
    prompt = (
        "Select the best API endpoint to call now based on the input and context. "
        "Never select authentication/login/token endpoints; login is handled automatically. "
        "Return JSON with a single key 'endpoint' whose value is the exact endpoint name, "
        "or 'NONE' if no endpoint should be called."
    )
    payload = {
        "purpose": purpose or "auto_select",
        "user_input": user_input,
        "flow": (flow_text or "").strip(),
        "recent_messages": _serialize_messages(messages),
        "endpoints": endpoints,
    }
    try:
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=200,
        )
    except Exception:
        return None
    content = ""
    try:
        content = response.choices[0].message.content or ""
    except Exception:
        content = ""
    endpoint_name = ""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            endpoint_name = str(parsed.get("endpoint") or "").strip()
    except Exception:
        endpoint_name = content.strip()
    if not endpoint_name or endpoint_name.lower() == "none":
        return None
    for req in requests:
        if not isinstance(req, dict):
            continue
        name = str(req.get("name") or "")
        if name and name.lower() == endpoint_name.lower():
            return req
    return None


def _resolve_selection_index_llm(options: list[str], user_input: str) -> int | None:
    if not options or not user_input:
        return None
    client = _selection_client()
    if client is None:
        return None
    options_text = "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)])
    prompt = (
        "Return only the 1-based index of the selected option. "
        "If the selection is unclear, return 0."
    )
    try:
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Options:\n{options_text}\nUser input: {user_input}"},
            ],
            temperature=0,
            max_tokens=3,
        )
    except Exception:
        return None
    content = ""
    try:
        content = response.choices[0].message.content or ""
    except Exception:
        content = ""
    match = re.search(r"\d+", content)
    if not match:
        return None
    idx = int(match.group(0)) - 1
    if 0 <= idx < len(options):
        return idx
    return None


def _selection_index_from_text(text: str) -> int | None:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return None
    if _has_date_like(cleaned):
        return None
    if cleaned.isdigit():
        return int(cleaned) - 1
    match = re.search(r"\d+", cleaned)
    if match:
        return int(match.group(0)) - 1
    return None


def _select_option_value(options: list[str], user_input: str) -> str:
    if not options or not user_input:
        return ""
    idx = _selection_index_from_text(user_input)
    if idx is not None and 0 <= idx < len(options):
        return options[idx]
    cleaned = user_input.strip().lower()
    for opt in options:
        if cleaned == opt.lower():
            return opt
    llm_idx = _resolve_selection_index_llm(options, user_input)
    if llm_idx is not None and 0 <= llm_idx < len(options):
        return options[llm_idx]
    return ""


def _is_services_query(text: str, services_raw: str | None = None, flow_text: str | None = None) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    services = []
    if isinstance(services_raw, str) and services_raw:
        services = [part.strip() for part in re.split(r"[\\n,;]+", services_raw) if part.strip()]
    client = _selection_client()
    if client is None:
        return False
    prompt = (
        "Determine whether the user is asking for a list of services/capabilities "
        "versus requesting a specific task. Reply only with 'services' or 'task'."
    )
    context = (flow_text or "").strip()
    services_block = "\n".join([f"- {item}" for item in services]) if services else ""
    user_text = f"Flow:\n{context}\n\nServices:\n{services_block}\n\nUser:\n{cleaned}"
    try:
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
            max_tokens=2,
        )
    except Exception:
        return False
    content = ""
    try:
        content = response.choices[0].message.content or ""
    except Exception:
        content = ""
    return "service" in content.lower()


def _flow_steps(flow_text: str) -> list[str]:
    steps: list[str] = []
    for part in re.split(r"[\\n]+", flow_text or ""):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            _, rest = part.split(":", 1)
            steps.extend([s.strip() for s in rest.split(";") if s.strip()])
        else:
            steps.extend([s.strip() for s in part.split(";") if s.strip()])
    return steps


def _has_date_like(text: str) -> bool:
    value = (text or "")
    if value:
        value = value.translate(
            str.maketrans(
                "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
                "01234567890123456789",
            )
        )
    if re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", value):
        return True
    if re.search(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", value):
        return True
    lowered = value.lower()
    month = (
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?|"
        r"يناير|فبراير|مارس|أبريل|ابريل|مايو|يونيو|يوليو|أغسطس|اغسطس|"
        r"سبتمبر|أكتوبر|اكتوبر|نوفمبر|ديسمبر)"
    )
    if re.search(rf"\b{month}\s+\d{{1,2}}(?:,?\s+\d{{2,4}})?\b", lowered):
        return True
    if re.search(rf"\b\d{{1,2}}\s+{month}(?:\s+\d{{2,4}})?\b", lowered):
        return True
    return False


def _history_has_date_like(messages: list[dict[str, str]] | None) -> bool:
    if not messages:
        return False
    for item in reversed(messages):
        if item.get("role") != "user":
            continue
        content = str(item.get("content") or "")
        if content and _has_date_like(content):
            return True
    return False


def _sanitize_messages_runtime(messages: list[dict[str, str]] | None) -> list[dict[str, str]] | None:
    if not messages:
        return messages
    cleaned: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        cleaned.append({"role": role or "user", "content": content})
    return cleaned


 


def _extract_last_options(messages: list[dict[str, str]] | None) -> list[str]:
    if not messages:
        return []
    for item in reversed(messages):
        if item.get("role") != "assistant":
            continue
        content = str(item.get("content") or "")
        if not content:
            continue
        options = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(?:-\s*)?Option\s*\d+\s*:\s*(.+)", line, flags=re.I)
            if match:
                options.append(match.group(1).strip())
                continue
            match = re.match(r"\d+\.\s*(.+)", line)
            if match:
                options.append(match.group(1).strip())
        return options
    return []


def _find_last_options_context(messages: list[dict[str, str]] | None) -> tuple[list[str], str, str, str]:
    if not messages:
        return ([], "", "", "")
    assistant_messages: list[str] = []
    for item in messages:
        if item.get("role") != "assistant":
            continue
        content = str(item.get("content") or "").strip()
        if content:
            assistant_messages.append(content)
    if not assistant_messages:
        return ([], "", "", "")
    options_list: list[str] = []
    options_text = ""
    options_index = None
    for idx in range(len(assistant_messages) - 1, -1, -1):
        content = assistant_messages[idx]
        options = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(?:-\s*)?Option\s*\d+\s*:\s*(.+)", line, flags=re.I)
            if match:
                options.append(match.group(1).strip())
                continue
            match = re.match(r"\d+\.\s*(.+)", line)
            if match:
                options.append(match.group(1).strip())
        if options:
            options_list = options
            options_text = content
            options_index = idx
            break
    if options_index is None:
        return ([], "", "", "")
    context_text = ""
    if options_index > 0:
        context_text = assistant_messages[options_index - 1]
    trigger_user = ""
    if messages:
        assistant_seen = 0
        target_assistant = options_index
        for item in messages:
            if item.get("role") == "assistant":
                if assistant_seen == target_assistant:
                    break
                assistant_seen += 1
                continue
            if item.get("role") == "user":
                trigger_user = str(item.get("content") or "").strip()
    return (options_list, options_text, context_text, trigger_user)


def _find_last_options_block(messages: list[dict[str, str]] | None) -> tuple[int, list[str]]:
    if not messages:
        return (-1, [])
    for idx in range(len(messages) - 1, -1, -1):
        item = messages[idx]
        if item.get("role") != "assistant":
            continue
        options = []
        content = str(item.get("content") or "")
        if not content:
            continue
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(?:-\s*)?Option\s*\d+\s*:\s*(.+)", line, flags=re.I)
            if match:
                options.append(match.group(1).strip())
                continue
            match = re.match(r"\d+\.\s*(.+)", line)
            if match:
                options.append(match.group(1).strip())
        if options:
            return (idx, options)
    return (-1, [])


def _find_last_selected_option_before(messages: list[dict[str, str]] | None, before_index: int) -> str:
    if not messages or before_index <= 0:
        return ""
    for idx in range(before_index - 1, -1, -1):
        item = messages[idx]
        if item.get("role") != "assistant":
            continue
        options = []
        content = str(item.get("content") or "")
        if not content:
            continue
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"(?:-\s*)?Option\s*\d+\s*:\s*(.+)", line, flags=re.I)
            if match:
                options.append(match.group(1).strip())
                continue
            match = re.match(r"\d+\.\s*(.+)", line)
            if match:
                options.append(match.group(1).strip())
        if not options:
            continue
        for j in range(idx + 1, before_index):
            if messages[j].get("role") != "user":
                continue
            selected = _select_option_value(options, str(messages[j].get("content") or ""))
            if selected:
                return selected
            break
    return ""


def _options_look_like_fields(options: list[str], context_text: str | None = None) -> bool:
    if not options:
        return False
    client = _selection_client()
    if client is None:
        return False
    prompt = (
        "You are classifying whether a list of items are form fields the user should fill "
        "or selectable choices. Reply only with 'fields' or 'choices'."
    )
    options_text = "\n".join([f"- {opt}" for opt in options])
    context = (context_text or "").strip()
    user_text = f"Context:\n{context}\n\nItems:\n{options_text}"
    try:
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
            max_tokens=2,
        )
    except Exception:
        return False
    content = ""
    try:
        content = response.choices[0].message.content or ""
    except Exception:
        content = ""
    return "field" in content.lower()


def _options_are_services(options: list[str], services_raw: str | None) -> bool:
    if not options or not services_raw:
        return False
    services = [part.strip() for part in re.split(r"[\\n,;]+", services_raw) if part.strip()]
    services_lower = {s.lower() for s in services}
    if not services_lower:
        return False
    return all(opt.strip().lower() in services_lower for opt in options)


def _assistant_requests_fields(messages: list[dict[str, str]] | None) -> bool:
    if not messages:
        return False
    last_assistant = ""
    for item in reversed(messages):
        if item.get("role") != "assistant":
            continue
        last_assistant = str(item.get("content") or "").strip()
        if last_assistant:
            break
    if not last_assistant:
        return False
    options = []
    for line in last_assistant.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"(?:-\\s*)?Option\\s*\\d+\\s*:\\s*(.+)", line, flags=re.I)
        if match:
            options.append(match.group(1).strip())
            continue
        match = re.match(r"\\d+\\.\\s*(.+)", line)
        if match:
            options.append(match.group(1).strip())
    if options:
        return _options_look_like_fields(options, last_assistant)
    client = _selection_client()
    if client is None:
        return False
    prompt = (
        "Classify if the assistant message is asking the user to fill in fields. "
        "Reply only with 'fields' or 'choices'."
    )
    try:
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": last_assistant},
            ],
            temperature=0,
            max_tokens=2,
        )
    except Exception:
        return False
    content = ""
    try:
        content = response.choices[0].message.content or ""
    except Exception:
        content = ""
    return "field" in content.lower()


def _parse_query_placeholders(url: str) -> list[str]:
    if not url or "?" not in url:
        return []
    query = url.split("?", 1)[1]
    params = []
    for pair in query.split("&"):
        if not pair:
            continue
        if "=" in pair:
            key, _ = pair.split("=", 1)
        else:
            key = pair
        key = key.strip()
        if not key:
            continue
        params.append(key)
    return params


def _pick_query_param_key(keys: list[str], endpoint_name: str, flow_text: str, selected: str) -> str | None:
    if not keys:
        return None
    if len(keys) == 1:
        return keys[0]
    client = _selection_client()
    if client is None:
        return max(keys, key=len)
    prompt = (
        "Choose which query parameter should receive the selected value. "
        "Prefer keys that represent the user's selection. "
        "Reply only with the parameter key, or 'none' if unsure."
    )
    user_text = (
        f"Endpoint: {endpoint_name}\nFlow: {flow_text}\n"
        f"Keys: {', '.join(keys)}\nSelected: {selected}"
    )
    try:
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
            max_tokens=5,
        )
    except Exception:
        return max(keys, key=len)
    content = ""
    try:
        content = response.choices[0].message.content or ""
    except Exception:
        content = ""
    choice = content.strip().split()[0] if content else ""
    if choice in keys:
        return choice
    return max(keys, key=len)


def _build_query_for_selection(url: str, endpoint_name: str, flow_text: str, selected: str) -> dict[str, str] | None:
    keys = _parse_query_placeholders(url)
    if not keys or not selected:
        return None
    key = _pick_query_param_key(keys, endpoint_name, flow_text, selected)
    if not key:
        return None
    return {key: selected}


def _extract_options_from_text(content: str) -> list[str]:
    if not content:
        return []
    options: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"(?:-\\s*)?Option\\s*\\d+\\s*:\\s*(.+)", line, flags=re.I)
        if match:
            options.append(match.group(1).strip())
            continue
        match = re.match(r"\\d+\\.\\s*(.+)", line)
        if match:
            options.append(match.group(1).strip())
    return options


def _selection_history(messages: list[dict[str, str]] | None) -> list[str]:
    if not messages:
        return []
    selections: list[str] = []
    for idx, item in enumerate(messages):
        if item.get("role") != "assistant":
            continue
        content = str(item.get("content") or "")
        if not content:
            continue
        options = _extract_options_from_text(content)
        if not options:
            continue
        for j in range(idx + 1, len(messages)):
            if messages[j].get("role") != "user":
                continue
            selected = _select_option_value(options, str(messages[j].get("content") or ""))
            if selected:
                selections.append(selected)
            break
    return selections


def _match_item_id(item: dict[str, Any], selected: str) -> Any:
    if not isinstance(item, dict):
        return None
    selected_norm = str(selected).strip().lower()
    for value in item.values():
        if isinstance(value, str) and value.strip().lower() == selected_norm:
            if "id" in item:
                return item.get("id")
            for key in item:
                if "id" in key.lower():
                    return item.get(key)
            break
    if "id" in item:
        return item.get("id")
    for key in item:
        if "id" in key.lower():
            return item.get(key)
    return None


def _resolve_selection_id(
    postman_call: Any,
    endpoint: dict[str, Any],
    parse_payload: Any,
    selected: str,
    query: dict[str, str] | None,
) -> Any:
    if not callable(postman_call):
        return None
    endpoint_name = str(endpoint.get("name") or "")
    if hasattr(postman_call, "invoke"):
        raw = postman_call.invoke({"endpoint_name": endpoint_name, "query": query})
    else:
        raw = postman_call(endpoint_name, query=query)
    payload = parse_payload(raw) if callable(parse_payload) else raw
    data = payload.get("data") if isinstance(payload, dict) else None
    if data is None:
        data = payload
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            match = _match_item_id(item, selected)
            if match is not None:
                return match
    if isinstance(data, dict):
        match = _match_item_id(data, selected)
        if match is not None:
            return match
    return None


def _flow_step_endpoints(
    requests: list[dict[str, Any]], flow_text: str
) -> list[tuple[str, dict[str, Any] | None]]:
    steps = _flow_steps(flow_text)
    if not steps:
        return []
    client = _selection_client()
    if client is None:
        return [(step, None) for step in steps]
    endpoints = []
    for req in requests:
        if not isinstance(req, dict):
            continue
        endpoints.append(
            {
                "name": str(req.get("name") or ""),
                "method": str(req.get("method") or "GET").upper(),
                "url": str(req.get("url") or ""),
                "description": str(req.get("description") or ""),
            }
        )
    prompt = (
        "Map each flow step to the best matching endpoint name. "
        "Return JSON where keys are the flow steps and values are endpoint names or 'NONE'."
    )
    try:
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"steps": steps, "endpoints": endpoints}, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=800,
        )
    except Exception:
        return [(step, None) for step in steps]
    content = ""
    try:
        content = response.choices[0].message.content or ""
    except Exception:
        content = ""
    try:
        mapping = json.loads(content)
    except Exception:
        mapping = {}
    mapped: list[tuple[str, dict[str, Any] | None]] = []
    for step in steps:
        endpoint = None
        name = ""
        if isinstance(mapping, dict):
            name = str(mapping.get(step) or "").strip()
        if name and name.lower() != "none":
            for req in requests:
                if not isinstance(req, dict):
                    continue
                req_name = str(req.get("name") or "")
                if req_name and req_name.lower() == name.lower():
                    endpoint = req
                    break
        mapped.append((step, endpoint))
    return mapped


def _extract_generic_values(messages: list[dict[str, str]] | None) -> dict[str, str]:
    if not messages:
        return {}
    name = ""
    date_start = ""
    date_end = ""
    date_pattern = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")
    for item in messages:
        if item.get("role") != "user":
            continue
        content = str(item.get("content") or "")
        if not content:
            continue
        dates = date_pattern.findall(content)
        if len(dates) >= 2:
            date_start, date_end = dates[0], dates[1]
            prefix = content.split(dates[0], 1)[0].strip(" ,;:-")
            if prefix:
                name = prefix
    values = {}
    if name:
        values["name"] = name
    if date_start:
        values["date_start"] = date_start
    if date_end:
        values["date_end"] = date_end
    return values


def _map_body_fields_with_values(
    body: dict[str, Any], values: dict[str, str], flow_text: str
) -> dict[str, str]:
    if not values or not body:
        return {}
    client = _selection_client()
    if client is None:
        return {}
    keys = list(body.keys())
    prompt = (
        "Map provided values to request body fields if appropriate. "
        "Return JSON mapping of body key to one of the provided value keys. "
        "Use null if no suitable value."
    )
    user_text = f"Flow:\n{flow_text}\n\nBody keys:\n{keys}\n\nValues:\n{values}"
    try:
        response = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
            max_tokens=200,
        )
    except Exception:
        return {}
    content = ""
    try:
        content = response.choices[0].message.content or ""
    except Exception:
        content = ""
    try:
        parsed = json.loads(content)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    mapping: dict[str, str] = {}
    for key, value_key in parsed.items():
        if key in body and isinstance(value_key, str) and value_key in values:
            mapping[key] = values[value_key]
    return mapping


def _auto_postman_get(module: Any, user_input: str) -> str:
    if not user_input:
        return ""
    requests = getattr(module, "POSTMAN_REQUESTS", None)
    if not isinstance(requests, list) or not requests:
        return ""
    best = _llm_select_endpoint(
        requests,
        user_input,
        allow_methods={"GET"},
        purpose="auto_get",
    )
    if not best:
        return ""
    postman_call = getattr(module, "postman_call", None)
    if not callable(postman_call):
        return ""
    raw = postman_call(str(best.get("name")))
    parse_payload = getattr(module, "_parse_postman_payload", None)
    format_list = getattr(module, "_format_postman_list", None)
    if callable(parse_payload) and callable(format_list):
        try:
            payload = parse_payload(raw)
            formatted = format_list(payload, user_input)
            if formatted:
                return formatted
        except Exception:
            pass
    return str(raw or "").strip()


def _auto_postman_from_flow(
    module: Any,
    messages: list[dict[str, str]] | None,
    user_input: str,
) -> str:
    if not user_input:
        return ""
    requests = getattr(module, "POSTMAN_REQUESTS", None)
    if not isinstance(requests, list) or not requests:
        return ""
    flow_text = ""
    agent_context = getattr(module, "AGENT_CONTEXT", None)
    services_raw = None
    if isinstance(agent_context, dict):
        flow_text = str(agent_context.get("flow") or "")
        services_raw = agent_context.get("services")
    if not flow_text and messages:
        for item in messages:
            if item.get("role") == "system":
                content = str(item.get("content") or "")
                if content:
                    flow_text = content
                    break
    if _is_services_query(user_input, services_raw, flow_text):
        return ""
    endpoint = _llm_select_endpoint(
        requests,
        user_input,
        messages=messages,
        flow_text=flow_text,
        allow_methods={"GET"},
        purpose="flow_get",
    )
    if not endpoint:
        return ""
    postman_call = getattr(module, "postman_call", None)
    if not callable(postman_call):
        return ""
    raw = postman_call(str(endpoint.get("name")))
    parse_payload = getattr(module, "_parse_postman_payload", None)
    format_list = getattr(module, "_format_postman_list", None)
    if callable(parse_payload) and callable(format_list):
        try:
            payload = parse_payload(raw)
            formatted = format_list(payload, user_input)
            if formatted:
                return formatted
        except Exception:
            pass
    return str(raw or "").strip()

def _auto_postman_from_flow_selection(module: Any, messages: list[dict[str, str]] | None, user_input: str) -> str:
    if not user_input:
        return ""
    if not messages:
        return ""
    requests = getattr(module, "POSTMAN_REQUESTS", None)
    if not isinstance(requests, list) or not requests:
        return ""
    cleaned = user_input.strip()
    context_message = ""
    finder = getattr(module, "_find_last_options_context", None)
    if callable(finder):
        try:
            opts = finder(messages)
            if isinstance(opts, tuple) and len(opts) >= 3:
                options, _, context_message = opts[0], opts[1], opts[2]
            else:
                options = opts or []
        except Exception:
            options = []
    else:
        options, _, context_message, _ = _find_last_options_context(messages)
    if not options:
        return ""
    is_selection = _selection_index_from_text(cleaned) is not None
    if not is_selection:
        is_selection = len(cleaned) <= 40 and not _has_date_like(cleaned)
    if not is_selection:
        return ""
    selected = _select_option_value(options, cleaned)
    if not selected:
        return ""
    current_options_index, _ = _find_last_options_block(messages)
    agent_context = getattr(module, "AGENT_CONTEXT", None)
    services_raw = None
    flow_text = ""
    if isinstance(agent_context, dict):
        flow_text = str(agent_context.get("flow") or "")
        services_raw = agent_context.get("services")
    services = []
    if isinstance(services_raw, str):
        services = [part.strip() for part in re.split(r"[\\n,;]+", services_raw) if part.strip()]
    services_lower = {s.lower() for s in services}
    options_are_services = bool(options) and services_lower and all(opt.strip().lower() in services_lower for opt in options)
    if options_are_services:
        return ""
    selection_input = f"User selected: {selected}\nUser input: {user_input}"
    endpoint = _llm_select_endpoint(
        requests,
        selection_input,
        messages=messages,
        flow_text=flow_text,
        allow_methods=None,
        purpose="flow_selection",
    )
    if not endpoint and flow_text:
        flow_lower = flow_text.lower()
        if "rate plan" in flow_lower or "rate plans" in flow_lower:
            for req in requests:
                name = str(req.get("name") or "").lower()
                if "rate plan" in name or "rate plans" in name:
                    endpoint = req
                    break
    if not endpoint:
        return ""
    postman_call = getattr(module, "postman_call", None)
    if not callable(postman_call):
        return ""
    url = str(endpoint.get("url") or "")
    endpoint_name = str(endpoint.get("name") or "")
    query = _build_query_for_selection(url, endpoint_name, flow_text, selected)
    if str(endpoint.get("method", "GET")).upper() == "POST":
        parse_payload = getattr(module, "_parse_postman_payload", None)
        body = {}
        raw_body = endpoint.get("body")
        if isinstance(raw_body, str) and raw_body.strip():
            try:
                body = json.loads(raw_body)
            except Exception:
                body = {}
        if not isinstance(body, dict):
            body = {}
        values = _extract_generic_values(messages)
        mapped = _map_body_fields_with_values(body, values, flow_text)
        for key, value in mapped.items():
            current = body.get(key)
            if current in (None, "", 0) or (isinstance(current, str) and "{{" in current):
                body[key] = value
        _apply_date_values(body, values)
        if not (values.get("date_start") and values.get("date_end")) and _body_needs_fresh_dates(body):
            if _is_arabic_text(user_input):
                return "من فضلك أدخل تاريخي تسجيل الوصول والمغادرة بصيغة YYYY-MM-DD (مثال: 2025-12-24 و 2025-12-30)."
            return "Please provide the check-in and check-out dates in YYYY-MM-DD format (e.g., 2025-12-24 and 2025-12-30)."
        target_container = body
        if isinstance(body.get("lines"), list):
            lines = body.get("lines") or []
            if lines and isinstance(lines[0], dict):
                target_container = lines[0]
            else:
                lines = [{}]
                target_container = lines[0]
                body["lines"] = lines
        flow_map = _flow_step_endpoints(requests, flow_text)
        current_idx = -1
        for idx, (_, mapped_endpoint) in enumerate(flow_map):
            if not mapped_endpoint:
                continue
            if str(mapped_endpoint.get("name") or "") == endpoint_name and str(mapped_endpoint.get("method", "")).upper() == "POST":
                current_idx = idx
                break
        prior_get_endpoints = []
        if current_idx >= 0:
            for _, mapped_endpoint in flow_map[:current_idx]:
                if not mapped_endpoint:
                    continue
                if str(mapped_endpoint.get("method", "GET")).upper() == "GET":
                    prior_get_endpoints.append(mapped_endpoint)
        selections = _selection_history(messages)
        if prior_get_endpoints:
            selection_slice = selections[-len(prior_get_endpoints):]
            resolved_ids: list[Any] = []
            for idx, req_endpoint in enumerate(prior_get_endpoints):
                if idx >= len(selection_slice):
                    continue
                selection_value = selection_slice[idx]
                dependency_value = selection_slice[idx - 1] if idx > 0 else selection_value
                dep_query = _build_query_for_selection(
                    str(req_endpoint.get("url") or ""),
                    str(req_endpoint.get("name") or ""),
                    flow_text,
                    dependency_value,
                )
                resolved = _resolve_selection_id(
                    postman_call, req_endpoint, parse_payload, selection_value, dep_query
                )
                if resolved is not None:
                    resolved_ids.append(resolved)
            id_keys = [k for k in target_container if "id" in str(k).lower()]
            for key, value in zip(id_keys, resolved_ids):
                current = target_container.get(key)
                if current in (None, "", 0) or (isinstance(current, str) and "{{" in current):
                    target_container[key] = value
        if hasattr(postman_call, "invoke"):
            raw = postman_call.invoke({"endpoint_name": endpoint_name, "json_body": body})
        else:
            raw = postman_call(endpoint_name, json_body=body)
    else:
        if hasattr(postman_call, "invoke"):
            raw = postman_call.invoke({"endpoint_name": endpoint_name, "query": query})
        else:
            raw = postman_call(endpoint_name, query=query)
    parse_payload = getattr(module, "_parse_postman_payload", None)
    format_list = getattr(module, "_format_postman_list", None)
    if callable(parse_payload) and callable(format_list):
        try:
            payload = parse_payload(raw)
            formatted = format_list(payload, user_input)
            if formatted:
                return formatted
        except Exception:
            pass
    return str(raw or "").strip()


def _maybe_add_rag_context(
    module: Any,
    device_id: str | int | None,
    name: str,
    user_input: str,
    messages: list[dict[str, str]] | None,
) -> list[dict[str, str]] | None:
    if not user_input:
        return messages
    uploads = list_agent_upload_paths(device_id, name)
    if not _has_vectorstore(device_id, name):
        if uploads:
            texts = collect_texts(uploads)
            if texts:
                try:
                    build_vectorstore(name, texts, device_id)
                except Exception:
                    pass
        if not _has_vectorstore(device_id, name) and not uploads:
            return messages
    try:
        store = load_vectorstore(name, device_id)
    except Exception:
        store = None

    def _runtime_rag_search(query: str) -> str:
        """Search across uploaded files and Postman context."""
        if store is None:
            return _simple_rag_context(uploads, query)
        docs = store.similarity_search(query, k=4)
        return "\n\n".join([doc.page_content for doc in docs])

    if hasattr(module, "rag_search"):
        rag_tool = module.rag_search
        if hasattr(rag_tool, "func"):
            rag_tool.func = _runtime_rag_search
        else:
            module.rag_search = _runtime_rag_search

    rag_text = ""
    if store is not None:
        try:
            rag_text = _runtime_rag_search(user_input)
        except Exception:
            rag_text = ""
    if not rag_text and uploads:
        rag_text = _simple_rag_context(uploads, user_input)
    rag_text = str(rag_text or "").strip()
    if not rag_text:
        return messages
    if messages:
        for item in messages:
            if item.get("role") == "system" and "Knowledge base context" in (item.get("content") or ""):
                return messages
    context_message = {"role": "system", "content": f"Knowledge base context:\n{rag_text}"}
    if messages:
        return [context_message] + list(messages)
    return [context_message, {"role": "user", "content": user_input}]


def _module_name(device_id: str | int | None, name: str) -> str:
    safe_device = normalize_device_id(device_id)
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return f"agent_{safe_device}_{safe_name}"


def load_agent_module(device_id: str | int | None, name: str) -> Any:
    agent_path = agent_file_path(device_id, name)
    if not os.path.isfile(agent_path):
        raise FileNotFoundError(f"Agent file not found: {agent_path}")
    spec = importlib.util.spec_from_file_location(_module_name(device_id, name), agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load agent module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_agent(
    device_id: str | int | None,
    name: str,
    user_input: str,
    messages: list[dict[str, str]] | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    module = load_agent_module(device_id, name)
    if runtime_context is not None:
        setattr(module, "RUNTIME_CONTEXT", runtime_context)
    skip_auto_postman = bool(runtime_context and runtime_context.get("skip_auto_postman"))
    setattr(module, "skip_auto", skip_auto_postman)
    messages = _sanitize_messages_runtime(messages)
    messages = _maybe_add_rag_context(module, device_id, name, user_input, messages)
    flow_text = ""
    services_raw = None
    if isinstance(getattr(module, "AGENT_CONTEXT", None), dict):
        flow_text = str(module.AGENT_CONTEXT.get("flow") or "")
        services_raw = module.AGENT_CONTEXT.get("services")
    services_query = _is_services_query(user_input, services_raw, flow_text)
    skip_auto = bool(runtime_context and runtime_context.get("skip_auto_postman"))
    if messages and hasattr(module, "_extract_last_options"):
        try:
            options = module._extract_last_options(messages)
        except Exception:
            options = []
        if options:
            is_sel = False
            if hasattr(module, "_llm_is_option_selection"):
                try:
                    is_sel = bool(module._llm_is_option_selection(options, user_input))
                except Exception:
                    is_sel = False
            elif hasattr(module, "_is_option_selection"):
                try:
                    is_sel = bool(module._is_option_selection(user_input))
                except Exception:
                    is_sel = False
            if is_sel:
                skip_auto = True
    if not services_query and not skip_auto:
        auto_result = _auto_postman_get(module, user_input)
        if auto_result:
            return auto_result
        flow_result = _auto_postman_from_flow(module, messages, user_input)
        if flow_result:
            return flow_result
    if messages and runtime_context and runtime_context.get("disable_followup") and hasattr(module, "run_with_history"):
        output = module.run_with_history(messages, user_input)
        return output
    if messages and hasattr(module, "run_with_history"):
        output = module.run_with_history(messages, user_input)
        return output
    if not hasattr(module, "run"):
        raise AttributeError("Agent module does not expose run(user_input)")
    output = module.run(user_input)
    return output


def run_agent_stream(
    device_id: str | int | None,
    name: str,
    user_input: str,
    messages: list[dict[str, str]] | None = None,
    runtime_context: dict[str, Any] | None = None,
):
    module = load_agent_module(device_id, name)
    if runtime_context is not None:
        setattr(module, "RUNTIME_CONTEXT", runtime_context)
    skip_auto_postman = bool(runtime_context and runtime_context.get("skip_auto_postman"))
    setattr(module, "skip_auto", skip_auto_postman)
    messages = _sanitize_messages_runtime(messages)
    messages = _maybe_add_rag_context(module, device_id, name, user_input, messages)
    flow_text = ""
    services_raw = None
    if isinstance(getattr(module, "AGENT_CONTEXT", None), dict):
        flow_text = str(module.AGENT_CONTEXT.get("flow") or "")
        services_raw = module.AGENT_CONTEXT.get("services")
    services_query = _is_services_query(user_input, services_raw, flow_text)
    if messages and hasattr(module, "_extract_last_options"):
        try:
            options = module._extract_last_options(messages)
        except Exception:
            options = []
        if options:
            is_sel = False
            if hasattr(module, "_llm_is_option_selection"):
                try:
                    is_sel = bool(module._llm_is_option_selection(options, user_input))
                except Exception:
                    is_sel = False
            elif hasattr(module, "_is_option_selection"):
                try:
                    is_sel = bool(module._is_option_selection(user_input))
                except Exception:
                    is_sel = False
            if is_sel:
                services_query = True
    if not services_query and (_has_date_like(user_input) or _history_has_date_like(messages)) and getattr(module, "POSTMAN_REQUESTS", None):
        forced = run_agent(device_id, name, user_input, messages, runtime_context)
        if forced:
            def _single():
                yield forced
            return _single(), module
    if not services_query:
        auto_result = _auto_postman_get(module, user_input)
        if auto_result:
            def _single():
                yield auto_result
            return _single(), module
        flow_result = _auto_postman_from_flow(module, messages, user_input)
        if flow_result:
            def _single():
                yield flow_result
            return _single(), module
    if messages and hasattr(module, "run_with_history_stream"):
        return module.run_with_history_stream(messages, user_input), module
    if hasattr(module, "run_stream"):
        return module.run_stream(user_input), module
    return None, module
