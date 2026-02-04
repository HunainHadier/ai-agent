import asyncio
import base64
import io
import json
import logging
import os
import re
import shutil
import tempfile
import traceback
import urllib.request
from typing import Any
from datetime import datetime

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from openai import OpenAI
import websockets
from fastapi.middleware.cors import CORSMiddleware 

from app.config import get_openai_api_key, get_openai_model
from app.builder import AgentBuilder, AgentSpec
from app.postman import missing_postman_variables
from app.runtime import run_agent, run_agent_stream
from app import runtime as runtime_helpers
from app import runtime as runtime_helpers
from app.storage import (
    agent_context_path,
    agent_file_path,
    agent_dir,
    copy_files_into_agent,
    delete_agent,
    delete_agent_upload,
    device_data_dir,
    find_agent_device_id,
    list_agent_names,
    list_agent_upload_paths,
    list_agent_uploads,
)
from app.rag import build_vectorstore, collect_texts
from app.tokens import count_message_tokens, count_tokens

app = FastAPI(title="Agent Builder")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger("assistant-ai")


def _ws_debug(message: str) -> None:
    try:
        with open("/tmp/assistant_ai_ws_debug.log", "a", encoding="utf-8") as handle:
            handle.write(f"{datetime.utcnow().isoformat()} {message}\n")
    except Exception:
        return


async def _safe_ws_send(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    try:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        return True
    except WebSocketDisconnect:
        return False
    except Exception as exc:
        _ws_debug(f"ws_send_failed: {exc}")
        _ws_debug(traceback.format_exc())
        return False


def _write_agent_log(device_id: str | int | None, name: str, filename: str, entry: dict[str, Any]) -> None:
    try:
        folder = agent_dir(device_id, name)
        path = os.path.join(folder, filename)
        entry = {**entry, "timestamp": datetime.utcnow().isoformat(), "agent": name}
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        return


def _pcm_duration_ms_from_base64(audio_b64: str, sample_rate: int) -> float:
    if not audio_b64 or sample_rate <= 0:
        return 0.0
    padding = 2 if audio_b64.endswith("==") else 1 if audio_b64.endswith("=") else 0
    byte_len = max(0, int((len(audio_b64) * 3) / 4) - padding)
    if byte_len <= 0:
        return 0.0
    samples = byte_len / 2
    return (samples / sample_rate) * 1000.0


def _pcm_byte_len_from_base64(audio_b64: str) -> int:
    if not audio_b64:
        return 0
    try:
        return len(base64.b64decode(audio_b64, validate=False))
    except Exception:
        return 0


async def _report_usage_to_api(
    api_base: str | None,
    auth_token: str | None,
    usage: int,
    input_usage: int,
    output_usage: int,
    integration_id: str,
    kind: str,
) -> bool:
    if not api_base or not auth_token:
        _ws_debug(
            f"usage_report skipped api_base={bool(api_base)} auth_token={bool(auth_token)}"
        )
        return False
    base = str(api_base).rstrip("/")
    url = f"{base}/ai/usage"
    payload = {
        "usage": max(0, int(usage)),
        "input_usage": max(0, int(input_usage)),
        "output_usage": max(0, int(output_usage)),
        "integration_id": integration_id,
        "kind": kind,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            if response.status_code >= 300:
                body = response.text
                if len(body) > 500:
                    body = body[:500] + "..."
                _ws_debug(
                    f"usage_report failed status={response.status_code} body={body}"
                )
                return False
            _ws_debug("usage_report ok")
            return True
    except Exception as exc:
        _ws_debug(f"usage_report error: {exc}")
        return False


def rebuild_agent_vectorstore(device_id: str | int | None, name: str) -> None:
    context_path = agent_context_path(device_id, name)
    if not os.path.isfile(context_path):
        raise HTTPException(status_code=404, detail="Agent context not found")
    uploads = list_agent_upload_paths(device_id, name)
    texts = collect_texts(uploads)
    try:
        build_vectorstore(name, texts, device_id)
    except Exception as exc:
        logger.warning("vectorstore rebuild failed name=%s device_id=%s error=%s", name, device_id, exc)


def _is_error_message(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    if normalized in {"no response.", "no response"}:
        return True
    if normalized.startswith("transcription failed"):
        return True
    if normalized.startswith("unable to reach the assistant"):
        return True
    if normalized.startswith("network error"):
        return True
    if normalized.startswith("assistant is not configured"):
        return True
    if normalized.startswith("auth token is missing"):
        return True
    return False


def _has_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _strip_image_disclaimer(
    text: str,
    user_input: str,
    attachments: list[dict[str, Any]] | None,
) -> str:
    content = str(text or "").strip()
    if not content:
        return content
    if attachments:
        return content
    if re.search(r"https?://\\S+", user_input or ""):
        return content
    if re.search(r"\\b(image|images|photo|photos|picture|pictures|img)\\b", user_input or "", flags=re.I):
        return content
    if re.search(r"[صس]ور|صورة|صور", user_input or ""):
        return content
    disclaimer_patterns = [
        r"i'?m unable to view or access external links or images\\.?\\s*",
        r"i'?m unable to view or analyze images\\.?\\s*",
        r"i can'?t view or access external links or images\\.?\\s*",
        r"i can'?t view or analyze images\\.?\\s*",
    ]
    cleaned = content
    for pattern in disclaimer_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.I)
    cleaned = cleaned.strip()
    if cleaned:
        return cleaned
    if _has_arabic_text(user_input):
        return "كيف يمكنني مساعدتك اليوم؟"
    return "How can I help you today?"


def _sanitize_messages(messages: list[dict[str, str]] | None) -> list[dict[str, str]] | None:
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
        if role == "assistant" and _is_error_message(content):
            continue
        cleaned.append({"role": role or "user", "content": content})
    return cleaned


def _load_agent_context(device_id: str | int | None, name: str) -> dict[str, Any]:
    context_path = agent_context_path(device_id, name)
    if not os.path.isfile(context_path):
        return {}
    try:
        with open(context_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        _ws_debug(f"load_agent_context_failed path={context_path} error={exc}")
        _ws_debug(traceback.format_exc())
        return {}


def _build_realtime_input_text(user_input: str, messages: list[dict[str, str]] | None) -> str:
    if not messages:
        return user_input
    lines: list[str] = []
    for item in messages:
        role = str(item.get("role") or "user").strip()
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append(f"user: {user_input}")
    return "\n".join(lines).strip()


def _parse_services_text(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;,\n]+", text or "") if item.strip()]


def _services_from_context(context: dict[str, Any]) -> list[str]:
    services_value = str(context.get("services") or "").strip()
    service_list = _parse_services_text(services_value)
    if service_list:
        return service_list
    flow = str(context.get("flow") or "").strip()
    flow_lines = [line.strip() for line in flow.splitlines() if line.strip()]
    for line in flow_lines:
        if ":" in line:
            name = line.split(":", 1)[0].strip()
            if name:
                service_list.append(name)
        else:
            service_list.extend(_parse_services_text(line))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in service_list:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _is_services_request(text: str) -> bool:
    value = (text or "").lower()
    keywords = [
        "services",
        "service list",
        "list services",
        "what services",
        "available services",
        "what can you do",
        "what do you do",
        "الخدمات",
        "خدماتك",
        "الخدمه",
        "الخدمة",
        "بتعمل ايه",
        "تقدر تعمل ايه",
    ]
    return any(keyword in value for keyword in keywords)


def _services_response(text: str, context: dict[str, Any]) -> str:
    services = _services_from_context(context)
    if _is_arabic_text(text):
        if not services:
            return "حالياً لا توجد خدمات مضافة للمساعد."
        header = "الخدمات المتاحة حالياً:"
    else:
        if not services:
            return "No services are configured yet."
        header = "Available services:"
    lines = [header]
    for idx, service in enumerate(services, start=1):
        lines.append(f"- Option {idx}: {service}")
    return "\n".join(lines)


def _is_arabic_text(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"[\\u0600-\\u06FF]", text))


def _extract_realtime_text(event: dict[str, Any], event_type: str) -> list[str]:
    texts: list[str] = []
    if "text" in event_type:
        direct_text = event.get("text")
        if isinstance(direct_text, str) and direct_text:
            texts.append(direct_text)
        delta_text = event.get("delta")
        if isinstance(delta_text, str) and delta_text:
            texts.append(delta_text)
    for key in ("item", "part", "content_part"):
        item = event.get(key)
        if isinstance(item, dict):
            content = item.get("content") or []
            if isinstance(content, list):
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    content_type = content_item.get("type")
                    text_value = content_item.get("text") or content_item.get("output_text")
                    if content_type in {"output_text", "text"} and text_value:
                        texts.append(str(text_value))
            text_value = item.get("text") or item.get("output_text")
            if text_value:
                texts.append(str(text_value))
    response_payload = event.get("response") or {}
    if isinstance(response_payload, dict):
        output_items = response_payload.get("output") or []
        if isinstance(output_items, list):
            for output_item in output_items:
                if not isinstance(output_item, dict):
                    continue
                content = output_item.get("content") or []
                if isinstance(content, list):
                    for content_item in content:
                        if not isinstance(content_item, dict):
                            continue
                        content_type = content_item.get("type")
                        text_value = content_item.get("text") or content_item.get("output_text")
                        if content_type in {"output_text", "text"} and text_value:
                            texts.append(str(text_value))
    return [text for text in texts if text]


def _normalize_attachments(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "url": item.get("url") or item.get("stored_url") or item.get("image_url"),
                "name": item.get("name") or item.get("filename"),
                "mime_type": item.get("mime_type") or item.get("mime"),
                "size": item.get("size"),
            }
        )
    return normalized


def _fast_response(user_input: str) -> str | None:
    cleaned = (user_input or "").strip()
    if not cleaned:
        return ""
    if len(cleaned) > 80:
        return None
    lowered = cleaned.lower()
    tokens = [token for token in re.split(r"\\s+", lowered) if token]
    arabic = bool(re.search(r"[\\u0600-\\u06FF]", cleaned))
    if arabic:
        if any(word in cleaned for word in ("مرحبا", "مرحباً", "اهلا", "أهلا", "السلام", "هاي")):
            options = [
                "أهلاً! وش حاب نشتغل عليه الآن؟",
                "أهلين! جاهز أساعد، ايش المطلوب؟",
                "يا هلا! شاركني الطلب ونكمل.",
                "مرحباً! قلّي كيف نكمل الشغل؟",
            ]
            return options[hash(cleaned) % len(options)]
        if cleaned in ("نعم", "لا", "تمام", "طيب", "شكرا", "شكراً"):
            options = [
                "تمام، وش الخطوة التالية؟",
                "واضح، خلّينا نكمل.",
                "ممتاز، قلّي المطلوب.",
            ]
            return options[hash(cleaned) % len(options)]
        return None
    if len(tokens) <= 2:
        if any(word in lowered for word in ("hi", "hello", "hey", "yo", "hola", "bonjour", "salut")):
            options = [
                "Hey! What should we tackle next?",
                "Hi there—what do you want me to handle?",
                "Hello! Share the task and I’ll take it.",
                "Hey! What are we working on now?",
            ]
            return options[hash(cleaned) % len(options)]
        if lowered in ("yes", "no", "ok", "okay", "thanks", "thank you"):
            options = [
                "Got it. What’s next?",
                "Okay—what should I do now?",
                "All set. Tell me the next step.",
            ]
            return options[hash(cleaned) % len(options)]
    return None


def _avoid_help_phrase(text: str, user_input: str) -> str:
    if not text:
        return text
    replacements_ar = [
        "وش المطلوب؟",
        "قلّي الطلب ونكمل.",
        "وش ننجز لك الآن؟",
        "جاهز، عطيني التفاصيل.",
    ]
    replacements_en = [
        "What should we tackle next?",
        "Share the task and I’ll take it.",
        "What do you want me to handle now?",
        "Give me the details and I’ll proceed.",
    ]
    normalized = text
    lowered = text.lower()
    use_arabic = bool(re.search(r"[\\u0600-\\u06FF]", text))
    pool = replacements_ar if use_arabic else replacements_en
    idx = abs(hash((user_input or "") + text)) % len(pool)
    replacement = pool[idx]
    patterns = [
        "كيف يمكنني مساعدتك اليوم؟",
        "كيف يمكنني مساعدتك اليوم",
        "كيف يمكنني مساعدتك؟",
        "كيف يمكنني مساعدتك",
        "how can i help you today?",
        "how can i help you today",
        "how can i help you?",
        "how can i help you",
    ]
    if any(pattern in lowered or pattern in normalized for pattern in patterns):
        for pattern in patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def _attachment_system_note(attachments: list[dict[str, Any]]) -> str:
    if not attachments:
        return ""
    lines = ["User attached files:"]
    for item in attachments:
        name = item.get("name") or "attachment"
        url = item.get("url")
        if url:
            lines.append(f"- {name} ({url})")
        else:
            lines.append(f"- {name}")
    lines.append(
        "Attachments are available for reference. "
        "If the user requests image generation based on an attachment, pass its URL as image_url."
    )
    return "\n".join(lines)


def _is_image_attachment(item: dict[str, Any]) -> bool:
    mime = str(item.get("mime_type") or "").lower()
    url = str(item.get("url") or item.get("stored_url") or "")
    if mime.startswith("image/"):
        return True
    return bool(re.search(r"\.(png|jpe?g|webp|gif)(?:\?.*)?$", url, re.I))


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


def _should_flow_autocall(user_input: str, messages: list[dict[str, str]] | None) -> bool:
    if not _has_date_like(user_input):
        return False
    if not messages:
        return True
    for item in reversed(messages):
        if item.get("role") != "assistant":
            continue
        content = str(item.get("content") or "").lower()
        if not content:
            continue
        has_checkin = "check-in" in content or "check in" in content or "checkin" in content or "تاريخ الدخول" in content
        has_checkout = "check-out" in content or "check out" in content or "checkout" in content or "تاريخ الخروج" in content
        has_guest = "guest name" in content or "اسم" in content
        return (has_checkin and has_checkout) or (has_guest and (has_checkin or has_checkout))
    return False


def _analyze_image_attachments(attachments: list[dict[str, Any]], user_input: str) -> str:
    images = [item for item in attachments if isinstance(item, dict) and _is_image_attachment(item)]
    if not images:
        return ""
    client = OpenAI(api_key=get_openai_api_key())
    model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
    prompt = (
        "Analyze the provided image(s) for actionable observations. "
        "Focus on cleanliness, clutter, safety hazards, missing items, and notable issues. "
        "Be concise and factual. If the user request mentions a task, align the analysis with it."
    )
    if user_input:
        prompt += f"\nUser request: {user_input}"
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for item in images[:3]:
        url = item.get("url") or item.get("stored_url")
        if not url:
            continue
        content.append({"type": "image_url", "image_url": {"url": url}})
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
        )
        text = response.choices[0].message.content or ""
        return text.strip()
    except Exception:
        return ""

def _parse_service_list(services: str) -> list[str]:
    return [item.strip() for item in re.split(r"[;,\n]+", services or "") if item.strip()]


def _clean_step(value: str) -> str:
    return re.sub(r"^[-*]\s+", "", re.sub(r"^\d+[\).\s-]+", "", value or "").strip()).strip()


def _split_steps(value: str) -> list[str]:
    chunks = re.split(r"\n|;|->|=>", value or "")
    parts: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.extend([item.strip() for item in re.split(r"\.\s+", chunk) if item.strip()])
    return parts


def _normalize_steps(steps: list[Any]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for step in steps:
        cleaned = _clean_step(str(step))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _apply_edit_notes(
    services: list[str],
    steps_by_service: dict[str, list[str]],
    edit_notes: str | None,
) -> dict[str, list[str]]:
    if not edit_notes:
        return steps_by_service
    notes_raw = edit_notes.strip()
    if not notes_raw:
        return steps_by_service
    notes_lower = notes_raw.lower()
    for service in services:
        service_lower = service.lower()
        if service_lower and (service_lower in notes_lower or ("inquiry" in notes_lower and "inquiry" in service_lower)):
            cleaned = notes_raw
            cleaned = re.sub(
                rf"^(in|for)\s+{re.escape(service)}\s*[:,\-]*\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()
            cleaned = re.sub(r"^\s*(need to|needs to)\s*", "", cleaned, flags=re.IGNORECASE).strip()
            if cleaned:
                steps = steps_by_service.get(service_lower, [])
                insert_idx = 0
                if "before booking number" in notes_lower:
                    for idx, step in enumerate(steps):
                        if "booking number" in step.lower():
                            insert_idx = idx
                            break
                steps_by_service[service_lower] = steps[:insert_idx] + [cleaned] + steps[insert_idx:]
    return steps_by_service


def _extract_json_payload(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _fallback_steps() -> list[str]:
    return [
        "Review the request and intent",
        "Collect missing details and confirm the action",
        "Execute the service workflow in the connected system",
        "Verify the result and summarize the outcome",
    ]


def _build_flow_steps_prompt(
    services: list[str],
    description: str | None,
    flow: str | None,
    edit_notes: str | None,
    language: str | None,
) -> list[dict[str, str]]:
    service_list = "\n".join([f"- {item}" for item in services])
    language_hint = f"Respond in {language}." if language else "Respond in the same language as the description."
    user_content = "\n".join(
        [
            "Generate AI-only workflow steps for each service.",
            "Use concise action verbs and avoid any mention of what the user says or does.",
            "If edit notes are provided, incorporate them into the updated steps.",
            "Return JSON only, no markdown.",
            "",
            "Services:",
            service_list,
            "",
            f"Description: {description or ''}",
            f"Existing flow (if any): {flow or ''}",
            f"Edit notes (if any): {edit_notes or ''}",
            "",
            language_hint,
            "JSON schema:",
            '{ "services": [ { "name": "Service name", "steps": ["Step 1", "Step 2"] } ] }',
            "Provide 3-6 steps per service.",
        ]
    )
    return [
        {
            "role": "system",
            "content": "You output strictly valid JSON with the specified schema and no extra keys.",
        },
        {"role": "user", "content": user_content},
    ]


def _generate_flow_steps(
    services: list[str],
    description: str | None,
    flow: str | None,
    edit_notes: str | None,
    language: str | None,
) -> list[dict[str, Any]]:
    client = OpenAI(api_key=get_openai_api_key())
    response = client.chat.completions.create(
        model=get_openai_model(),
        messages=_build_flow_steps_prompt(services, description, flow, edit_notes, language),
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    payload = _extract_json_payload(content)
    raw_services = payload.get("services") if isinstance(payload, dict) else []
    if not isinstance(raw_services, list):
        raw_services = []

    mapped: dict[str, list[str]] = {}
    for item in raw_services:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("service") or "").strip()
            steps_value = item.get("steps") or item.get("flow_steps") or []
        else:
            name = ""
            steps_value = []
        if not name:
            continue
        if isinstance(steps_value, str):
            steps = _split_steps(steps_value)
        elif isinstance(steps_value, list):
            steps = steps_value
        else:
            steps = []
        mapped[name.lower()] = _normalize_steps(steps)

    mapped = _apply_edit_notes(services, mapped, edit_notes)

    results: list[dict[str, Any]] = []
    for service in services:
        key = service.lower()
        steps = mapped.get(key) or _fallback_steps()
        results.append({"name": service, "steps": steps})
    return results


class RunRequest(BaseModel):
    input: str
    messages: list[dict[str, str]] | None = Field(default=None)
    device_id: str | int | None = Field(default=None)
    assistant_id: str | int | None = Field(default=None)
    agent_id: str | int | None = Field(default=None)
    auth_token: str | None = Field(default=None)
    api_base: str | None = Field(default=None)
    stream: bool | None = Field(default=None)
    attachments: list[dict[str, Any]] | None = Field(default=None)


class FlowStepsRequest(BaseModel):
    services: str
    description: str | None = Field(default=None)
    flow: str | None = Field(default=None)
    edit_notes: str | None = Field(default=None)
    language: str | None = Field(default=None)


@app.get("/agents")
def list_agents(device_id: str | None = None) -> list[str]:
    return list_agent_names(device_id)


@app.post("/agents")
def create_agent(
    name: str = Form(...),
    description: str = Form(...),
    services: str = Form(...),
    flow: str = Form(...),
    device_id: str | None = Form(None),
    language: str | None = Form(None),
    disable_design: bool | None = Form(None),
    enable_design: bool | None = Form(None),
    postman_collection: UploadFile | None = File(None),
    postman_variables: str | None = Form(None),
    brand_profile: str | None = Form(None),
    integrations: str | None = Form(None),
    files: list[UploadFile] | None = File(None),
) -> dict[str, str]:
    if os.path.isfile(agent_context_path(device_id, name)):
        raise HTTPException(status_code=409, detail="Agent already exists")
    builder = AgentBuilder()
    tmp_dir = tempfile.mkdtemp(prefix="agent_builder_")

    postman_path = None
    if postman_collection:
        postman_path = os.path.join(tmp_dir, postman_collection.filename)
        with open(postman_path, "wb") as handle:
            handle.write(postman_collection.file.read())

    file_paths: list[str] = []
    if files:
        for upload in files:
            if not upload.filename:
                continue
            dest = os.path.join(tmp_dir, upload.filename)
            with open(dest, "wb") as handle:
                handle.write(upload.file.read())
            file_paths.append(dest)

    variables = None
    if postman_variables:
        try:
            variables = json.loads(postman_variables)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="postman_variables must be JSON") from exc

    brand_profile_payload = None
    if brand_profile:
        try:
            brand_profile_payload = json.loads(brand_profile)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="brand_profile must be JSON") from exc

    integrations_payload = None
    if integrations:
        try:
            integrations_payload = json.loads(integrations)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="integrations must be JSON") from exc

    try:
        spec = AgentSpec(
            name=name,
            description=description,
            services=services,
            flow=flow,
            device_id=device_id,
            postman_path=postman_path,
            file_paths=file_paths,
            postman_variables=variables,
            language=language,
            integrations=integrations_payload,
            brand_profile=brand_profile_payload,
            disable_design=disable_design,
            enable_design=enable_design,
        )
        agent_path = builder.build(spec)
    except json.JSONDecodeError as exc:
        detail = f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        raise HTTPException(status_code=400, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"agent": name, "path": agent_path}


@app.put("/agents/{name}")
def update_agent(
    name: str,
    description: str | None = Form(None),
    services: str | None = Form(None),
    flow: str | None = Form(None),
    device_id: str | None = Form(None),
    language: str | None = Form(None),
    disable_design: bool | None = Form(None),
    enable_design: bool | None = Form(None),
    postman_collection: UploadFile | None = File(None),
    postman_variables: str | None = Form(None),
    brand_profile: str | None = Form(None),
    integrations: str | None = Form(None),
    files: list[UploadFile] | None = File(None),
) -> dict[str, str]:
    context_path = agent_context_path(device_id, name)
    if not os.path.isfile(context_path):
        raise HTTPException(status_code=404, detail="Agent not found")

    builder = AgentBuilder()
    tmp_dir = tempfile.mkdtemp(prefix="agent_builder_")

    postman_path = None
    if postman_collection:
        postman_path = os.path.join(tmp_dir, postman_collection.filename)
        with open(postman_path, "wb") as handle:
            handle.write(postman_collection.file.read())

    file_paths: list[str] = []
    if files:
        for upload in files:
            if not upload.filename:
                continue
            dest = os.path.join(tmp_dir, upload.filename)
            with open(dest, "wb") as handle:
                handle.write(upload.file.read())
            file_paths.append(dest)

    variables = None
    if postman_variables:
        try:
            variables = json.loads(postman_variables)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="postman_variables must be JSON") from exc

    brand_profile_payload = None
    if brand_profile:
        try:
            brand_profile_payload = json.loads(brand_profile)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="brand_profile must be JSON") from exc

    integrations_payload = None
    if integrations:
        try:
            integrations_payload = json.loads(integrations)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="integrations must be JSON") from exc

    try:
        spec = AgentSpec(
            name=name,
            description=description,
            services=services,
            flow=flow,
            device_id=device_id,
            postman_path=postman_path,
            file_paths=file_paths,
            postman_variables=variables,
            language=language,
            integrations=integrations_payload,
            brand_profile=brand_profile_payload,
            disable_design=disable_design,
            enable_design=enable_design,
        )
        agent_path = builder.build(spec)
    except json.JSONDecodeError as exc:
        detail = f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        raise HTTPException(status_code=400, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"agent": name, "path": agent_path}


@app.post("/agents/{name}/rename")
def rename_agent(
    name: str,
    new_name: str = Form(...),
    device_id: str | None = Form(None),
) -> dict[str, str]:
    new_name = new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New name is required")
    if new_name == name:
        return {"status": "unchanged", "agent": name}
    if not os.path.isfile(agent_context_path(device_id, name)):
        raise HTTPException(status_code=404, detail="Agent not found")
    if os.path.isfile(agent_context_path(device_id, new_name)):
        raise HTTPException(status_code=409, detail="Agent already exists")

    old_data_dir = os.path.join(device_data_dir(device_id), name)
    new_data_dir = os.path.join(device_data_dir(device_id), new_name)
    if os.path.isdir(old_data_dir):
        shutil.move(old_data_dir, new_data_dir)

    old_agent_path = agent_file_path(device_id, name)
    if os.path.isfile(old_agent_path):
        os.remove(old_agent_path)

    builder = AgentBuilder()
    spec = AgentSpec(
        name=new_name,
        description=None,
        services=None,
        flow=None,
        device_id=device_id,
        postman_path=None,
        file_paths=None,
        postman_variables=None,
        language=None,
    )
    agent_path = builder.build(spec)
    return {"status": "renamed", "agent": new_name, "path": agent_path}


@app.get("/agents/{name}/knowledge")
def list_agent_knowledge(name: str, device_id: str | None = None) -> dict[str, list[dict[str, int | str]]]:
    if not os.path.isfile(agent_context_path(device_id, name)):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"files": list_agent_uploads(device_id, name)}


@app.post("/agents/{name}/knowledge")
def upload_agent_knowledge(
    name: str,
    device_id: str | None = Form(None),
    files: list[UploadFile] | None = File(None),
) -> dict[str, int | str]:
    if not os.path.isfile(agent_context_path(device_id, name)):
        raise HTTPException(status_code=404, detail="Agent not found")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    tmp_dir = tempfile.mkdtemp(prefix="agent_builder_")
    file_paths: list[str] = []
    for upload in files:
        if not upload.filename:
            continue
        dest = os.path.join(tmp_dir, upload.filename)
        with open(dest, "wb") as handle:
            handle.write(upload.file.read())
        file_paths.append(dest)

    copy_files_into_agent(device_id, name, file_paths)
    rebuild_agent_vectorstore(device_id, name)

    return {"status": "uploaded", "count": len(file_paths)}


@app.delete("/agents/{name}/knowledge/{filename}")
def delete_agent_knowledge(name: str, filename: str, device_id: str | None = None) -> dict[str, str]:
    if not os.path.isfile(agent_context_path(device_id, name)):
        raise HTTPException(status_code=404, detail="Agent not found")
    delete_agent_upload(device_id, name, filename)
    rebuild_agent_vectorstore(device_id, name)
    return {"status": "deleted", "name": filename}

@app.post("/postman/inspect")
def inspect_postman(collection: UploadFile = File(...)) -> dict[str, list[dict[str, str]]]:
    tmp_dir = tempfile.mkdtemp(prefix="agent_builder_")
    collection_path = os.path.join(tmp_dir, collection.filename)
    with open(collection_path, "wb") as handle:
        handle.write(collection.file.read())
    missing = missing_postman_variables(collection_path)
    return {"missing": missing}


@app.post("/images/white-background")
def white_background(image: UploadFile = File(...)) -> dict[str, str]:
    filename = image.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="Image file is required.")
    client = OpenAI(api_key=get_openai_api_key())
    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip() or "gpt-image-1"
    prompt = (
        "Replace the background with pure white (#FFFFFF). "
        "Keep the character exactly the same with no changes, no cropping, and no color shifts."
    )
    try:
        image.file.seek(0)
        raw = image.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        wrapped = io.BytesIO(raw)
        wrapped.name = image.filename or "image.png"
        result = client.images.edit(
            model=model,
            image=wrapped,
            prompt=prompt,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("white_background_failed error=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    data = None
    image_url = None
    if result and getattr(result, "data", None):
        if result.data:
            data = result.data[0].b64_json
            image_url = result.data[0].url
    if not data and image_url:
        try:
            with urllib.request.urlopen(image_url, timeout=30) as handle:
                fetched = handle.read()
            if fetched:
                data = base64.b64encode(fetched).decode("utf-8")
        except Exception as exc:
            logger.warning("white_background_fetch_failed error=%s url=%s", exc, image_url)
    if not data:
        raise HTTPException(status_code=400, detail="Image background update failed.")
    return {"b64": data, "mime_type": "image/png"}


async def _speech_transcribe(file: UploadFile) -> tuple[dict[str, str], dict[str, str]]:
    api_key = get_openai_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured.")
    if not file:
        raise HTTPException(status_code=400, detail="file is required.")
    try:
        file.file.seek(0)
        client = OpenAI(api_key=api_key)
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=(file.filename or "audio.webm", file.file, file.content_type or "application/octet-stream"),
        )
        text = getattr(transcription, "text", "") or ""
        output_usage = count_tokens(text)
        headers = {
            "X-Token-Usage-Audio": str(output_usage),
            "X-Token-Usage-Output": str(output_usage),
            "X-Token-Usage-Input": "0",
        }
        return {"text": text, "model": "gpt-4o-mini-transcribe"}, headers
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ai/speech/transcribe")
async def speech_transcribe_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    payload, headers = await _speech_transcribe(file)
    return JSONResponse(payload, headers=headers)


@app.post("/speech/transcribe")
async def speech_transcribe_public_endpoint(file: UploadFile = File(...)) -> JSONResponse:
    payload, headers = await _speech_transcribe(file)
    return JSONResponse(payload, headers=headers)


@app.post("/flows/steps")
def generate_flow_steps(payload: FlowStepsRequest) -> dict[str, Any]:
    services_list = _parse_service_list(payload.services)
    if not services_list:
        raise HTTPException(status_code=400, detail="Services list is required")
    try:
        services = _generate_flow_steps(
            services_list,
            payload.description,
            payload.flow,
            payload.edit_notes,
            payload.language,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"services": services}


@app.post("/agents/{name}/run")
def run_agent_endpoint(
    name: str,
    payload: RunRequest,
    request: Request,
    device_id: str | None = None,
) -> Response:
    resolved_device_id = device_id or payload.device_id
    if resolved_device_id is None or str(resolved_device_id).strip() == "":
        resolved_device_id = find_agent_device_id(name)
    payload.messages = _sanitize_messages(payload.messages)
    attachments = _normalize_attachments(payload.attachments)
    attachment_note = _attachment_system_note(attachments)
    analysis_note = _analyze_image_attachments(attachments, payload.input)
    if analysis_note:
        attachment_note = (attachment_note + "\n\nImage analysis:\n" + analysis_note).strip()
    if attachment_note:
        payload.messages = [{"role": "system", "content": attachment_note}] + (payload.messages or [])
    resolved_assistant_id = (
        payload.assistant_id
        or payload.agent_id
        or request.query_params.get("assistant_id")
        or request.query_params.get("agent_id")
    )
    auth_token = payload.auth_token or request.query_params.get("auth_token")
    api_base = payload.api_base or request.query_params.get("api_base")
    runtime_context = {
        "auth_token": auth_token,
        "api_base": api_base,
        "device_id": resolved_device_id,
        "assistant_id": resolved_assistant_id,
        "attachments": attachments,
    }
    _write_agent_log(
        resolved_device_id,
        name,
        "prompt_logs.jsonl",
        {
            "event": "user_prompt",
            "input": payload.input,
            "messages": payload.messages or [],
            "attachments": attachments,
            "assistant_id": resolved_assistant_id,
        },
    )
    accepts_stream = "text/event-stream" in request.headers.get("accept", "").lower()
    wants_stream = bool(payload.stream) or accepts_stream

    def chunk_text(text: str) -> list[str]:
        if not text:
            return []
        parts = re.split(r"(\\s+)", text)
        return [part for part in parts if part]

    def sse_event(data: dict[str, Any]) -> str:
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    fast_reply = _fast_response(payload.input) if not (payload.messages or []) else None
    if fast_reply is not None:
        message_tokens = count_message_tokens(payload.messages or [])
        input_usage = message_tokens + count_tokens(payload.input)
        output_usage = count_tokens(fast_reply)
        usage = input_usage + output_usage
        if wants_stream:
            def generator():
                if fast_reply:
                    yield sse_event({"delta": fast_reply})
                yield sse_event({
                    "done": True,
                    "output": fast_reply or "No response.",
                    "usage": usage,
                    "input_usage": input_usage,
                    "output_usage": output_usage,
                })

            return StreamingResponse(
                generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return JSONResponse(
            {"output": fast_reply or "No response."},
            headers={
                "X-Token-Usage-Text": str(usage),
                "X-Token-Usage-Input": str(input_usage),
                "X-Token-Usage-Output": str(output_usage),
            },
        )

    if wants_stream:
        if runtime_helpers._has_date_like(payload.input):
            try:
                forced = run_agent(
                    resolved_device_id,
                    name,
                    payload.input,
                    payload.messages,
                    runtime_context=runtime_context,
                )
                if runtime_helpers._looks_like_postman_list(forced):
                    def generator():
                        yield sse_event({"delta": forced})
                        yield sse_event({"done": True, "output": forced, "usage": 0})

                    return StreamingResponse(
                        generator(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache, no-transform",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )
            except Exception:
                pass
        stream_result, module = run_agent_stream(
            resolved_device_id,
            name,
            payload.input,
            payload.messages,
            runtime_context=runtime_context,
        )

        def generator():
            output_parts: list[str] = []
            audio_transcript_parts: list[str] = []
            try:
                if stream_result is not None:
                    for chunk in stream_result:
                        if chunk is None:
                            continue
                        token = str(chunk)
                        output_parts.append(token)
                        yield sse_event({"delta": token})
                else:
                    output = run_agent(
                        resolved_device_id,
                        name,
                        payload.input,
                        payload.messages,
                        runtime_context=runtime_context,
                    )
                    output_parts.append(output)
                    for part in chunk_text(output):
                        yield sse_event({"delta": part})
            except Exception as exc:
                yield sse_event({"error": str(exc)})

            final_output = getattr(module, "LAST_OUTPUT", "") if module is not None else ""
            if not final_output:
                final_output = "".join(output_parts).strip()
            final_output = _avoid_help_phrase(final_output, payload.input)
            if not final_output:
                try:
                    fallback_output = run_agent(
                        resolved_device_id,
                        name,
                        payload.input,
                        payload.messages,
                        runtime_context=runtime_context,
                    )
                    if fallback_output and str(fallback_output).strip():
                        final_output = _avoid_help_phrase(str(fallback_output).strip(), payload.input)
                except Exception as exc:
                    logger.warning(
                        "stream_fallback_failed name=%s device_id=%s error=%s",
                        name,
                        resolved_device_id,
                        exc,
                    )
            if not final_output:
                final_output = "No response."
            _write_agent_log(
                resolved_device_id,
                name,
                "chat_logs.jsonl",
                {
                    "event": "assistant_reply",
                    "output": final_output,
                    "assistant_id": resolved_assistant_id,
                },
            )
            message_tokens = count_message_tokens(payload.messages or [])
            input_usage = message_tokens + count_tokens(payload.input)
            output_usage = count_tokens(final_output)
            usage = input_usage + output_usage
            yield sse_event(
                {
                    "done": True,
                    "output": final_output,
                    "usage": usage,
                    "input_usage": input_usage,
                    "output_usage": output_usage,
                }
            )

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        output = run_agent(
            resolved_device_id,
            name,
            payload.input,
            payload.messages,
            runtime_context=runtime_context,
        )
        output = _avoid_help_phrase(output, payload.input)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _write_agent_log(
        resolved_device_id,
        name,
        "chat_logs.jsonl",
        {
            "event": "assistant_reply",
            "output": output,
            "assistant_id": resolved_assistant_id,
        },
    )
    message_tokens = count_message_tokens(payload.messages or [])
    input_usage = message_tokens + count_tokens(payload.input)
    output_usage = count_tokens(output)
    usage = input_usage + output_usage
    return JSONResponse(
        {"output": output},
        headers={
            "X-Token-Usage-Text": str(usage),
            "X-Token-Usage-Input": str(input_usage),
            "X-Token-Usage-Output": str(output_usage),
        },
    )


@app.websocket("/ws/agents/{name}/run")
async def run_agent_websocket(websocket: WebSocket, name: str) -> None:
    await websocket.accept()
    query_params = websocket.query_params

    def query_value(*keys: str) -> str | None:
        for key in keys:
            value = query_params.get(key)
            if value is None:
                continue
            value = str(value).strip()
            if value:
                return value
        return None

    async def stream_audio_from_text(response_text: str, payload: dict[str, Any]) -> bool:
        audio_enabled = str(payload.get("audio_enabled") or "").strip().lower() in {"1", "true", "yes"}
        text = str(response_text or "").strip()
        _ws_debug(f"audio_from_text start enabled={audio_enabled} text_len={len(text)} name={name}")
        if not audio_enabled or not api_key or not text:
            return False
        try:
            async with websockets.connect(
                f"wss://api.openai.com/v1/realtime?model={realtime_model}",
                extra_headers={
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
            ) as rt:
                session_update = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "voice": payload.get("voice") or voice,
                        "input_audio_format": "pcm16",
                        "output_audio_format": audio_format,
                    },
                }
                await rt.send(json.dumps(session_update))
                response_create = {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"],
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}],
                            }
                        ],
                    },
                }
                await rt.send(json.dumps(response_create))
                async for message in rt:
                    event = json.loads(message)
                    event_type = str(event.get("type") or "")
                    _ws_debug(f"audio_from_text_event={event_type} keys={list(event.keys())}")
                    if event_type == "session.updated":
                        session_info = event.get("session") or {}
                        _ws_debug(
                            f"realtime_audio_from_text_session modalities={session_info.get('modalities')} audio={session_info.get('audio')}"
                        )
                    if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                        delta = event.get("delta") or event.get("audio") or event.get("data") or ""
                        if delta:
                            if not await _safe_ws_send(
                                websocket,
                                {
                                    "type": "audio_delta",
                                    "data": delta,
                                    "format": audio_format,
                                    "sample_rate": sample_rate,
                                },
                            ):
                                return False
                        continue
                    if event_type in {"response.audio.done", "response.output_audio.done"}:
                        await _safe_ws_send(websocket, {"type": "audio_done"})
                        break
                    if event_type in {"response.completed", "response.done", "error"}:
                        break
        except Exception as exc:
            _ws_debug(f"audio_from_text error: {exc}")
            _ws_debug(traceback.format_exc())
            return False
        _ws_debug("audio_from_text done")
        return True

    async def stream_audio_from_text(response_text: str, payload: dict[str, Any]) -> bool:
        audio_enabled = str(payload.get("audio_enabled") or "").strip().lower() in {"1", "true", "yes"}
        text = str(response_text or "").strip()
        _ws_debug(
            f"audio_from_text start enabled={audio_enabled} text_len={len(text)} name={name}"
        )
        if not audio_enabled or not api_key or not text:
            return False
        try:
            async with websockets.connect(
                f"wss://api.openai.com/v1/realtime?model={realtime_model}",
                extra_headers={
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
            ) as rt:
                session_update = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "voice": payload.get("voice") or voice,
                        "input_audio_format": "pcm16",
                        "output_audio_format": audio_format,
                    },
                }
                await rt.send(json.dumps(session_update))
                response_create = {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"],
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}],
                            }
                        ],
                    },
                }
                await rt.send(json.dumps(response_create))
                async for message in rt:
                    event = json.loads(message)
                    event_type = str(event.get("type") or "")
                    _ws_debug(f"audio_from_text_event={event_type} keys={list(event.keys())}")
                    if event_type == "session.updated":
                        session_info = event.get("session") or {}
                        _ws_debug(
                            f"audio_from_text_session modalities={session_info.get('modalities')} audio={session_info.get('audio')}"
                        )
                    if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                        delta = event.get("delta") or event.get("audio") or event.get("data") or ""
                        if delta:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "audio_delta",
                                        "data": delta,
                                        "format": audio_format,
                                        "sample_rate": sample_rate,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        continue
                    if event_type in {"response.audio.done", "response.output_audio.done"}:
                        await websocket.send_text(json.dumps({"type": "audio_done"}, ensure_ascii=False))
                        break
                    if event_type in {"response.completed", "response.done", "error"}:
                        break
        except Exception as exc:
            _ws_debug(f"audio_from_text error: {exc}")
            _ws_debug(traceback.format_exc())
            return False
        _ws_debug("audio_from_text done")
        return True

    async def stream_audio_from_text(response_text: str, payload: dict[str, Any]) -> bool:
        audio_enabled = str(payload.get("audio_enabled") or "").strip().lower() in {"1", "true", "yes"}
        text = str(response_text or "").strip()
        _ws_debug(
            f"audio_from_text start enabled={audio_enabled} text_len={len(text)} name={name} device_id={resolved_device_id}"
        )
        if not audio_enabled or not api_key or not text:
            return False
        try:
            async with websockets.connect(
                f"wss://api.openai.com/v1/realtime?model={realtime_model}",
                extra_headers={
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
            ) as rt:
                session_update = {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "voice": payload.get("voice") or voice,
                        "input_audio_format": "pcm16",
                        "output_audio_format": audio_format,
                    },
                }
                await rt.send(json.dumps(session_update))
                response_create = {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"],
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}],
                            }
                        ],
                    },
                }
                await rt.send(json.dumps(response_create))
                async for message in rt:
                    event = json.loads(message)
                    event_type = str(event.get("type") or "")
                    _ws_debug(f"audio_from_text_event={event_type} keys={list(event.keys())}")
                    if event_type == "session.updated":
                        session_info = event.get("session") or {}
                        _ws_debug(f"audio_from_text_session modalities={session_info.get('modalities')} audio={session_info.get('audio')}")
                    if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                        delta = event.get("delta") or event.get("audio") or event.get("data") or ""
                        if delta:
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "audio_delta",
                                        "data": delta,
                                        "format": audio_format,
                                        "sample_rate": sample_rate,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        continue
                    if event_type in {"response.audio.done", "response.output_audio.done"}:
                        await websocket.send_text(json.dumps({"type": "audio_done"}, ensure_ascii=False))
                        break
                    if event_type in {"response.completed", "response.done", "error"}:
                        break
        except Exception as exc:
            _ws_debug(f"audio_from_text error: {exc}")
            _ws_debug(traceback.format_exc())
            return False
        _ws_debug("audio_from_text done")
        return True

    def chunk_text(text: str) -> list[str]:
        if not text:
            return []
        parts = re.split(r"(\\s+)", text)
        return [part for part in parts if part]

    while True:
        try:
            raw_payload = await websocket.receive_text()
        except WebSocketDisconnect:
            break

        payload: dict[str, Any] = {}
        if raw_payload:
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                payload = {"input": raw_payload}

        user_input = str(payload.get("input") or payload.get("text") or payload.get("message") or "").strip()
        if not user_input:
            await websocket.send_text(json.dumps({"error": "input is required."}))
            continue

        resolved_device_id = payload.get("device_id")
        if resolved_device_id is None or str(resolved_device_id).strip() == "":
            resolved_device_id = query_value("device_id", "deviceId")
        if resolved_device_id is None or str(resolved_device_id).strip() == "":
            resolved_device_id = find_agent_device_id(name)
        resolved_assistant_id = payload.get("assistant_id") or payload.get("agent_id")
        if resolved_assistant_id is None or str(resolved_assistant_id).strip() == "":
            resolved_assistant_id = query_value("assistant_id", "assistantId", "agent_id", "agentId")
        auth_token = payload.get("auth_token") or query_value("auth_token", "authToken", "token")
        api_base = payload.get("api_base") or query_value("api_base", "apiBase")
        attachments = _normalize_attachments(payload.get("attachments"))
        fast_reply = _fast_response(user_input) if not (payload.get("messages") or []) else None
        if fast_reply is not None:
            message_tokens = count_message_tokens(payload.get("messages") or [])
            usage = message_tokens + count_tokens(user_input) + count_tokens(fast_reply)
            fast_reply = _avoid_help_phrase(fast_reply, user_input)
            fast_reply = _strip_image_disclaimer(fast_reply, user_input, attachments)
            _write_agent_log(
                resolved_device_id,
                name,
                "prompt_logs.jsonl",
                {
                    "event": "user_prompt",
                    "input": user_input,
                    "messages": payload.get("messages") or [],
                    "attachments": attachments,
                    "assistant_id": resolved_assistant_id,
                },
            )
            _write_agent_log(
                resolved_device_id,
                name,
                "chat_logs.jsonl",
                {
                    "event": "assistant_reply",
                    "output": fast_reply or "No response.",
                    "assistant_id": resolved_assistant_id,
                },
            )
            if fast_reply:
                await websocket.send_text(json.dumps({"delta": fast_reply}, ensure_ascii=False))
            await websocket.send_text(
                json.dumps(
                    {"done": True, "output": fast_reply or "No response.", "usage": usage},
                    ensure_ascii=False,
                )
            )
            continue

        messages = _sanitize_messages(payload.get("messages"))
        attachment_note = _attachment_system_note(attachments)
        analysis_note = _analyze_image_attachments(attachments, user_input)
        if analysis_note:
            if attachment_note:
                attachment_note = (attachment_note + "\n\nImage analysis:\n" + analysis_note).strip()
            else:
                attachment_note = ("Image analysis:\n" + analysis_note).strip()
        if attachment_note:
            messages = [{"role": "system", "content": attachment_note}] + (messages or [])
        runtime_context = {
            "auth_token": auth_token,
            "api_base": api_base,
            "device_id": resolved_device_id,
            "assistant_id": resolved_assistant_id,
            "attachments": attachments,
        }
        _write_agent_log(
            resolved_device_id,
            name,
            "prompt_logs.jsonl",
            {
                "event": "user_prompt",
                "input": user_input,
                "messages": messages or [],
                "attachments": attachments,
                "assistant_id": resolved_assistant_id,
            },
        )
        logger.info(
            "ws_request name=%s device_id=%s input_len=%s messages_len=%s",
            name,
            resolved_device_id,
            len(user_input),
            len(messages or []),
        )

        stream_result = None
        module = None
        try:
            if (not runtime_helpers._is_services_query(user_input)) and (
                runtime_helpers._has_date_like(user_input)
                or runtime_helpers._history_has_date_like(messages)
            ):
                try:
                    forced = run_agent(
                        resolved_device_id,
                        name,
                        user_input,
                        messages,
                        runtime_context=runtime_context,
                    )
                    if forced and runtime_helpers._looks_like_postman_list(forced):
                        final_output = _avoid_help_phrase(str(forced).strip(), user_input)
                        final_output = _strip_image_disclaimer(final_output, user_input, attachments)
                        _write_agent_log(
                            resolved_device_id,
                            name,
                            "chat_logs.jsonl",
                            {
                                "event": "assistant_reply",
                                "output": final_output,
                                "assistant_id": resolved_assistant_id,
                            },
                        )
                        message_tokens = count_message_tokens(messages or [])
                        usage = message_tokens + count_tokens(user_input) + count_tokens(final_output)
                        for part in chunk_text(final_output):
                            await websocket.send_text(json.dumps({"delta": part}, ensure_ascii=False))
                        await websocket.send_text(
                            json.dumps(
                                {"done": True, "output": final_output, "usage": usage},
                                ensure_ascii=False,
                            )
                        )
                        continue
                except Exception:
                    pass
            stream_result, module = run_agent_stream(
                resolved_device_id,
                name,
                user_input,
                messages,
                runtime_context=runtime_context,
            )
            output_parts: list[str] = []
            if stream_result is not None:
                for chunk in stream_result:
                    if chunk is None:
                        continue
                    token = str(chunk)
                    output_parts.append(token)
                    await websocket.send_text(json.dumps({"delta": token}, ensure_ascii=False))
            else:
                output = run_agent(
                    resolved_device_id,
                    name,
                    user_input,
                    messages,
                    runtime_context=runtime_context,
                )
                output_parts.append(output)
                for part in chunk_text(output):
                    await websocket.send_text(json.dumps({"delta": part}, ensure_ascii=False))

            final_output = getattr(module, "LAST_OUTPUT", "") if module is not None else ""
            if not final_output:
                final_output = "".join(output_parts).strip()
            final_output = _avoid_help_phrase(final_output, user_input)
            final_output = _strip_image_disclaimer(final_output, user_input, attachments)
            if not final_output:
                try:
                    fallback_output = run_agent(
                        resolved_device_id,
                        name,
                        user_input,
                        messages,
                        runtime_context=runtime_context,
                    )
                    if fallback_output and str(fallback_output).strip():
                        final_output = _avoid_help_phrase(str(fallback_output).strip(), user_input)
                        final_output = _strip_image_disclaimer(final_output, user_input, attachments)
                except Exception as exc:
                    logger.warning(
                        "ws_fallback_failed name=%s device_id=%s error=%s",
                        name,
                        resolved_device_id,
                        exc,
                    )
            if not final_output:
                final_output = "No response."
            _write_agent_log(
                resolved_device_id,
                name,
                "chat_logs.jsonl",
                {
                    "event": "assistant_reply",
                    "output": final_output,
                    "assistant_id": resolved_assistant_id,
                },
            )
            message_tokens = count_message_tokens(messages or [])
            usage = message_tokens + count_tokens(user_input) + count_tokens(final_output)
            await websocket.send_text(
                json.dumps(
                    {"done": True, "output": final_output, "usage": usage},
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            await websocket.send_text(json.dumps({"error": str(exc)}, ensure_ascii=False))


@app.websocket("/ws/realtime/agents/{name}/run")
async def run_agent_realtime_websocket(websocket: WebSocket, name: str) -> None:
    await websocket.accept()
    query_params = websocket.query_params
    api_key = get_openai_api_key()
    realtime_model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-mini-realtime-preview")
    voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
    audio_format = os.getenv("OPENAI_REALTIME_AUDIO_FORMAT", "pcm16")
    sample_rate = int(os.getenv("OPENAI_REALTIME_SAMPLE_RATE", "24000"))

    def query_value(*keys: str) -> str | None:
        for key in keys:
            value = query_params.get(key)
            if value is None:
                continue
            value = str(value).strip()
            if value:
                return value
        return None

    async def stream_audio_from_text(response_text: str, payload: dict[str, Any]) -> tuple[bool, str]:
        audio_enabled = str(payload.get("audio_enabled") or "").strip().lower() in {"1", "true", "yes"}
        text_value = str(response_text or "").strip()
        tts_instructions = (
            "You are a text-to-speech engine. Speak the user's text verbatim, "
            "without adding, removing, or paraphrasing any words. Output only the spoken text."
        )
        _ws_debug(
            f"realtime_audio_from_text start enabled={audio_enabled} text_len={len(text_value)} name={name} device_id={resolved_device_id}"
        )
        if not audio_enabled or not api_key or not text_value:
            return False, ""
        audio_sent = False
        transcript_parts: list[str] = []
        try:
            async with websockets.connect(
                f"wss://api.openai.com/v1/realtime?model={realtime_model}",
                extra_headers={
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
            ) as rt:
                session_update = {
                    "type": "session.update",
                    "session": {
                        "instructions": tts_instructions,
                        "modalities": ["text", "audio"],
                        "voice": payload.get("voice") or voice,
                        "input_audio_format": "pcm16",
                        "output_audio_format": audio_format,
                    },
                }
                await rt.send(json.dumps(session_update))
                response_create = {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text", "audio"],
                        "instructions": tts_instructions,
                        "temperature": 0.6,
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text_value}],
                            }
                        ],
                    },
                }
                await rt.send(json.dumps(response_create))
                async for message in rt:
                    event = json.loads(message)
                    event_type = str(event.get("type") or "")
                    _ws_debug(f"realtime_audio_from_text_event={event_type} keys={list(event.keys())}")
                    if event_type == "session.updated":
                        session_info = event.get("session") or {}
                        _ws_debug(
                            f"realtime_audio_from_text_session modalities={session_info.get('modalities')} audio={session_info.get('audio')}"
                        )
                    if event_type == "error":
                        _ws_debug(f"realtime_audio_from_text_error={event.get('error')}")
                        break
                    if event_type in {"response.audio_transcript.delta", "response.output_audio_transcript.delta"}:
                        transcript_delta = event.get("delta") or event.get("text") or ""
                        if transcript_delta:
                            transcript_parts.append(str(transcript_delta))
                        continue
                    if event_type in {"response.audio_transcript.done", "response.output_audio_transcript.done"}:
                        transcript_done = event.get("text") or ""
                        if transcript_done:
                            transcript_parts.append(str(transcript_done))
                        continue
                    if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                        delta = event.get("delta") or event.get("audio") or event.get("data") or ""
                        if delta:
                            audio_sent = True
                            if not await _safe_ws_send(
                                websocket,
                                {
                                    "type": "audio_delta",
                                    "data": delta,
                                    "format": audio_format,
                                    "sample_rate": sample_rate,
                                },
                            ):
                                return False
                        continue
                    if event_type in {"response.audio.done", "response.output_audio.done"}:
                        await _safe_ws_send(websocket, {"type": "audio_done"})
                        break
                    if event_type in {"response.completed", "response.done", "error"}:
                        break
        except Exception as exc:
            _ws_debug(f"realtime_audio_from_text error: {exc}")
            _ws_debug(traceback.format_exc())
            return False, ""
        _ws_debug("realtime_audio_from_text done")
        transcript_text = "".join(transcript_parts).strip()
        _ws_debug(f"realtime_audio_from_text_transcript_len={len(transcript_text)}")
        return audio_sent, transcript_text

    while True:
        try:
            raw_payload = await websocket.receive_text()
        except WebSocketDisconnect:
            break

        try:
            payload: dict[str, Any] = {}
            if raw_payload:
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    payload = {"input": raw_payload}

            user_input = str(payload.get("input") or payload.get("text") or payload.get("message") or "").strip()
            tts_only = str(payload.get("tts_only") or "").strip().lower() in {"1", "true", "yes"}
            speak_text = str(payload.get("speak_text") or "").strip()
            audio_base64 = payload.get("audio") or payload.get("data") or ""
            is_audio = str(payload.get("type") or "").strip().lower() == "audio" and bool(audio_base64)
            _ws_debug(
                f"payload name={name} input_len={len(user_input)} audio_enabled={payload.get('audio_enabled')}"
            )
            if tts_only and not speak_text:
                speak_text = user_input
            if not user_input and not is_audio and not (tts_only and speak_text):
                await websocket.send_text(json.dumps({"type": "error", "error": "input is required."}))
                continue

            resolved_device_id = payload.get("device_id")
            if resolved_device_id is None or str(resolved_device_id).strip() == "":
                resolved_device_id = query_value("device_id", "deviceId")
            if resolved_device_id is None or str(resolved_device_id).strip() == "":
                resolved_device_id = find_agent_device_id(name)
            context = _load_agent_context(resolved_device_id, name)
            if not (isinstance(context, dict) and context.get("postman_requests")):
                fallback_device_id = find_agent_device_id(name)
                if fallback_device_id and str(fallback_device_id) != str(resolved_device_id):
                    resolved_device_id = fallback_device_id
                    context = _load_agent_context(resolved_device_id, name)
            sanitized_messages = _sanitize_messages(payload.get("messages"))
            attachments = _normalize_attachments(payload.get("attachments"))
            audio_enabled_flag = str(payload.get("audio_enabled") or "").strip().lower() in {"1", "true", "yes"}
            runtime_context = {
                "auth_token": payload.get("auth_token") or query_value("auth_token", "authToken", "token"),
                "api_base": payload.get("api_base") or query_value("api_base", "apiBase"),
                "device_id": resolved_device_id,
                "assistant_id": payload.get("assistant_id") or payload.get("agent_id"),
                "attachments": attachments,
            }
        except Exception as exc:
            _ws_debug(f"realtime_ws_payload_error: {exc}")
            _ws_debug(traceback.format_exc())
            await _safe_ws_send(websocket, {"type": "error", "error": str(exc)})
            continue
        async def _attach_usage(
            done_payload: dict[str, Any],
            usage_messages: list[dict[str, str]] | None,
            output_text: str,
            audio_sent_flag: bool,
        ) -> None:
            input_usage = count_message_tokens(usage_messages or []) + count_tokens(user_input)
            output_usage = count_tokens(output_text)
            usage = input_usage + output_usage
            usage_reported = await _report_usage_to_api(
                runtime_context.get("api_base"),
                runtime_context.get("auth_token"),
                usage,
                input_usage,
                output_usage,
                "ai-audio" if audio_sent_flag else "ai-text",
                "audio" if audio_sent_flag else "text",
            )
            done_payload["usage"] = usage
            done_payload["input_usage"] = input_usage
            done_payload["output_usage"] = output_usage
            done_payload["model"] = realtime_model
            done_payload["usage_reported"] = usage_reported
        if tts_only and speak_text:
            audio_sent, transcript_text = await stream_audio_from_text(speak_text, payload)
            done_text = transcript_text.strip() or speak_text
            await _safe_ws_send(
                websocket,
                {
                    "type": "done",
                    "output": done_text,
                    "audio_transcript": transcript_text.strip() or done_text,
                    "audio_sent": audio_sent,
                    "model": realtime_model,
                },
            )
            continue
        if is_audio:
            sample_rate = int(payload.get("sample_rate") or payload.get("sampleRate") or sample_rate)
            audio_b64 = str(audio_base64)
            byte_len = _pcm_byte_len_from_base64(audio_b64)
            if byte_len <= 0:
                await _safe_ws_send(
                    websocket,
                    {
                        "type": "error",
                        "error": "Invalid audio payload.",
                        "code": "input_audio_too_short",
                    },
                )
                continue
            duration_ms = _pcm_duration_ms_from_base64(audio_b64, sample_rate)
            if duration_ms < 100:
                await _safe_ws_send(
                    websocket,
                    {
                        "type": "error",
                        "error": "Audio too short. Please speak a bit longer.",
                        "code": "input_audio_too_short",
                    },
                )
                continue
            instructions = str(context.get("realtime_instructions") or "").strip()
            if not instructions:
                instructions = f"You are the assistant named {name}."

            messages = _sanitize_messages(payload.get("messages"))
            attachment_note = _attachment_system_note(attachments)
            analysis_note = _analyze_image_attachments(attachments, user_input)
            if analysis_note:
                if attachment_note:
                    attachment_note = (attachment_note + "\n\nImage analysis:\n" + analysis_note).strip()
                else:
                    attachment_note = ("Image analysis:\n" + analysis_note).strip()
            if attachment_note:
                messages = [{"role": "system", "content": attachment_note}] + (messages or [])

            _write_agent_log(
                resolved_device_id,
                name,
                "prompt_logs.jsonl",
                {
                    "event": "user_prompt",
                    "input": "voice_input",
                    "messages": messages or [],
                    "attachments": attachments,
                },
            )


            try:
                pcm_bytes = base64.b64decode(audio_b64, validate=False)
                wav_bytes = _pcm16_to_wav_bytes(pcm_bytes, sample_rate)
                client = OpenAI(api_key=api_key)
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
                )
                transcript = getattr(transcription, "text", "") or ""
            except Exception as exc:
                _ws_debug(f"realtime_transcribe_error: {exc}")
                await websocket.send_text(json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False))
                continue

            transcript = transcript.strip()
            if not transcript:
                await websocket.send_text(json.dumps({"type": "error", "error": "Empty transcription."}, ensure_ascii=False))
                continue

            await websocket.send_text(
                json.dumps({"type": "input_transcript_delta", "delta": transcript}, ensure_ascii=False)
            )
            await websocket.send_text(
                json.dumps({"type": "input_transcript_done"}, ensure_ascii=False)
            )

            try:
                response_text = run_agent(
                    resolved_device_id,
                    name,
                    transcript,
                    messages=sanitized_messages,
                    runtime_context=runtime_context,
                )
            except Exception as exc:
                response_text = f"Unable to fetch the requested data: {exc}"

            response_text = _strip_image_disclaimer(response_text, transcript, attachments)
            _write_agent_log(
                resolved_device_id,
                name,
                "chat_logs.jsonl",
                {
                    "event": "assistant_reply",
                    "output": response_text,
                },
            )
            await websocket.send_text(
                json.dumps({"type": "text_delta", "delta": response_text}, ensure_ascii=False)
            )
            audio_transcript = ""
            audio_sent, audio_transcript = await stream_audio_from_text(response_text, payload)
            final_text = response_text
            payload_done = {"type": "done", "output": final_text}
            if audio_sent:
                payload_done["audio_transcript"] = final_text
            await _attach_usage(payload_done, sanitized_messages, final_text, audio_sent)
            await websocket.send_text(json.dumps(payload_done, ensure_ascii=False))
            if audio_sent:
                await websocket.send_text(json.dumps({"type": "audio_done"}, ensure_ascii=False))
            continue
        if _is_services_request(user_input):
            response_text = _services_response(user_input, context)
            response_text = _strip_image_disclaimer(response_text, user_input, attachments)
            _write_agent_log(
                resolved_device_id,
                name,
                "prompt_logs.jsonl",
                {
                    "event": "user_prompt",
                    "input": user_input,
                    "messages": sanitized_messages,
                    "attachments": attachments,
                },
            )
            _write_agent_log(
                resolved_device_id,
                name,
                "chat_logs.jsonl",
                {
                    "event": "assistant_reply",
                    "output": response_text,
                },
            )
            await websocket.send_text(
                json.dumps({"type": "text_delta", "delta": response_text}, ensure_ascii=False)
            )
            audio_transcript = ""
            audio_sent, audio_transcript = await stream_audio_from_text(response_text, payload)
            final_text = response_text
            payload_done = {"type": "done", "output": final_text}
            if audio_sent:
                payload_done["audio_transcript"] = final_text
            await _attach_usage(payload_done, sanitized_messages, final_text, audio_sent)
            await websocket.send_text(json.dumps(payload_done, ensure_ascii=False))
            continue
        if (not runtime_helpers._is_services_query(user_input)) and runtime_helpers._has_date_like(
            user_input
        ) and isinstance(context, dict) and context.get("postman_requests"):
            try:
                response_text = run_agent(
                    resolved_device_id,
                    name,
                    user_input,
                    messages=sanitized_messages,
                    runtime_context={**runtime_context, "disable_followup": True},
                )
            except Exception as exc:
                response_text = f"Unable to fetch the requested data: {exc}"
            response_text = _strip_image_disclaimer(response_text, user_input, attachments)
            _write_agent_log(
                resolved_device_id,
                name,
                "prompt_logs.jsonl",
                {
                    "event": "user_prompt",
                    "input": user_input,
                    "messages": sanitized_messages,
                    "attachments": attachments,
                },
            )
            _write_agent_log(
                resolved_device_id,
                name,
                "chat_logs.jsonl",
                {
                    "event": "assistant_reply",
                    "output": response_text,
                },
            )
            await websocket.send_text(
                json.dumps({"type": "text_delta", "delta": response_text}, ensure_ascii=False)
            )
            audio_transcript = ""
            audio_sent, audio_transcript = await stream_audio_from_text(response_text, payload)
            final_text = response_text
            payload_done = {"type": "done", "output": final_text}
            if audio_sent:
                payload_done["audio_transcript"] = final_text
            await _attach_usage(payload_done, sanitized_messages, final_text, audio_sent)
            await websocket.send_text(json.dumps(payload_done, ensure_ascii=False))
            continue
        if isinstance(context, dict) and context.get("postman_requests"):
            messages_for_check = sanitized_messages
            is_selection = bool(re.fullmatch(r"\s*\d+\s*", user_input))
            last_options = ""
            options_list = []
            for item in reversed(messages_for_check or []):
                if item.get("role") != "assistant":
                    continue
                content = str(item.get("content") or "")
                if content:
                    last_options = content
                    for line in content.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        match = re.match(r"(?:-?\s*)?(?:Option\s*)?\d+\s*[:.)-]\s*(.+)", line, flags=re.I)
                        if match:
                            options_list.append(match.group(1).strip())
                            continue
                    break
            if not is_selection and last_options:
                is_selection = len(user_input.strip()) <= 40 and not runtime_helpers._has_date_like(user_input)
            services_list = [s.lower() for s in _services_from_context(context)] if isinstance(context, dict) else []
            options_are_services = bool(options_list) and services_list and all(
                opt.lower() in services_list for opt in options_list
            )
            if last_options and services_list:
                last_lower = last_options.lower()
                if any(s in last_lower for s in services_list):
                    options_are_services = True
            if is_selection and last_options and options_are_services:
                minimal_history = messages_for_check
                if last_options:
                    minimal_history = [{"role": "assistant", "content": last_options}]
                try:
                    response_text = run_agent(
                        resolved_device_id,
                        name,
                        user_input,
                        messages=minimal_history,
                        runtime_context={**runtime_context, "skip_auto_postman": True},
                    )
                except Exception as exc:
                    response_text = f"Unable to fetch the requested data: {exc}"
                response_text = _strip_image_disclaimer(response_text, user_input, attachments)
                _write_agent_log(
                    resolved_device_id,
                    name,
                    "prompt_logs.jsonl",
                    {
                        "event": "user_prompt",
                        "input": user_input,
                        "messages": messages_for_check,
                        "attachments": attachments,
                    },
                )
                _write_agent_log(
                    resolved_device_id,
                    name,
                    "chat_logs.jsonl",
                    {
                        "event": "assistant_reply",
                        "output": response_text,
                    },
                )
                await websocket.send_text(
                    json.dumps({"type": "text_delta", "delta": response_text}, ensure_ascii=False)
                )
                audio_transcript = ""
                audio_sent, audio_transcript = await stream_audio_from_text(response_text, payload)
                final_text = response_text
                payload_done = {"type": "done", "output": final_text}
                if audio_sent:
                    payload_done["audio_transcript"] = final_text
                await _attach_usage(payload_done, messages_for_check, final_text, audio_sent)
                await websocket.send_text(json.dumps(payload_done, ensure_ascii=False))
                continue
            if is_selection and last_options and not options_are_services:
                try:
                    module = runtime_helpers.load_agent_module(resolved_device_id, name)
                    setattr(module, "RUNTIME_CONTEXT", runtime_context)
                    flow_result = runtime_helpers._auto_postman_from_flow_selection(
                        module, messages_for_check, user_input
                    )
                    if flow_result:
                        flow_result = _strip_image_disclaimer(flow_result, user_input, attachments)
                        _write_agent_log(
                            resolved_device_id,
                            name,
                            "prompt_logs.jsonl",
                            {
                                "event": "user_prompt",
                                "input": user_input,
                                "messages": messages_for_check,
                                "attachments": attachments,
                            },
                        )
                        _write_agent_log(
                            resolved_device_id,
                            name,
                            "chat_logs.jsonl",
                            {
                                "event": "assistant_reply",
                                "output": flow_result,
                            },
                        )
                        await websocket.send_text(
                            json.dumps({"type": "text_delta", "delta": flow_result}, ensure_ascii=False)
                        )
                        audio_transcript = ""
                        audio_sent, audio_transcript = await stream_audio_from_text(flow_result, payload)
                        final_text = flow_result
                        payload_done = {"type": "done", "output": final_text}
                        if audio_sent:
                            payload_done["audio_transcript"] = final_text
                        await _attach_usage(payload_done, messages_for_check, final_text, audio_sent)
                        await websocket.send_text(json.dumps(payload_done, ensure_ascii=False))
                        continue
                except Exception:
                    pass
            if is_selection and last_options:
                try:
                    response_text = run_agent(
                        resolved_device_id,
                        name,
                        user_input,
                        messages=messages_for_check,
                        runtime_context=runtime_context,
                    )
                except Exception as exc:
                    response_text = f"Unable to fetch the requested data: {exc}"
                response_text = _strip_image_disclaimer(response_text, user_input, attachments)
                _write_agent_log(
                    resolved_device_id,
                    name,
                    "prompt_logs.jsonl",
                    {
                        "event": "user_prompt",
                        "input": user_input,
                        "messages": messages_for_check,
                        "attachments": attachments,
                    },
                )
                _write_agent_log(
                    resolved_device_id,
                    name,
                    "chat_logs.jsonl",
                    {
                        "event": "assistant_reply",
                        "output": response_text,
                    },
                )
                await websocket.send_text(
                    json.dumps({"type": "text_delta", "delta": response_text}, ensure_ascii=False)
                )
                audio_transcript = ""
                audio_sent, audio_transcript = await stream_audio_from_text(response_text, payload)
                final_text = response_text
                payload_done = {"type": "done", "output": final_text}
                if audio_sent:
                    payload_done["audio_transcript"] = final_text
                await _attach_usage(payload_done, messages_for_check, final_text, audio_sent)
                await websocket.send_text(json.dumps(payload_done, ensure_ascii=False))
                continue
        if isinstance(context, dict) and context.get("postman_requests"):
            endpoint = runtime_helpers._llm_select_endpoint(
                context.get("postman_requests") or [],
                user_input,
                flow_text=str(context.get("flow") or ""),
                allow_methods=None,
                purpose="realtime_auto",
            )
        else:
            endpoint = None
        if endpoint:
            try:
                response_text = run_agent(
                    resolved_device_id,
                    name,
                    user_input,
                    messages=None,
                    runtime_context=runtime_context,
                )
            except Exception as exc:
                response_text = f"Unable to fetch the requested data: {exc}"
            response_text = _strip_image_disclaimer(response_text, user_input, attachments)
            _write_agent_log(
                resolved_device_id,
                name,
                "prompt_logs.jsonl",
                {
                    "event": "user_prompt",
                    "input": user_input,
                    "messages": None,
                    "attachments": attachments,
                },
            )
            _write_agent_log(
                resolved_device_id,
                name,
                "chat_logs.jsonl",
                {
                    "event": "assistant_reply",
                    "output": response_text,
                },
            )
            await websocket.send_text(
                json.dumps({"type": "text_delta", "delta": response_text}, ensure_ascii=False)
            )
            audio_transcript = ""
            audio_sent, audio_transcript = await stream_audio_from_text(response_text, payload)
            final_text = response_text
            payload_done = {"type": "done", "output": final_text}
            if audio_sent:
                payload_done["audio_transcript"] = final_text
            await _attach_usage(payload_done, None, final_text, audio_sent)
            await websocket.send_text(json.dumps(payload_done, ensure_ascii=False))
            continue
        if isinstance(context, dict) and context.get("postman_requests"):
            try:
                response_text = run_agent(
                    resolved_device_id,
                    name,
                    user_input,
                    messages=sanitized_messages,
                    runtime_context=runtime_context,
                )
            except Exception as exc:
                response_text = f"Unable to fetch the requested data: {exc}"
            response_text = _strip_image_disclaimer(response_text, user_input, attachments)
            if not str(response_text or "").strip():
                response_text = "How can I help you today?"
                if _has_arabic_text(user_input):
                    response_text = "كيف يمكنني مساعدتك اليوم؟"
            _write_agent_log(
                resolved_device_id,
                name,
                "prompt_logs.jsonl",
                {
                    "event": "user_prompt",
                    "input": user_input,
                    "messages": sanitized_messages,
                    "attachments": attachments,
                },
            )
            _write_agent_log(
                resolved_device_id,
                name,
                "chat_logs.jsonl",
                {
                    "event": "assistant_reply",
                    "output": response_text,
                },
            )
            await websocket.send_text(
                json.dumps({"type": "text_delta", "delta": response_text}, ensure_ascii=False)
            )
            audio_transcript = ""
            audio_sent, audio_transcript = await stream_audio_from_text(response_text, payload)
            final_text = response_text
            payload_done = {"type": "done", "output": final_text}
            if audio_sent:
                payload_done["audio_transcript"] = final_text
            await _attach_usage(payload_done, sanitized_messages, final_text, audio_sent)
            await websocket.send_text(json.dumps(payload_done, ensure_ascii=False))
            continue
        # For agents with image/WhatsApp integrations, bypass realtime model and use agent flow.
        integrations = []
        if isinstance(context, dict):
            integrations = context.get("integrations") or []
        integration_ids = {str(item.get("id")).strip() for item in integrations if isinstance(item, dict)}
        if integration_ids.intersection({"gemini-image", "nano-banan-pro", "whatsapp"}):
            try:
                response_text = run_agent(
                    resolved_device_id,
                    name,
                    user_input,
                    messages=sanitized_messages,
                    runtime_context=runtime_context,
                )
            except Exception as exc:
                response_text = f"Unable to fetch the requested data: {exc}"
            response_text = _strip_image_disclaimer(response_text, user_input, attachments)
            _write_agent_log(
                resolved_device_id,
                name,
                "prompt_logs.jsonl",
                {
                    "event": "user_prompt",
                    "input": user_input,
                    "messages": sanitized_messages,
                    "attachments": attachments,
                },
            )
            _write_agent_log(
                resolved_device_id,
                name,
                "chat_logs.jsonl",
                {
                    "event": "assistant_reply",
                    "output": response_text,
                },
            )
            await websocket.send_text(
                json.dumps({"type": "text_delta", "delta": response_text}, ensure_ascii=False)
            )
            audio_transcript = ""
            audio_sent, audio_transcript = await stream_audio_from_text(response_text, payload)
            final_text = response_text
            payload_done = {"type": "done", "output": final_text}
            if audio_sent:
                payload_done["audio_transcript"] = final_text
            await _attach_usage(payload_done, sanitized_messages, final_text, audio_sent)
            await websocket.send_text(json.dumps(payload_done, ensure_ascii=False))
            continue
        # If the agent's LLM indicates a tool-driven action (e.g., image/WhatsApp), use the agent flow
        try:
            module = runtime_helpers.load_agent_module(resolved_device_id, name)
            setattr(module, "RUNTIME_CONTEXT", runtime_context)
            use_agent = False
            if hasattr(module, "_llm_send_requested"):
                use_agent = bool(module._llm_send_requested(sanitized_messages or [], user_input))
            if not use_agent and hasattr(module, "_llm_style_specified"):
                use_agent = bool(module._llm_style_specified(user_input))
            if use_agent:
                response_text = run_agent(
                    resolved_device_id,
                    name,
                    user_input,
                    messages=sanitized_messages,
                    runtime_context=runtime_context,
                )
                response_text = _strip_image_disclaimer(response_text, user_input, attachments)
                _write_agent_log(
                    resolved_device_id,
                    name,
                    "prompt_logs.jsonl",
                    {
                        "event": "user_prompt",
                        "input": user_input,
                        "messages": sanitized_messages,
                        "attachments": attachments,
                    },
                )
                _write_agent_log(
                    resolved_device_id,
                    name,
                    "chat_logs.jsonl",
                    {
                        "event": "assistant_reply",
                        "output": response_text,
                    },
                )
                await websocket.send_text(
                    json.dumps({"type": "text_delta", "delta": response_text}, ensure_ascii=False)
                )
                audio_transcript = ""
                audio_sent, audio_transcript = await stream_audio_from_text(response_text, payload)
                final_text = response_text
                payload_done = {"type": "done", "output": final_text}
                if audio_sent:
                    payload_done["audio_transcript"] = final_text
                await _attach_usage(payload_done, sanitized_messages, final_text, audio_sent)
                await websocket.send_text(json.dumps(payload_done, ensure_ascii=False))
                continue
        except Exception:
            pass
        instructions = str(context.get("realtime_instructions") or "").strip()
        if not instructions:
            instructions = f"You are the assistant named {name}."

        messages = _sanitize_messages(payload.get("messages"))
        attachment_note = _attachment_system_note(attachments)
        analysis_note = _analyze_image_attachments(attachments, user_input)
        if analysis_note:
            if attachment_note:
                attachment_note = (attachment_note + "\n\nImage analysis:\n" + analysis_note).strip()
            else:
                attachment_note = ("Image analysis:\n" + analysis_note).strip()
        if attachment_note:
            messages = [{"role": "system", "content": attachment_note}] + (messages or [])
        input_text = _build_realtime_input_text(user_input, messages)
        _write_agent_log(
            resolved_device_id,
            name,
            "prompt_logs.jsonl",
            {
                "event": "user_prompt",
                "input": user_input,
                "messages": messages or [],
                "attachments": attachments,
            },
        )

        audio_enabled = str(payload.get("audio_enabled") or "").strip().lower() in {"1", "true", "yes"}
        output_parts: list[str] = []
        audio_transcript_parts: list[str] = []
        audio_sent = False
        last_response_event: dict[str, Any] | None = None

        try:
            async with websockets.connect(
                f"wss://api.openai.com/v1/realtime?model={realtime_model}",
                extra_headers={
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
            ) as rt:
                session_modalities = ["text", "audio"] if audio_enabled else ["text"]
                session = {
                    "instructions": instructions,
                    "modalities": session_modalities,
                }
                if audio_enabled:
                    session["voice"] = payload.get("voice") or voice
                    session["input_audio_format"] = "pcm16"
                    session["output_audio_format"] = audio_format
                session_update = {"type": "session.update", "session": session}
                await rt.send(json.dumps(session_update))

                response_create = {
                    "type": "response.create",
                    "response": {
                        "modalities": session_modalities,
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": input_text}],
                            }
                        ],
                    },
                }
                await rt.send(json.dumps(response_create))

                async for message in rt:
                    event = json.loads(message)
                    event_type = str(event.get("type") or "")
                    _ws_debug(f"realtime_event={event_type} keys={list(event.keys())}")
                    if event_type == "session.updated":
                        session_info = event.get("session") or {}
                        _ws_debug(f"session_updated modalities={session_info.get('modalities')} audio={session_info.get('audio')}")
                        if event_type in {
                            "response.input_audio_transcription.delta",
                            "response.input_audio_transcript.delta",
                            "input_audio_transcription.delta",
                            "conversation.item.input_audio_transcription.delta",
                        }:
                            delta = event.get("delta") or event.get("text") or ""
                            if delta:
                                await websocket.send_text(
                                    json.dumps({"type": "input_transcript_delta", "delta": delta}, ensure_ascii=False)
                                )
                            continue
                        if event_type in {
                            "response.input_audio_transcription.done",
                            "response.input_audio_transcript.done",
                            "input_audio_transcription.done",
                            "conversation.item.input_audio_transcription.completed",
                        }:
                            await websocket.send_text(json.dumps({"type": "input_transcript_done"}, ensure_ascii=False))
                            continue
                    if event_type in {"response.text.delta", "response.output_text.delta"}:
                        delta = event.get("delta") or event.get("text") or ""
                        if delta:
                            output_parts.append(str(delta))
                            await websocket.send_text(
                                json.dumps({"type": "text_delta", "delta": delta}, ensure_ascii=False)
                            )
                        continue
                    if event_type in {"response.text.done", "response.output_text.done"}:
                        done_text = event.get("text") or ""
                        if done_text and not output_parts:
                            output_parts.append(str(done_text))
                            await websocket.send_text(
                                json.dumps({"type": "text_delta", "delta": done_text}, ensure_ascii=False)
                            )
                        continue
                    if event_type in {"response.audio_transcript.delta", "response.output_audio_transcript.delta"}:
                        transcript_delta = event.get("delta") or event.get("text") or ""
                        if transcript_delta:
                            audio_transcript_parts.append(str(transcript_delta))
                        continue
                    if event_type in {"response.audio_transcript.done", "response.output_audio_transcript.done"}:
                        transcript_done = event.get("text") or ""
                        if transcript_done:
                            audio_transcript_parts.append(str(transcript_done))
                        continue
                    if audio_enabled and event_type in {"response.audio.delta", "response.output_audio.delta"}:
                        delta = event.get("delta") or event.get("audio") or event.get("data") or ""
                        if delta:
                            audio_sent = True
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "audio_delta",
                                        "data": delta,
                                        "format": audio_format,
                                        "sample_rate": sample_rate,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        continue
                    if audio_enabled and event_type in {"response.audio.done", "response.output_audio.done"}:
                        audio_sent = True
                        await websocket.send_text(json.dumps({"type": "audio_done"}, ensure_ascii=False))
                        continue
                    if not output_parts and event_type.startswith("response") and "audio" not in event_type:
                        extracted = _extract_realtime_text(event, event_type)
                        if extracted:
                            for text_value in extracted:
                                output_parts.append(text_value)
                                await websocket.send_text(
                                    json.dumps({"type": "text_delta", "delta": text_value}, ensure_ascii=False)
                                )
                    if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                        delta = event.get("delta") or event.get("audio") or event.get("data") or ""
                        if delta:
                            audio_sent = True
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "audio_delta",
                                        "data": delta,
                                        "format": audio_format,
                                        "sample_rate": sample_rate,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        continue
                    if event_type in {"response.audio.done", "response.output_audio.done"}:
                        await websocket.send_text(json.dumps({"type": "audio_done"}, ensure_ascii=False))
                        continue
                    if event_type in {"response.completed", "response.done"}:
                        last_response_event = event
                        break
                    if event_type == "error":
                        await websocket.send_text(
                            json.dumps({"type": "error", "error": str(event.get("error") or "")}, ensure_ascii=False)
                        )
                        break

            final_output = "".join(output_parts).strip()
            if not final_output and last_response_event:
                response_payload = last_response_event.get("response") or {}
                output_items = response_payload.get("output") or []
                for item in output_items:
                    for content_item in item.get("content") or []:
                        content_type = content_item.get("type")
                        content_text = content_item.get("text") or content_item.get("output_text")
                        if content_type in {"output_text", "text"} and content_text:
                            final_output = str(content_text).strip()
                            break
                    if final_output:
                        break
            if not final_output:
                final_output = ""
            audio_transcript = "".join(audio_transcript_parts).strip()
            final_output = _strip_image_disclaimer(final_output, user_input, attachments)
            _write_agent_log(
                resolved_device_id,
                name,
                "chat_logs.jsonl",
                {
                    "event": "assistant_reply",
                    "output": final_output,
                },
            )
            input_usage = count_message_tokens(messages or []) + count_tokens(user_input)
            output_usage = count_tokens(final_output)
            usage = input_usage + output_usage
            audio_transcript = ""
            if not audio_sent:
                audio_sent, audio_transcript = await stream_audio_from_text(final_output, payload)
            integration_id = "ai-audio" if audio_sent else "ai-text"
            usage_reported = await _report_usage_to_api(
                runtime_context.get("api_base"),
                runtime_context.get("auth_token"),
                usage,
                input_usage,
                output_usage,
                integration_id,
                "audio" if audio_sent else "text",
            )
            payload_done = {
                "type": "done",
                "output": final_output,
                "usage": usage,
                "input_usage": input_usage,
                "output_usage": output_usage,
                "model": realtime_model,
                "usage_reported": usage_reported,
            }
            if audio_sent:
                payload_done["audio_transcript"] = final_output
            await websocket.send_text(
                json.dumps(payload_done, ensure_ascii=False)
            )
        except Exception as exc:
            _ws_debug(f"realtime_text_error: {exc}")
            _ws_debug(traceback.format_exc())
            await websocket.send_text(json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False))


@app.delete("/agents/{name}")
def delete_agent_endpoint(name: str, device_id: str | None = None) -> dict[str, str]:
    try:
        delete_agent(device_id, name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "deleted", "name": name}
