import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class PostmanRequest:
    name: str
    method: str
    url: str
    headers: dict[str, str]
    body: str | None
    responses: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "url": self.url,
            "headers": self.headers,
            "body": self.body,
            "responses": self.responses,
        }


def _collect_items(items: Iterable[dict[str, Any]], out: list[PostmanRequest]) -> None:
    for item in items:
        if "item" in item:
            _collect_items(item["item"], out)
            continue
        request = item.get("request", {})
        method = request.get("method", "GET")
        url = request.get("url", {})
        if isinstance(url, dict):
            raw_url = url.get("raw") or ""
        else:
            raw_url = str(url)
        headers = {h.get("key", ""): h.get("value", "") for h in request.get("header", [])}
        body = None
        if request.get("body"):
            body = request["body"].get("raw")
        responses = []
        for resp in item.get("response", []) or []:
            responses.append(
                {
                    "name": resp.get("name", ""),
                    "status": resp.get("status", ""),
                    "code": resp.get("code", ""),
                    "body": resp.get("body", ""),
                }
            )
        out.append(
            PostmanRequest(
                name=item.get("name", ""),
                method=method,
                url=raw_url,
                headers=headers,
                body=body,
                responses=responses,
            )
        )


def load_postman_collection(path: str) -> list[PostmanRequest]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    items = data.get("item", [])
    requests: list[PostmanRequest] = []
    _collect_items(items, requests)
    return requests


def load_postman_variables(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    variables = {}
    for item in data.get("variable", []) or []:
        key = item.get("key")
        value = item.get("value")
        if key and value is not None:
            variables[key] = str(value)
    return variables


def missing_postman_variables(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    missing = []
    for item in data.get("variable", []) or []:
        key = item.get("key")
        value = item.get("value")
        if key and (value is None or str(value).strip() == ""):
            missing.append({"key": key, "value": ""})
    return missing


def pick_login_request(requests: list[PostmanRequest]) -> PostmanRequest | None:
    best: PostmanRequest | None = None
    best_score = 0.0
    for req in requests:
        name = req.name.lower()
        url = req.url.lower()
        score = 0.0
        if req.method.upper() in {"POST", "PUT"}:
            score += 0.2
        if "login" in name or "/login" in url:
            score += 2.0
        if "sign in" in name or "signin" in name or "/signin" in url:
            score += 1.8
        if "auth" in name or "/auth" in url:
            score += 1.5
        if "token" in name or "/token" in url:
            score += 1.2
        if "oauth" in name or "/oauth" in url:
            score += 1.2
        if "session" in name or "/session" in url:
            score += 0.8
        if score > best_score:
            best_score = score
            best = req
    return best if best_score > 0 else None


def summarize_postman(requests: list[PostmanRequest]) -> str:
    lines: list[str] = []
    lines.append("Postman Collection Summary")
    for req in requests:
        lines.append(f"- {req.name} [{req.method}] {req.url}")
        if req.headers:
            lines.append("  Headers:")
            for key, value in req.headers.items():
                if key:
                    lines.append(f"    {key}: {value}")
        if req.body:
            lines.append("  Body:")
            lines.append(req.body.strip())
        for resp in req.responses:
            lines.append(f"  Response {resp.get('status', '')} ({resp.get('code', '')})")
            body = resp.get("body", "")
            if body:
                lines.append(body.strip())
    return "\n".join(lines)
