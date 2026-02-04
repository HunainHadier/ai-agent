import json
import os
from dataclasses import dataclass
from typing import Iterable

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from app.postman import (
    load_postman_collection,
    load_postman_variables,
    pick_login_request,
    summarize_postman,
)
from app.rag import build_vectorstore, collect_texts
from app.config import get_openai_api_key, get_openai_model
from app.storage import (
    agent_context_path,
    agent_file_path,
    copy_files_into_agent,
    list_agent_upload_paths,
    normalize_device_id,
)


@dataclass
class AgentSpec:
    name: str
    description: str | None
    services: str | None
    flow: str | None
    device_id: str | None = None
    postman_path: str | None = None
    file_paths: list[str] | None = None
    postman_variables: dict[str, str] | None = None
    language: str | None = None
    integrations: list[dict] | None = None
    brand_profile: dict | None = None
    disable_design: bool | None = None
    enable_design: bool | None = None


class AgentBuilder:
    def __init__(self, template_dir: str | None = None) -> None:
        template_root = template_dir or os.path.join(os.path.dirname(__file__), os.pardir, "templates")
        self.env = Environment(loader=FileSystemLoader(template_root), autoescape=False)

    @staticmethod
    def _llm_client() -> OpenAI | None:
        try:
            api_key = get_openai_api_key()
        except RuntimeError:
            return None
        return OpenAI(api_key=api_key)

    @staticmethod
    def _service_endpoint_map(services: str, requests: list[dict[str, str]]) -> dict[str, list[str]]:
        service_lines = [s.strip() for s in services.replace(";", "\n").splitlines() if s.strip()]
        mapping: dict[str, list[str]] = {service: [] for service in service_lines}
        if not service_lines or not requests:
            return mapping
        client = AgentBuilder._llm_client()
        if client is None:
            return mapping
        endpoints = [
            {
                "name": str(req.get("name") or ""),
                "method": str(req.get("method") or "GET"),
                "url": str(req.get("url") or ""),
                "description": str(req.get("description") or ""),
            }
            for req in requests
            if isinstance(req, dict)
        ]
        prompt = (
            "You map services to API endpoints. "
            "Return JSON where each key is the service name and each value is a list of endpoint names "
            "chosen from the provided endpoints. If none apply, return an empty list."
        )
        user_text = json.dumps(
            {"services": service_lines, "endpoints": endpoints},
            ensure_ascii=False,
        )
        try:
            response = client.chat.completions.create(
                model=get_openai_model(),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=0,
                max_tokens=800,
            )
        except Exception:
            return mapping
        content = ""
        try:
            content = response.choices[0].message.content or ""
        except Exception:
            content = ""
        try:
            parsed = json.loads(content)
        except Exception:
            return mapping
        if not isinstance(parsed, dict):
            return mapping
        endpoint_names = {e["name"] for e in endpoints if e.get("name")}
        for service, value in parsed.items():
            if service not in mapping:
                continue
            if isinstance(value, list):
                mapping[service] = [name for name in value if name in endpoint_names]
        return mapping

    @staticmethod
    def _build_realtime_instructions(context: dict) -> str:
        description = str(context.get("description") or "").strip()
        services = str(context.get("services") or "").strip()
        flow = str(context.get("flow") or "").strip()
        integrations = context.get("integrations") or []
        language = str(context.get("language") or "").strip()
        disable_design = bool(context.get("disable_design"))
        requirements = str(context.get("requirements") or "").strip()
        behavior = str(context.get("behavior") or "").strip()
        parts: list[str] = [
            "You are the assistant.",
            "Answer as a helpful colleague working with the user.",
            "Match the user's language and dialect automatically.",
            "If a service requires integrations or knowledge base data that are not available, use reasonable dummy data and proceed.",
        ]
        if description:
            parts.append(f"Description:\n{description}")
        if services:
            parts.append(f"Services:\n{services}")
        if flow:
            parts.append(f"Flow steps:\n{flow}")
        if requirements:
            parts.append(f"Requirements:\n{requirements}")
        if behavior:
            parts.append(f"Behavior:\n{behavior}")
        if isinstance(integrations, list):
            summary = []
            for item in integrations:
                if not isinstance(item, dict):
                    continue
                name_value = str(item.get("name") or item.get("id") or "").strip()
                if name_value:
                    summary.append(name_value)
            if summary:
                parts.append("Integrations available: " + ", ".join(summary))
        if language:
            parts.append(f"Preferred language: {language}.")
        if disable_design:
            parts.append("Do not ask about design or marketing details.")
        return "\n\n".join(parts).strip()

    def build(self, spec: AgentSpec) -> str:
        device_id = spec.device_id
        files = spec.file_paths or []
        copy_files_into_agent(device_id, spec.name, files)
        context_path = agent_context_path(device_id, spec.name)
        existing_context = {}
        if os.path.isfile(context_path):
            with open(context_path, "r", encoding="utf-8") as handle:
                existing_context = json.load(handle)

        postman_summary = ""
        postman_requests: list[dict[str, str]] = []
        postman_variables: dict[str, str] = {}
        postman_login: dict[str, str] | None = None
        if spec.postman_path:
            requests = load_postman_collection(spec.postman_path)
            postman_summary = summarize_postman(requests)
            postman_requests = [req.to_dict() for req in requests]
            postman_variables = load_postman_variables(spec.postman_path)
            login_request = pick_login_request(requests)
            postman_login = login_request.to_dict() if login_request else None
        elif existing_context:
            postman_summary = existing_context.get("postman_summary", "")
            postman_requests = existing_context.get("postman_requests", [])
            postman_variables = existing_context.get("postman_variables", {})
            postman_login = existing_context.get("postman_login")
        if spec.postman_variables:
            postman_variables.update(spec.postman_variables)

        description = spec.description or existing_context.get("description", "")
        services = spec.services or existing_context.get("services", "")
        flow = spec.flow or existing_context.get("flow", "")
        language = spec.language or existing_context.get("language")
        disable_design = (
            spec.disable_design
            if spec.disable_design is not None
            else existing_context.get("disable_design")
        )
        enable_design = (
            spec.enable_design
            if spec.enable_design is not None
            else existing_context.get("enable_design")
        )
        integrations = (
            spec.integrations
            if spec.integrations is not None
            else existing_context.get("integrations", [])
        )
        if isinstance(integrations, list):
            integrations = [
                item
                for item in integrations
                if isinstance(item, dict) and item.get("enabled", True)
            ]
        else:
            integrations = []
        brand_profile = (
            spec.brand_profile
            if spec.brand_profile is not None
            else existing_context.get("brand_profile", {})
        )
        if not isinstance(brand_profile, dict):
            brand_profile = {}

        service_map = self._service_endpoint_map(services, postman_requests)

        context = {
            "description": description,
            "services": services,
            "flow": flow,
            "language": language,
            "postman_summary": postman_summary,
            "postman_requests": postman_requests,
            "postman_variables": postman_variables,
            "postman_login": postman_login,
            "service_endpoint_map": service_map,
            "has_postman": bool(postman_requests),
            "integrations": integrations,
            "brand_profile": brand_profile,
            "device_id": spec.device_id,
            "disable_design": disable_design,
            "enable_design": enable_design,
            "realtime_instructions": "",
        }

        all_uploads = list_agent_upload_paths(device_id, spec.name)
        texts = collect_texts(all_uploads)
        try:
            build_vectorstore(spec.name, texts, device_id)
        except Exception:
            pass

        context["realtime_instructions"] = self._build_realtime_instructions(context)
        with open(context_path, "w", encoding="utf-8") as handle:
            json.dump(context, handle, indent=2)
        _apply_file_permissions(context_path)

        template = self.env.get_template("agent_template.py.j2")
        rendered = template.render(
            agent_name=spec.name,
            device_id=normalize_device_id(device_id),
            context=context,
        )

        agent_path = agent_file_path(device_id, spec.name)
        with open(agent_path, "w", encoding="utf-8") as handle:
            handle.write(rendered)
        _apply_file_permissions(agent_path)
        return agent_path


def _apply_file_permissions(path: str, mode: int = 0o664) -> None:
    try:
        os.chmod(path, mode)
    except Exception:
        pass
