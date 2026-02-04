import os
import re
import shutil
from typing import Iterable

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")
_DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
_FALLBACK_DATA_DIR = os.getenv("ASSISTANT_DATA_DIR_FALLBACK", "/tmp/assistant-ai-data")


def _resolve_data_dir() -> str:
    candidate = os.getenv("ASSISTANT_DATA_DIR", _DEFAULT_DATA_DIR)
    try:
        os.makedirs(candidate, exist_ok=True)
        return candidate
    except PermissionError:
        os.makedirs(_FALLBACK_DATA_DIR, exist_ok=True)
        return _FALLBACK_DATA_DIR


DATA_DIR = _resolve_data_dir()
DEFAULT_DEVICE = "default"
DEFAULT_AGENTS_DIR = os.path.join(AGENTS_DIR, DEFAULT_DEVICE)
DEFAULT_DATA_DIR = os.path.join(DATA_DIR, DEFAULT_DEVICE)
_DEFAULT_MIGRATED = False


def ensure_dir(path: str, mode: int = 0o775) -> str:
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, mode)
    except Exception:
        pass
    return path


def normalize_device_id(device_id: str | int | None) -> str:
    if device_id is None:
        return DEFAULT_DEVICE
    value = str(device_id).strip()
    if not value:
        return DEFAULT_DEVICE
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def _migrate_default_storage() -> None:
    global _DEFAULT_MIGRATED
    if _DEFAULT_MIGRATED:
        return
    _DEFAULT_MIGRATED = True

    ensure_dir(AGENTS_DIR)
    ensure_dir(DATA_DIR)
    ensure_dir(DEFAULT_AGENTS_DIR)
    ensure_dir(DEFAULT_DATA_DIR)

    for entry in os.listdir(AGENTS_DIR):
        if entry in {DEFAULT_DEVICE, "__pycache__"}:
            continue
        if not entry.endswith(".py"):
            continue
        src = os.path.join(AGENTS_DIR, entry)
        dest = os.path.join(DEFAULT_AGENTS_DIR, entry)
        if os.path.isfile(src) and not os.path.exists(dest):
            shutil.move(src, dest)

    for entry in os.listdir(DATA_DIR):
        if entry in {DEFAULT_DEVICE, "__pycache__"}:
            continue
        src = os.path.join(DATA_DIR, entry)
        if not os.path.isdir(src):
            continue
        if re.fullmatch(r"\d+", entry):
            continue
        dest = os.path.join(DEFAULT_DATA_DIR, entry)
        if not os.path.exists(dest):
            shutil.move(src, dest)

    for entry in os.listdir(DEFAULT_DATA_DIR):
        if not re.fullmatch(r"\d+", entry):
            continue
        src = os.path.join(DEFAULT_DATA_DIR, entry)
        if not os.path.isdir(src):
            continue
        dest = os.path.join(DATA_DIR, entry)
        if os.path.exists(dest):
            continue
        agent_path = os.path.join(DEFAULT_AGENTS_DIR, f"{entry}.py")
        if os.path.isfile(agent_path):
            continue
        shutil.move(src, dest)


def device_agents_dir(device_id: str | int | None) -> str:
    _migrate_default_storage()
    normalized = normalize_device_id(device_id)
    if normalized == DEFAULT_DEVICE:
        return DEFAULT_AGENTS_DIR
    return os.path.join(AGENTS_DIR, normalized)


def ensure_device_agents_dir(device_id: str | int | None) -> str:
    return ensure_dir(device_agents_dir(device_id))


def agent_file_path(device_id: str | int | None, name: str) -> str:
    return os.path.join(ensure_device_agents_dir(device_id), f"{name}.py")


def device_data_dir(device_id: str | int | None) -> str:
    _migrate_default_storage()
    normalized = normalize_device_id(device_id)
    if normalized == DEFAULT_DEVICE:
        return DEFAULT_DATA_DIR
    return os.path.join(DATA_DIR, normalized)


def ensure_device_data_dir(device_id: str | int | None) -> str:
    return ensure_dir(device_data_dir(device_id))


def agent_dir(device_id: str | int | None, name: str) -> str:
    return ensure_dir(os.path.join(ensure_device_data_dir(device_id), name))


def agent_uploads_dir(device_id: str | int | None, name: str) -> str:
    return ensure_dir(os.path.join(agent_dir(device_id, name), "uploads"))


def list_agent_upload_paths(device_id: str | int | None, name: str) -> list[str]:
    uploads_dir = agent_uploads_dir(device_id, name)
    if not os.path.isdir(uploads_dir):
        return []
    files = []
    for entry in sorted(os.listdir(uploads_dir)):
        path = os.path.join(uploads_dir, entry)
        if os.path.isfile(path):
            files.append(path)
    return files


def list_agent_uploads(device_id: str | int | None, name: str) -> list[dict[str, int | str]]:
    uploads_dir = agent_uploads_dir(device_id, name)
    if not os.path.isdir(uploads_dir):
        return []
    items = []
    for entry in sorted(os.listdir(uploads_dir)):
        path = os.path.join(uploads_dir, entry)
        if not os.path.isfile(path):
            continue
        stat = os.stat(path)
        items.append(
            {
                "name": entry,
                "size": stat.st_size,
                "updated_at": int(stat.st_mtime),
            }
        )
    return items


def delete_agent_upload(device_id: str | int | None, name: str, filename: str) -> None:
    if not filename:
        return
    safe_name = os.path.basename(filename)
    if not safe_name:
        return
    path = os.path.join(agent_uploads_dir(device_id, name), safe_name)
    if os.path.isfile(path):
        os.remove(path)


def agent_vector_dir(device_id: str | int | None, name: str) -> str:
    return ensure_dir(os.path.join(agent_dir(device_id, name), "vectorstore"))


def agent_context_path(device_id: str | int | None, name: str) -> str:
    return os.path.join(agent_dir(device_id, name), "context.json")


def copy_files_into_agent(device_id: str | int | None, name: str, files: Iterable[str]) -> list[str]:
    uploads_dir = agent_uploads_dir(device_id, name)
    copied = []
    for path in files:
        if not os.path.isfile(path):
            continue
        dest = os.path.join(uploads_dir, os.path.basename(path))
        shutil.copy2(path, dest)
        copied.append(dest)
    return copied


def list_agent_names(device_id: str | int | None) -> list[str]:
    root = device_agents_dir(device_id)
    if not os.path.isdir(root):
        return []
    return sorted(
        [os.path.splitext(name)[0] for name in os.listdir(root) if name.endswith(".py")]
    )


def delete_agent(device_id: str | int | None, name: str) -> None:
    agent_path = os.path.join(device_agents_dir(device_id), f"{name}.py")
    if os.path.isfile(agent_path):
        os.remove(agent_path)
    data_path = os.path.join(device_data_dir(device_id), name)
    if os.path.isdir(data_path):
        shutil.rmtree(data_path)


def find_agent_device_id(name: str) -> str | None:
    """Locate the device_id that owns the agent file, if unambiguous."""
    if not name:
        return None
    _migrate_default_storage()
    candidates: list[str] = []
    if os.path.isdir(AGENTS_DIR):
        for entry in os.listdir(AGENTS_DIR):
            if entry in {"__pycache__"}:
                continue
            entry_path = os.path.join(AGENTS_DIR, entry)
            if not os.path.isdir(entry_path):
                continue
            agent_path = os.path.join(entry_path, f"{name}.py")
            if os.path.isfile(agent_path):
                candidates.append(entry)
    if len(candidates) == 1:
        return candidates[0]
    return None
