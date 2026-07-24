from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{os.getpid()}"


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"request must contain a mapping: {path}")
    return data


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_yaml(path: Path, value: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def command_text(argv: Iterable[str]) -> str:
    return shlex.join([str(item) for item in argv])


def run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    started = utc_now()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command_text(argv),
            "started_at": started,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command_text(argv),
            "started_at": started,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def import_probe(module: str) -> dict[str, Any]:
    code = (
        "import importlib,json;"
        f"m=importlib.import_module({module!r});"
        "print(json.dumps({'version':getattr(m,'__version__','installed'),"
        "'path':getattr(m,'__file__',None)}))"
    )
    result = run_command([os.sys.executable, "-c", code], timeout=20)
    if result["returncode"] == 0:
        try:
            result.update(json.loads(result["stdout"]))
            result["available"] = True
        except json.JSONDecodeError:
            result["available"] = False
    else:
        result["available"] = False
    return result


def directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def human_bytes(value: int | float | None) -> str:
    if value is None:
        return "-"
    amount = float(value)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or suffix == "TiB":
            return f"{amount:.2f} {suffix}"
        amount /= 1024
    return f"{amount:.2f} TiB"


def model_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    candidates = [
        path / "config.json",
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            digest.update(candidate.name.encode())
            digest.update(candidate.read_bytes())
    files = sorted(
        item
        for item in path.glob("*")
        if item.is_file() and item.suffix in {".safetensors", ".bin"}
    )
    for item in files:
        stat = item.stat()
        digest.update(item.name.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def source_tree_fingerprint(path: Path) -> str:
    """Hash distributable source content without depending on Git metadata."""
    digest = hashlib.sha256()
    source_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".txt",
        ".yaml",
        ".yml",
    }
    ignored_parts = {
        ".git",
        "__pycache__",
        "artifacts",
        "cache",
        "checkpoints",
        "logs",
        "out",
        "output",
        "runs",
        "weights",
    }
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path)
        if any(part in ignored_parts for part in relative.parts):
            continue
        if item.suffix.lower() not in source_suffixes:
            continue
        digest.update(relative.as_posix().encode())
        size = item.stat().st_size
        digest.update(str(size).encode())
        if size <= 4 * 1024 * 1024:
            digest.update(item.read_bytes())
    return digest.hexdigest()


def copy_metadata_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    patterns = (
        "*.json",
        "tokenizer.model",
        "*.txt",
        "*.py",
        "LICENSE*",
    )
    for pattern in patterns:
        for item in source.glob(pattern):
            if item.is_file():
                shutil.copy2(item, destination / item.name)


def ensure_within(path: Path, roots: Iterable[Path], *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    resolved_roots = [root.expanduser().resolve() for root in roots]
    if not any(resolved == root or root in resolved.parents for root in resolved_roots):
        allowed = ", ".join(str(root) for root in resolved_roots)
        raise ValueError(f"{label} must be under one of: {allowed}; got {resolved}")
    return resolved
