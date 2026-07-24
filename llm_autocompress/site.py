from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "llm-autocompress" / "site.yaml"


def _first_existing(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists():
            return expanded.resolve()
    return fallback.expanduser().resolve()


def _configured_path(
    data: dict[str, Any],
    key: str,
    env_name: str,
    candidates: list[Path],
    fallback: Path,
    *,
    config_dir: Path,
) -> Path:
    raw = os.environ.get(env_name) or data.get(key)
    if raw:
        value = Path(os.path.expandvars(str(raw))).expanduser()
        if not value.is_absolute():
            value = config_dir / value
        return value.resolve()
    return _first_existing(candidates, fallback)


@dataclass(frozen=True, slots=True)
class SiteConfig:
    config_path: str
    model_root: Path
    data_root: Path
    run_root: Path
    dependency_root: Path
    d2prune_root: Path
    spinfer_root: Path
    samoyeds_root: Path
    runtime_env: str
    quant_env: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key, item in list(value.items()):
            if isinstance(item, Path):
                value[key] = str(item)
        return value

    def placeholders(self) -> dict[str, str]:
        return {
            "${MODEL_ROOT}": str(self.model_root),
            "${DATA_ROOT}": str(self.data_root),
            "${RUN_ROOT}": str(self.run_root),
            "${DEPENDENCY_ROOT}": str(self.dependency_root),
            "${D2PRUNE_ROOT}": str(self.d2prune_root),
            "${SPINFER_ROOT}": str(self.spinfer_root),
            "${SAMOYEDS_ROOT}": str(self.samoyeds_root),
        }


def config_path(explicit: str | Path | None = None) -> Path:
    raw = explicit or os.environ.get("LLM_AUTOCOMPRESS_CONFIG")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_CONFIG_PATH


def load_site_config(explicit: str | Path | None = None) -> SiteConfig:
    path = config_path(explicit)
    data: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"site config must be a mapping: {path}")
        data = loaded

    config_dir = path.parent
    workspace = REPO_ROOT.parent.parent
    dependencies = _configured_path(
        data,
        "dependency_root",
        "LLM_AUTOCOMPRESS_DEPENDENCY_ROOT",
        [REPO_ROOT / "dependencies", workspace / "dependencies"],
        REPO_ROOT / "dependencies",
        config_dir=config_dir,
    )
    model_root = _configured_path(
        data,
        "model_root",
        "LLM_AUTOCOMPRESS_MODEL_ROOT",
        [workspace / "cache" / "llm_weights", REPO_ROOT / "models"],
        REPO_ROOT / "models",
        config_dir=config_dir,
    )
    data_root = _configured_path(
        data,
        "data_root",
        "LLM_AUTOCOMPRESS_DATA_ROOT",
        [workspace / "cache" / "data", REPO_ROOT / "data"],
        REPO_ROOT / "data",
        config_dir=config_dir,
    )
    run_root = _configured_path(
        data,
        "run_root",
        "LLM_AUTOCOMPRESS_RUN_ROOT",
        [REPO_ROOT / "runs"],
        REPO_ROOT / "runs",
        config_dir=config_dir,
    )
    d2prune_root = _configured_path(
        data,
        "d2prune_root",
        "LLM_AUTOCOMPRESS_D2PRUNE_ROOT",
        [
            REPO_ROOT / "third_party" / "d2prune_core",
            workspace / "D2Prune",
            dependencies / "D2Prune",
        ],
        REPO_ROOT / "third_party" / "d2prune_core",
        config_dir=config_dir,
    )
    spinfer_root = _configured_path(
        data,
        "spinfer_root",
        "LLM_AUTOCOMPRESS_SPINFER_ROOT",
        [workspace / "SpInfer", dependencies / "SpInfer"],
        dependencies / "SpInfer",
        config_dir=config_dir,
    )
    samoyeds_root = _configured_path(
        data,
        "samoyeds_root",
        "LLM_AUTOCOMPRESS_SAMOYEDS_ROOT",
        [REPO_ROOT.parent, workspace / "Samoyeds", dependencies / "Samoyeds"],
        dependencies / "Samoyeds",
        config_dir=config_dir,
    )
    return SiteConfig(
        config_path=str(path),
        model_root=model_root,
        data_root=data_root,
        run_root=run_root,
        dependency_root=dependencies,
        d2prune_root=d2prune_root,
        spinfer_root=spinfer_root,
        samoyeds_root=samoyeds_root,
        runtime_env=str(
            os.environ.get("LLM_AUTOCOMPRESS_RUNTIME_ENV")
            or data.get("runtime_env")
            or "llm-autocompress-runtime"
        ),
        quant_env=str(
            os.environ.get("LLM_AUTOCOMPRESS_QUANT_ENV")
            or data.get("quant_env")
            or "llm-autocompress-quant"
        ),
    )


def expand_site_tokens(value: Any, site: SiteConfig | None = None) -> Any:
    current = site or load_site_config()
    if isinstance(value, str):
        expanded = value
        for token, replacement in current.placeholders().items():
            expanded = expanded.replace(token, replacement)
        return os.path.expandvars(expanded)
    if isinstance(value, list):
        return [expand_site_tokens(item, current) for item in value]
    if isinstance(value, dict):
        return {
            key: expand_site_tokens(item, current)
            for key, item in value.items()
        }
    return value


def save_site_config(
    values: dict[str, Any],
    explicit: str | Path | None = None,
) -> Path:
    path = config_path(explicit)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            existing.update(loaded)
    existing.update(
        {
            key: str(value)
            for key, value in values.items()
            if value is not None
        }
    )
    path.write_text(
        yaml.safe_dump(existing, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path
