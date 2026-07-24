from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelInfo:
    requested_path: str
    resolved_path: str
    family: str
    kind: str
    architecture: str
    model_type: str
    dtype: str
    hidden_size: int | None
    intermediate_size: int | None
    layers: int | None
    experts: int | None
    experts_per_token: int | None
    parameter_bytes: int
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_model_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "config.json").exists():
        return candidate
    refs_main = candidate / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = candidate / "snapshots" / revision
        if (snapshot / "config.json").exists():
            return snapshot.resolve()
    snapshots = candidate / "snapshots"
    if snapshots.is_dir():
        choices = sorted(
            (item for item in snapshots.iterdir() if (item / "config.json").exists()),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        if choices:
            return choices[0].resolve()
    raise ValueError(f"cannot locate Hugging Face config.json under {candidate}")


def _int(config: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = config.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _family(model_type: str, architecture: str, config: dict[str, Any]) -> tuple[str, str]:
    text = f"{model_type} {architecture}".lower()
    experts = _int(config, "n_routed_experts", "num_local_experts", "num_experts")
    kind = "moe" if experts and experts > 1 else "dense"
    if "deepseek" in text:
        return "deepseek", kind
    if "mixtral" in text:
        return "mixtral", "moe"
    if "qwen" in text:
        return "qwen", kind
    if "llama" in text:
        return "llama", kind
    if model_type == "opt" or "opt" in architecture.lower():
        return "opt", kind
    if "mistral" in text:
        return "mistral", kind
    return model_type or "unknown", kind


def _parameter_bytes(path: Path) -> int:
    safetensors = [item for item in path.glob("*.safetensors") if item.is_file()]
    if safetensors:
        return sum(item.stat().st_size for item in safetensors)
    suffixes = {".bin", ".msgpack", ".h5"}
    return sum(
        item.stat().st_size
        for item in path.glob("*")
        if item.is_file() and item.suffix in suffixes
    )


def inspect_model(path: str | Path, family: str = "auto", kind: str = "auto") -> ModelInfo:
    resolved = resolve_model_path(path)
    config = json.loads((resolved / "config.json").read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    architecture = str(architectures[0]) if architectures else ""
    model_type = str(config.get("model_type", ""))
    inferred_family, inferred_kind = _family(model_type, architecture, config)
    return ModelInfo(
        requested_path=str(Path(path).expanduser().resolve()),
        resolved_path=str(resolved),
        family=inferred_family if family == "auto" else family,
        kind=inferred_kind if kind == "auto" else kind,
        architecture=architecture,
        model_type=model_type,
        dtype=str(config.get("torch_dtype", "auto")),
        hidden_size=_int(config, "hidden_size", "d_model"),
        intermediate_size=_int(
            config, "moe_intermediate_size", "intermediate_size", "ffn_dim"
        ),
        layers=_int(config, "num_hidden_layers", "n_layer"),
        experts=_int(config, "n_routed_experts", "num_local_experts", "num_experts"),
        experts_per_token=_int(
            config, "num_experts_per_tok", "num_experts_per_token", "top_k"
        ),
        parameter_bytes=_parameter_bytes(resolved),
        config=config,
    )
