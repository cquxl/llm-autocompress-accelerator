from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import DEFAULT_DATA_ROOT, DEFAULT_WEIGHTS_ROOT, SITE_CONFIG


MODEL_ALIASES = {
    "opt-125m": DEFAULT_WEIGHTS_ROOT / "opt-125m",
    "deepseek-v2-lite": DEFAULT_WEIGHTS_ROOT / "DeepSeek-V2-Lite",
    "deepseek-16b-moe": DEFAULT_WEIGHTS_ROOT / "DeepSeek-V2-Lite",
    "llama-3-8b": DEFAULT_WEIGHTS_ROOT / "Meta-Llama-3-8b",
    "llama-2-70b": DEFAULT_WEIGHTS_ROOT / "llama-2-70b",
    "mixtral": DEFAULT_WEIGHTS_ROOT / "Mistral-8x7B-v0.1",
}


def resolve_model_alias(value: str) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    normalized = value.strip().lower()
    if normalized in MODEL_ALIASES:
        return MODEL_ALIASES[normalized].resolve()
    matches = [
        item
        for key, item in MODEL_ALIASES.items()
        if normalized in key or key in normalized
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve local model alias: {value}")


def request_mapping_from_business(
    *,
    model: str,
    prompt: str = "",
    profile: str = "interactive",
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Translate an EdgeLite-style business request into the fixed request schema."""
    model_path = resolve_model_alias(model)
    text = f"{model} {prompt}".lower()
    is_moe = any(word in text for word in ("moe", "deepseek", "mixtral", "专家"))
    selected: list[str] = []
    keyword_methods = (
        (("gptq",), "gptq_w4a16"),
        (("awq",), "awq_w4a16"),
        (("smoothquant", "smooth quant"), "smoothquant_w8a8"),
        (("wanda", "万达"), "wanda_2_4"),
        (("sparsegpt",), "sparsegpt_unstructured"),
        (("d2prune",), "d2prune_2_4"),
        (("rose", "专家剪枝"), "rose_expert"),
    )
    for keywords, method in keyword_methods:
        if any(keyword in text for keyword in keywords):
            selected.append(method)
    if not selected:
        selected = (
            ["rose_expert", "rose_2_4", "rose_channel", "rose_unstructured"]
            if is_moe
            else [
                "gptq_w4a16",
                "awq_w4a16",
                "smoothquant_w8a8",
                "wanda_2_4",
                "sparsegpt_unstructured",
                "d2prune_2_4",
            ]
        )
    if "非结构" in text or "unstructured" in text:
        selected = [
            method.replace("_2_4", "_unstructured")
            if method.startswith(("wanda_", "d2prune_", "sparsegpt_"))
            else method
            for method in selected
        ]
    backends = ["auto"]
    explicit_backends = [
        name
        for name in ("vllm", "cusparselt", "samoyeds", "spinfer", "cublas")
        if name in text
    ]
    if explicit_backends:
        backends = explicit_backends
    return {
        "name": model_path.name.lower().replace("_", "-"),
        "model": {
            "path": str(model_path),
            "trust_remote_code": is_moe,
        },
        "methods": list(dict.fromkeys(selected)),
        "backends": backends,
        "calibration": {
            "dataset": str(DEFAULT_DATA_ROOT / "c4"),
            "samples": 128,
            "sequence_length": 2048,
        },
        "evaluation": {
            "dataset": str(DEFAULT_DATA_ROOT / "wikitext" / "wikitext-2-raw-v1"),
        },
        "constraints": {
            "max_relative_ppl_increase": 0.05,
            "min_same_backend_speedup": 1.0,
        },
        "pruning": {
            "sparsity_ratio": 0.5,
            "expert_ratio": 0.25,
            "channel_ratio": 0.25,
            "only_moe": is_moe,
            "tune_router": True,
        },
        "search": {
            "enabled": True,
            "quantization_first": True,
            "target_checkpoint_ratio": 2.0,
            "allow_pruning_fallback": True,
            "combine_with_best_quant": True,
            "pruning_granularity": "2:4",
            "max_trials": 3,
        },
        "workload": {
            "profile": profile,
            "batch_sizes": [1, 8],
            "input_lengths": [128, 512, 2048],
            "output_lengths": [32, 128],
            "warmup": 10,
            "iterations": 30,
        },
        "execution": {
            "device": "cuda:0",
            "tensor_parallel_size": 1,
            "offline": True,
        },
        "output_dir": output_dir or str(SITE_CONFIG.run_root),
        "business_request": prompt,
    }


def apply_web_preset(
    mapping: dict[str, Any],
    preset: str,
    *,
    pruning_granularity: str = "2:4",
    profile: str = "interactive",
) -> dict[str, Any]:
    """Apply an explicit Web demo scope without hiding the resulting method list."""
    result = {**mapping}
    model_path = str((result.get("model") or {}).get("path", "")).lower()
    is_moe = any(name in model_path for name in ("deepseek", "mixtral", "moe"))
    pruning_fallback = {
        "2:4": "rose_2_4" if is_moe else "wanda_2_4",
        "unstructured": (
            "rose_unstructured" if is_moe else "sparsegpt_unstructured"
        ),
        "channel": "rose_channel" if is_moe else "d2prune_channel",
        "expert": "rose_expert",
    }.get(pruning_granularity)
    if pruning_fallback is None:
        raise ValueError(f"unsupported pruning granularity: {pruning_granularity}")
    quant_probe = (
        ["smoothquant_w8a8", "awq_w4a16"]
        if profile == "prefill-heavy"
        else ["awq_w4a16", "gptq_w4a16"]
    )
    presets = {
        "auto-smoke": [quant_probe[0], pruning_fallback],
        "auto-full": [*quant_probe, pruning_fallback],
        "quantization": [
            "gptq_w4a16",
            "awq_w4a16",
            "smoothquant_w8a8",
        ],
        "pruning": (
            ["rose_expert", "rose_2_4", "rose_channel", "rose_unstructured"]
            if is_moe
            else [
                "wanda_2_4",
                "sparsegpt_unstructured",
                "d2prune_2_4",
            ]
        ),
        "dense": ["dense"],
    }
    if preset not in presets:
        raise ValueError(f"unknown Web preset: {preset}")
    result["methods"] = presets[preset]
    # Automatic comparison always needs dense baselines. "auto" keeps
    # Transformers/vLLM plus the method-specific deployment kernels in plan.
    result["backends"] = (
        ["transformers", "vllm"] if preset == "dense" else ["auto"]
    )
    result["name"] = f"{result['name']}-{preset}"
    result["web_preset"] = preset
    result["search"] = {
        **(result.get("search") or {}),
        "enabled": preset in {"auto-smoke", "auto-full"},
        "pruning_granularity": pruning_granularity,
        "max_trials": 2 if preset == "auto-smoke" else 3,
    }
    return result
