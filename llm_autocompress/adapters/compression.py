from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from ..models import ModelInfo
from ..schema import SPINFER_ROOT, CompressionRequest
from ..utils import (
    directory_size,
    model_fingerprint,
    utc_now,
    write_json,
)


def _symbol(modules: list[str], name: str) -> Any:
    errors = []
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(f"cannot import {name}; tried {errors}")


def _manifest_base(
    method: str,
    model: ModelInfo,
    request: CompressionRequest,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "running",
        "method": method,
        "source_model": model.resolved_path,
        "source_fingerprint": model_fingerprint(Path(model.resolved_path)),
        "source_bytes": model.parameter_bytes,
        "output_dir": str(output_dir),
        "calibration": {
            "dataset": request.calibration.dataset,
            "samples": request.calibration.samples,
            "sequence_length": request.calibration.sequence_length,
            "seed": request.calibration.seed,
        },
        "software": {"python": sys.version, "executable": sys.executable},
        "synthetic_weights": False,
    }


def _load_calibration_dataset(path: Path, samples: int, seed: int):
    from datasets import Dataset, concatenate_datasets

    if path.is_file() and path.suffix == ".arrow":
        dataset = Dataset.from_file(str(path))
    elif path.is_dir():
        arrow_files = sorted(path.rglob("*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(f"no Arrow dataset files below {path}")
        datasets = [Dataset.from_file(str(item)) for item in arrow_files[:4]]
        dataset = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    else:
        raise FileNotFoundError(f"calibration dataset not found: {path}")
    text_column = next(
        (name for name in ("text", "content", "sentence") if name in dataset.column_names),
        None,
    )
    if not text_column:
        raise ValueError(f"dataset has no supported text column: {dataset.column_names}")
    dataset = dataset.filter(
        lambda row: row[text_column] is not None
        and bool(str(row[text_column]).strip()),
        load_from_cache_file=False,
    )
    count = min(samples, len(dataset))
    dataset = dataset.shuffle(seed=seed).select(range(count))
    return Dataset.from_dict(
        {"text": [str(value) for value in dataset[text_column]]}
    )


def prune_2_4_tensor(weight: Any):
    """Build a legal 2:4 layout for kernel-only validation.

    Formal checkpoint pruning is delegated to Wanda/SparseGPT/D2Prune/ROSE.
    """
    import torch

    if weight.ndim != 2 or weight.shape[-1] % 4:
        raise ValueError(f"2:4 requires a 2D tensor with K divisible by 4, got {weight.shape}")
    original_shape = weight.shape
    grouped = weight.detach().reshape(-1, 4)
    keep = torch.topk(grouped.abs(), k=2, dim=1, largest=True).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(1, keep, True)
    return (grouped * mask).reshape(original_shape)


def validate_2_4_tensor(weight: Any) -> dict[str, Any]:
    grouped = weight.detach().reshape(-1, 4)
    nonzero = grouped.ne(0).sum(dim=1)
    valid = bool((nonzero <= 2).all().item())
    return {
        "valid": valid,
        "groups": int(nonzero.numel()),
        "groups_over_two_nonzeros": int((nonzero > 2).sum().item()),
        "groups_exactly_two_nonzeros": int((nonzero == 2).sum().item()),
        "sparsity": float(weight.eq(0).sum().item() / weight.numel()),
    }


def _quant_recipe(method: str, model_family: str | None = None):
    QuantizationModifier = _symbol(
        [
            "llmcompressor.modifiers.quantization",
            "llmcompressor.modifiers.quantization.modifiers",
        ],
        "QuantizationModifier",
    )
    if method == "gptq_w4a16":
        GPTQModifier = _symbol(
            [
                "llmcompressor.modifiers.quantization.gptq",
                "llmcompressor.modifiers.quantization",
            ],
            "GPTQModifier",
        )
        try:
            return [
                GPTQModifier(
                    targets="Linear",
                    scheme="W4A16",
                    ignore=["lm_head"],
                )
            ]
        except TypeError:
            return [
                QuantizationModifier(
                    targets="Linear",
                    scheme="W4A16",
                    ignore=["lm_head"],
                ),
                GPTQModifier(block_size=128, dampening_frac=0.01),
            ]
    if method == "awq_w4a16":
        AWQModifier = _symbol(
            [
                "llmcompressor.modifiers.awq",
                "llmcompressor.modifiers.quantization.awq",
            ],
            "AWQModifier",
        )
        try:
            return [
                AWQModifier(
                    targets="Linear",
                    scheme="W4A16",
                    ignore=["lm_head"],
                )
            ]
        except (TypeError, ValueError):
            return [
                AWQModifier(
                    config_groups={
                        "group_0": {
                            "targets": ["Linear"],
                            "weights": {
                                "num_bits": 4,
                                "type": "int",
                                "symmetric": True,
                                "strategy": "group",
                                "group_size": 128,
                            },
                        }
                    }
                ),
                QuantizationModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=["lm_head"],
                ),
            ]
    if method == "smoothquant_w8a8":
        SmoothQuantModifier = _symbol(
            [
                "llmcompressor.modifiers.smoothquant",
                "llmcompressor.modifiers.quantization.smoothquant",
            ],
            "SmoothQuantModifier",
        )
        smooth_kwargs: dict[str, Any] = {"smoothing_strength": 0.8}
        if model_family == "opt":
            smooth_kwargs["mappings"] = [
                [
                    ["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
                    "re:.*self_attn_layer_norm",
                ],
                [["re:.*fc1"], "re:.*final_layer_norm"],
            ]
        return [
            SmoothQuantModifier(**smooth_kwargs),
            QuantizationModifier(
                targets="Linear",
                scheme="W8A8",
                ignore=["lm_head"],
            ),
        ]
    raise ValueError(f"unsupported quantization method: {method}")


def compress_with_llmcompressor(
    method: str,
    model: ModelInfo,
    request: CompressionRequest,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _manifest_base(method, model, request, output_dir)
    manifest_path = output_dir / "compression_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    try:
        import compressed_tensors
        import llmcompressor
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        oneshot = _symbol(["llmcompressor"], "oneshot")
        manifest["software"].update(
            {
                "implementation": "llmcompressor",
                "llmcompressor": getattr(llmcompressor, "__version__", "unknown"),
                "compressed_tensors": getattr(
                    compressed_tensors, "__version__", "unknown"
                ),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }
        )
        write_json(manifest_path, manifest)
        source = model.resolved_path
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=request.model.trust_remote_code,
            local_files_only=request.execution.offline,
        )
        loaded = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype="auto",
            trust_remote_code=request.model.trust_remote_code,
            local_files_only=request.execution.offline,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )
        dataset = _load_calibration_dataset(
            Path(request.calibration.dataset),
            request.calibration.samples,
            request.calibration.seed,
        )
        recipe = _quant_recipe(method, model.family)
        oneshot(
            model=loaded,
            tokenizer=tokenizer,
            dataset=dataset,
            recipe=recipe,
            max_seq_length=request.calibration.sequence_length,
            num_calibration_samples=request.calibration.samples,
        )
        save_kwargs = {"safe_serialization": True, "max_shard_size": "5GB"}
        try:
            loaded.save_pretrained(output_dir, save_compressed=True, **save_kwargs)
        except TypeError:
            loaded.save_pretrained(output_dir, **save_kwargs)
        tokenizer.save_pretrained(output_dir)
        output_bytes = directory_size(output_dir)
        config = {}
        config_path = output_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "completed",
                "finished_at": utc_now(),
                "output_bytes": output_bytes,
                "compression_ratio": model.parameter_bytes / max(output_bytes, 1),
                "effective_weight_compression_ratio": (
                    4.0
                    if method in {"gptq_w4a16", "awq_w4a16"}
                    else 2.0
                ),
                "format": "compressed-tensors",
                "quantization_config": config.get("quantization_config"),
                "recipe": [type(item).__name__ for item in recipe],
            }
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    write_json(manifest_path, manifest)
    return manifest


def prepare_spinfer_phase2_script(destination: Path) -> dict[str, Any]:
    """Copy SpInfer's converter and disable its explicit fake-sparsity switch."""
    source = (
        SPINFER_ROOT
        / "end2end_inference"
        / "ft_tools"
        / "huggingface_opt_convert_Phase2.py"
    )
    if not source.exists():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8")
    marker = "FAKE_SPARSITY = True"
    if marker not in text:
        raise RuntimeError(
            "SpInfer converter layout changed; refusing to run because fake sparsity "
            "cannot be disabled deterministically"
        )
    patched = text.replace(marker, "FAKE_SPARSITY = False", 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(patched, encoding="utf-8")
    return {
        "source": str(source),
        "copy": str(destination),
        "fake_sparsity": False,
        "source_unchanged": True,
    }


def execute_compression(
    method: str,
    model: ModelInfo,
    request: CompressionRequest,
    output_dir: Path,
) -> dict[str, Any]:
    if method in {"gptq_w4a16", "awq_w4a16", "smoothquant_w8a8"}:
        return compress_with_llmcompressor(method, model, request, output_dir)
    raise ValueError(f"no direct compression adapter for {method}")
