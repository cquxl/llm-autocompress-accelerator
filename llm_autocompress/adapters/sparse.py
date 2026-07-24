from __future__ import annotations

import importlib
import importlib.util
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .compression import prune_2_4_tensor, validate_2_4_tensor
from ..schema import SAMOYEDS_ROOT, SITE_CONFIG, SKILL_ROOT
from ..utils import source_tree_fingerprint, utc_now


KERNEL_MODULE = "cusparselt24_kernel"
KERNEL_SOURCE = (
    SKILL_ROOT / "native" / "samoyeds_cusparselt" / "cusparselt24_mod.cu"
)


def cusparselt_extension_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("LLM_AUTOCOMPRESS_CUSPARSELT_EXTENSION")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            candidates.append(path)
        else:
            candidates.extend(path.glob(f"{KERNEL_MODULE}*.so"))
    build_root = SITE_CONFIG.dependency_root / "samoyeds-cusparselt"
    candidates.extend(build_root.glob(f"{KERNEL_MODULE}*.so"))
    candidates.extend(
        SAMOYEDS_ROOT.glob(
            f"build*/**/{KERNEL_MODULE}*.so"
        )
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def load_samoyeds_cusparselt_extension():
    # Import torch first so its shared libraries are globally visible before
    # loading a host-built extension with importlib.
    import torch  # noqa: F401

    candidates = cusparselt_extension_candidates()
    if candidates:
        path = candidates[0]
        existing = sys.modules.get(KERNEL_MODULE)
        if existing is not None and Path(existing.__file__).resolve() == path:
            return existing
        spec = importlib.util.spec_from_file_location(KERNEL_MODULE, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load Samoyeds cuSPARSELt extension: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[KERNEL_MODULE] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "CuSparseLtLinear"):
            raise ImportError(f"{path} does not expose CuSparseLtLinear")
        return module
    try:
        module = importlib.import_module(KERNEL_MODULE)
        if hasattr(module, "CuSparseLtLinear"):
            return module
    except ImportError:
        pass
    raise ImportError(
        "Samoyeds cusparselt24_kernel is not built. Run "
        "scripts/build_samoyeds_cusparselt.py --output-dir "
        f"{SITE_CONFIG.dependency_root / 'samoyeds-cusparselt'} --yes"
    )


def samoyeds_cusparselt_probe() -> dict[str, Any]:
    try:
        module = load_samoyeds_cusparselt_extension()
        return {
            "available": True,
            "module": KERNEL_MODULE,
            "path": str(Path(module.__file__).resolve()),
            "api": "CuSparseLtLinear",
            "implementation": "Samoyeds direct cuSPARSELt C API",
            "uses_torch_private_cslt": False,
        }
    except Exception as exc:
        return {
            "available": False,
            "module": KERNEL_MODULE,
            "candidates": [str(item) for item in cusparselt_extension_candidates()],
            "error": f"{type(exc).__name__}: {exc}",
            "uses_torch_private_cslt": False,
        }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def benchmark_cusparselt_weight(
    weight: Any,
    *,
    token_counts: list[int],
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuSPARSELt benchmarking")
    kernel = load_samoyeds_cusparselt_extension()
    target = torch.device(device)
    dense_weight = weight.detach().to(device=target, dtype=torch.float16).contiguous()
    pruned = prune_2_4_tensor(dense_weight).contiguous()
    validation = validate_2_4_tensor(pruned)
    if not validation["valid"]:
        raise RuntimeError("generated weight does not satisfy 2:4")
    if pruned.shape[0] % 16 or pruned.shape[1] % 16:
        raise RuntimeError(
            f"Samoyeds cuSPARSELt requires M/K multiples of 16, got {pruned.shape}"
        )
    packed_weight = kernel.CuSparseLtLinear(pruned)
    records = []
    for tokens in token_counts:
        inputs = torch.randn(
            tokens,
            pruned.shape[1],
            device=target,
            dtype=torch.float16,
        )
        padded_tokens = ((tokens + 15) // 16) * 16
        padded_inputs = (
            F.pad(inputs, (0, 0, 0, padded_tokens - tokens))
            if padded_tokens != tokens
            else inputs
        ).contiguous()
        dense_reference = F.linear(inputs, pruned)
        torch.cuda.synchronize(target)
        setup_started = time.perf_counter()
        sparse_reference = packed_weight.forward(padded_inputs)[:tokens]
        torch.cuda.synchronize(target)
        setup_ms = (time.perf_counter() - setup_started) * 1000
        max_abs_error = float((dense_reference - sparse_reference).abs().max().item())
        denom = dense_reference.abs().clamp_min(1e-5)
        max_rel_error = float(
            ((dense_reference - sparse_reference).abs() / denom).max().item()
        )

        def measure(fn):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize(target)
            samples: list[float] = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize(target)
                samples.append(float(start.elapsed_time(end)))
            return samples

        dense_ms = measure(lambda: F.linear(inputs, pruned))
        sparse_ms = measure(lambda: packed_weight.forward(padded_inputs))
        dense_median = statistics.median(dense_ms)
        sparse_median = statistics.median(sparse_ms)
        records.append(
            {
                "tokens": tokens,
                "m": tokens,
                "k": int(pruned.shape[1]),
                "n": int(pruned.shape[0]),
                "padded_tokens": padded_tokens,
                "plan_compress_search_ms": setup_ms,
                "dense_p50_ms": dense_median,
                "dense_p95_ms": _percentile(dense_ms, 0.95),
                "cusparselt_p50_ms": sparse_median,
                "cusparselt_p95_ms": _percentile(sparse_ms, 0.95),
                "same_backend_speedup": dense_median / max(sparse_median, 1e-12),
                "max_abs_error": max_abs_error,
                "max_rel_error": max_rel_error,
            }
        )
    return {
        "generated_at": utc_now(),
        "status": "completed",
        "backend": "cusparselt",
        "implementation": "samoyeds_cusparselt24_kernel",
        "kernel_api": "CuSparseLtLinear",
        "native_operator": "cusparseLtMatmul",
        "compression_api": "cusparseLtSpMMACompress",
        "algorithm_search_api": "cusparseLtMatmulSearch",
        "extension_path": str(Path(kernel.__file__).resolve()),
        "kernel_source": str(KERNEL_SOURCE),
        "kernel_source_sha256": source_tree_fingerprint(KERNEL_SOURCE.parent),
        "uses_torch_private_cslt": False,
        "uses_pytorch_cutlass_fallback": False,
        "weight_shape": list(pruned.shape),
        "weight_validation": validation,
        "records": records,
    }


def benchmark_cusparselt_checkpoint(
    model_path: Path,
    *,
    token_counts: list[int],
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "status": "running",
        "model_path": str(model_path),
        "backend": "cusparselt",
    }
    try:
        import torch.nn as nn
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        selected_name = None
        selected_weight = None
        for name, module in model.named_modules():
            if (
                isinstance(module, nn.Linear)
                and module.weight.ndim == 2
                and module.weight.shape[0] % 64 == 0
                and module.weight.shape[1] % 64 == 0
                and "lm_head" not in name
            ):
                selected_name = name
                selected_weight = module.weight
                break
        if selected_weight is None:
            raise RuntimeError("no cuSPARSELt-compatible linear weight found")
        result.update(
            benchmark_cusparselt_weight(
                selected_weight,
                token_counts=token_counts,
                warmup=warmup,
                iterations=iterations,
                device=device,
            )
        )
        result["linear_module"] = selected_name
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return result
