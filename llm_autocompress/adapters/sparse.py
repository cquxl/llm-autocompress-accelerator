from __future__ import annotations

import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from .compression import prune_2_4_tensor, validate_2_4_tensor
from ..utils import utc_now


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
    target = torch.device(device)
    dense_weight = weight.detach().to(device=target, dtype=torch.float16).contiguous()
    pruned = prune_2_4_tensor(dense_weight).contiguous()
    validation = validate_2_4_tensor(pruned)
    if not validation["valid"]:
        raise RuntimeError("generated weight does not satisfy 2:4")

    from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured

    SparseSemiStructuredTensor._FORCE_CUTLASS = False
    sparse_weight = to_sparse_semi_structured(pruned)
    records = []
    for tokens in token_counts:
        inputs = torch.randn(
            tokens,
            pruned.shape[1],
            device=target,
            dtype=torch.float16,
        )
        dense_reference = F.linear(inputs, pruned)
        sparse_reference = F.linear(inputs, sparse_weight)
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
        sparse_ms = measure(lambda: F.linear(inputs, sparse_weight))
        dense_median = statistics.median(dense_ms)
        sparse_median = statistics.median(sparse_ms)
        records.append(
            {
                "tokens": tokens,
                "m": tokens,
                "k": int(pruned.shape[1]),
                "n": int(pruned.shape[0]),
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
        "torch_operator": "torch._cslt_sparse_mm",
        "force_cutlass": False,
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

