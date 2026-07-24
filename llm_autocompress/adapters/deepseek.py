from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .compression import prune_2_4_tensor, validate_2_4_tensor
from .sparse import benchmark_cusparselt_weight
from ..schema import SAMOYEDS_ROOT
from ..utils import utc_now


def _weight_map(model_path: Path) -> dict[str, str]:
    index = model_path / "model.safetensors.index.json"
    if not index.exists():
        raise FileNotFoundError(f"missing sharded safetensors index: {index}")
    data = json.loads(index.read_text(encoding="utf-8"))
    return dict(data["weight_map"])


def load_weight(model_path: Path, key: str):
    from safetensors import safe_open

    mapping = _weight_map(model_path)
    shard = mapping.get(key)
    if shard is None:
        raise KeyError(f"weight not found: {key}")
    with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def expert_keys(model_path: Path, layer: int, experts: int = 2) -> list[str]:
    mapping = _weight_map(model_path)
    keys = []
    for expert in range(experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
            if key in mapping:
                keys.append(key)
    return keys


def benchmark_real_expert(
    model_path: Path,
    *,
    layer: int,
    expert: int,
    token_counts: list[int],
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "status": "running",
        "scope": "real_checkpoint_expert_linear",
        "model_path": str(model_path),
        "layer": layer,
        "expert": expert,
        "synthetic_weights": False,
    }
    key = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
    try:
        weight = load_weight(model_path, key)
        pruned = prune_2_4_tensor(weight)
        original_error = (weight.float() - pruned.float()).pow(2).mean().sqrt()
        reference_norm = weight.float().pow(2).mean().sqrt().clamp_min(1e-12)
        result.update(
            {
                "weight_key": key,
                "weight_shape": list(weight.shape),
                "two_of_four": validate_2_4_tensor(pruned),
                "relative_pruning_rmse": float((original_error / reference_norm).item()),
                "cusparselt": benchmark_cusparselt_weight(
                    weight,
                    token_counts=token_counts,
                    warmup=warmup,
                    iterations=iterations,
                    device=device,
                ),
                "status": "completed",
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return result


def pack_real_weight_with_samoyeds(
    model_path: Path,
    *,
    layer: int,
    expert: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "status": "running",
        "scope": "samoyeds_real_weight_pack",
        "synthetic_weights": False,
    }
    key = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
    try:
        if str(SAMOYEDS_ROOT) not in sys.path:
            sys.path.insert(0, str(SAMOYEDS_ROOT))
        from module.util import sparsifier

        weight = load_weight(model_path, key).half().cuda()
        values, indices, metadata = sparsifier(weight)
        result.update(
            {
                "status": "completed",
                "weight_key": key,
                "weight_shape": list(weight.shape),
                "packed_shapes": {
                    "values": list(values.shape),
                    "indices": list(indices.shape),
                    "metadata": list(metadata.shape),
                },
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return result


def benchmark_samoyeds_real_weight(
    model_path: Path,
    *,
    layer: int,
    expert: int,
    token_counts: list[int],
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    """Benchmark the Samoyeds CUDA extension with an actual checkpoint tensor."""
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "status": "running",
        "backend": "samoyeds",
        "scope": "real_checkpoint_expert_linear",
        "synthetic_weights": False,
        "full_model_claim": False,
    }
    key = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
    try:
        import torch

        extension_dir = SAMOYEDS_ROOT / "build" / "lib.linux-x86_64-cpython-310"
        for path in (str(extension_dir), str(SAMOYEDS_ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        import samoyeds_kernel
        from module.util import M, N, vector_length, sparsifier

        original_cpu = load_weight(model_path, key).half().contiguous()
        rows, columns = original_cpu.shape
        if rows % 2 or columns % 128:
            raise ValueError(
                f"Samoyeds requires rows divisible by 2 and K divisible by 128; "
                f"got {tuple(original_cpu.shape)}"
            )
        values_cpu, indices_cpu, metadata_cpu = sparsifier(original_cpu)
        pruned_cpu = samoyeds_kernel.get_pruned_value(
            values_cpu,
            indices_cpu,
            metadata_cpu,
            rows,
            columns,
            vector_length,
            N,
            M,
        )
        values = values_cpu.to(device)
        indices = indices_cpu.to(device)
        metadata = metadata_cpu.to(device)
        pruned = pruned_cpu.to(device)
        records: list[dict[str, Any]] = []

        for tokens in token_counts:
            aligned_tokens = ((int(tokens) + 63) // 64) * 64
            inputs = torch.randn(
                columns,
                aligned_tokens,
                dtype=torch.float16,
                device=device,
            )

            def samoyeds_call():
                return samoyeds_kernel.spmm_dense(
                    values,
                    indices,
                    metadata,
                    inputs,
                    rows,
                    columns,
                    aligned_tokens,
                    vector_length,
                    N,
                    M,
                )

            def dense_call():
                return pruned @ inputs

            dense_reference = dense_call()
            sparse_output = samoyeds_call()
            if sparse_output.shape != dense_reference.shape:
                raise RuntimeError(
                    f"Samoyeds output shape {tuple(sparse_output.shape)} != "
                    f"dense reference {tuple(dense_reference.shape)}"
                )
            max_abs_error = float(
                (dense_reference - sparse_output).abs().max().item()
            )
            relative_rmse = float(
                (
                    (dense_reference.float() - sparse_output.float())
                    .pow(2)
                    .mean()
                    .sqrt()
                    / dense_reference.float().pow(2).mean().sqrt().clamp_min(1e-12)
                ).item()
            )

            def measure(function):
                for _ in range(warmup):
                    function()
                torch.cuda.synchronize(device)
                samples = []
                for _ in range(iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    function()
                    end.record()
                    torch.cuda.synchronize(device)
                    samples.append(float(start.elapsed_time(end)))
                return samples

            dense_ms = measure(dense_call)
            sparse_ms = measure(samoyeds_call)
            records.append(
                {
                    "tokens_requested": int(tokens),
                    "tokens_measured": aligned_tokens,
                    "m": rows,
                    "k": columns,
                    "n": aligned_tokens,
                    "cublas_pruned_p50_ms": statistics.median(dense_ms),
                    "samoyeds_p50_ms": statistics.median(sparse_ms),
                    "same_mask_speedup": statistics.median(dense_ms)
                    / max(statistics.median(sparse_ms), 1e-12),
                    "max_abs_error_vs_same_mask_dense": max_abs_error,
                    "relative_rmse_vs_same_mask_dense": relative_rmse,
                    "numeric_validation_passed": relative_rmse <= 0.05,
                }
            )
        result.update(
            {
                "status": "completed",
                "software": {
                    "torch": torch.__version__,
                    "samoyeds_extension": getattr(
                        samoyeds_kernel, "__file__", "installed"
                    ),
                },
                "weight_key": key,
                "weight_shape": [rows, columns],
                "packed_shapes": {
                    "values": list(values.shape),
                    "indices": list(indices.shape),
                    "metadata": list(metadata.shape),
                },
                "effective_sparsity": float(
                    pruned_cpu.eq(0).sum().item() / pruned_cpu.numel()
                ),
                "records": records,
                "timing_note": "CUDA events and synchronization; dense reference uses the identical Samoyeds pruning mask.",
                "numeric_validation_passed": all(
                    record["numeric_validation_passed"] for record in records
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return result


def capture_router_activations(
    model_path: Path,
    *,
    prompts: list[str],
    layer: int,
    device: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "status": "running",
        "scope": "real_checkpoint_router_capture",
        "synthetic_activations": False,
        "layer": layer,
    }
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map={"": device},
        )
        layers = getattr(getattr(model, "model", model), "layers")
        gate = layers[layer].mlp.gate
        captured: list[dict[str, Any]] = []

        def hook(_module, args, output):
            hidden = args[0].detach().float().cpu()
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.detach().float().cpu()
            topk = min(6, logits.shape[-1])
            counts = torch.bincount(
                logits.topk(topk, dim=-1).indices.reshape(-1),
                minlength=logits.shape[-1],
            )
            captured.append(
                {
                    "hidden_shape": list(hidden.shape),
                    "hidden_mean": float(hidden.mean().item()),
                    "hidden_std": float(hidden.std().item()),
                    "router_shape": list(logits.shape),
                    "expert_token_counts": counts.tolist(),
                }
            )

        handle = gate.register_forward_hook(hook)
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                model(**tokens, use_cache=False)
        handle.remove()
        result.update({"status": "completed", "captures": captured})
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return result


def select_rose_experts(
    captures: list[dict[str, Any]],
    *,
    prune_ratio: float,
) -> dict[str, Any]:
    if not captures:
        raise ValueError("ROSE expert selection requires captured real router activations")
    counts = captures[0]["expert_token_counts"]
    total = [0] * len(counts)
    for capture in captures:
        values = capture["expert_token_counts"]
        if len(values) != len(total):
            raise ValueError("router captures use inconsistent expert counts")
        total = [left + int(right) for left, right in zip(total, values)]
    prune_count = min(len(total) - 1, max(0, round(len(total) * prune_ratio)))
    ordered = sorted(range(len(total)), key=lambda index: (total[index], index))
    pruned = ordered[:prune_count]
    kept = sorted(set(range(len(total))) - set(pruned))
    return {
        "algorithm": "rose",
        "structure": "expert",
        "source": "real_router_activations",
        "expert_token_counts": total,
        "pruned_experts": pruned,
        "kept_experts": kept,
        "prune_ratio_requested": prune_ratio,
        "prune_ratio_effective": prune_count / len(total),
        "full_model_claim": False,
    }


def benchmark_rose_channel(
    model_path: Path,
    *,
    layer: int,
    expert: int,
    channel_ratio: float,
    token_counts: list[int],
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "status": "running",
        "scope": "real_checkpoint_expert_channel",
        "algorithm": "rose",
        "structure": "channel",
        "synthetic_weights": False,
        "full_model_claim": False,
    }
    try:
        import statistics
        import time

        import torch
        import torch.nn.functional as F

        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        gate = load_weight(model_path, f"{prefix}.gate_proj.weight").float()
        up = load_weight(model_path, f"{prefix}.up_proj.weight").float()
        down = load_weight(model_path, f"{prefix}.down_proj.weight").float()
        score = gate.norm(dim=1) + up.norm(dim=1) + down.norm(dim=0)
        original_channels = int(score.numel())
        keep_channels = max(64, round(original_channels * (1 - channel_ratio) / 64) * 64)
        keep_channels = min(original_channels, keep_channels)
        keep = torch.topk(score, k=keep_channels, largest=True).indices.sort().values
        gate_small = gate.index_select(0, keep).half().to(device).contiguous()
        up_small = up.index_select(0, keep).half().to(device).contiguous()
        down_small = down.index_select(1, keep).half().to(device).contiguous()
        gate_full = gate.half().to(device).contiguous()
        up_full = up.half().to(device).contiguous()
        down_full = down.half().to(device).contiguous()
        records = []

        def mlp(x, g, u, d):
            return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)

        for tokens in token_counts:
            x = torch.randn(tokens, gate.shape[1], device=device, dtype=torch.float16)
            original = mlp(x, gate_full, up_full, down_full)
            reduced = mlp(x, gate_small, up_small, down_small)
            gate_masked = torch.zeros_like(gate_full)
            up_masked = torch.zeros_like(up_full)
            down_masked = torch.zeros_like(down_full)
            keep_device = keep.to(device)
            gate_masked.index_copy_(0, keep_device, gate_small)
            up_masked.index_copy_(0, keep_device, up_small)
            down_masked.index_copy_(1, keep_device, down_small)
            masked_reference = mlp(x, gate_masked, up_masked, down_masked)
            equivalence_error = float(
                (masked_reference - reduced).abs().max().item()
            )
            original_relative_rmse = float(
                (
                    (original.float() - reduced.float()).pow(2).mean().sqrt()
                    / original.float().pow(2).mean().sqrt().clamp_min(1e-12)
                ).item()
            )

            def measure(fn):
                for _ in range(warmup):
                    fn()
                torch.cuda.synchronize(device)
                samples = []
                for _ in range(iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    fn()
                    end.record()
                    torch.cuda.synchronize(device)
                    samples.append(float(start.elapsed_time(end)))
                return samples

            full_ms = measure(lambda: mlp(x, gate_full, up_full, down_full))
            small_ms = measure(lambda: mlp(x, gate_small, up_small, down_small))
            records.append(
                {
                    "tokens": tokens,
                    "original_channels": original_channels,
                    "kept_channels": keep_channels,
                    "effective_channel_ratio": 1
                    - (keep_channels / original_channels),
                    "cublas_original_p50_ms": statistics.median(full_ms),
                    "cublas_reduced_p50_ms": statistics.median(small_ms),
                    "same_backend_speedup": statistics.median(full_ms)
                    / max(statistics.median(small_ms), 1e-12),
                    "masked_reference_max_abs_error": equivalence_error,
                    "relative_rmse_vs_original": original_relative_rmse,
                }
            )
        result.update(
            {
                "status": "completed",
                "layer": layer,
                "expert": expert,
                "original_channels": original_channels,
                "kept_channels": keep_channels,
                "records": records,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    return result


def prune_rose_unstructured_weight(
    model_path: Path,
    *,
    layer: int,
    expert: int,
    sparsity_ratio: float,
) -> dict[str, Any]:
    import torch

    key = f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
    weight = load_weight(model_path, key)
    count = min(weight.numel() - 1, max(0, round(weight.numel() * sparsity_ratio)))
    flat = weight.abs().flatten()
    threshold = torch.kthvalue(flat, k=max(1, count)).values if count else -1
    pruned = weight * weight.abs().gt(threshold)
    return {
        "generated_at": utc_now(),
        "status": "completed",
        "algorithm": "rose",
        "structure": "unstructured",
        "scope": "real_checkpoint_expert_weight",
        "weight_key": key,
        "weight_shape": list(weight.shape),
        "synthetic_weights": False,
        "sparsity": float(pruned.eq(0).sum().item() / pruned.numel()),
        "nonzero": int(pruned.ne(0).sum().item()),
        "full_model_claim": False,
    }
