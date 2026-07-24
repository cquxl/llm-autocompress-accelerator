from __future__ import annotations

import threading
import traceback
import os
from pathlib import Path
from typing import Any, Callable

from .adapters.compression import execute_compression, prepare_spinfer_phase2_script
from .adapters.deepseek import (
    benchmark_real_expert,
    benchmark_rose_channel,
    benchmark_samoyeds_real_weight,
    capture_router_activations,
    prune_rose_unstructured_weight,
    select_rose_experts,
)
from .adapters.inference import (
    benchmark_transformers,
    benchmark_vllm,
    evaluate_transformers_artifact,
)
from .adapters.pruning import execute_pruning
from .adapters.sparse import benchmark_cusparselt_checkpoint
from .environment import inspect_environment
from .models import ModelInfo, inspect_model
from .planner import build_plan
from .reporting import evaluate_run, render_report, summarize_benchmark, write_results_csv
from .schema import CompressionRequest, SITE_CONFIG, SKILL_ROOT, load_request
from .schema import request_from_mapping
from .utils import (
    atomic_write_text,
    read_json,
    run_command,
    run_id,
    utc_now,
    write_json,
    write_yaml,
)


Log = Callable[[str], None]
Event = Callable[[dict[str, Any]], None]


def _emit(logger: Log | None, message: str) -> None:
    if logger:
        logger(f"[{utc_now()}] {message}")


def _notify(handler: Event | None, value: dict[str, Any]) -> None:
    if handler:
        handler({"time": utc_now(), **value})


def _cancelled(cancel_event: threading.Event | None) -> bool:
    return bool(cancel_event and cancel_event.is_set())


def _profile_performance(result: dict[str, Any], profile: str) -> float | None:
    summary = summarize_benchmark(result)
    if profile == "throughput":
        return summary.get("decode_tokens_per_second")
    if profile == "prefill-heavy":
        return summary.get("prefill_tokens_per_second")
    tpot = summary.get("tpot_ms")
    return None if tpot in (None, 0) else 1000 / float(tpot)


def _kernel_speedup(result: dict[str, Any]) -> float | None:
    values: list[float] = []
    for records in (
        result.get("records") or [],
        (result.get("cusparselt") or {}).get("records") or [],
        (result.get("kernel") or {}).get("records") or [],
    ):
        for record in records:
            value = record.get("same_backend_speedup")
            if value is None:
                value = record.get("same_mask_speedup")
            if value is not None:
                values.append(float(value))
    if not values:
        return None
    values.sort()
    return values[len(values) // 2]


def _counts(request: CompressionRequest, mode: str) -> tuple[list[int], int, int]:
    tokens = request.workload.input_lengths[:1] if mode == "smoke" else request.workload.input_lengths
    warmup = min(2, request.workload.warmup) if mode == "smoke" else request.workload.warmup
    iterations = min(5, request.workload.iterations) if mode == "smoke" else request.workload.iterations
    return tokens, warmup, iterations


def create_run_dir(request: CompressionRequest, explicit: Path | None = None) -> Path:
    root = Path(request.output_dir).expanduser().resolve()
    run_dir = explicit.resolve() if explicit else root / run_id(request.name)
    if not request.execution.allow_external_output:
        safe_root = SITE_CONFIG.run_root.resolve()
        if run_dir != safe_root and safe_root not in run_dir.parents:
            raise ValueError(
                f"output must stay below {safe_root}; set "
                "execution.allow_external_output=true to override"
            )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    for directory in (
        run_dir,
        run_dir / "artifacts",
        run_dir / "benchmarks",
        run_dir / "quality",
        run_dir / "logs",
        run_dir / "charts",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return run_dir


def inspect_request(request: CompressionRequest) -> dict[str, Any]:
    request.validate()
    return {
        "environment": inspect_environment(),
        "model": inspect_model(
            request.model.path,
            family=request.model.family,
            kind=request.model.kind,
        ).to_dict(),
    }


def plan_request(
    request: CompressionRequest,
    *,
    run_dir: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any], ModelInfo]:
    request.validate()
    target = create_run_dir(request, run_dir)
    env = inspect_environment()
    model = inspect_model(
        request.model.path,
        family=request.model.family,
        kind=request.model.kind,
    )
    plan = build_plan(request, model, env, target)
    write_yaml(target / "request.yaml", request.to_dict())
    write_json(target / "env.json", env)
    write_json(target / "plan.json", plan)
    return target, plan, env, model


def _failed_result(candidate: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "status": "failed",
        "backend": candidate["backend"],
        "scope": candidate["scope"],
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def _write_candidate(
    run_dir: Path,
    candidate: dict[str, Any],
    value: dict[str, Any],
) -> None:
    write_json(run_dir / "benchmarks" / f"{candidate['id']}.json", value)


def _quantization_name(method: str) -> str | None:
    return {
        "gptq_w4a16": "compressed-tensors",
        "awq_w4a16": "compressed-tensors",
        "smoothquant_w8a8": "compressed-tensors",
    }.get(method)


def _execute_deepseek_micro(
    candidate: dict[str, Any],
    request: CompressionRequest,
    model: ModelInfo,
    *,
    mode: str,
) -> dict[str, Any]:
    model_path = Path(model.resolved_path)
    tokens, warmup, iterations = _counts(request, mode)
    method = candidate["method"]
    if method == "rose_2_4":
        if candidate["backend"] == "samoyeds":
            return benchmark_samoyeds_real_weight(
                model_path,
                layer=1,
                expert=0,
                token_counts=tokens,
                warmup=warmup,
                iterations=iterations,
                device=request.execution.device,
            )
        return benchmark_real_expert(
            model_path,
            layer=1,
            expert=0,
            token_counts=tokens,
            warmup=warmup,
            iterations=iterations,
            device=request.execution.device,
        )
    if method == "rose_channel":
        return benchmark_rose_channel(
            model_path,
            layer=1,
            expert=0,
            channel_ratio=request.pruning.channel_ratio,
            token_counts=tokens,
            warmup=warmup,
            iterations=iterations,
            device=request.execution.device,
        )
    if method == "rose_unstructured":
        value = prune_rose_unstructured_weight(
            model_path,
            layer=1,
            expert=0,
            sparsity_ratio=request.pruning.sparsity_ratio,
        )
        value.update(
            {
                "backend": candidate["backend"],
                "status": "failed",
                "error": (
                    "real expert pruning completed, but this checkout has no "
                    "shape-compatible DeepSeek SpInfer launcher; no timing was fabricated"
                ),
            }
        )
        return value
    if method == "rose_expert":
        capture = capture_router_activations(
            model_path,
            prompts=request.evaluation.generation_prompts[:1]
            if mode == "smoke"
            else request.evaluation.generation_prompts,
            layer=1,
            device=request.execution.device,
        )
        if capture.get("status") != "completed":
            return capture
        selection = select_rose_experts(
            capture["captures"],
            prune_ratio=request.pruning.expert_ratio,
        )
        kernel = benchmark_samoyeds_real_weight(
            model_path,
            layer=1,
            expert=selection["kept_experts"][0],
            token_counts=tokens,
            warmup=warmup,
            iterations=iterations,
            device=request.execution.device,
        )
        return {
            "generated_at": utc_now(),
            "status": (
                "completed" if kernel.get("status") == "completed" else "failed"
            ),
            "backend": "samoyeds",
            "scope": "expert_selection_and_real_expert_kernel",
            "router_capture": capture,
            "expert_selection": selection,
            "kernel": kernel,
            "records": kernel.get("records", []),
            "full_model_claim": False,
            "error": kernel.get("error"),
        }
    raise ValueError(f"no DeepSeek micro adapter for {method}")


def _execute_candidate(
    candidate: dict[str, Any],
    request: CompressionRequest,
    model: ModelInfo,
    run_dir: Path,
    *,
    mode: str,
    event_handler: Event | None = None,
) -> dict[str, Any]:
    method = candidate["method"]
    backend = candidate["backend"]
    artifact_dir = (
        Path(candidate["artifact_dir"]) if candidate.get("artifact_dir") else None
    )

    if method.startswith("rose_") and model.kind == "moe":
        return _execute_deepseek_micro(
            candidate,
            request,
            model,
            mode=mode,
        )

    if method in {"gptq_w4a16", "awq_w4a16", "smoothquant_w8a8"}:
        if artifact_dir is None:
            raise RuntimeError("compression candidate has no artifact directory")
        completed = run_command(
            [
                "conda",
                "run",
                "-n",
                SITE_CONFIG.quant_env,
                "python",
                str(SKILL_ROOT / "scripts" / "quant_worker.py"),
                "--method",
                method,
                "--request",
                str(run_dir / "request.yaml"),
                "--output",
                str(artifact_dir),
            ],
            timeout=request.execution.timeout_seconds,
            env={
                **os.environ,
                "HF_DATASETS_CACHE": str(artifact_dir / "datasets_cache"),
            },
        )
        manifest = read_json(
            artifact_dir / "compression_manifest.json",
            default={
                "status": "failed",
                "error": completed.get("stderr") or completed.get("stdout"),
            },
        )
        manifest["converter_process"] = completed
        write_json(artifact_dir / "compression_manifest.json", manifest)
    elif method != "dense":
        if artifact_dir is None:
            raise RuntimeError("pruning candidate has no artifact directory")
        manifest = execute_pruning(
            method,
            model,
            request,
            artifact_dir,
            mode=mode,
        )
    else:
        manifest = {"status": "completed"}

    if manifest.get("status") != "completed":
        return {
            "generated_at": utc_now(),
            "status": "failed",
            "backend": backend,
            "scope": candidate["scope"],
            "error": manifest.get("error", "compression failed"),
        }

    model_path = artifact_dir or Path(model.resolved_path)
    if method != "dense":
        quality = evaluate_transformers_artifact(
            model_path,
            request,
            mode=mode,
            event_callback=event_handler,
        )
        write_json(run_dir / "quality" / f"{candidate['id']}.json", quality)
    if backend == "vllm":
        return benchmark_vllm(
            model_path,
            request,
            mode=mode,
            quantization=_quantization_name(method),
            event_callback=event_handler,
        )
    if backend == "cusparselt":
        tokens, warmup, iterations = _counts(request, mode)
        return benchmark_cusparselt_checkpoint(
            model_path,
            token_counts=tokens,
            warmup=warmup,
            iterations=iterations,
            device=request.execution.device,
        )
    if backend == "spinfer":
        safety = prepare_spinfer_phase2_script(artifact_dir / "spinfer_phase2.py")
        return {
            "generated_at": utc_now(),
            "status": "failed",
            "backend": "spinfer",
            "scope": candidate["scope"],
            "conversion_safety": safety,
            "error": (
                "real pruned checkpoint was created and fake sparsity was disabled, "
                "but no compatible FasterTransformer/SpInfer executable was found for "
                f"{model.family}; no end-to-end metric was fabricated"
            ),
        }
    if backend == "transformers":
        return benchmark_transformers(
            model_path,
            request,
            mode=mode,
            event_callback=event_handler,
        )
    raise RuntimeError(f"backend adapter is not implemented: {backend}")


def _svg_chart(evaluation: dict[str, Any]) -> str:
    items = [
        item
        for item in evaluation.get("evaluated", [])
        if item.get("same_backend_speedup") is not None
    ]
    width, height = 900, max(220, 80 + len(items) * 42)
    max_value = max([1.0] + [float(item["same_backend_speedup"]) for item in items])
    bars = []
    for index, item in enumerate(items):
        y = 55 + index * 42
        bar_width = 600 * float(item["same_backend_speedup"]) / max_value
        color = "#15803d" if item.get("accepted") else "#64748b"
        bars.append(
            f'<text x="10" y="{y + 18}" font-size="13">{item["id"]}</text>'
            f'<rect x="260" y="{y}" width="{bar_width:.1f}" height="24" fill="{color}"/>'
            f'<text x="{270 + bar_width:.1f}" y="{y + 18}" font-size="13">'
            f'{item["same_backend_speedup"]:.3f}×</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="10" y="28" font-size="18" font-weight="bold">'
        'Same-backend compression speedup</text>'
        + "".join(bars)
        + "</svg>"
    )


def regenerate_report(run_dir: Path) -> dict[str, Any]:
    request = load_request(run_dir / "request.yaml")
    plan = read_json(run_dir / "plan.json")
    env = read_json(run_dir / "env.json")
    if not plan or not env:
        raise FileNotFoundError("run directory lacks plan.json or env.json")
    evaluation = evaluate_run(run_dir, request, plan)
    report = render_report(request, plan, evaluation, env)
    write_json(run_dir / "evaluation.json", evaluation)
    atomic_write_text(run_dir / "report.md", report)
    write_results_csv(run_dir / "results.csv", evaluation)
    atomic_write_text(run_dir / "charts" / "speedup.svg", _svg_chart(evaluation))
    return evaluation


def run_request(
    request: CompressionRequest,
    *,
    mode: str,
    yes: bool,
    run_dir: Path | None = None,
    cancel_event: threading.Event | None = None,
    logger: Log | None = print,
    event_handler: Event | None = None,
) -> Path:
    if not yes:
        raise PermissionError("model conversion and benchmark execution require --yes")
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    _notify(
        event_handler,
        {
            "type": "phase",
            "phase": "inspect",
            "message": "检查环境、模型、数据和推理后端",
        },
    )
    _emit(logger, "inspecting environment, model, data and backend capabilities")
    target, plan, _env, model = plan_request(request, run_dir=run_dir)
    _notify(
        event_handler,
        {
            "type": "plan_ready",
            "run_dir": str(target),
            "model": model.to_dict(),
            "candidates": plan["candidates"],
        },
    )
    _emit(logger, f"run directory: {target}")
    candidates = plan["candidates"]
    quant_methods = {"gptq_w4a16", "awq_w4a16", "smoothquant_w8a8"}
    pruning_prefixes = ("wanda_", "sparsegpt_", "d2prune_", "rose_")
    baseline_ppl: float | None = None
    quant_probes: list[dict[str, Any]] = []
    baseline_results: dict[str, dict[str, Any]] = {}
    trials_executed = 0
    business_target_met = False
    selected_candidate: str | None = None
    search_trace: dict[str, Any] = {
        "strategy": plan["selection_policy"].get("staged_search", {}),
        "stages": [
            {
                "stage": "quantization_probe",
                "status": "running",
                "methods": [
                    item["method"]
                    for item in candidates
                    if item["method"] in quant_methods
                ],
            }
        ],
        "quantization_probes": quant_probes,
        "pruning_fallback_triggered": False,
        "target_met_by_quantization": False,
        "business_target_met": False,
        "selected_candidate": None,
        "decisions": [],
    }
    write_json(target / "search_trace.json", search_trace)
    _emit(
        logger,
        f"plan ready: {sum(bool(item['runnable']) for item in candidates)} "
        f"runnable / {len(candidates)} total candidates",
    )
    for index, candidate in enumerate(candidates, start=1):
        if _cancelled(cancel_event):
            _emit(logger, "cancellation requested")
            break
        path = target / "benchmarks" / f"{candidate['id']}.json"
        if path.exists():
            continue
        is_pruning = candidate["method"].startswith(pruning_prefixes)
        if (
            request.search.enabled
            and candidate["method"] != "dense"
            and business_target_met
        ):
            reason = (
                f"early stop: {selected_candidate} already satisfies the "
                "business quality, compression and speed constraints"
            )
            _write_candidate(
                target,
                candidate,
                {
                    "generated_at": utc_now(),
                    "status": "skipped",
                    "backend": candidate["backend"],
                    "scope": candidate["scope"],
                    "reasons": [reason],
                },
            )
            _emit(
                logger,
                f"candidate [{index}/{len(candidates)}] {candidate['id']}: "
                "skipped (business target already met)",
            )
            _notify(
                event_handler,
                {
                    "type": "candidate_complete",
                    "index": index,
                    "total": len(candidates),
                    "candidate": candidate,
                    "status": "skipped",
                    "metrics": {},
                    "reasons": [reason],
                },
            )
            continue
        if (
            request.search.enabled
            and candidate["method"] != "dense"
            and trials_executed >= request.search.max_trials
        ):
            reason = (
                f"trial budget exhausted ({request.search.max_trials}); "
                "candidate not executed"
            )
            _write_candidate(
                target,
                candidate,
                {
                    "generated_at": utc_now(),
                    "status": "skipped",
                    "backend": candidate["backend"],
                    "scope": candidate["scope"],
                    "reasons": [reason],
                },
            )
            _emit(
                logger,
                f"candidate [{index}/{len(candidates)}] {candidate['id']}: "
                "skipped (trial budget)",
            )
            _notify(
                event_handler,
                {
                    "type": "candidate_complete",
                    "index": index,
                    "total": len(candidates),
                    "candidate": candidate,
                    "status": "skipped",
                    "metrics": {},
                    "reasons": [reason],
                },
            )
            continue
        quant_target_met = any(
            bool(item.get("target_met")) for item in quant_probes
        )
        if (
            is_pruning
            and request.search.enabled
            and request.search.quantization_first
            and quant_target_met
        ):
            reason = (
                "skipped by staged search: a quantization candidate already "
                f"met quality and {request.search.target_checkpoint_ratio:.2f}x "
                "checkpoint target"
            )
            _write_candidate(
                target,
                candidate,
                {
                    "generated_at": utc_now(),
                    "status": "skipped",
                    "backend": candidate["backend"],
                    "scope": candidate["scope"],
                    "reasons": [reason],
                },
            )
            _emit(
                logger,
                f"candidate [{index}/{len(candidates)}] {candidate['id']}: "
                "skipped (quantization target met)",
            )
            _notify(
                event_handler,
                {
                    "type": "candidate_complete",
                    "index": index,
                    "total": len(candidates),
                    "candidate": candidate,
                    "status": "skipped",
                    "metrics": {},
                    "reasons": [reason],
                },
            )
            continue
        if (
            is_pruning
            and request.search.enabled
            and request.search.quantization_first
            and not request.search.allow_pruning_fallback
        ):
            reason = "pruning fallback is disabled by search policy"
            _write_candidate(
                target,
                candidate,
                {
                    "generated_at": utc_now(),
                    "status": "skipped",
                    "backend": candidate["backend"],
                    "scope": candidate["scope"],
                    "reasons": [reason],
                },
            )
            continue
        if (
            is_pruning
            and request.search.enabled
            and request.search.quantization_first
        ):
            search_trace["pruning_fallback_triggered"] = True
            _notify(
                event_handler,
                {
                    "type": "phase",
                    "phase": "pruning_fallback",
                    "message": (
                        "量化候选未满足质量/压缩目标，进入 "
                        f"{request.search.pruning_granularity} 剪枝回退"
                    ),
                },
            )
        _emit(
            logger,
            f"candidate [{index}/{len(candidates)}] {candidate['id']}: "
            f"{candidate['algorithm']} / {candidate['structure']} -> "
            f"{candidate['backend']}",
        )
        _notify(
            event_handler,
            {
                "type": "candidate_start",
                "index": index,
                "total": len(candidates),
                "candidate": candidate,
            },
        )
        if not candidate["runnable"]:
            _write_candidate(
                target,
                candidate,
                {
                    "generated_at": utc_now(),
                    "status": "skipped",
                    "backend": candidate["backend"],
                    "scope": candidate["scope"],
                    "reasons": candidate["reasons"],
                },
            )
            _emit(
                logger,
                f"candidate [{index}/{len(candidates)}] {candidate['id']}: skipped",
            )
            _notify(
                event_handler,
                {
                    "type": "candidate_complete",
                    "index": index,
                    "total": len(candidates),
                    "candidate": candidate,
                    "status": "skipped",
                    "metrics": {},
                    "reasons": candidate["reasons"],
                },
            )
            continue
        try:
            if candidate["method"] != "dense":
                trials_executed += 1
            if candidate["method"] == "dense":
                if candidate["backend"] == "transformers":
                    result = benchmark_transformers(
                        Path(model.resolved_path),
                        request,
                        mode=mode,
                        event_callback=event_handler,
                    )
                elif candidate["backend"] == "vllm":
                    result = benchmark_vllm(
                        Path(model.resolved_path),
                        request,
                        mode=mode,
                        event_callback=event_handler,
                    )
                else:
                    result = {
                        "generated_at": utc_now(),
                        "status": "skipped",
                        "backend": candidate["backend"],
                        "scope": candidate["scope"],
                        "reasons": ["dense micro baseline is produced inside its paired kernel result"],
                    }
            else:
                result = _execute_candidate(
                    candidate,
                    request,
                    model,
                    target,
                    mode=mode,
                    event_handler=event_handler,
                )
        except Exception as exc:
            result = _failed_result(candidate, exc)
        _write_candidate(target, candidate, result)
        _emit(
            logger,
            f"candidate [{index}/{len(candidates)}] {candidate['id']}: "
            f"{result.get('status')}",
        )
        quality_result = read_json(
            target / "quality" / f"{candidate['id']}.json",
            default={},
        )
        manifest_result = read_json(
            target
            / "artifacts"
            / candidate["id"]
            / "compression_manifest.json",
            default={},
        )
        if candidate["method"] == "dense" and candidate["backend"] == "transformers":
            baseline_ppl = (result.get("quality") or {}).get("perplexity")
        if candidate["method"] == "dense":
            baseline_results[candidate["backend"]] = result
        if candidate["method"] in quant_methods:
            candidate_ppl = (quality_result.get("quality") or {}).get(
                "perplexity"
            )
            checkpoint_ratio = manifest_result.get("compression_ratio")
            effective_ratio = manifest_result.get(
                "effective_weight_compression_ratio"
            )
            if effective_ratio is None:
                effective_ratio = {
                    "gptq_w4a16": 4.0,
                    "awq_w4a16": 4.0,
                    "smoothquant_w8a8": 2.0,
                }.get(candidate["method"], checkpoint_ratio)
            relative_ppl = (
                None
                if baseline_ppl in (None, 0) or candidate_ppl is None
                else (candidate_ppl - baseline_ppl) / baseline_ppl
            )
            probe = {
                "candidate": candidate["id"],
                "status": result.get("status"),
                "checkpoint_ratio": checkpoint_ratio,
                "effective_weight_compression_ratio": effective_ratio,
                "target_checkpoint_ratio": request.search.target_checkpoint_ratio,
                "perplexity": candidate_ppl,
                "relative_perplexity_increase": relative_ppl,
                "quality_pass": (
                    relative_ppl is not None
                    and relative_ppl
                    <= request.constraints.max_relative_ppl_increase
                ),
            }
            candidate_performance = _profile_performance(
                result, request.workload.profile
            )
            baseline_performance = _profile_performance(
                baseline_results.get(candidate["backend"], {}),
                request.workload.profile,
            )
            speedup = (
                None
                if candidate_performance is None
                or baseline_performance in (None, 0)
                else candidate_performance / baseline_performance
            )
            probe["same_backend_speedup"] = speedup
            probe["speed_pass"] = bool(
                speedup is not None
                and speedup >= request.constraints.min_same_backend_speedup
            )
            probe["target_met"] = bool(
                probe["status"] == "completed"
                and probe["quality_pass"]
                and probe["speed_pass"]
                and effective_ratio is not None
                and effective_ratio
                >= request.search.target_checkpoint_ratio
            )
            quant_probes.append(probe)
            search_trace["target_met_by_quantization"] = any(
                item["target_met"] for item in quant_probes
            )
            write_json(target / "search_trace.json", search_trace)
            _notify(
                event_handler,
                {
                    "type": "search_probe",
                    "probe": probe,
                    "target_met": probe["target_met"],
                },
            )
        if request.search.enabled and candidate["method"] != "dense":
            quality = quality_result.get("quality") or result.get("quality") or {}
            candidate_ppl = quality.get("perplexity")
            relative_ppl = (
                None
                if baseline_ppl in (None, 0) or candidate_ppl is None
                else (candidate_ppl - baseline_ppl) / baseline_ppl
            )
            effective_ratio = manifest_result.get(
                "effective_weight_compression_ratio"
            )
            sparsity = (manifest_result.get("tensors") or {}).get(
                "sparsity_after"
            )
            if effective_ratio is None and sparsity is not None and sparsity < 1:
                effective_ratio = 1 / max(1 - float(sparsity), 1e-12)
            if effective_ratio is None:
                effective_ratio = manifest_result.get("compression_ratio")
            candidate_performance = _profile_performance(
                result, request.workload.profile
            )
            baseline_performance = _profile_performance(
                baseline_results.get(candidate["backend"], {}),
                request.workload.profile,
            )
            measured_speedup = (
                None
                if candidate_performance is None
                or baseline_performance in (None, 0)
                else candidate_performance / baseline_performance
            )
            if measured_speedup is None:
                measured_speedup = _kernel_speedup(result)
            decision = {
                "candidate": candidate["id"],
                "trial": trials_executed,
                "quality_pass": (
                    relative_ppl is not None
                    and relative_ppl
                    <= request.constraints.max_relative_ppl_increase
                ),
                "compression_pass": (
                    effective_ratio is not None
                    and effective_ratio
                    >= request.search.target_checkpoint_ratio
                ),
                "speed_pass": (
                    measured_speedup is not None
                    and measured_speedup
                    >= request.constraints.min_same_backend_speedup
                ),
                "effective_weight_compression_ratio": effective_ratio,
                "same_backend_or_kernel_speedup": measured_speedup,
                "relative_perplexity_increase": relative_ppl,
            }
            decision["business_target_met"] = bool(
                result.get("status") == "completed"
                and decision["quality_pass"]
                and decision["compression_pass"]
                and decision["speed_pass"]
                and candidate["scope"] in {"end_to_end", "optional_end_to_end"}
            )
            search_trace["decisions"].append(decision)
            if decision["business_target_met"]:
                business_target_met = True
                selected_candidate = candidate["id"]
                search_trace["business_target_met"] = True
                search_trace["selected_candidate"] = selected_candidate
                _notify(
                    event_handler,
                    {
                        "type": "phase",
                        "phase": "early_stop",
                        "message": (
                            f"{selected_candidate} 已满足业务目标，提前停止搜索"
                        ),
                    },
                )
            write_json(target / "search_trace.json", search_trace)
        _notify(
            event_handler,
            {
                "type": "candidate_complete",
                "index": index,
                "total": len(candidates),
                "candidate": candidate,
                "status": result.get("status"),
                "metrics": summarize_benchmark(result),
                "quality": (
                    quality_result.get("quality")
                    or result.get("quality")
                    or {}
                ),
                "generation_sample": (
                    quality_result.get("generation_sample")
                    or result.get("generation_sample")
                ),
                "artifact": {
                    "path": manifest_result.get("output_dir")
                    or candidate.get("artifact_dir"),
                    "status": manifest_result.get("status"),
                    "source_bytes": manifest_result.get("source_bytes"),
                    "output_bytes": manifest_result.get("output_bytes"),
                    "compression_ratio": manifest_result.get("compression_ratio"),
                    "effective_weight_compression_ratio": manifest_result.get(
                        "effective_weight_compression_ratio"
                    )
                    or {
                        "gptq_w4a16": 4.0,
                        "awq_w4a16": 4.0,
                        "smoothquant_w8a8": 2.0,
                    }.get(candidate["method"]),
                    "sparsity_after": (
                        manifest_result.get("tensors") or {}
                    ).get("sparsity_after"),
                    "sparsity_valid": (
                        manifest_result.get("tensors") or {}
                    ).get("all_groups_valid"),
                },
                "error": result.get("error"),
            },
        )
    _notify(
        event_handler,
        {
            "type": "phase",
            "phase": "report",
            "message": "汇总质量、吞吐、延迟与加速结果",
        },
    )
    _emit(logger, "generating evaluation and report")
    search_trace["stages"][0]["status"] = "completed"
    search_trace["trials_executed"] = trials_executed
    search_trace["stages"].append(
        {
            "stage": "pruning_fallback",
            "status": (
                "executed"
                if search_trace["pruning_fallback_triggered"]
                else "not_needed"
                if search_trace["target_met_by_quantization"]
                else "not_requested"
            ),
            "granularity": request.search.pruning_granularity,
        }
    )
    write_json(target / "search_trace.json", search_trace)
    evaluation = regenerate_report(target)
    _notify(
        event_handler,
        {
            "type": "report_ready",
            "run_dir": str(target),
            "evaluation": evaluation,
        },
    )
    speedups = [
        item["same_backend_speedup"]
        for item in evaluation.get("evaluated", [])
        if item.get("same_backend_speedup") is not None
    ]
    if (
        mode == "full"
        and model.family == "opt"
        and speedups
        and max(speedups) < 1.0
        and request.execution.fallback_model
        and Path(request.execution.fallback_model).exists()
        and not _cancelled(cancel_event)
    ):
        _emit(
            logger,
            "all measured OPT same-backend speedups are below 1x; "
            "running the configured Llama-2-7B quantization fallback",
        )
        fallback_mapping = request.to_dict()
        fallback_mapping["name"] = f"{request.name}-llama2-7b-fallback"
        fallback_mapping["model"] = {
            "path": request.execution.fallback_model,
            "family": "llama",
            "kind": "dense",
            "trust_remote_code": False,
        }
        fallback_mapping["methods"] = [
            "gptq_w4a16",
            "awq_w4a16",
            "smoothquant_w8a8",
        ]
        fallback_request = request_from_mapping(fallback_mapping)
        fallback_dir = run_request(
            fallback_request,
            mode=mode,
            yes=True,
            cancel_event=cancel_event,
            logger=logger,
            event_handler=event_handler,
        )
        write_json(
            target / "fallback.json",
            {
                "trigger": "all OPT same-backend speedups below 1x",
                "run_dir": str(fallback_dir),
            },
        )
    write_json(
        target / "status.json",
        {
            "status": "cancelled" if _cancelled(cancel_event) else "completed",
            "finished_at": utc_now(),
            "run_dir": str(target),
        },
    )
    _emit(logger, f"report: {target / 'report.md'}")
    return target


def default_run_root() -> Path:
    return SKILL_ROOT / "runs"
