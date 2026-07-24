from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from .schema import CompressionRequest
from .utils import human_bytes, read_json, utc_now, write_json


def _median(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(record[key])
        for record in records
        if record.get(key) is not None
    ]
    return statistics.median(values) if values else None


def summarize_benchmark(benchmark: dict[str, Any] | None) -> dict[str, Any]:
    if not benchmark or benchmark.get("status") != "completed":
        return {}
    records = benchmark.get("records") or []
    return {
        "ttft_ms": _median(records, "ttft_p50_ms"),
        "tpot_ms": _median(records, "tpot_p50_ms"),
        "prefill_tokens_per_second": _median(records, "prefill_tokens_per_second"),
        "decode_tokens_per_second": _median(records, "decode_tokens_per_second"),
        "request_throughput_per_second": _median(
            records, "request_throughput_per_second"
        ),
        "end_to_end_ms": _median(records, "end_to_end_p50_ms"),
        "peak_vram_bytes": benchmark.get("peak_vram_bytes"),
    }


def _performance(summary: dict[str, Any], profile: str) -> tuple[str, float | None]:
    if profile == "throughput":
        return "decode_tokens_per_second", summary.get("decode_tokens_per_second")
    if profile == "prefill-heavy":
        return "prefill_tokens_per_second", summary.get("prefill_tokens_per_second")
    tpot = summary.get("tpot_ms")
    return "inverse_tpot", None if tpot in (None, 0) else 1000 / float(tpot)


def _micro_speedup(result: dict[str, Any]) -> float | None:
    record_sets = [result.get("records") or []]
    for key in ("cusparselt", "kernel"):
        nested = result.get(key)
        if isinstance(nested, dict):
            record_sets.append(nested.get("records") or [])
    values = []
    for records in record_sets:
        for record in records:
            for key in (
                "same_backend_speedup",
                "same_mask_speedup",
            ):
                if record.get(key) is not None:
                    values.append(float(record[key]))
                    break
    return statistics.median(values) if values else None


def evaluate_run(
    run_dir: Path,
    request: CompressionRequest,
    plan: dict[str, Any],
) -> dict[str, Any]:
    search_trace = read_json(run_dir / "search_trace.json", default={})
    baseline_results: dict[str, dict[str, Any]] = {}
    for backend in ("transformers", "vllm"):
        value = read_json(run_dir / "benchmarks" / f"dense__{backend}.json")
        if value:
            baseline_results[backend] = value
    dense_quality = (
        baseline_results.get("transformers", {}).get("quality") or {}
    )
    baseline_ppl = dense_quality.get("perplexity")

    evaluated = []
    for candidate in plan.get("candidates", []):
        if candidate["method"] == "dense":
            continue
        result = read_json(
            run_dir / "benchmarks" / f"{candidate['id']}.json",
            default={},
        )
        manifest = read_json(
            run_dir
            / "artifacts"
            / candidate["id"]
            / "compression_manifest.json",
            default={},
        )
        quality_result = read_json(
            run_dir / "quality" / f"{candidate['id']}.json",
            default={},
        )
        quality = quality_result.get("quality") or result.get("quality") or {}
        candidate_ppl = quality.get("perplexity")
        source_bytes = manifest.get("source_bytes")
        artifact_bytes = manifest.get("output_bytes")
        compression_ratio = manifest.get("compression_ratio")
        effective_compression_ratio = manifest.get(
            "effective_weight_compression_ratio"
        )
        if effective_compression_ratio is None:
            effective_compression_ratio = {
                "gptq_w4a16": 4.0,
                "awq_w4a16": 4.0,
                "smoothquant_w8a8": 2.0,
            }.get(candidate["method"])
        if (
            compression_ratio is None
            and source_bytes not in (None, 0)
            and artifact_bytes not in (None, 0)
        ):
            compression_ratio = float(source_bytes) / float(artifact_bytes)
        tensor_validation = manifest.get("tensors") or {}
        artifact_validation = manifest.get("artifact_validation") or {}
        relative_ppl = (
            None
            if baseline_ppl in (None, 0) or candidate_ppl is None
            else (candidate_ppl - baseline_ppl) / baseline_ppl
        )
        quality_pass = (
            relative_ppl is not None
            and relative_ppl <= request.constraints.max_relative_ppl_increase
        )
        summary = summarize_benchmark(result)
        baseline_summary = summarize_benchmark(
            baseline_results.get(candidate["backend"])
        )
        metric_name, performance = _performance(
            summary, request.workload.profile
        )
        _, baseline_performance = _performance(
            baseline_summary, request.workload.profile
        )
        same_backend_speedup = (
            None
            if performance is None or baseline_performance in (None, 0)
            else performance / baseline_performance
        )
        transformers_metric = _performance(
            summarize_benchmark(baseline_results.get("transformers")),
            request.workload.profile,
        )[1]
        deployment_speedup = (
            None
            if performance is None or transformers_metric in (None, 0)
            else performance / transformers_metric
        )
        end_to_end = candidate["scope"] in {
            "end_to_end",
            "optional_end_to_end",
        }
        compression_pass = bool(
            effective_compression_ratio is not None
            and effective_compression_ratio
            >= request.search.target_checkpoint_ratio
        )
        accepted = bool(
            end_to_end
            and result.get("status") == "completed"
            and quality_pass
            and (compression_pass or not request.search.enabled)
            and same_backend_speedup is not None
            and same_backend_speedup
            >= request.constraints.min_same_backend_speedup
        )
        evaluated.append(
            {
                **candidate,
                "execution_status": result.get(
                    "status", manifest.get("status", candidate.get("status"))
                ),
                "quality_status": (
                    "pass"
                    if quality_pass
                    else "missing"
                    if relative_ppl is None
                    else "fail"
                ),
                "baseline_perplexity": baseline_ppl,
                "perplexity": candidate_ppl,
                "relative_perplexity_increase": relative_ppl,
                "performance_metric": metric_name,
                "performance_value": performance,
                "same_backend_speedup": same_backend_speedup,
                "deployment_speedup": deployment_speedup,
                "micro_kernel_speedup": _micro_speedup(result),
                "peak_vram_bytes": summary.get("peak_vram_bytes"),
                "benchmark_summary": summary,
                "source_bytes": source_bytes,
                "artifact_bytes": artifact_bytes,
                "compression_ratio": compression_ratio,
                "effective_weight_compression_ratio": effective_compression_ratio,
                "sparsity_after": tensor_validation.get("sparsity_after"),
                "sparsity_valid": tensor_validation.get("all_groups_valid"),
                "artifact_valid": artifact_validation.get(
                    "valid",
                    bool(
                        manifest.get("status") == "completed"
                        and artifact_bytes
                    ),
                ),
                "artifact_path": manifest.get("output_dir")
                or candidate.get("artifact_dir"),
                "compression_pass": compression_pass,
                "generation_sample": quality_result.get("generation_sample")
                or result.get("generation_sample"),
                "reasons": result.get("reasons") or candidate.get("reasons") or [],
                "accepted": accepted,
                "error": result.get("error") or manifest.get("error"),
            }
        )

    eligible = [item for item in evaluated if item["accepted"]]
    eligible.sort(
        key=lambda item: (
            item.get("same_backend_speedup") or 0,
            -(item.get("peak_vram_bytes") or 10**30),
            -(item.get("artifact_bytes") or 10**30),
        ),
        reverse=True,
    )
    recommended = eligible[0] if eligible else None
    pareto = sorted(
        (
            item
            for item in evaluated
            if item.get("execution_status") == "completed"
        ),
        key=lambda item: (
            item.get("relative_perplexity_increase")
            if item.get("relative_perplexity_increase") is not None
            else float("inf"),
            -(item.get("same_backend_speedup") or 0),
        ),
    )[:5]
    return {
        "generated_at": utc_now(),
        "baseline": {
            "transformers": summarize_benchmark(
                baseline_results.get("transformers")
            ),
            "vllm": summarize_benchmark(baseline_results.get("vllm")),
            "perplexity": baseline_ppl,
            "generation_sample": baseline_results.get("transformers", {}).get(
                "generation_sample"
            ),
            "generation_metrics": baseline_results.get("transformers", {}).get(
                "generation_metrics"
            ),
        },
        "evaluated": evaluated,
        "recommended": recommended,
        "pareto_frontier": pareto,
        "policy": plan.get("selection_policy"),
        "search_trace": search_trace,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(
    request: CompressionRequest,
    plan: dict[str, Any],
    evaluation: dict[str, Any],
    env: dict[str, Any],
) -> str:
    recommended = evaluation.get("recommended")
    summary = (
        f"`{recommended['id']}`"
        if recommended
        else "无候选同时满足质量与同后端加速门槛"
    )
    rows = []
    generation_rows = []
    for item in evaluation.get("evaluated", []):
        rows.append(
            "| {id} | {algorithm} | {structure} | {backend} | {scope} | {status} | "
            "{artifact} | {ratio} | {effective} | {sparsity} | {ppl} | {ppl_delta} | "
            "{same} | {deploy} | {micro} | {accepted} |".format(
                id=item["id"],
                algorithm=item["algorithm"],
                structure=item["structure"],
                backend=item["backend"],
                scope=item["scope"],
                status=item["execution_status"],
                artifact=(
                    "-"
                    if item.get("artifact_bytes") is None
                    else human_bytes(item["artifact_bytes"])
                ),
                ratio=_fmt(item.get("compression_ratio")),
                effective=_fmt(
                    item.get("effective_weight_compression_ratio")
                ),
                sparsity=(
                    "-"
                    if item.get("sparsity_after") is None
                    else f"{item['sparsity_after'] * 100:.2f}%"
                ),
                ppl=_fmt(item.get("perplexity")),
                ppl_delta=(
                    "-"
                    if item.get("relative_perplexity_increase") is None
                    else f"{item['relative_perplexity_increase'] * 100:.2f}%"
                ),
                same=_fmt(item.get("same_backend_speedup")),
                deploy=_fmt(item.get("deployment_speedup")),
                micro=_fmt(item.get("micro_kernel_speedup")),
                accepted="yes" if item["accepted"] else "no",
            )
        )
        if item.get("generation_sample"):
            generation_rows.append(
                f"- `{item['id']}`: {item['generation_sample']}"
            )
    gpu_lines = [
        f"- GPU {gpu['index']}: {gpu['name']}, CC {gpu['compute_capability']}, "
        f"{gpu['memory_used_mb']}/{gpu['memory_total_mb']} MiB used"
        for gpu in env.get("gpus", [])
    ]
    failures = []
    for item in evaluation.get("evaluated", []):
        if item.get("error"):
            failures.append(f"- `{item['id']}`: {item['error']}")
            continue
        if item.get("execution_status") == "skipped":
            reasons = item.get("reasons") or []
            failures.append(
                f"- `{item['id']}`: skipped"
                + (f" — {'; '.join(str(reason) for reason in reasons)}" if reasons else "")
            )
    baseline = evaluation.get("baseline") or {}
    baseline_transformers = baseline.get("transformers") or {}
    baseline_vllm = baseline.get("vllm") or {}
    report_kind = (
        "Automatic compression comparison"
        if evaluation.get("evaluated")
        else "Dense baseline validation (no compression candidate)"
    )
    search_trace = evaluation.get("search_trace") or {}
    search_strategy = search_trace.get("strategy") or {}
    search_decisions = search_trace.get("decisions") or []
    decision_rows = [
        "| {trial} | {candidate} | {quality} | {compression} | {speed} | "
        "{ratio} | {speedup} | {ppl_delta} | {result} |".format(
            trial=item.get("trial", "-"),
            candidate=item.get("candidate", "-"),
            quality="pass" if item.get("quality_pass") else "fail",
            compression="pass" if item.get("compression_pass") else "fail",
            speed="pass" if item.get("speed_pass") else "fail",
            ratio=_fmt(item.get("effective_weight_compression_ratio")),
            speedup=_fmt(item.get("same_backend_or_kernel_speedup")),
            ppl_delta=(
                "-"
                if item.get("relative_perplexity_increase") is None
                else f"{item['relative_perplexity_increase'] * 100:.2f}%"
            ),
            result=(
                "target met; early stop"
                if item.get("business_target_met")
                else "continue"
            ),
        )
        for item in search_decisions
    ]
    automatic_search = (
        f"""## Automatic Search Trace

- Strategy: quantization probe first; run pruning fallback only when needed.
- Trial budget: {search_trace.get('trials_executed', 0)} executed / {search_strategy.get('max_trials', '-')} maximum.
- Business target: effective weight compression ≥ {_fmt(search_strategy.get('target_checkpoint_ratio'))}×, PPL relative increase ≤ {_fmt(search_strategy.get('maximum_relative_ppl_increase') * 100 if search_strategy.get('maximum_relative_ppl_increase') is not None else None)}%, same-backend speedup ≥ {_fmt(search_strategy.get('minimum_same_backend_speedup'))}×.
- Result: {"passed; selected `" + str(search_trace.get("selected_candidate")) + "` and stopped early" if search_trace.get("business_target_met") else "no candidate met every business constraint within the trial budget"}.
- Pruning fallback: {"executed" if search_trace.get("pruning_fallback_triggered") else "not needed"}.

| Trial | Candidate | Quality | Compression | Speed | Effective compression | Same-backend/kernel speedup | PPL Δ | Decision |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(decision_rows) if decision_rows else "| - | - | - | - | - | - | - | - | No compression trial |"}
"""
        if search_strategy.get("enabled")
        else ""
    )
    return f"""# LLM AutoCompress Acceleration Report

Generated: {evaluation['generated_at']}

## Summary

- Request: `{request.name}`
- Model: `{plan['model']['resolved_path']}`
- Family/kind: `{plan['model']['family']}` / `{plan['model']['kind']}`
- Report kind: {report_kind}
- Workload profile: `{request.workload.profile}`
- Quality gate: WikiText2 PPL relative increase ≤ {request.constraints.max_relative_ppl_increase * 100:.2f}%
- Recommendation: {summary}
- Metric policy: same-backend compression speedup and Transformers-to-deployment speedup are reported separately.

{automatic_search}

## Environment

{chr(10).join(gpu_lines) if gpu_lines else "- No NVIDIA GPU detected"}

## Dense Baselines

- Transformers: PPL {_fmt(baseline.get("perplexity"))}, TTFT {_fmt(baseline_transformers.get("ttft_ms"))} ms, TPOT {_fmt(baseline_transformers.get("tpot_ms"))} ms, Prefill {_fmt(baseline_transformers.get("prefill_tokens_per_second"))} tokens/s, Decode {_fmt(baseline_transformers.get("decode_tokens_per_second"))} tokens/s
- vLLM: TPOT {_fmt(baseline_vllm.get("tpot_ms"))} ms, Decode {_fmt(baseline_vllm.get("decode_tokens_per_second"))} tokens/s, End-to-end {_fmt(baseline_vllm.get("end_to_end_ms"))} ms
- Transformers generation: {baseline.get("generation_sample") or "-"}

## Candidate Results

| Candidate | Algorithm | Structure | Backend | Scope | Status | Artifact | Checkpoint size ratio | Effective weight compression | Sparsity | PPL | PPL Δ | Same-backend speedup | Deployment speedup | Micro/kernel speedup | Accepted |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows) if rows else "| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |"}

## Generation Samples

- `dense__transformers`: {baseline.get("generation_sample") or "-"}
{chr(10).join(generation_rows) if generation_rows else "- No compressed candidate generation sample"}

## Interpretation

- `end_to_end` candidates may be recommended only when quality, target compression, and same-backend speed pass.
- `linear`, `expert_moe_layer`, and `structured_checkpoint_and_linear` results are microbenchmarks and are never presented as full-model acceleration.
- Checkpoint size ratio is source bytes / saved artifact bytes. A dense-format 2:4 HF checkpoint may remain near 1× on disk; cuSPARSELt packs it at runtime.
- Missing metrics remain missing; no historical or synthetic value is substituted.

## Failures and Skips

{chr(10).join(failures) if failures else "- None"}

## Reproducibility

- `request.yaml`: normalized request.
- `env.json`: hardware and software snapshot.
- `plan.json`: capability decisions and rejected reasons.
- `search_trace.json`: business constraints, executed trials, fallback and early-stop decisions.
- `artifacts/*/compression_manifest.json`: source fingerprint, algorithm, structure, and output validation.
- `benchmarks/*.json` and `quality/*.json`: raw measured values.
- `evaluation.json` and `results.csv`: selection output.
"""


def write_results_csv(path: Path, evaluation: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "id",
        "algorithm",
        "structure",
        "backend",
        "scope",
        "execution_status",
        "quality_status",
        "perplexity",
        "relative_perplexity_increase",
        "performance_metric",
        "performance_value",
        "same_backend_speedup",
        "deployment_speedup",
        "micro_kernel_speedup",
        "peak_vram_bytes",
        "artifact_bytes",
        "source_bytes",
        "compression_ratio",
        "effective_weight_compression_ratio",
        "sparsity_after",
        "sparsity_valid",
        "artifact_valid",
        "artifact_path",
        "accepted",
        "error",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evaluation.get("evaluated", []))
