from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .capabilities import assess, matching_capabilities
from .models import ModelInfo
from .schema import CompressionRequest
from .utils import utc_now


@dataclass(slots=True)
class Candidate:
    id: str
    method: str
    algorithm: str
    structure: str
    backend: str
    scope: str
    status: str
    runnable: bool
    reasons: list[str]
    artifact_dir: str | None
    real_artifact: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _method_parts(method: str) -> tuple[str, str]:
    if method in {"dense", "gptq_w4a16", "awq_w4a16", "smoothquant_w8a8"}:
        return method.split("_", 1)[0], "dense" if method == "dense" else "quantized"
    known_suffixes = (
        "unstructured",
        "2_4",
        "channel",
        "expert",
    )
    for suffix in known_suffixes:
        marker = f"_{suffix}"
        if method.endswith(marker):
            return method[: -len(marker)], suffix.replace("_", ":")
    return method, "unknown"


def build_plan(
    request: CompressionRequest,
    model: ModelInfo,
    env: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    candidates: list[Candidate] = []
    for capability in matching_capabilities(request.methods, request.backends):
        runnable, reasons = assess(capability, model, env)
        candidate_id = f"{capability.method}__{capability.backend}"
        algorithm, structure = _method_parts(capability.method)
        artifact = (
            run_dir / "artifacts" / candidate_id
            if capability.method != "dense" and capability.real_artifact
            else None
        )
        candidates.append(
            Candidate(
                id=candidate_id,
                method=capability.method,
                algorithm=algorithm,
                structure=structure,
                backend=capability.backend,
                scope=capability.scope,
                status="planned" if runnable else "skipped",
                runnable=runnable,
                reasons=reasons,
                artifact_dir=str(artifact) if artifact else None,
                real_artifact=capability.real_artifact,
                notes=capability.notes,
            )
        )

    requested_order = {
        method: index for index, method in enumerate(request.methods, start=1)
    }
    candidates.sort(
        key=lambda item: (
            0 if item.method == "dense" else requested_order.get(item.method, 999),
            item.backend,
        )
    )
    return {
        "generated_at": utc_now(),
        "request": request.to_dict(),
        "model": model.to_dict(),
        "selection_policy": {
            "quality_gate": {
                "metric": "wikitext2_perplexity_relative_increase",
                "maximum": request.constraints.max_relative_ppl_increase,
            },
            "workload_profile": request.workload.profile,
            "same_backend_speedup_is_distinct": True,
            "no_passing_candidate_behavior": "return_pareto_frontier_without_recommendation",
            "staged_search": {
                "enabled": request.search.enabled,
                "quantization_first": request.search.quantization_first,
                "target_checkpoint_ratio": request.search.target_checkpoint_ratio,
                "minimum_same_backend_speedup": (
                    request.constraints.min_same_backend_speedup
                ),
                "maximum_relative_ppl_increase": (
                    request.constraints.max_relative_ppl_increase
                ),
                "allow_pruning_fallback": request.search.allow_pruning_fallback,
                "combine_with_best_quant": request.search.combine_with_best_quant,
                "pruning_granularity": request.search.pruning_granularity,
                "max_trials": request.search.max_trials,
            },
        },
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
