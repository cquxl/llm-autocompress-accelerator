from __future__ import annotations

import ast
import os
import shutil
import traceback
from pathlib import Path
from typing import Any

from ..models import ModelInfo
from ..schema import D2PRUNE_ROOT, CompressionRequest
from ..utils import (
    directory_size,
    model_fingerprint,
    run_command,
    source_tree_fingerprint,
    utc_now,
    write_json,
)


def prepare_d2prune_worktree(output_dir: Path) -> tuple[Path, list[str]]:
    """Create a run-local compatibility copy without modifying the user's repo."""
    run_dir = output_dir.parent.parent
    worktree = run_dir / "logs" / f"{output_dir.name}-d2prune-source"
    if worktree.exists():
        shutil.rmtree(worktree)
    shutil.copytree(
        D2PRUNE_ROOT,
        worktree,
        ignore=shutil.ignore_patterns(
            ".git",
            "out",
            "cache",
            "__pycache__",
            "*.pyc",
            "*.pt",
            "*.pth",
            "*.safetensors",
        ),
    )
    patches: list[str] = []
    opt_source = worktree / "prune" / "prune_opt.py"
    if opt_source.exists():
        text = opt_source.read_text(encoding="utf-8")
        old = "self.model.decoder.final_layer_norm"
        count = text.count(old)
        if count:
            opt_source.write_text(
                text.replace(old, "self.model.model.decoder.final_layer_norm"),
                encoding="utf-8",
            )
            patches.append(
                f"prune/prune_opt.py: normalized {count} OPT decoder access(es) "
                "for current Transformers"
            )
    return worktree, patches


def pruning_method_parts(method: str) -> tuple[str, str]:
    for suffix in ("unstructured", "2_4", "channel", "expert"):
        marker = f"_{suffix}"
        if method.endswith(marker):
            return method[: -len(marker)], suffix
    raise ValueError(f"not a composite pruning method: {method}")


def build_d2prune_command(
    method: str,
    model: ModelInfo,
    request: CompressionRequest,
    output_dir: Path,
    *,
    mode: str,
) -> list[str]:
    algorithm, structure = pruning_method_parts(method)
    if algorithm == "rose":
        base_method = "d2prune"
    elif algorithm in {"wanda", "sparsegpt", "d2prune"}:
        base_method = algorithm
    else:
        raise ValueError(f"D2Prune repo does not implement {algorithm}")
    if structure in {"channel", "expert"}:
        raise ValueError(
            f"{method} requires a structural MoE adapter; it is not a D2Prune CLI mode"
        )

    command = [
        os.sys.executable,
        "main.py",
        "--model",
        model.resolved_path,
        "--sparsity_ratio",
        str(request.pruning.sparsity_ratio),
        "--nsamples",
        str(min(request.calibration.samples, 16) if mode == "smoke" else request.calibration.samples),
        "--seqlen",
        str(min(request.calibration.sequence_length, 512) if mode == "smoke" else request.calibration.sequence_length),
        "--sparsity_type",
        "2:4" if structure == "2_4" else "unstructured",
        "--prune_method",
        base_method,
        "--device",
        request.execution.device,
        "--cali_dataset",
        "c4",
        "--cali_data_path",
        request.calibration.dataset,
        "--eval_dataset",
        "wikitext2",
        "--eval_data_path",
        request.evaluation.dataset,
        "--output_dir",
        str(output_dir / "d2prune_logs"),
        "--save_model",
        str(output_dir),
        "--seed",
        str(request.calibration.seed),
        "--target_layer_names",
        repr(request.pruning.target_layer_names),
        "--free",
    ]
    if structure == "2_4":
        command += ["--prune_n", "2", "--prune_m", "4"]
    if base_method == "d2prune":
        command += ["--d2_wanda", "--d2_sparsegpt", "--d2_admm"]
    if algorithm == "rose":
        command += ["--prune_moe", "--only_prune_moe"]
        if request.pruning.tune_router:
            command += ["--tune_router"]
    return command


def validate_pruned_artifact(path: Path) -> dict[str, Any]:
    weight_files = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin"))
    config = path / "config.json"
    return {
        "config_exists": config.exists(),
        "weight_files": [item.name for item in weight_files],
        "weight_bytes": sum(item.stat().st_size for item in weight_files),
        "valid": config.exists() and bool(weight_files),
    }


def run_d2prune(
    method: str,
    model: ModelInfo,
    request: CompressionRequest,
    output_dir: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "compression_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "running",
        "method": method,
        "source_model": model.resolved_path,
        "source_fingerprint": model_fingerprint(Path(model.resolved_path)),
        "source_bytes": model.parameter_bytes,
        "output_dir": str(output_dir),
        "synthetic_weights": False,
        "repo": str(D2PRUNE_ROOT),
        "repo_fingerprint_sha256": source_tree_fingerprint(D2PRUNE_ROOT),
        "bundled_source": (
            D2PRUNE_ROOT.resolve()
            == (Path(__file__).resolve().parents[2] / "third_party" / "d2prune_core").resolve()
        ),
    }
    write_json(manifest_path, manifest)
    try:
        if not (D2PRUNE_ROOT / "main.py").exists():
            raise FileNotFoundError(D2PRUNE_ROOT / "main.py")
        if method.startswith("rose_") and model.family != "mixtral":
            raise RuntimeError(
                "the local ROSE/D2Prune implementation currently dispatches MoE pruning "
                "only for Mixtral; DeepSeek is handled by the real-weight MoE micro adapter"
            )
        worktree, compatibility_patches = prepare_d2prune_worktree(output_dir)
        command = build_d2prune_command(
            method, model, request, output_dir, mode=mode
        )
        manifest["command"] = command
        manifest["execution_source"] = str(worktree)
        manifest["compatibility_patches"] = compatibility_patches
        manifest["source_repo_modified"] = False
        write_json(manifest_path, manifest)
        completed = run_command(
            command,
            cwd=worktree,
            timeout=request.execution.timeout_seconds,
        )
        (output_dir / "stdout.log").write_text(
            completed.get("stdout", ""), encoding="utf-8"
        )
        (output_dir / "stderr.log").write_text(
            completed.get("stderr", ""), encoding="utf-8"
        )
        validation = validate_pruned_artifact(output_dir)
        manifest.update(
            {
                "status": (
                    "completed"
                    if completed["returncode"] == 0 and validation["valid"]
                    else "failed"
                ),
                "finished_at": utc_now(),
                "returncode": completed["returncode"],
                "artifact_validation": validation,
                "output_bytes": directory_size(output_dir),
                "effective_weight_compression_ratio": (
                    1.0 / max(1.0 - request.pruning.sparsity_ratio, 1e-12)
                ),
                "fake_sparsity": False,
            }
        )
        if manifest["status"] == "failed":
            manifest["error"] = (
                completed.get("stderr")
                or "D2Prune did not produce a reloadable model artifact"
            )[-4000:]
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


def execute_pruning(
    method: str,
    model: ModelInfo,
    request: CompressionRequest,
    output_dir: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    algorithm, structure = pruning_method_parts(method)
    if algorithm in {"wanda", "sparsegpt", "d2prune"}:
        return run_d2prune(method, model, request, output_dir, mode=mode)
    if algorithm == "rose" and structure in {"unstructured", "2_4"}:
        return run_d2prune(method, model, request, output_dir, mode=mode)
    raise ValueError(
        f"{method} is executed by the MoE structural microbenchmark adapter"
    )
