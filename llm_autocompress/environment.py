from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from .schema import (
    D2PRUNE_ROOT,
    SAMOYEDS_ROOT,
    SITE_CONFIG,
    SKILL_ROOT,
    SPINFER_ROOT,
)
from .utils import import_probe, run_command, utc_now


MODULES = (
    "torch",
    "transformers",
    "datasets",
    "safetensors",
    "vllm",
    "llmcompressor",
    "compressed_tensors",
    "auto_gptq",
    "gptqmodel",
    "awq",
    "tensorrt_llm",
)


def _nvidia_gpus() -> list[dict[str, Any]]:
    query = (
        "index,name,memory.total,memory.used,compute_cap,driver_version,"
        "temperature.gpu,utilization.gpu"
    )
    result = run_command(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        timeout=15,
    )
    gpus: list[dict[str, Any]] = []
    if result["returncode"] != 0:
        return gpus
    for line in result["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": int(parts[2]),
                "memory_used_mb": int(parts[3]),
                "compute_capability": parts[4],
                "driver_version": parts[5],
                "temperature_c": int(parts[6]),
                "utilization_pct": int(parts[7]),
            }
        )
    return gpus


def _library_probe() -> dict[str, Any]:
    result = run_command(["ldconfig", "-p"], timeout=15)
    text = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".lower()
    return {
        "cusparselt": "libcusparselt" in text,
        "cublas": "libcublas" in text,
        "tensorrt": "libnvinfer" in text,
    }


def _gpu_profile(gpus: list[dict[str, Any]]) -> dict[str, Any]:
    if not gpus:
        return {
            "available": False,
            "compute_capability": None,
            "supports_2_4": False,
            "supports_awq": False,
            "recommended_runtime": "cpu-inspection-only",
        }
    capability = max(float(item["compute_capability"]) for item in gpus)
    total_memory_mb = sum(int(item["memory_total_mb"]) for item in gpus)
    names = sorted({str(item["name"]) for item in gpus})
    return {
        "available": True,
        "names": names,
        "count": len(gpus),
        "compute_capability": capability,
        "total_memory_mb": total_memory_mb,
        "supports_2_4": capability >= 8.0,
        "supports_awq": capability >= 7.5,
        "supports_samoyeds_source_build": capability in {8.0, 8.6, 8.9, 9.0},
        "recommended_runtime": "vllm",
        "note": (
            "A40/Ampere path: compile CUDA extensions for sm_86"
            if capability == 8.6
            else f"compile CUDA extensions for sm_{str(capability).replace('.', '')}"
        ),
    }


def _repo_probe(path: Path, markers: tuple[str, ...]) -> dict[str, Any]:
    revision = run_command(["git", "-C", str(path), "rev-parse", "HEAD"], timeout=10)
    dirty = run_command(
        ["git", "-C", str(path), "status", "--porcelain"],
        timeout=10,
    )
    return {
        "path": str(path),
        "exists": path.exists(),
        "markers": {
            marker: (path / marker).exists()
            for marker in markers
        },
        "markers_valid": path.exists()
        and all((path / marker).exists() for marker in markers),
        "git_revision": (
            revision.get("stdout", "").strip()
            if revision["returncode"] == 0
            else None
        ),
        "git_dirty": (
            bool(dirty.get("stdout", "").strip())
            if dirty["returncode"] == 0
            else None
        ),
    }


def _samoyeds_extension_probe() -> dict[str, Any]:
    extension_candidates = sorted(
        (SAMOYEDS_ROOT / "build").glob("lib.*")
    )
    extension = (
        extension_candidates[-1]
        if extension_candidates
        else SAMOYEDS_ROOT / "build"
    )
    probe_env = dict(os.environ)
    existing = probe_env.get("PYTHONPATH", "")
    probe_env["PYTHONPATH"] = os.pathsep.join(
        [str(extension), str(SAMOYEDS_ROOT), existing]
    ).rstrip(os.pathsep)
    result = run_command(
        [
            sys.executable,
            "-c",
            "import torch, samoyeds_kernel; print(torch.__version__)",
        ],
        timeout=20,
        env=probe_env,
    )
    return {
        "available": result["returncode"] == 0,
        "path": str(extension),
        "torch_abi_error": (
            None if result["returncode"] == 0 else result.get("stderr") or result.get("stdout")
        ),
    }


def _quant_converter_probe(conda_envs: list[str]) -> dict[str, Any]:
    name = SITE_CONFIG.quant_env
    exists = any(Path(item).name == name for item in conda_envs)
    if not exists:
        return {"environment": name, "available": False}
    result = run_command(
        [
            "conda",
            "run",
            "-n",
            name,
            "python",
            "-c",
            "import llmcompressor, torch; print(torch.__version__)",
        ],
        timeout=30,
    )
    return {
        "environment": name,
        "available": result["returncode"] == 0,
        "version": result.get("stdout"),
        "error": None if result["returncode"] == 0 else result.get("stderr"),
    }


def inspect_environment() -> dict[str, Any]:
    disk = shutil.disk_usage(SKILL_ROOT)
    gpus = _nvidia_gpus()
    nvcc = run_command(["nvcc", "--version"], timeout=15)
    conda = run_command(["conda", "env", "list", "--json"], timeout=20)
    conda_envs: list[str] = []
    if conda["returncode"] == 0:
        try:
            conda_envs = json.loads(conda["stdout"]).get("envs", [])
        except json.JSONDecodeError:
            pass
    return {
        "generated_at": utc_now(),
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "cwd": os.getcwd(),
        },
        "site_config": SITE_CONFIG.to_dict(),
        "gpus": gpus,
        "gpu_profile": _gpu_profile(gpus),
        "cuda": {
            "nvcc_available": nvcc["returncode"] == 0,
            "nvcc": nvcc["stdout"] or nvcc["stderr"],
        },
        "libraries": _library_probe(),
        "modules": {name: import_probe(name) for name in MODULES},
        "extensions": {"samoyeds": _samoyeds_extension_probe()},
        "converter_envs": {"quant": _quant_converter_probe(conda_envs)},
        "conda_envs": conda_envs,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "repos": {
            "samoyeds": _repo_probe(
                SAMOYEDS_ROOT, ("deepseek_Samoyeds.py", "Samoyeds-Kernel")
            ),
            "d2prune": _repo_probe(
                D2PRUNE_ROOT, ("main.py", "cfg/args.py", "prune/wanda")
            ),
            "spinfer": _repo_probe(
                SPINFER_ROOT,
                ("README.md", "end2end_inference/ft_tools"),
            ),
        },
    }


def module_available(env: dict[str, Any], module: str) -> bool:
    return bool(env.get("modules", {}).get(module, {}).get("available"))


def readiness_report(env: dict[str, Any]) -> dict[str, Any]:
    site = env.get("site_config") or {}
    repos = env.get("repos") or {}
    modules = env.get("modules") or {}
    gpu = env.get("gpu_profile") or {}
    checks = {
        "gpu": bool(gpu.get("available")),
        "model_root": Path(str(site.get("model_root", ""))).is_dir(),
        "data_root": Path(str(site.get("data_root", ""))).is_dir(),
        "runtime_torch": bool(modules.get("torch", {}).get("available")),
        "runtime_transformers": bool(
            modules.get("transformers", {}).get("available")
        ),
        "runtime_vllm": bool(modules.get("vllm", {}).get("available")),
        "quant_converter": bool(
            env.get("converter_envs", {}).get("quant", {}).get("available")
        ),
        "d2prune_source": bool(
            repos.get("d2prune", {}).get("markers_valid")
        ),
        "spinfer_source": bool(
            repos.get("spinfer", {}).get("markers_valid")
        ),
        "samoyeds_source": bool(
            repos.get("samoyeds", {}).get("markers_valid")
        ),
        "samoyeds_extension": bool(
            env.get("extensions", {}).get("samoyeds", {}).get("available")
        ),
        "structured_2_4_hardware": bool(gpu.get("supports_2_4")),
    }
    return {
        "checks": checks,
        "ready_for_quantization_autopilot": all(
            checks[name]
            for name in (
                "gpu",
                "model_root",
                "data_root",
                "runtime_torch",
                "runtime_transformers",
                "runtime_vllm",
                "quant_converter",
            )
        ),
        "ready_for_pruning_fallback": all(
            checks[name]
            for name in (
                "gpu",
                "model_root",
                "data_root",
                "runtime_torch",
                "runtime_transformers",
                "d2prune_source",
            )
        ),
        "ready_for_cusparselt": all(
            checks[name]
            for name in (
                "gpu",
                "runtime_torch",
                "structured_2_4_hardware",
            )
        ),
        "ready_for_samoyeds": all(
            checks[name]
            for name in (
                "gpu",
                "samoyeds_source",
                "samoyeds_extension",
            )
        ),
        "missing": [name for name, passed in checks.items() if not passed],
    }


def idle_gpu(env: dict[str, Any], minimum_free_mb: int = 1024) -> int | None:
    candidates = []
    for gpu in env.get("gpus", []):
        free = gpu["memory_total_mb"] - gpu["memory_used_mb"]
        if free >= minimum_free_mb:
            candidates.append((free, -gpu["utilization_pct"], gpu["index"]))
    if not candidates:
        return None
    return max(candidates)[2]
