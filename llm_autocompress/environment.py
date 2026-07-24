from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from .schema import PROJECTS_ROOT, SAMOYEDS_ROOT, SKILL_ROOT
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


def _repo_probe(path: Path, markers: tuple[str, ...]) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "markers": {
            marker: (path / marker).exists()
            for marker in markers
        },
    }


def _samoyeds_extension_probe() -> dict[str, Any]:
    extension = SAMOYEDS_ROOT / "build" / "lib.linux-x86_64-cpython-310"
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
    name = "llm-autocompress-quant"
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
        "gpus": _nvidia_gpus(),
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
                PROJECTS_ROOT / "D2Prune", ("main.py", "cfg/args.py")
            ),
            "spinfer": _repo_probe(
                PROJECTS_ROOT / "SpInfer",
                ("README.md", "end2end_inference/ft_tools"),
            ),
        },
    }


def module_available(env: dict[str, Any], module: str) -> bool:
    return bool(env.get("modules", {}).get(module, {}).get("available"))


def idle_gpu(env: dict[str, Any], minimum_free_mb: int = 1024) -> int | None:
    candidates = []
    for gpu in env.get("gpus", []):
        free = gpu["memory_total_mb"] - gpu["memory_used_mb"]
        if free >= minimum_free_mb:
            candidates.append((free, -gpu["utilization_pct"], gpu["index"]))
    if not candidates:
        return None
    return max(candidates)[2]
