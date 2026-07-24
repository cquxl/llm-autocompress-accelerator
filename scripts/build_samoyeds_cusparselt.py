#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "samoyeds_cusparselt" / "cusparselt24_mod.cu"
MODULE_NAME = "cusparselt24_kernel"


def _cusparselt_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    raw = (
        explicit
        or os.environ.get("CUSPARSELT_ROOT")
        or os.environ.get("CUSPARSELT_DIR")
    )
    if raw:
        candidates.append(Path(raw).expanduser())
    spec = importlib.util.find_spec("nvidia.cusparselt")
    if spec and spec.submodule_search_locations:
        candidates.extend(Path(item) for item in spec.submodule_search_locations)
    candidates.extend(
        [
            Path(sys.prefix)
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "nvidia"
            / "cusparselt",
            ROOT
            / "dependencies"
            / "Samoyeds"
            / "Samoyeds-Kernel"
            / "cusparselt"
            / "libcusparse_lt-linux-x86_64-0.5.2.1-archive",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if (
            (resolved / "include" / "cusparseLt.h").is_file()
            and list((resolved / "lib").glob("libcusparseLt.so*"))
        ):
            return resolved
    raise FileNotFoundError(
        "cuSPARSELt headers/library not found; install nvidia-cusparselt-cu12 "
        "or pass --cusparselt-root"
    )


def _cuda_arches() -> str | None:
    configured = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if configured:
        return configured
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    values = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ";".join(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Samoyeds direct-cuSPARSELt 2:4 extension."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cusparselt-root")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    cusparselt = _cusparselt_root(args.cusparselt_root)
    library = sorted((cusparselt / "lib").glob("libcusparseLt.so*"))[0]
    arches = _cuda_arches()
    print(f"Source: {SOURCE}")
    print(f"Output: {output_dir}")
    print(f"cuSPARSELt: {cusparselt}")
    print(f"CUDA architectures: {arches or '<PyTorch default>'}")
    if not args.yes:
        print("Dry run only. Re-run with --yes to compile.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if arches:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arches
    from torch.utils.cpp_extension import load

    extension = load(
        name=MODULE_NAME,
        sources=[str(SOURCE)],
        build_directory=str(output_dir),
        extra_include_paths=[str(cusparselt / "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_ldflags=[
            str(library),
            f"-Wl,-rpath,{cusparselt / 'lib'}",
        ],
        with_cuda=True,
        verbose=True,
    )
    extension_path = Path(extension.__file__).resolve()
    marker = output_dir / "build-info.txt"
    marker.write_text(
        "\n".join(
            [
                f"module={MODULE_NAME}",
                f"extension={extension_path}",
                f"source={SOURCE}",
                f"cusparselt={cusparselt}",
                f"torch_cuda_arch_list={arches or ''}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(extension_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
