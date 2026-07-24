#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets" / "dependencies.yaml"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def valid_checkout(path: Path, markers: list[str]) -> bool:
    return path.is_dir() and all((path / marker).exists() for marker in markers)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare pinned external algorithm/kernel source repositories."
    )
    parser.add_argument("--destination", required=True)
    parser.add_argument(
        "--only",
        action="append",
        choices=(
            "wanda",
            "sparsegpt",
            "gptq",
            "awq",
            "smoothquant",
            "llm_compressor",
            "samoyeds",
            "spinfer",
            "d2prune",
        ),
    )
    parser.add_argument("--runtime-env", default="llm-autocompress-runtime")
    parser.add_argument("--build-samoyeds", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    repositories = manifest["repositories"]
    selected = args.only or [
        name
        for name, spec in repositories.items()
        if spec.get("install_default")
    ]
    destination = Path(args.destination).expanduser().resolve()
    print(f"Dependency root: {destination}")
    for name in selected:
        spec = repositories[name]
        bundled = spec.get("bundled")
        if bundled:
            bundled_path = (ROOT / bundled).resolve()
            markers = list(spec.get("markers") or [])
            if not valid_checkout(bundled_path, markers):
                raise RuntimeError(
                    f"{name} bundled source lacks required markers: {markers}"
                )
            print(f"{name}: using bundled source at {bundled_path}")
            continue
        target = destination / spec["directory"]
        markers = list(spec.get("markers") or [])
        if valid_checkout(target, markers):
            print(f"{name}: existing valid checkout at {target}")
            continue
        if not spec.get("url"):
            print(
                f"{name}: no distributable source URL is configured; "
                f"provide an existing checkout with markers {markers}"
            )
            continue
        print(f"{name}: clone {spec['url']} @ {spec['revision']} -> {target}")
        if not args.yes:
            continue
        destination.mkdir(parents=True, exist_ok=True)
        command = ["git", "clone", "--filter=blob:none"]
        if spec.get("recursive"):
            command.append("--recurse-submodules")
        command.extend([spec["url"], str(target)])
        run(command)
        run(["git", "checkout", spec["revision"]], cwd=target)
        if spec.get("recursive"):
            run(
                [
                    "git",
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                ],
                cwd=target,
            )
        if not valid_checkout(target, markers):
            raise RuntimeError(f"{name} checkout lacks required markers: {markers}")

    samoyeds = destination / repositories["samoyeds"]["directory"]
    if args.build_samoyeds and valid_checkout(
        samoyeds, list(repositories["samoyeds"]["markers"])
    ):
        print(
            "Samoyeds will be compiled on this host for the detected GPU "
            "(A40=sm_86, L40=sm_89)."
        )
        if args.yes:
            run(
                [
                    "conda",
                    "run",
                    "-n",
                    args.runtime_env,
                    "python",
                    str(ROOT / "scripts" / "build_samoyeds_cusparselt.py"),
                    "--output-dir",
                    str(destination / "samoyeds-cusparselt"),
                    "--yes",
                ]
            )
            run(
                [
                    "conda",
                    "run",
                    "-n",
                    args.runtime_env,
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "-v",
                    ".",
                ],
                cwd=samoyeds,
            )
    elif args.build_samoyeds:
        print("Samoyeds build skipped because its source checkout is unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
