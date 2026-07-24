from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .business import request_mapping_from_business
from .adapters.inference import benchmark_transformers, benchmark_vllm
from .environment import inspect_environment, readiness_report
from .executor import plan_request, regenerate_report, run_request
from .models import inspect_model
from .schema import SKILL_ROOT, load_request, request_from_mapping
from .utils import write_yaml
from .utils import write_json
from .site import load_site_config, save_site_config


DEMO_REQUESTS = {
    "opt-125m": SKILL_ROOT / "assets" / "requests" / "opt-125m.yaml",
    "deepseek-v2-lite": SKILL_ROOT
    / "assets"
    / "requests"
    / "deepseek-v2-lite.yaml",
}


def _json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-autopilot",
        description="EdgeLite-style LLM/MoE automatic compression and acceleration.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="read-only environment/model inspection")
    inspect.add_argument("--model")
    inspect.add_argument("--request")

    doctor = commands.add_parser(
        "doctor",
        help="check whether this host can run quantization, pruning and kernels",
    )

    configure = commands.add_parser(
        "configure-host",
        help="write portable host paths and conda environment names",
    )
    configure.add_argument("--config")
    configure.add_argument("--model-root")
    configure.add_argument("--data-root")
    configure.add_argument("--run-root")
    configure.add_argument("--dependency-root")
    configure.add_argument("--d2prune-root")
    configure.add_argument("--spinfer-root")
    configure.add_argument("--samoyeds-root")
    configure.add_argument("--runtime-env")
    configure.add_argument("--quant-env")
    configure.add_argument("--yes", action="store_true")

    bootstrap = commands.add_parser(
        "bootstrap", help="convert a business description into request YAML"
    )
    bootstrap.add_argument("--model", required=True)
    bootstrap.add_argument("--prompt", default="")
    bootstrap.add_argument(
        "--profile",
        choices=("interactive", "throughput", "prefill-heavy"),
        default="interactive",
    )
    bootstrap.add_argument("--output", required=True)

    plan = commands.add_parser("plan", help="build a capability-filtered candidate plan")
    plan.add_argument("--request", required=True)
    plan.add_argument("--run-dir")

    run = commands.add_parser("run", help="execute conversion, quality and benchmarks")
    run.add_argument("--request", required=True)
    run.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    run.add_argument("--run-dir")
    run.add_argument("--yes", action="store_true")

    demo = commands.add_parser("demo", help="run a built-in real-model demo")
    demo.add_argument("name", choices=tuple(DEMO_REQUESTS))
    demo.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    demo.add_argument("--yes", action="store_true")

    report = commands.add_parser("report", help="regenerate report from raw artifacts")
    report.add_argument("--run-dir", required=True)

    benchmark = commands.add_parser(
        "benchmark-artifact", help="benchmark an existing local artifact"
    )
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--request", required=True)
    benchmark.add_argument("--backend", choices=("transformers", "vllm"), required=True)
    benchmark.add_argument("--quantization")
    benchmark.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    benchmark.add_argument("--output", required=True)

    serve = commands.add_parser("serve", help="start the localhost Web workbench")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        environment = inspect_environment()
        _json(
            {
                "site": load_site_config().to_dict(),
                "environment": environment,
                "readiness": readiness_report(environment),
            }
        )
        return 0
    if args.command == "configure-host":
        values = {
            "model_root": args.model_root,
            "data_root": args.data_root,
            "run_root": args.run_root,
            "dependency_root": args.dependency_root,
            "d2prune_root": args.d2prune_root,
            "spinfer_root": args.spinfer_root,
            "samoyeds_root": args.samoyeds_root,
            "runtime_env": args.runtime_env,
            "quant_env": args.quant_env,
        }
        preview = {
            key: value for key, value in values.items() if value is not None
        }
        if not args.yes:
            _json(
                {
                    "status": "dry-run",
                    "config_path": str(
                        Path(args.config).expanduser().resolve()
                        if args.config
                        else load_site_config().config_path
                    ),
                    "changes": preview,
                    "message": "re-run with --yes to write host configuration",
                }
            )
            return 0
        path = save_site_config(preview, args.config)
        _json(
            {
                "status": "configured",
                "config_path": str(path),
                "site": load_site_config(path).to_dict(),
            }
        )
        return 0
    if args.command == "inspect":
        value = {"environment": inspect_environment()}
        if args.request:
            request = load_request(args.request)
            value["model"] = inspect_model(
                request.model.path,
                request.model.family,
                request.model.kind,
            ).to_dict()
        elif args.model:
            value["model"] = inspect_model(args.model).to_dict()
        _json(value)
        return 0
    if args.command == "bootstrap":
        mapping = request_mapping_from_business(
            model=args.model,
            prompt=args.prompt,
            profile=args.profile,
        )
        request_from_mapping(mapping)
        output = Path(args.output).expanduser().resolve()
        write_yaml(output, mapping)
        print(output)
        return 0
    if args.command == "plan":
        request = load_request(args.request)
        run_dir, plan, _env, _model = plan_request(
            request,
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
        _json({"run_dir": str(run_dir), "plan": plan})
        return 0
    if args.command == "run":
        target = run_request(
            load_request(args.request),
            mode=args.mode,
            yes=args.yes,
            run_dir=Path(args.run_dir) if args.run_dir else None,
        )
        print(target)
        return 0
    if args.command == "demo":
        target = run_request(
            load_request(DEMO_REQUESTS[args.name]),
            mode=args.mode,
            yes=args.yes,
        )
        print(target)
        return 0
    if args.command == "report":
        run_dir = Path(args.run_dir).expanduser().resolve()
        _json(regenerate_report(run_dir))
        return 0
    if args.command == "benchmark-artifact":
        request = load_request(args.request)
        model_path = Path(args.model).expanduser().resolve()
        if args.backend == "vllm":
            value = benchmark_vllm(
                model_path,
                request,
                mode=args.mode,
                quantization=args.quantization,
            )
        else:
            value = benchmark_transformers(model_path, request, mode=args.mode)
        write_json(Path(args.output).expanduser().resolve(), value)
        _json(value)
        return 0 if value.get("status") == "completed" else 1
    if args.command == "serve":
        from .webapp import serve

        serve(args.host, args.port)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
