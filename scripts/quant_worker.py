#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from llm_autocompress.adapters.compression import execute_compression
from llm_autocompress.models import inspect_model
from llm_autocompress.schema import load_request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    request = load_request(args.request)
    model = inspect_model(
        request.model.path,
        request.model.family,
        request.model.kind,
    )
    manifest = execute_compression(
        args.method,
        model,
        request,
        Path(args.output),
    )
    return 0 if manifest.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
