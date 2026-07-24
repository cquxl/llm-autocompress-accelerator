#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="llm-autocompress-quant"
CONFIRM=0
for arg in "$@"; do
  case "$arg" in
    --env=*) ENV_NAME="${arg#*=}" ;;
    --yes) CONFIRM=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Isolated quantization environment: ${ENV_NAME}"
if [[ "$CONFIRM" -ne 1 ]]; then
  echo "Dry run only. Re-run with --yes to create/install."
  exit 0
fi
if ! conda env list --json | grep -q "\"[^\"]*/${ENV_NAME}\""; then
  mkdir -p "${ROOT}/artifacts/cache/conda-pkgs"
  CONDA_PKGS_DIRS="${ROOT}/artifacts/cache/conda-pkgs" \
    conda create -n "$ENV_NAME" python=3.10 pip -y
fi
conda run -n "$ENV_NAME" python -m pip install -e "${ROOT}[quant]"
conda run -n "$ENV_NAME" python -m pip freeze > "${ROOT}/environment.${ENV_NAME}.lock.txt"
echo "Installed converter and lock file."
