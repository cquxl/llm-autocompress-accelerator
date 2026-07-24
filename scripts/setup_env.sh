#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="flatquant"
INSTALL_QUANT=0
CONFIRM=0
for arg in "$@"; do
  case "$arg" in
    --env=*) ENV_NAME="${arg#*=}" ;;
    --quant) INSTALL_QUANT=1 ;;
    --yes) CONFIRM=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Target conda environment: ${ENV_NAME}"
echo "Project: ${ROOT}"
if [[ "$CONFIRM" -ne 1 ]]; then
  echo "Dry run only. Re-run with --yes to install the local package."
  exit 0
fi
if [[ "$INSTALL_QUANT" -eq 1 ]]; then
  echo "Quant conversion dependencies conflict with vLLM in flatquant." >&2
  echo "Use scripts/setup_quant_env.sh --yes for the isolated converter." >&2
  exit 1
fi
if ! conda env list --json | grep -q "\"[^\"]*/${ENV_NAME}\""; then
  echo "Conda environment does not exist: ${ENV_NAME}" >&2
  exit 1
fi
EXTRA="runtime,test"
conda run -n "$ENV_NAME" python -m pip install -e "${ROOT}[${EXTRA}]"
conda run -n "$ENV_NAME" python -m pip freeze > "${ROOT}/environment.${ENV_NAME}.lock.txt"
echo "Installed and locked: ${ROOT}/environment.${ENV_NAME}.lock.txt"
