#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ENV="llm-autocompress-runtime"
QUANT_ENV="llm-autocompress-quant"
MODEL_ROOT=""
DATA_ROOT=""
RUN_ROOT=""
DEPENDENCY_ROOT=""
D2PRUNE_ROOT=""
CONFIG_PATH=""
WITH_DEPS=0
BUILD_SAMOYEDS=0
CONFIRM=0

for arg in "$@"; do
  case "$arg" in
    --runtime-env=*) RUNTIME_ENV="${arg#*=}" ;;
    --quant-env=*) QUANT_ENV="${arg#*=}" ;;
    --model-root=*) MODEL_ROOT="${arg#*=}" ;;
    --data-root=*) DATA_ROOT="${arg#*=}" ;;
    --run-root=*) RUN_ROOT="${arg#*=}" ;;
    --dependency-root=*) DEPENDENCY_ROOT="${arg#*=}" ;;
    --d2prune-root=*) D2PRUNE_ROOT="${arg#*=}" ;;
    --config=*) CONFIG_PATH="${arg#*=}" ;;
    --with-dependencies) WITH_DEPS=1 ;;
    --build-samoyeds) BUILD_SAMOYEDS=1 ;;
    --yes) CONFIRM=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "$DEPENDENCY_ROOT" ]]; then DEPENDENCY_ROOT="${ROOT}/dependencies"; fi
if [[ -z "$RUN_ROOT" ]]; then RUN_ROOT="${ROOT}/runs"; fi
if [[ -z "$D2PRUNE_ROOT" ]]; then D2PRUNE_ROOT="${ROOT}/third_party/d2prune_core"; fi

echo "Project: ${ROOT}"
echo "Runtime env: ${RUNTIME_ENV}"
echo "Quant env: ${QUANT_ENV}"
echo "Model root: ${MODEL_ROOT:-<required for real runs>}"
echo "Data root: ${DATA_ROOT:-<required for calibration/evaluation>}"
echo "Dependency root: ${DEPENDENCY_ROOT}"
echo "D2Prune root: ${D2PRUNE_ROOT}"
if [[ "$CONFIRM" -ne 1 ]]; then
  echo "Dry run only. Re-run with --yes to create environments and configuration."
  exit 0
fi
if [[ -z "$MODEL_ROOT" || -z "$DATA_ROOT" ]]; then
  echo "--model-root and --data-root are required with --yes" >&2
  exit 2
fi

"${ROOT}/scripts/setup_env.sh" --env="${RUNTIME_ENV}" --create --yes
"${ROOT}/scripts/setup_quant_env.sh" --env="${QUANT_ENV}" --yes

if [[ "$WITH_DEPS" -eq 1 ]]; then
  DEP_ARGS=(
    --destination "${DEPENDENCY_ROOT}"
    --runtime-env "${RUNTIME_ENV}"
    --yes
  )
  if [[ "$BUILD_SAMOYEDS" -eq 1 ]]; then DEP_ARGS+=(--build-samoyeds); fi
  conda run -n "${RUNTIME_ENV}" \
    python "${ROOT}/scripts/bootstrap_dependencies.py" "${DEP_ARGS[@]}"
fi

CONFIG_ARGS=(
  configure-host
  --model-root "${MODEL_ROOT}"
  --data-root "${DATA_ROOT}"
  --run-root "${RUN_ROOT}"
  --dependency-root "${DEPENDENCY_ROOT}"
  --d2prune-root "${D2PRUNE_ROOT}"
  --spinfer-root "${DEPENDENCY_ROOT}/SpInfer"
  --samoyeds-root "${DEPENDENCY_ROOT}/Samoyeds"
  --runtime-env "${RUNTIME_ENV}"
  --quant-env "${QUANT_ENV}"
  --yes
)
if [[ -n "$CONFIG_PATH" ]]; then CONFIG_ARGS+=(--config "${CONFIG_PATH}"); fi
conda run -n "${RUNTIME_ENV}" llm-autopilot "${CONFIG_ARGS[@]}"

if [[ -n "$CONFIG_PATH" ]]; then
  LLM_AUTOCOMPRESS_CONFIG="${CONFIG_PATH}" \
    conda run -n "${RUNTIME_ENV}" llm-autopilot doctor
else
  conda run -n "${RUNTIME_ENV}" llm-autopilot doctor
fi
