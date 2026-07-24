#!/usr/bin/env bash
set -euo pipefail

MODE="link"
CONFIRM=0
for arg in "$@"; do
  case "$arg" in
    --yes) CONFIRM=1 ;;
    --copy) MODE="copy" ;;
    --link) MODE="link" ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODEX_ROOT="${CODEX_HOME:-/data/xl/.codex}"
TARGET="${CODEX_ROOT}/skills/llm-autocompress-accelerator"
echo "Mode: ${MODE}"
echo "Source: ${SOURCE}"
echo "Target: ${TARGET}"
if [[ -e "$TARGET" || -L "$TARGET" ]]; then
  echo "Refusing to overwrite existing skill: ${TARGET}" >&2
  exit 1
fi
if [[ "$CONFIRM" -ne 1 ]]; then
  echo "Dry run only. Re-run with --yes to install."
  exit 0
fi
mkdir -p "$(dirname "$TARGET")"
if [[ "$MODE" == "link" ]]; then ln -s "$SOURCE" "$TARGET"; else cp -a "$SOURCE" "$TARGET"; fi
echo "Installed Skill at ${TARGET}"
