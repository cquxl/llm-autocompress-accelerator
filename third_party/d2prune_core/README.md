# Bundled D2Prune core

This directory is a source-only snapshot of the local CQU D2Prune research
implementation used by this project. It is included so a new GPU host can run
Wanda, SparseGPT, D2Prune/ADMM, N:M (including 2:4), and MoE/ROSE pruning
without depending on an absolute path from the original development server.

Snapshot contents:

- `main.py`, `cfg/`, `data/`, `model/`, `prune/`, and `utils/`
- Wanda and SparseGPT implementations integrated by D2Prune
- D2Prune/ADMM and MoE/ROSE pruning implementations
- the upstream MIT license and Python requirements

Deliberately excluded:

- model checkpoints and compressed weights
- calibration/evaluation datasets
- generated outputs, logs, caches, notebooks, and bytecode

The autopilot never edits this snapshot in place. It copies it into the
current run directory, applies any recorded Transformers compatibility patch
there, and records the execution source and patch list in the conversion
manifest.

For an intentionally different D2Prune checkout, set
`LLM_AUTOCOMPRESS_D2PRUNE_ROOT` or `d2prune_root` in the host site config.
