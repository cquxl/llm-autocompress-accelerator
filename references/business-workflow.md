# EdgeLite-style business workflow

1. Translate natural language or the Web form into the fixed YAML schema.
2. Inspect the local checkpoint, GPU/CUDA stack, datasets, repositories and Python modules.
3. Rank a short, ordered candidate list instead of exhaustively running it:
   AWQ/GPTQ (or SmoothQuant for prefill-heavy) first, then one pruning fallback
   matching the requested granularity.
4. Run one candidate, then check quality, effective compression and same-backend
   speed. Stop immediately when all three pass.
5. Enter Wanda/SparseGPT/D2Prune/ROSE only when quantization misses a gate; do not
   use magnitude pruning as a formal fallback.
6. Mark unsupported or early-stopped candidates `skipped` with the exact reason.
7. Write a conversion manifest before changing weights.
8. Run real conversion, reload/generation/PPL checks, then the compatible deployment or kernel benchmark.
9. Keep failures isolated. Never substitute historical or synthetic measurements.
10. Preserve `search_trace.json`, raw JSON, CSV, logs and manifests; regenerate Markdown without rerunning models.

The independent layers are:

- Algorithm: Wanda, SparseGPT, D2Prune, ROSE, GPTQ, AWQ, SmoothQuant.
- Structure/format: unstructured, 2:4, channel, expert, W4A16, W8A8.
- Execution: Transformers, vLLM, cuBLAS, cuSPARSELt, Samoyeds, SpInfer, optional TensorRT-LLM.
