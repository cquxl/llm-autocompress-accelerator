# Samoyeds cuSPARSELt 2:4 extension

This is the source-only Samoyeds `cusparselt24_kernel` binding used by the
autopilot's `cusparselt` backend. It was integrated from
`samoyeds_mod/api/cusparselt24_mod.cu` in the local Samoyeds `codex/kernel`
worktree.

The extension accepts a real FP16 2:4 weight, compresses it with
`cusparseLtSpMMACompress`, searches the algorithm with
`cusparseLtMatmulSearch`, caches a plan per padded token shape, and executes
`cusparseLtMatmul`. PyTorch is used only for tensor ownership, CUDA streams,
and the Python extension ABI. The implementation does not call
`torch._cslt_sparse_mm`, `to_sparse_semi_structured`, or the PyTorch CUTLASS
fallback.

Build it for each target host/GPU:

```bash
conda run -n llm-autocompress-runtime \
  python scripts/build_samoyeds_cusparselt.py \
  --output-dir dependencies/samoyeds-cusparselt --yes
```

The resulting shared object is host-specific and is intentionally not stored
in Git.
