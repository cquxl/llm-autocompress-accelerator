# New GPU host bootstrap

The Skill separates portable project code from host-local paths:

- bundled: orchestration, Web UI, reporting, quantization adapters, and the
  source-only D2Prune core (Wanda, SparseGPT, D2Prune/ADMM, 2:4, ROSE);
- provisioned per host: pinned Samoyeds and SpInfer source, CUDA extensions, vLLM,
  and the isolated GPTQ/AWQ/SmoothQuant converter built from pinned
  `llmcompressor`/`compressed-tensors` packages;
- never bundled: checkpoints, datasets, compressed artifacts, caches, or secrets.

## One-time A40/L40 setup

From a fresh clone:

```bash
./scripts/setup_host.sh \
  --model-root=/models/llm_weights \
  --data-root=/datasets/llm \
  --run-root=/results/llm-autocompress \
  --with-dependencies \
  --build-samoyeds \
  --yes
```

The command defaults to `llm-autocompress-runtime` and
`llm-autocompress-quant`. To reuse an existing environment such as `flatquant`,
add `--runtime-env=flatquant`. It writes
`~/.config/llm-autocompress/site.yaml`; use `--config=/path/site.yaml` for an
alternate location.

On A40, the doctor identifies compute capability 8.6 and the Samoyeds build is
compiled locally for `sm_86`. On L40, it compiles for `sm_89`. Do not copy a
compiled extension between those architectures.

`--build-samoyeds` also compiles the source-only Samoyeds
`cusparselt24_kernel` into `<dependency_root>/samoyeds-cusparselt`. This extension
uses the direct NVIDIA cuSPARSELt C API and is mandatory for the `cusparselt`
backend; there is no PyTorch private-operator fallback.

## Discovery precedence

Every path can be supplied by the site YAML or environment variables:

```text
LLM_AUTOCOMPRESS_MODEL_ROOT
LLM_AUTOCOMPRESS_DATA_ROOT
LLM_AUTOCOMPRESS_RUN_ROOT
LLM_AUTOCOMPRESS_DEPENDENCY_ROOT
LLM_AUTOCOMPRESS_D2PRUNE_ROOT
LLM_AUTOCOMPRESS_SPINFER_ROOT
LLM_AUTOCOMPRESS_SAMOYEDS_ROOT
LLM_AUTOCOMPRESS_RUNTIME_ENV
LLM_AUTOCOMPRESS_QUANT_ENV
```

If D2Prune is not explicitly configured, the bundled core is selected. Request
files use `${MODEL_ROOT}`, `${DATA_ROOT}`, and `${RUN_ROOT}` tokens, so the same
YAML works on different servers.

## Readiness gate

Run this before conversion:

```bash
conda run -n llm-autocompress-runtime llm-autopilot doctor
```

Do not launch an unavailable candidate. The doctor reports separate readiness
for quantization autopilot, pruning fallback, cuSPARSELt, Samoyeds, and SpInfer.
Missing optional backends become explicit `skipped` candidates rather than
fabricated measurements.
