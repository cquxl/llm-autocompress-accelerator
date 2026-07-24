---
name: llm-autocompress-accelerator
description: Automate real local LLM and MoE checkpoint compression, capability-aware inference backend selection, quality evaluation, CUDA kernel or end-to-end benchmarking, and honest acceleration reporting. Use for Wanda, SparseGPT, D2Prune, ROSE, GPTQ, AWQ, SmoothQuant, 2:4, expert/channel pruning, vLLM, cuSPARSELt, Samoyeds, SpInfer, cuBLAS, OPT, Llama, Qwen, Mixtral, or DeepSeek workflows.
---

# LLM AutoCompress Accelerator

Turn an EdgeLite-style business requirement into a reproducible local compression and acceleration run. Keep compression algorithms, resulting sparsity/quantization formats, and execution kernels as three distinct layers.

Use `flatquant` for serving/benchmarking. Run `scripts/setup_quant_env.sh --yes` once to create the isolated GPTQ/AWQ/SmoothQuant converter; do not install its conflicting dependencies into `flatquant`.

## Workflow

1. Read [business-workflow.md](references/business-workflow.md), then translate the
   quality, effective-compression and same-backend speed requirements:

   ```bash
   conda run -n flatquant llm-autopilot bootstrap \
     --model opt-125m --prompt "权重压缩4倍，PPL增幅不超过20%，吞吐不低于Dense" \
     --output /tmp/opt-request.yaml
   ```

2. Inspect without modifying models:

   ```bash
   conda run -n flatquant llm-autopilot inspect --request /tmp/opt-request.yaml
   conda run -n flatquant llm-autopilot plan --request /tmp/opt-request.yaml
   ```

3. Read [compression-methods.md](references/compression-methods.md) and
   [backend-matrix.md](references/backend-matrix.md). Probe the first ranked
   quantization candidate and stop as soon as every business gate passes. Enter
   Wanda/SparseGPT/D2Prune/ROSE pruning only when quantization misses the target.
   Do not describe compression algorithms as kernels or kernels as algorithms.

4. Execute only after explicit approval:

   ```bash
   conda run -n flatquant llm-autopilot run \
     --request /tmp/opt-request.yaml --mode smoke --yes
   ```

5. Interpret results under [benchmark-protocol.md](references/benchmark-protocol.md).
   Read `search_trace.json` before recommending. Recommend only end-to-end candidates
   that pass quality, effective-compression and same-backend speed gates. Clearly
   label expert, MoE block, layer, and linear microbenchmarks.

## Built-in demos

```bash
conda run -n flatquant llm-autopilot demo opt-125m --mode smoke --yes
conda run -n flatquant llm-autopilot demo deepseek-v2-lite --mode smoke --yes
conda run -n flatquant llm-autopilot serve --host 127.0.0.1 --port 7860
```

The OPT demo targets reloadable compression and end-to-end quality. The DeepSeek demo loads real checkpoint tensors and activations for expert/kernel evidence; it must not claim compressed end-to-end generation until that path is implemented and verified.

## Guardrails

- Reuse local weights and Arrow datasets; do not download when a local copy exists.
- Never enable SpInfer fake sparsity. Never invent timing, PPL, or memory data.
- Never use magnitude pruning as a formal candidate. Use the local Wanda,
  SparseGPT, D2Prune or ROSE implementation; a layout-only mask is permitted only
  for isolated kernel validation and must be labeled as such.
- Record skipped/failed candidates and continue.
- Preserve existing Samoyeds, D2Prune and SpInfer repositories; write only under the run directory.
- Require `--yes` for installs, compilation, conversion, and model writes.
- Use [request-schema.md](references/request-schema.md) when editing requests manually.
