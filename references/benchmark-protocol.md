# Benchmark protocol

- Full: batch 1/8, input 128/512/2048, output 32/128, 10 warmups, 30 iterations.
- Smoke: first shape, up to 2 warmups and 5 iterations; record actual counts.
- Synchronize CUDA around timings; kernel paths use CUDA events.
- Report checkpoint bytes, compression ratio, peak VRAM, PPL, TTFT, prefill tokens/s,
  TPOT, decode tokens/s, request throughput and end-to-end latency when exposed.
- Default quality gate: relative WikiText2 PPL increase no more than 5%.
- Interactive ranks inverse TPOT; throughput ranks tokens/s; prefill-heavy ranks prefill tokens/s.
- Rank same-backend compression gains first. Separately report Transformers-to-deployment gains.
- Missing metrics remain null. Failed candidates cannot be recommended.
- With staged search enabled, require the effective weight compression target in
  addition to PPL and speed, and record each decision in `search_trace.json`.
