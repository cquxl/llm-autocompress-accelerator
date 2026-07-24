# Backend and kernel matrix

| Compressed form | Preferred backend | Measurement scope |
| --- | --- | --- |
| GPTQ/AWQ/SmoothQuant | vLLM | end-to-end generation |
| 2:4 linear | Samoyeds `cusparselt24_kernel` (`CuSparseLtLinear`) | real checkpoint linear; identical-mask dense cuBLAS reference |
| Samoyeds N:M expert | Samoyeds extension | real checkpoint expert/MoE microbenchmark |
| unstructured OPT | SpInfer + FasterTransformer compatibility path | end-to-end after real conversion; force `FAKE_SPARSITY=False` |
| channel-reduced shapes | cuBLAS | real expert/channel microbenchmark |
| dense baseline | Transformers and vLLM | end-to-end |
| TensorRT-LLM | optional | enable only when capability probe succeeds |

Never compare a microbenchmark speedup to an end-to-end baseline. Never label block, expert,
or decoder-layer measurements as full-model acceleration.

The cuSPARSELt row must use the source in
`native/samoyeds_cusparselt/cusparselt24_mod.cu`, which directly calls
`cusparseLtSpMMACompress`, `cusparseLtMatmulSearch`, and `cusparseLtMatmul`.
`torch._cslt_sparse_mm`, PyTorch semi-structured tensors, and the PyTorch
CUTLASS fallback are not permitted substitutes.
