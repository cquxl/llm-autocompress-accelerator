# Backend and kernel matrix

| Compressed form | Preferred backend | Measurement scope |
| --- | --- | --- |
| GPTQ/AWQ/SmoothQuant | vLLM | end-to-end generation |
| 2:4 linear | cuSPARSELt (`torch._cslt_sparse_mm`) | real checkpoint linear; identical-mask dense reference |
| Samoyeds N:M expert | Samoyeds extension | real checkpoint expert/MoE microbenchmark |
| unstructured OPT | SpInfer + FasterTransformer compatibility path | end-to-end after real conversion; force `FAKE_SPARSITY=False` |
| channel-reduced shapes | cuBLAS | real expert/channel microbenchmark |
| dense baseline | Transformers and vLLM | end-to-end |
| TensorRT-LLM | optional | enable only when capability probe succeeds |

Never compare a microbenchmark speedup to an end-to-end baseline. Never label block, expert,
or decoder-layer measurements as full-model acceleration.
