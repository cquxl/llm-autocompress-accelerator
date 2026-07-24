# LLM AutoCompress Accelerator

面向本地 LLM/MoE 的 EdgeLite 风格自动化业务闭环：

`业务要求 → 环境/模型检查 → 候选矩阵 → 真实压缩 → 质量门槛 → Kernel/端到端测速 → 报告`

压缩算法（Wanda、SparseGPT、D2Prune、ROSE、GPTQ、AWQ、SmoothQuant）与加速后端
（vLLM、cuSPARSELt、Samoyeds、SpInfer、cuBLAS）严格分层。

仓库内已包含 D2Prune 的纯源码核心（Wanda、SparseGPT、D2Prune/ADMM、2:4、
MoE/ROSE），不包含模型、数据、缓存或历史结果。Samoyeds 和 SpInfer 按固定
commit 在每台 GPU 服务器上准备，Samoyeds 必须针对当前 GPU 重新编译。
其中 2:4 的 cuSPARSELt 后端固定使用仓库内的 Samoyeds
`cusparselt24_kernel` 原生扩展，直接调用 NVIDIA cuSPARSELt C API，不调用
PyTorch 私有的 `_cslt_sparse_mm`，也不回退到 PyTorch CUTLASS。

新 A40/L40 服务器只需从 GitHub clone 后执行一次：

```bash
./scripts/setup_host.sh \
  --model-root=/models/llm_weights \
  --data-root=/datasets/llm \
  --run-root=/results/llm-autocompress \
  --with-dependencies --build-samoyeds --yes

conda run -n llm-autocompress-runtime llm-autopilot doctor
conda run -n llm-autocompress-runtime llm-autopilot demo opt-125m --mode smoke --yes
conda run -n llm-autocompress-runtime llm-autopilot serve
```

当前服务器需要复用 `flatquant` 时，在初始化命令中增加
`--runtime-env=flatquant`。

运行保存原始 JSON、CSV、manifest、日志、Markdown 报告和 SVG 图表。失败或缺失指标不会被伪造。

runtime 环境用于 vLLM、Transformers、cuSPARSELt、Samoyeds 和报告；隔离的
`llm-autocompress-quant` 仅做 GPTQ/AWQ/SmoothQuant 转换，避免与 vLLM 的
固定依赖冲突。请求 YAML 使用 `${MODEL_ROOT}`、`${DATA_ROOT}`、`${RUN_ROOT}`，
相同请求可直接迁移到不同服务器。

量化核心 adapter 位于
`llm_autocompress/adapters/compression.py`，实际调用锁定版本的
`llmcompressor`/`compressed-tensors`；每个转换 manifest 都记录实现与版本。
可选的论文原始仓库及固定 commit 记录在 `assets/dependencies.yaml`。
