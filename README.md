# LLM AutoCompress Accelerator

面向本地 LLM/MoE 的 EdgeLite 风格自动化业务闭环：

`业务要求 → 环境/模型检查 → 候选矩阵 → 真实压缩 → 质量门槛 → Kernel/端到端测速 → 报告`

压缩算法（Wanda、SparseGPT、D2Prune、ROSE、GPTQ、AWQ、SmoothQuant）与加速后端
（vLLM、cuSPARSELt、Samoyeds、SpInfer、cuBLAS）严格分层。

```bash
./scripts/setup_env.sh --env=flatquant --yes
./scripts/setup_quant_env.sh --yes
conda run -n flatquant llm-autopilot inspect --request assets/requests/opt-125m.yaml
conda run -n flatquant llm-autopilot demo opt-125m --mode smoke --yes
conda run -n flatquant llm-autopilot serve
```

运行保存原始 JSON、CSV、manifest、日志、Markdown 报告和 SVG 图表。失败或缺失指标不会被伪造。

`flatquant` 用于 vLLM、Transformers、cuSPARSELt 和报告；隔离的
`llm-autocompress-quant` 仅做 GPTQ/AWQ/SmoothQuant 转换，避免与 vLLM 的固定依赖冲突。
