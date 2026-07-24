# Request YAML

Required: `name`, `model.path`.

- `model`: local path, optional family/kind, `trust_remote_code`.
- `methods`: identifiers such as `wanda_2_4`, `rose_expert`, `gptq_w4a16`.
- `backends`: `auto` or explicit backend names.
- `calibration`: local Arrow directory, sample count, sequence length, seed.
- `evaluation`: local WikiText2 directory, token budgets, prompts.
- `constraints`: PPL increase, minimum speedup, optional VRAM limit.
- `search`: staged-search switch, effective weight compression target,
  quantization-first policy, pruning fallback granularity, and trial budget.
- `pruning`: sparsity, expert/channel ratios, target layers, router tuning.
- `workload`: profile, batch/input/output shapes, warmups and iterations.
- `execution`: device, tensor parallel size, offline mode, timeout.
- `output_dir`: run root.

Use `llm-autopilot bootstrap` for business-language input and inspect YAML before a full run.
