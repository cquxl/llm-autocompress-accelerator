from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .environment import module_available
from .models import ModelInfo


@dataclass(frozen=True, slots=True)
class Capability:
    method: str
    backend: str
    families: tuple[str, ...]
    kinds: tuple[str, ...]
    scope: str
    required_modules: tuple[str, ...] = ()
    required_repo: str | None = None
    min_compute_capability: float | None = None
    real_artifact: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REGISTRY: tuple[Capability, ...] = (
    Capability(
        "dense",
        "transformers",
        ("*",),
        ("dense", "moe"),
        "end_to_end",
        ("torch", "transformers"),
    ),
    Capability(
        "dense",
        "vllm",
        ("opt", "llama", "mistral", "mixtral", "qwen", "deepseek"),
        ("dense", "moe"),
        "end_to_end",
        ("vllm",),
    ),
    Capability(
        "gptq_w4a16",
        "vllm",
        ("opt", "llama", "mistral", "qwen", "deepseek", "mixtral"),
        ("dense", "moe"),
        "end_to_end",
        ("torch", "transformers", "llmcompressor", "vllm"),
        min_compute_capability=7.0,
    ),
    Capability(
        "awq_w4a16",
        "vllm",
        ("opt", "llama", "mistral", "qwen", "deepseek", "mixtral"),
        ("dense", "moe"),
        "end_to_end",
        ("torch", "transformers", "llmcompressor", "vllm"),
        min_compute_capability=7.5,
    ),
    Capability(
        "smoothquant_w8a8",
        "vllm",
        ("opt", "llama", "mistral", "qwen", "deepseek", "mixtral"),
        ("dense", "moe"),
        "end_to_end",
        ("torch", "transformers", "llmcompressor", "vllm"),
        min_compute_capability=7.5,
    ),
    Capability(
        "wanda_2_4",
        "cusparselt",
        ("opt", "llama", "mistral", "mixtral", "qwen", "deepseek"),
        ("dense", "moe"),
        "linear_and_checkpoint",
        ("torch", "transformers"),
        required_repo="d2prune",
        min_compute_capability=8.0,
    ),
    Capability(
        "sparsegpt_2_4",
        "cusparselt",
        ("opt", "llama", "mistral", "mixtral", "qwen", "deepseek"),
        ("dense", "moe"),
        "linear_and_checkpoint",
        ("torch", "transformers"),
        required_repo="d2prune",
        min_compute_capability=8.0,
    ),
    Capability(
        "d2prune_2_4",
        "cusparselt",
        ("opt", "llama", "mistral", "mixtral", "qwen", "deepseek"),
        ("dense", "moe"),
        "linear_and_checkpoint",
        ("torch", "transformers"),
        required_repo="d2prune",
        min_compute_capability=8.0,
    ),
    Capability(
        "wanda_unstructured",
        "spinfer",
        ("opt",),
        ("dense",),
        "end_to_end",
        ("torch", "transformers"),
        required_repo="spinfer",
        min_compute_capability=8.0,
    ),
    Capability(
        "sparsegpt_unstructured",
        "spinfer",
        ("opt",),
        ("dense",),
        "end_to_end",
        ("torch", "transformers"),
        required_repo="spinfer",
        min_compute_capability=8.0,
    ),
    Capability(
        "d2prune_unstructured",
        "spinfer",
        ("opt",),
        ("dense",),
        "end_to_end",
        ("torch", "transformers"),
        required_repo="spinfer",
        min_compute_capability=8.0,
    ),
    Capability(
        "d2prune_channel",
        "cublas",
        ("opt", "llama", "mistral", "mixtral", "qwen", "deepseek"),
        ("dense", "moe"),
        "structured_checkpoint_and_linear",
        ("torch", "transformers"),
        required_repo="d2prune",
        min_compute_capability=7.0,
        notes="Physically reduced channel shapes use dense cuBLAS kernels.",
    ),
    Capability(
        "rose_2_4",
        "cusparselt",
        ("deepseek", "mixtral", "qwen"),
        ("moe",),
        "expert_moe_layer",
        ("torch",),
        min_compute_capability=8.0,
        notes="ROSE 2:4 mask on a real expert weight with cuSPARSELt timing.",
    ),
    Capability(
        "rose_2_4",
        "samoyeds",
        ("deepseek", "mixtral", "qwen"),
        ("moe",),
        "expert_moe_layer",
        ("torch",),
        required_repo="samoyeds",
        min_compute_capability=8.0,
    ),
    Capability(
        "rose_unstructured",
        "spinfer",
        ("deepseek", "mixtral", "qwen"),
        ("moe",),
        "expert_moe_layer",
        ("torch",),
        required_repo="spinfer",
        min_compute_capability=8.0,
        notes="ROSE expert-weight pruning with SpInfer-compatible sparse linear microbenchmark.",
    ),
    Capability(
        "rose_expert",
        "samoyeds",
        ("deepseek", "mixtral", "qwen"),
        ("moe",),
        "expert_moe_layer",
        ("torch",),
        required_repo="samoyeds",
        min_compute_capability=8.0,
        notes="Router-calibrated expert selection; not reported as full-model acceleration.",
    ),
    Capability(
        "rose_channel",
        "cublas",
        ("deepseek", "mixtral", "qwen"),
        ("moe",),
        "expert_moe_layer",
        ("torch",),
        required_repo="d2prune",
        min_compute_capability=7.0,
        notes="ROSE channel importance with physically reduced expert linear shapes.",
    ),
    Capability(
        "dense",
        "cublas",
        ("*",),
        ("dense", "moe"),
        "linear",
        ("torch",),
        min_compute_capability=7.0,
    ),
    Capability(
        "dense",
        "samoyeds",
        ("deepseek", "mixtral", "qwen"),
        ("moe",),
        "expert_moe_layer",
        ("torch",),
        required_repo="samoyeds",
        min_compute_capability=8.0,
        real_artifact=False,
        notes="Dense comparison path for Samoyeds microbenchmarks.",
    ),
    Capability(
        "dense",
        "tensorrt_llm",
        ("llama", "mistral", "qwen", "deepseek"),
        ("dense", "moe"),
        "optional_end_to_end",
        ("tensorrt_llm",),
        min_compute_capability=8.0,
        notes="Optional only; not a v1 acceptance backend.",
    ),
)


def _compute_capability(env: dict[str, Any]) -> float | None:
    gpus = env.get("gpus") or []
    if not gpus:
        return None
    try:
        return max(float(gpu["compute_capability"]) for gpu in gpus)
    except (KeyError, TypeError, ValueError):
        return None


def assess(
    capability: Capability, model: ModelInfo, env: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if "*" not in capability.families and model.family not in capability.families:
        reasons.append(f"model family {model.family} is unsupported")
    if model.kind not in capability.kinds:
        reasons.append(f"model kind {model.kind} is unsupported")
    for module in capability.required_modules:
        alternate_quant = (
            module == "llmcompressor"
            and env.get("converter_envs", {}).get("quant", {}).get("available")
        )
        if not module_available(env, module) and not alternate_quant:
            reasons.append(f"python module {module} is missing")
    if capability.required_repo:
        repo = env.get("repos", {}).get(capability.required_repo, {})
        if not repo.get("markers_valid", repo.get("exists")):
            reasons.append(
                f"local repo {capability.required_repo} is missing or incomplete "
                f"at {repo.get('path', '<unconfigured>')}"
            )
    if capability.backend == "samoyeds":
        extension = env.get("extensions", {}).get("samoyeds", {})
        if not extension.get("available"):
            reasons.append(
                "Samoyeds extension is unavailable or ABI-incompatible with this PyTorch"
            )
    required_cc = capability.min_compute_capability
    current_cc = _compute_capability(env)
    if required_cc is not None:
        if current_cc is None:
            reasons.append("no NVIDIA GPU detected")
        elif current_cc < required_cc:
            reasons.append(f"compute capability {current_cc} < {required_cc}")
    return not reasons, reasons


def matching_capabilities(
    methods: Iterable[str],
    backends: Iterable[str],
) -> list[Capability]:
    method_set = set(methods) | {"dense"}
    backend_set = set(backends)
    auto = "auto" in backend_set
    return [
        capability
        for capability in REGISTRY
        if capability.method in method_set
        and (auto or capability.backend in backend_set)
    ]
