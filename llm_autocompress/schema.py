from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .site import expand_site_tokens, load_site_config
from .utils import load_mapping


SKILL_ROOT = Path(__file__).resolve().parents[1]
SITE_CONFIG = load_site_config()
SAMOYEDS_ROOT = SITE_CONFIG.samoyeds_root
PROJECTS_ROOT = SITE_CONFIG.dependency_root
D2PRUNE_ROOT = SITE_CONFIG.d2prune_root
SPINFER_ROOT = SITE_CONFIG.spinfer_root
DEFAULT_WEIGHTS_ROOT = SITE_CONFIG.model_root
DEFAULT_DATA_ROOT = SITE_CONFIG.data_root

KNOWN_METHODS = {
    "dense",
    "gptq_w4a16",
    "awq_w4a16",
    "smoothquant_w8a8",
    "wanda_unstructured",
    "wanda_2_4",
    "sparsegpt_unstructured",
    "sparsegpt_2_4",
    "d2prune_unstructured",
    "d2prune_2_4",
    "d2prune_channel",
    "rose_expert",
    "rose_unstructured",
    "rose_2_4",
    "rose_channel",
}
KNOWN_BACKENDS = {
    "auto",
    "transformers",
    "vllm",
    "cublas",
    "cusparselt",
    "samoyeds",
    "spinfer",
    "tensorrt_llm",
}
KNOWN_PROFILES = {"interactive", "throughput", "prefill-heavy"}


def _list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (str, int, float)):
        return [value]
    return list(value)


@dataclass(slots=True)
class ModelSpec:
    path: str
    family: str = "auto"
    kind: str = "auto"
    trust_remote_code: bool = False


@dataclass(slots=True)
class CalibrationSpec:
    dataset: str = str(DEFAULT_DATA_ROOT / "c4")
    samples: int = 128
    sequence_length: int = 2048
    seed: int = 42


@dataclass(slots=True)
class EvaluationSpec:
    dataset: str = str(DEFAULT_DATA_ROOT / "wikitext" / "wikitext-2-raw-v1")
    max_tokens_smoke: int = 4096
    max_tokens_full: int = 32768
    generation_prompts: list[str] = field(
        default_factory=lambda: [
            "The future of efficient language model inference is",
            "Sparse mixture-of-experts models are useful because",
        ]
    )


@dataclass(slots=True)
class ConstraintSpec:
    max_relative_ppl_increase: float = 0.05
    min_same_backend_speedup: float = 1.0
    max_vram_gb: float | None = None


@dataclass(slots=True)
class PruningSpec:
    sparsity_ratio: float = 0.5
    expert_ratio: float = 0.25
    channel_ratio: float = 0.25
    target_layer_names: list[str] = field(default_factory=list)
    only_moe: bool = False
    tune_router: bool = True


@dataclass(slots=True)
class SearchSpec:
    enabled: bool = False
    quantization_first: bool = True
    target_checkpoint_ratio: float = 2.0
    allow_pruning_fallback: bool = True
    combine_with_best_quant: bool = True
    pruning_granularity: str = "2:4"
    max_trials: int = 3


@dataclass(slots=True)
class WorkloadSpec:
    profile: str = "interactive"
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8])
    input_lengths: list[int] = field(default_factory=lambda: [128, 512, 2048])
    output_lengths: list[int] = field(default_factory=lambda: [32, 128])
    warmup: int = 10
    iterations: int = 30


@dataclass(slots=True)
class ExecutionSpec:
    device: str = "cuda:0"
    tensor_parallel_size: int = 1
    offline: bool = True
    timeout_seconds: int = 7200
    fallback_model: str | None = str(
        DEFAULT_WEIGHTS_ROOT / "models--meta-llama--Llama-2-7b-hf"
    )
    allow_external_output: bool = False


@dataclass(slots=True)
class CompressionRequest:
    name: str
    model: ModelSpec
    calibration: CalibrationSpec = field(default_factory=CalibrationSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    constraints: ConstraintSpec = field(default_factory=ConstraintSpec)
    pruning: PruningSpec = field(default_factory=PruningSpec)
    search: SearchSpec = field(default_factory=SearchSpec)
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    methods: list[str] = field(
        default_factory=lambda: [
            "gptq_w4a16",
            "awq_w4a16",
            "smoothquant_w8a8",
            "wanda_2_4",
            "sparsegpt_unstructured",
            "d2prune_2_4",
            "rose_expert",
        ]
    )
    backends: list[str] = field(default_factory=lambda: ["auto"])
    output_dir: str = str(SITE_CONFIG.run_root)

    def validate(self, *, require_model: bool = True) -> None:
        if require_model and not Path(self.model.path).expanduser().exists():
            raise ValueError(f"model path does not exist: {self.model.path}")
        unknown_methods = set(self.methods) - KNOWN_METHODS
        if unknown_methods:
            raise ValueError(f"unknown compression methods: {sorted(unknown_methods)}")
        unknown_backends = set(self.backends) - KNOWN_BACKENDS
        if unknown_backends:
            raise ValueError(f"unknown inference backends: {sorted(unknown_backends)}")
        if self.workload.profile not in KNOWN_PROFILES:
            raise ValueError(f"unknown workload profile: {self.workload.profile}")
        if not 0 <= self.constraints.max_relative_ppl_increase <= 10:
            raise ValueError("max_relative_ppl_increase must be between 0 and 10")
        if self.calibration.samples <= 0 or self.calibration.sequence_length <= 0:
            raise ValueError("calibration samples and sequence length must be positive")
        for ratio, label in (
            (self.pruning.sparsity_ratio, "sparsity_ratio"),
            (self.pruning.expert_ratio, "expert_ratio"),
            (self.pruning.channel_ratio, "channel_ratio"),
        ):
            if not 0 <= ratio < 1:
                raise ValueError(f"{label} must be in [0, 1)")
        if self.search.target_checkpoint_ratio < 1:
            raise ValueError("search.target_checkpoint_ratio must be >= 1")
        if self.search.max_trials <= 0:
            raise ValueError("search.max_trials must be positive")
        if self.search.pruning_granularity not in {
            "2:4",
            "unstructured",
            "channel",
            "expert",
        }:
            raise ValueError("unsupported search.pruning_granularity")
        for values, label in (
            (self.workload.batch_sizes, "batch_sizes"),
            (self.workload.input_lengths, "input_lengths"),
            (self.workload.output_lengths, "output_lengths"),
        ):
            if not values or any(int(item) <= 0 for item in values):
                raise ValueError(f"{label} must contain positive integers")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def request_from_mapping(data: dict[str, Any]) -> CompressionRequest:
    data = expand_site_tokens(data)
    model_data = data.get("model")
    if isinstance(model_data, str):
        model_data = {"path": model_data}
    if not isinstance(model_data, dict) or not model_data.get("path"):
        raise ValueError("request.model.path is required")

    calibration_data = data.get("calibration") or {}
    evaluation_data = data.get("evaluation") or {}
    constraints_data = data.get("constraints") or {}
    pruning_data = data.get("pruning") or {}
    search_data = data.get("search") or {}
    workload_data = data.get("workload") or {}
    execution_data = data.get("execution") or {}

    request = CompressionRequest(
        name=str(data.get("name") or Path(model_data["path"]).name),
        model=ModelSpec(
            path=str(Path(model_data["path"]).expanduser().resolve()),
            family=str(model_data.get("family", "auto")),
            kind=str(model_data.get("kind", "auto")),
            trust_remote_code=bool(model_data.get("trust_remote_code", False)),
        ),
        calibration=CalibrationSpec(
            dataset=str(calibration_data.get("dataset", CalibrationSpec().dataset)),
            samples=int(calibration_data.get("samples", 128)),
            sequence_length=int(calibration_data.get("sequence_length", 2048)),
            seed=int(calibration_data.get("seed", 42)),
        ),
        evaluation=EvaluationSpec(
            dataset=str(evaluation_data.get("dataset", EvaluationSpec().dataset)),
            max_tokens_smoke=int(evaluation_data.get("max_tokens_smoke", 4096)),
            max_tokens_full=int(evaluation_data.get("max_tokens_full", 32768)),
            generation_prompts=[
                str(item)
                for item in _list(
                    evaluation_data.get("generation_prompts"),
                    EvaluationSpec().generation_prompts,
                )
            ],
        ),
        constraints=ConstraintSpec(
            max_relative_ppl_increase=float(
                constraints_data.get("max_relative_ppl_increase", 0.05)
            ),
            min_same_backend_speedup=float(
                constraints_data.get("min_same_backend_speedup", 1.0)
            ),
            max_vram_gb=(
                None
                if constraints_data.get("max_vram_gb") is None
                else float(constraints_data["max_vram_gb"])
            ),
        ),
        pruning=PruningSpec(
            sparsity_ratio=float(pruning_data.get("sparsity_ratio", 0.5)),
            expert_ratio=float(pruning_data.get("expert_ratio", 0.25)),
            channel_ratio=float(pruning_data.get("channel_ratio", 0.25)),
            target_layer_names=[
                str(item)
                for item in _list(pruning_data.get("target_layer_names"), [])
            ],
            only_moe=bool(pruning_data.get("only_moe", False)),
            tune_router=bool(pruning_data.get("tune_router", True)),
        ),
        search=SearchSpec(
            enabled=bool(search_data.get("enabled", False)),
            quantization_first=bool(
                search_data.get("quantization_first", True)
            ),
            target_checkpoint_ratio=float(
                search_data.get("target_checkpoint_ratio", 2.0)
            ),
            allow_pruning_fallback=bool(
                search_data.get("allow_pruning_fallback", True)
            ),
            combine_with_best_quant=bool(
                search_data.get("combine_with_best_quant", True)
            ),
            pruning_granularity=str(
                search_data.get("pruning_granularity", "2:4")
            ),
            max_trials=int(search_data.get("max_trials", 3)),
        ),
        workload=WorkloadSpec(
            profile=str(workload_data.get("profile", "interactive")),
            batch_sizes=[int(item) for item in _list(workload_data.get("batch_sizes"), [1, 8])],
            input_lengths=[
                int(item)
                for item in _list(workload_data.get("input_lengths"), [128, 512, 2048])
            ],
            output_lengths=[
                int(item)
                for item in _list(workload_data.get("output_lengths"), [32, 128])
            ],
            warmup=int(workload_data.get("warmup", 10)),
            iterations=int(workload_data.get("iterations", 30)),
        ),
        execution=ExecutionSpec(
            device=str(execution_data.get("device", "cuda:0")),
            tensor_parallel_size=int(execution_data.get("tensor_parallel_size", 1)),
            offline=bool(execution_data.get("offline", True)),
            timeout_seconds=int(execution_data.get("timeout_seconds", 7200)),
            fallback_model=execution_data.get(
                "fallback_model", ExecutionSpec().fallback_model
            ),
            allow_external_output=bool(
                execution_data.get("allow_external_output", False)
            ),
        ),
        methods=[str(item) for item in _list(data.get("methods"), CompressionRequest.__dataclass_fields__["methods"].default_factory())],
        backends=[str(item) for item in _list(data.get("backends"), ["auto"])],
        output_dir=str(data.get("output_dir", SITE_CONFIG.run_root)),
    )
    request.validate()
    return request


def load_request(path: str | Path) -> CompressionRequest:
    request_path = Path(path).expanduser().resolve()
    return request_from_mapping(load_mapping(request_path))
