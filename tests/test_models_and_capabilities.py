from llm_autocompress.capabilities import assess, matching_capabilities
from llm_autocompress.models import inspect_model
from llm_autocompress.schema import DEFAULT_WEIGHTS_ROOT


def test_opt_model_detection_and_weight_size():
    model = inspect_model(DEFAULT_WEIGHTS_ROOT / "opt-125m")
    assert model.family == "opt"
    assert model.kind == "dense"
    assert model.parameter_bytes > 200_000_000
    assert model.parameter_bytes < 400_000_000


def test_missing_quant_dependency_is_explained():
    model = inspect_model(DEFAULT_WEIGHTS_ROOT / "opt-125m")
    capability = next(
        item
        for item in matching_capabilities(["gptq_w4a16"], ["vllm"])
        if item.method == "gptq_w4a16"
    )
    env = {
        "gpus": [{"compute_capability": "8.9"}],
        "modules": {
            "torch": {"available": True},
            "transformers": {"available": True},
            "vllm": {"available": True},
            "llmcompressor": {"available": False},
        },
        "repos": {},
    }
    runnable, reasons = assess(capability, model, env)
    assert not runnable
    assert "python module llmcompressor is missing" in reasons
