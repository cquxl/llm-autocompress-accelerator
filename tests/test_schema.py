from pathlib import Path

import pytest

from llm_autocompress.business import apply_web_preset, request_mapping_from_business
from llm_autocompress.schema import SKILL_ROOT, load_request, request_from_mapping


def test_builtin_opt_request_is_valid():
    request = load_request(SKILL_ROOT / "assets" / "requests" / "opt-125m.yaml")
    assert request.model.path.endswith("opt-125m")
    assert "wanda_2_4" in request.methods
    assert request.workload.iterations == 30


def test_business_request_keeps_algorithm_and_backend_separate():
    mapping = request_mapping_from_business(
        model="opt-125m",
        prompt="用 Wanda 2:4 并通过 cuSPARSELt 测速",
    )
    assert mapping["methods"] == ["wanda_2_4"]
    assert mapping["backends"] == ["cusparselt"]


def test_unknown_method_is_rejected():
    mapping = request_mapping_from_business(model="opt-125m")
    mapping["methods"] = ["not-a-method"]
    with pytest.raises(ValueError, match="unknown compression"):
        request_from_mapping(mapping)


def test_web_autocompress_preset_is_not_dense_only():
    mapping = apply_web_preset(
        request_mapping_from_business(model="opt-125m"),
        "auto-smoke",
    )
    assert mapping["methods"] == ["awq_w4a16", "wanda_2_4"]
    assert mapping["backends"] == ["auto"]
    assert mapping["web_preset"] == "auto-smoke"
    assert mapping["search"]["enabled"] is True
