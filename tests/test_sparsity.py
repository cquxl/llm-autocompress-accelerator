from pathlib import Path

import pytest
import torch

from llm_autocompress.adapters.compression import (
    prune_2_4_tensor,
    validate_2_4_tensor,
)
from llm_autocompress.adapters.sparse import (
    benchmark_cusparselt_weight,
    samoyeds_cusparselt_probe,
)


def test_two_of_four_pruning_is_real_and_valid():
    weight = torch.randn(64, 128)
    pruned = prune_2_4_tensor(weight)
    validation = validate_2_4_tensor(pruned)
    assert validation["valid"]
    assert validation["groups_exactly_two_nonzeros"] == validation["groups"]
    assert validation["sparsity"] == 0.5
    assert not torch.equal(weight, pruned)


def test_cusparselt_adapter_has_no_pytorch_private_operator():
    source = (
        Path(__file__).resolve().parents[1]
        / "llm_autocompress"
        / "adapters"
        / "sparse.py"
    ).read_text(encoding="utf-8")
    assert "torch._cslt_sparse_mm" not in source
    assert "to_sparse_semi_structured" not in source


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not samoyeds_cusparselt_probe().get("available"),
    reason="host-built Samoyeds cuSPARSELt extension is unavailable",
)
def test_samoyeds_cusparselt_executes_and_matches_dense_reference():
    result = benchmark_cusparselt_weight(
        torch.randn(64, 64),
        token_counts=[16],
        warmup=1,
        iterations=2,
        device="cuda:0",
    )
    assert result["implementation"] == "samoyeds_cusparselt24_kernel"
    assert result["uses_torch_private_cslt"] is False
    assert result["uses_pytorch_cutlass_fallback"] is False
    assert result["records"][0]["max_abs_error"] < 0.1
