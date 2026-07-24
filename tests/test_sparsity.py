import torch

from llm_autocompress.adapters.compression import (
    prune_2_4_tensor,
    validate_2_4_tensor,
)


def test_two_of_four_pruning_is_real_and_valid():
    weight = torch.randn(64, 128)
    pruned = prune_2_4_tensor(weight)
    validation = validate_2_4_tensor(pruned)
    assert validation["valid"]
    assert validation["groups_exactly_two_nonzeros"] == validation["groups"]
    assert validation["sparsity"] == 0.5
    assert not torch.equal(weight, pruned)
