# Compression methods

| Algorithm | Structures | Importance/calibration | Artifact |
| --- | --- | --- | --- |
| Wanda | unstructured, 2:4 | activation-aware weights × activation norms | reloadable HF checkpoint through bundled D2Prune core |
| SparseGPT | unstructured, 2:4 | approximate second-order reconstruction | reloadable HF checkpoint through bundled D2Prune core |
| D2Prune | unstructured, 2:4, channel | Wanda + SparseGPT + ADMM flags | reloadable when the bundled dispatcher supports the family |
| ROSE | expert, unstructured, 2:4, channel | real router counts and expert weights | structural plan or real-weight micro artifact |
| GPTQ | W4A16 | local C4 | compressed-tensors checkpoint |
| AWQ | W4A16 | local C4 activation statistics | compressed-tensors checkpoint |
| SmoothQuant | W8A8 | local C4 activation statistics | compressed-tensors checkpoint |

For 2:4, validate every group along K has at most two nonzeros. For expert/channel pruning,
compare against both the identical masked dense reference and original dense weights.

The algorithm is not the acceleration backend. For example, Wanda may produce a
2:4 checkpoint and cuSPARSELt executes that format; ROSE may produce an expert
or channel plan and Samoyeds/cuBLAS measures the compatible operator.
