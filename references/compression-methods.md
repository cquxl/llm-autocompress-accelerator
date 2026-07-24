# Compression methods

| Algorithm | Structures | Importance/calibration | Artifact |
| --- | --- | --- | --- |
| Wanda | unstructured, 2:4 | activation-aware magnitude | reloadable HF checkpoint through local D2Prune |
| SparseGPT | unstructured, 2:4 | approximate second-order reconstruction | reloadable HF checkpoint through local D2Prune |
| D2Prune | unstructured, 2:4, channel | Wanda + SparseGPT + ADMM flags | reloadable when the local dispatcher supports the family |
| ROSE | expert, unstructured, 2:4, channel | real router counts and expert weights | structural plan or real-weight micro artifact |
| GPTQ | W4A16 | local C4 | compressed-tensors checkpoint |
| AWQ | W4A16 | local C4 activation statistics | compressed-tensors checkpoint |
| SmoothQuant | W8A8 | local C4 activation statistics | compressed-tensors checkpoint |

For 2:4, validate every group along K has at most two nonzeros. For expert/channel pruning,
compare against both the identical masked dense reference and original dense weights.
