---
license: mit
language:
- en
- zh
pipeline_tag: text-generation
library_name: transformers
base_model:
- cyankiwi/GLM-5.2-AWQ-INT4
- zai-org/GLM-5.2
tags:
- glm
- glm-5.2
- moe
- awq
- int4
- pruned
- gb10
- vllm
---

# GLM-5.2-AWQ-INT4-15pct

15% expert-pruned `cyankiwi/GLM-5.2-AWQ-INT4` (256→218 experts/layer), to free
KV-cache headroom for 256k context on memory-constrained clusters (built for 4×
GB10 / DGX Spark). The prune is data-free. ~636B total / ~40B active, AWQ
compressed-tensors W4A16, ~378 GB.

Quality is coherence-checked, not benchmarked. Evaluate before production use; for
guaranteed quality use the unpruned `cyankiwi/GLM-5.2-AWQ-INT4` or
`zai-org/GLM-5.2`.

## Method (data-free)

GLM/DeepSeek routers carry a learned `e_score_correction_bias` per expert: a high
bias means the router had to boost that expert to select it (least favored).
`awq_surgery.py` drops the 38 highest-bias experts per layer, keeps the 218
lowest, re-indexes survivors, and row-slices the router. No calibration data, no
forward passes. Both `num_experts` and `n_routed_experts` become 218. This is not
REAP (REAP needs calibration data and was infeasible on this hardware).

## Serving

A 256k 4-bit MoE for multi-node TP, not a single-GPU model. Needs sm_121 Triton
sparse-MLA kernels (native `_flashmla_C` is Hopper-only). Stack and bootstrap:
github.com/CosmicRaisins/glm-5.2-gb10. Runtime: TP=4, `--kv-cache-dtype
fp8_ds_mla`, `--reasoning-parser glm45 --tool-call-parser glm47
--enable-auto-tool-choice`, cudagraph FULL, `gpu-memory-utilization 0.93`.

## License

MIT, inherited. Retain upstream notices on redistribution. GLM-5.2 © Z.ai (MIT);
GLM-5.2-AWQ-INT4 © cyankiwi (MIT). The data-free prune is the only modification.
Not affiliated with Z.ai or cyankiwi.
