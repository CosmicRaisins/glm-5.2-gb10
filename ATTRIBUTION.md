# Attribution

This stack builds on open-source work. Licensing is in `NOTICE`; the full
sequence of fixes is in `docs/retrospective.md`.

- **jasl** — the portable Triton sparse-MLA kernels and the sm12x DeepGEMM
  fallback. GLM-5.2's attention can't run on sm_121 without these. (Apache-2.0,
  vLLM lineage)
- **cyankiwi** — `GLM-5.2-AWQ-INT4`, the INT4 weights this prunes. (MIT)
- **Z.ai / Zhipu AI** — GLM-5.2: model, `GlmMoeDsa` arch, native MTP, `glm45`/
  `glm47` formats. (MIT)
- **hazyumps** (`deepseek-v4-flash-gb10`) — GB10 runbook: NCCL 2.30.4, RDMA/
  `IPC_LOCK` passthrough, bf16-indexer.
- **vLLM project** — `GlmMoeDsa`, Marlin WNA16, b12x MoE (#40082), the parsers,
  the NVFP4 oracle. (Apache-2.0)
- **eugr** — `spark-vllm-docker` build harness and `llama-benchy`.
- **aidendle94 / local-inference-lab** — B12X kernel lineage, raw-entrypoint
  serving pattern.
- **0xSero** — NVFP4-REAP checkpoints, MTP layer-78 reference.
- **brandonmmusic-max / voipmonitor** — GLM-5.2 consumer-Blackwell patches
  reference.
- **NVIDIA** — DGX Spark / GB10, CUDA 13 / FlashInfer / cutlass, NCCL 2.30.4
  aarch64.

REAP (CerebrasResearch) was evaluated and not used; the prune here is a different,
data-free method.

Original to this repo: the data-free `e_score_correction_bias` prune
(`prune/awq_surgery.py`), the int32→int64 prefill fix and index-bounds guards, the
fused gather-dequant prefill kernel, the separate-draft MTP reconstruction
(`mtp/`), the V3.2 monkeypatch adaptation, the recipe, and the bootstrap. Built by
CosmicRaisins with agentic assistance. Not affiliated with the parties above.
