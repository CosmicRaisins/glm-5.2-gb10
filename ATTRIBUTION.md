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
- **yewentao256 / vLLM** — `fused_indexer_q_rope_quant` (upstream PR #46862): the
  fused indexer Q rope+fp8-quant Triton kernel, vendored in
  `kernels/sparse_attn_indexer.py` with its call-site in `kernels/deepseek_v2.py`.
  Not original to this repo. (Apache-2.0)
- **eugr** — `spark-vllm-docker` build harness and `llama-benchy`.
- **aidendle94 / local-inference-lab** — B12X kernel lineage, raw-entrypoint
  serving pattern.
- **back199640** (GB10 user forum) — the fp8 decode head-padding fix
  (`_compute_fp8_decode_padded_heads`: pad to 32, not 64, when heads/rank ≤ 32),
  vendored in `kernels/flashmla_sparse.py`. At TP=4 GLM-5.2 has 16 heads/rank, so
  the old 64-pad wasted 75% of the fp8 attention compute; a ~+28–34% prefill win
  measured on GB10. The subsequent 32 → 16 step (`CHANGES.md` #6, zero padding at
  TP=4 via the b12x `mg_n_hg=1` path, a further +6–10% prefill) is this project's
  follow-up to their insight.
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
