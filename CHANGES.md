# Modifications to third-party source

This file records prominent changes to Apache-2.0 source as required by Apache
License 2.0 Section 4(b). Original SPDX and copyright notices remain in modified
files. `NOTICE` records upstream licensing; `ATTRIBUTION.md` explains authorship.

## Current DCP and quantized-draft integration

The production image is based on a pinned Apache-2.0 vLLM fork carrying DCP and
B12X sparse attention.

- `patches/pr72-1-draft-dcp-config-propagation.patch` and
  `patches/pr72-2-glm-dcp-draft-path.patch` preserve m9e / voipmonitor PR #72.
  They propagate DCP configuration into the draft and provide the GLM sparse
  indexer buffers required by the DCP MTP path. These patches are third-party
  work, not authored by this repository.
- `patches/draft-quant-packed-mapping.patch` adds fused-module packed mappings
  when vLLM constructs a quantized NextN draft config. Without it, the draft can
  silently instantiate unquantized projections and lose acceptance.
- The production recipes configure TP4, DCP2, the sparse GLM indexer pattern, a
  compressed-tensors draft, probabilistic sampling, and the B12X sparse backend.

## Adaptive speculative decoding

`patches/adaptive-mtp-vllm-hooks.patch` and
`adaptive-mtp/overlay/.../acceptance_length.py` modify Apache-2.0 vLLM source.
They forward-port Aiden Le / aidendle94 and local-inference-lab's
acceptance-length adaptive-depth foundation to the pinned DCP scheduler.

Repository-specific changes to that foundation are:

1. Replace the continuous/ratcheted target with a configurable discrete ladder
   and initialize at its lowest point (`2,4,5` in production).
2. Use k2 as a 32-step baseline. A draft-token acceptance ratio of 0.85 probes
   k4.
3. Judge k4/k5 from unconditional marginal accepted tokens at p2+p3 and p4:
   `0.70` probes k5, `0.35` retains k4, and `0.15` at p4 retains k5.
4. Shorten exploratory k4/k5 windows to 16 steps so the controller retreats
   quickly when content changes.
5. Capture CUDA graphs for every allowed depth.
6. Add opt-in `MTP_WINDOW_JSON` telemetry from existing CPU-side accept counts;
   no extra model work or synchronization is introduced.

The controller remains batch-wide. Production sets `max_num_seqs=1`, making it
session-equivalent for the workload it was tuned for.

## GLM-5.2 vision integration

The `vision/` overlay modifies and specializes Apache-2.0 vLLM multimodal code.
It uses unmodified, separately licensed QuantTrio and Baseten weights.

1. Register a local `Glm5vConfig` and `Glm5vForConditionalGeneration` around
   `GlmMoeDsaForCausalLM`.
2. Preserve QuantTrio tensor names and compressed-tensors mappings below the
   multimodal wrapper.
3. Expose the nested target `lm_head` so the MTP draft shares the correct output
   projection; otherwise speculative acceptance is zero.
4. Route GLM's MTP proposer through the text-token path even though the target
   also accepts images.
5. Propagate `index_topk_pattern` through the outer vision config.
6. Assemble a zero-copy composite with pinned sources, relative symlinks, a
   merged tensor index, and a hash/revision manifest. No weights are trained or
   requantized by the assembler.

Baseten supplied the trained PatchMerger projector, frozen MoonViT packaging,
processing/configuration reference, and chat template. Moonshot AI supplied the
MoonViT/Kimi lineage. Those weights retain their upstream terms.

## Legacy sm12x kernel port

The `kernels/` tree is retained as the original no-DCP GB10 path. It derives
from Apache-2.0 vLLM and jasl portable sm12x sparse-MLA/DeepGEMM-fallback work.
The current DCP production image uses its branch-native backend instead of this
vendored tree, but these modifications remain substantive and reproducible.

1. **GLM V3.2 adapter and monkeypatch** (`patch_flashmla_ops.py`,
   `flashmla_sparse.py`): adapt the portable wrappers to V3.2
   `FlashMLASparseImpl` signatures and route sm12x away from unavailable
   `_flashmla_C` kernels.
2. **int32 overflow fix** (`sm12x_sparse_mla_attn.py`): promote gathered-KV
   program-id/stride arithmetic to int64. At TP4 and T=2048 the prior product
   crossed 2^31 and corrupted long-context prefill.
3. **Indexer bounds guards** (`sparse_mla_kernels.py`): reject negative and
   out-of-range sparse KV indices in scalar, multihead, and d512-split paths.
4. **Fused gather/dequantize/attention prefill**
   (`sm12x_sparse_mla_attn.py`): split 576 into NoPE 512 + RoPE 64, use
   tensor-core `tl.dot` with online softmax, and avoid materializing the full
   gathered tensor. This flattened the legacy depth curve and raised cold
   prefill from roughly 336 to 508 tok/s in the recorded configuration.
5. **fp8 decode head padding 64 to 32** (`flashmla_sparse.py`): credited to
   back199640. On TP4 this removed half of the unnecessary padded heads and
   measured a 28–34% legacy prefill gain.
6. **fp8 decode head padding 32 to 16** (`flashmla_sparse.py`): this
   repository's follow-up uses the B12X `mg_n_hg=1` path at 16 heads/rank,
   removing padding at TP4 and adding a further 6–10% legacy prefill gain.

`kernels/sparse_attn_indexer.py` and `kernels/deepseek_v2.py` also carry vLLM PR
#46862 (`fused_indexer_q_rope_quant`, yewentao256). That code is vendored
upstream work, not an original contribution here.

## Original source and generated artifacts

The repository's original scripts, recipes, documentation, and modifications
are Apache-2.0 unless marked otherwise. This includes the vision assembler,
adaptive policy code, quantized-draft mapping fix, launch integration, and the
historical prune/reconstruction scripts.

The license on source tooling does not relicense its input or output model
weights. Generated composites and legacy model artifacts retain all applicable
Z.ai, QuantTrio, cyankiwi, Baseten, Moonshot, and other upstream terms described
in `NOTICE`.
