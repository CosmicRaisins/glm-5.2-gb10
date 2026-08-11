# GLM-5.2 on 4× GB10 — current retrospective

The repository began as a way to run GLM-5.2's sparse-MLA architecture on
consumer Blackwell (`sm_121`) and now centers on a TP4/DCP2 QuantTrio stack with
adaptive MTP and vision. The production profile is
`recipes/glm52-quanttrio-vision-dcp2-290816.yaml`.

## 1. Making sparse GLM attention run on GB10

Stock and early mainline vLLM paths depended on kernels that did not support
`sm_121`. The working foundation came from community Apache-2.0 work:

- jasl's portable sm12x sparse-MLA and DeepGEMM fallbacks;
- Aiden Le / local-inference-lab's B12X serving and kernel lineage;
- vLLM's `GlmMoeDsa`, sparse indexer, Marlin, and parser implementations;
- hazyumps' GB10 transport/runbook findings.

The legacy `kernels/` port adapted that work to GLM's V3.2 interfaces. The main
repository-specific fixes were int64 gathered-KV indexing, explicit sparse-index
bounds, a fused gather/dequantize/attention prefill kernel, and the 32-to-16
head-padding follow-up. The earlier 64-to-32 fix is credited to back199640.
Those changes made long-context prefill coherent and removed much of the padded
attention work at TP4.

The current DCP image no longer mounts the legacy tree; its pinned branch carries
native B12X sparse attention. The legacy kernels remain useful as a documented,
Apache-2.0 no-DCP implementation and as the record of the GB10 fixes.

## 2. Moving to DCP and unpruned QuantTrio

The first production route used pruned AWQ weights to make room for KV cache.
DCP changed that trade-off by sharding the MLA KV cache across decode-context
ranks. The current stack therefore serves the unpruned
`QuantTrio/GLM-5.2-Int4-Int8Mix` checkpoint and its quantized in-checkpoint MTP
head.

The DCP foundation and GLM draft path are credited to m9e / voipmonitor and
vLLM. The two PR #72 patches in this repository preserve that work. This
repository adds the reproducible integration, the quantized-NextN packed-module
mapping fix, the sparse `index_topk_pattern` override, recipes, and GB10
validation. Zatz's 655k demonstration provided the public high-context reference
that was independently reproduced here.

The production choice is DCP2 at 290,816 configured context. DCP4 can expose
roughly twice the context, but its per-layer collectives reduce prefill. Decode
stays around the low-20 tok/s range because it is dominated by active-weight
bandwidth rather than KV placement.

## 3. Adaptive MTP

Fixed speculative depth is content-sensitive: prose often does not repay a wide
verification step, while code and structured output can. The current controller
therefore derives from Aiden Le / aidendle94 and local-inference-lab's
acceptance-length adaptive speculative-decoding work.

This repository forward-ports that foundation to the pinned DCP scheduler and
adds the production policy:

- k2 is the safe baseline;
- strong head acceptance probes k4;
- marginal accepted tokens at p2+p3 decide whether k4 pays;
- p4's marginal gain decides whether k5 pays;
- exploratory k4/k5 windows are shorter than the baseline window;
- every allowed depth has CUDA-graph coverage;
- `MTP_WINDOW_JSON` exposes the decisions without additional model work.

This is intentionally described as a modification and productionization of
Aiden's work, not as an independently invented adaptive-decoding method. The
exact thresholds and state transitions are in `adaptive-mtp/README.md`.

## 4. Vision without replacing the text model

The vision path keeps QuantTrio's text and MTP tensors unchanged. A zero-copy
assembler attaches Baseten's pinned MoonViT tower and trained PatchMerger
projector, then records source revisions, file hashes, and the merged tensor
index.

The local vLLM wrapper preserves QuantTrio quantization mappings, exposes the
nested target `lm_head` to the MTP loader, routes the draft through the text
proposer, and propagates GLM's sparse indexer pattern. Baseten and Moonshot retain
credit and their upstream weight terms; the wrapper, assembler, and integration
changes are Apache-2.0 source in this repository.

## Historical path

The 15% data-free prune and standalone INT4 MTP reconstruction remain published
for reproducibility but are no longer production. They solved the pre-DCP memory
constraint. The old draft also passed through a 0xSero checkpoint with no
explicit repository license, so its weights are not presented as Apache-2.0.
REAP was evaluated and not used.

## Current contribution boundary

The current repository contribution is the GB10 integration layer: adaptive
2/4/5 policy and instrumentation, quantized-draft fix, GLM-5V wrapper and
assembler, legacy kernel fixes, production recipes, and coherent long-context
validation. The base model, quantized weights, DCP foundation, adaptive-depth
foundation, B12X/jasl kernels, and vision weights are credited to their authors
in `../ATTRIBUTION.md` and `../NOTICE`.
