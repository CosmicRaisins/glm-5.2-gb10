# GLM-5.2 on DGX Spark (GB10, sm_121)

The current production profile adds vision and adaptive MTP without replacing
the QuantTrio text backbone: TP4 + DCP2, 290,816 context, 299,648 tokens of raw
KV capacity, and an acceptance-driven k=2/4/5 controller. The vision tower and
projector come from
[baseten/GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4);
the adaptive-depth foundation is credited to
[Aiden Le / aidendle94](https://huggingface.co/aidendle94), with the production
policy modifications documented in [`adaptive-mtp/`](adaptive-mtp/README.md).

Serves GLM-5.2 (744B/40B MoE, `GlmMoeDsa`) on a 4-node GB10 cluster. The
text-only reference config serves the **unpruned**
[QuantTrio Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)
(full 256 experts, 406 GB, in-checkpoint MTP) at **320k context, ~600 t/s
prefill / ~22 t/s decode, flat to depth** via TP4 + decode-context-parallel
(DCP2, KV sharded 2-way). No pruning, no quality compromise.

Context vs prefill is a dial — one flag (`--decode-context-parallel-size`):
Benchmarked with llama-benchy. Mixed agentic coding workloads can sustain about
28 tok/s when draft acceptance is stronger than book prose.

| Config | Weights | Context | Prefill (d0/8k/32k) | Decode | Recipe |
|---|---|---|---|---|---|
| **DCP2 + vision + adaptive MTP** (production) | unpruned QuantTrio + MoonViT/PatchMerger | **290,816** | 604 / 610 / — | 22.3 / 21.3 / — mean | `glm52-quanttrio-vision-dcp2-290816.yaml` |
| **DCP2** (reference) | unpruned | **327,680** | 598 / 603 / 598 | ~22  | `glm52-quanttrio-unpruned-dcp2-320k.yaml` |
| DCP4 | unpruned | **655,360** | ~430 flat | ~22 | `glm52-quanttrio-unpruned-dcp4-640k.yaml` |
| No DCP (legacy stack) | 10%-pruned | 327,680 | 722 / 736 / 626 | ~22 | `glm52-quanttrio-10pct-prod.yaml` |

**The trade-off:** DCP shards the MLA KV cache across ranks instead of
replicating it (per-rank KV ÷ N ⇒ context × N), but prefill attention then pays
a per-layer cross-rank LSE all-gather + reduce-scatter, so prefill falls
roughly linearly with DCP degree: ~720 → ~600 → ~430 t/s. Decode is unaffected
(~21–24 everywhere; it's MoE-bandwidth-bound). Without DCP the unpruned
checkpoint only fits ~160k of KV — the no-DCP row uses the 10% expert prune to
reach 327k, which is also the max-prefill option if you accept the prune (see
Weights).

## Requirements

4× GB10 / DGX Spark (sm_121, aarch64), node-to-node RoCE, ~410 GB of weights on
every node (or NFS). Not portable to single-GPU, x86, or datacenter Blackwell
(sm_100).

## Two stacks

**DCP stack (reference).** Built from public sources; this repo's `kernels/`
are *not* used (the branch's native `B12X_MLA_SPARSE` backend includes the
sm_121 sparse-MLA work, head-padding included):

- vLLM: [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm)
  branch `codex/dcp-globaltopk-sharddraft-defaults-20260622` @ `e232d26`, plus
  the three patches vendored in `patches/` (apply with `git apply -p1` against
  the vLLM tree, in order):
  - `pr72-1-draft-dcp-config-propagation.patch` +
    `pr72-2-glm-dcp-draft-path.patch` (= upstream PR #72) — without these the
    drafter crashes with `requires topk_scores_buffer` under DCP.
  - `draft-quant-packed-mapping.patch` — without it quantized-NextN drafts
    silently build unquantized and MTP acceptance collapses.
  - `adaptive-mtp-vllm-hooks.patch` + `adaptive-mtp/overlay/` — Aiden Le's
    acceptance-length adaptation forward-ported to this scheduler, with our
    2/4/5 production policy, per-position marginal-gain decisions,
    instrumentation, and CUDA graphs for every active depth.
- b12x @ `9cd63a7` (`pip install --no-deps git+https://github.com/lukealonso/b12x@9cd63a7...`).
- Non-negotiable launch requirements (all in the recipes):
  - `--hf-overrides '{"index_topk_pattern":"FFFSSS…"}'` — 78 chars derived from
    the checkpoint's `indexer_types` (`full`→`F`, `shared`→`S`). GLM-5.2 only
    trains indexer weights on 22/78 layers; the QuantTrio config ships
    `index_topk_pattern: null`, and without the override the other 56 layers
    top-k through **uninitialized weights** — coherent under ~2k tokens
    (top-k ≡ select-all), garbage beyond, MTP acceptance craters at depth.
  - `VLLM_USE_V2_MODEL_RUNNER=1` (the DCP+MTP drafting path lives in the V2
    runner; the V1 runner drops DCP from the draft config).
  - `VLLM_USE_B12X_SPARSE_INDEXER=1`, `--attention-backend B12X_MLA_SPARSE`,
    and the same backend pinned as `draft_attention_backend`.
  - A compressed-tensors draft with probabilistic sampling. Fixed-depth
    reference recipes use k=3; production sets maximum k=5 plus
    `adaptive_speculative_tokens_window` and the `2,4,5` ladder.
  - `VLLM_MARLIN_USE_ATOMIC_ADD=1` in the current production profile.

Full credit for the DCP branch, kernels, and the 640k demonstration: m9e /
voipmonitor (vLLM branch, PR #72) and Zatz (GB10 forum, 655k single-boot
result). Independently reproduced here: coherent with exact long-range
retrieval at depth, MTP acceptance ~2.0–2.4 accepted/draft.

**Legacy kernel stack.** The original no-DCP port remains under `kernels/` as a
modified Apache-2.0 implementation of the vLLM/jasl portable sm12x path. It
contains the GLM V3.2 adapter, int64 and bounds fixes, fused prefill kernel, and
head-padding work documented in `CHANGES.md`. Two patch steps from my
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) fork wire
them in:

- `mods/glm52-sm12x-sparse` copies `kernels/` into vLLM and routes
  DeepGEMM-only paths to the `sm12x_*` fallbacks.
- `mods/glm52-b12x-sparse` installs B12X and provides the capture-safe
  sparse-MLA decode path.

## Run

Build the pinned DCP base with the patches listed above, then build the adaptive
and vision child images from the repository root. Exact commands and base-image
pins are in `adaptive-mtp/README.md` and `vision/README.md`. The current serving
spec is `recipes/glm52-quanttrio-vision-dcp2-290816.yaml`.

**Prefix-cache tradeoff for agent clients:** the stock GLM-5.2 chat template
clears reasoning from turns before the latest user message. This saves context,
but it also rewrites the prompt at each user-turn boundary and invalidates the
cached suffix after the first cleared reasoning block. Clients that prioritize
cross-turn TTFT can opt out per request:

```json
{"chat_template_kwargs":{"clear_thinking":false}}
```

This keeps prior reasoning in subsequent prompts so the prefix stays
append-only, at the cost of consuming more context and replaying that reasoning
to the model. The recipes in this repo do not set it; the default remains
reasoning-clearing behavior.

The recipe is the production source of truth. Replace the cluster-specific RoCE
interfaces and node addresses before launching it through a compatible
raw-entrypoint harness or transcribing its environment and command into your
launcher. Clear page caches on every node before a large boot; vLLM's unified-
memory startup guard can otherwise reject the 0.90 allocation. The API listens
on `:8210`.

`bootstrap.sh` and `launch.sh` are retained for the legacy no-DCP kernel stack.
`launch.sh` is a plain per-node `docker run`; it is not the current DCP image
builder.

## Weights

- **Unpruned (reference):** [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)
  — 256 experts, w4a16/w8a16, in-checkpoint MTP (layer-78 nextn).
- **Vision production composite:** the same QuantTrio text and MTP tensors plus
  Baseten's pinned MoonViT-3d tower and trained PatchMerger projector. The
  assembler symlinks the existing text shards, so it does not duplicate or
  requantize the 406 GB backbone; see [`vision/`](vision/README.md).
- **Historical only:** the 10% QuantTrio prune and legacy
  [AWQ 15% prune](https://huggingface.co/CosmicRaisins/GLM-5.2-AWQ-INT4-15pct)
  plus its separate MTP draft predate DCP. They remain reproducible under
  `prune/` and `mtp/`, but are not the production path.

## Performance notes

**Methodology.** Tables use
[llama-benchy](https://github.com/eugr/llama-benchy), one stream
(`max_num_seqs=1`, concurrency 1), pp2048/tg512, and a coherent book corpus.
The production table uses five runs; historical reference tables use three.
Random-token corpora misstate speculative acceptance badly (3–5× in my tests).
Decode therefore moves with content and acceptance; prefill is comparatively
tight.

**MTP k:** the current production recipe configures a maximum of k=5 but adapts
on a 2/4/5 ladder. k=2 is the prose-safe baseline; strong head acceptance probes
k4, then the unconditional marginal gain at p2/p3 and p4 decides whether k4 or
k5 pays for itself. Text-only historical recipes remain fixed-k unless they set
`adaptive_speculative_tokens_window`; see [`adaptive-mtp/`](adaptive-mtp/README.md).

**Bench vs real workload:** the book-corpus result is the conservative decode
number. Mixed coding-agent traffic can sustain about 28 tok/s when code and
structured output raise draft acceptance; treat that as workload-specific.

### DCP2 + vision + adaptive MTP — 290,816 ctx (production)

The current atomic-add run used llama-benchy v0.3.7, pp2048/tg512, five runs,
concurrency 1, and prefix-cache bypass. The controller exercised k2/k4/k5 and
accepted 2,944 of 4,676 drafted tokens across warmup and measurement.

| Depth | Prefill mean | Decode mean | Decode median | TTFR median |
|---|---:|---:|---:|---:|
| 0 | 603.5 | 22.3 ± 3.0 | 21.5 | 3.42 s |
| 8K | 609.9 | 21.3 ± 1.1 | 21.7 | 16.80 s |

The earlier vision-v6 sweep included 32K and measured prefill 629/643/640 and
decode means 21.9/19.4/21.9 at depth 0/8K/32K. Raw results are under
`benchmarks/atomic-20260810/` and `benchmarks/vision-v6-20260724/`.

### DCP2 — unpruned, 327,680 ctx (text-only reference)

| Depth | Prefill (pp2048) | Decode (tg512) | Peak | TTFR |
|---|---|---|---|---|
| 0   | 597.9 ± 6.4 | 21.7 ± 0.6 | 31.3 | 3.4 s |
| 8K  | 602.6 ± 0.8 | 21.5 ± 0.8 | 31.0 | 17.0 s |
| 32K | 597.7 ± 0.2 | 21.8 ± 0.6 | 30.7 | 58.2 s |

### DCP4 — 655,360 ctx

The table is the retained historical sweep; the unpruned checkpoint has also
been validated for fit, coherence, and long-range retrieval on DCP4, but not with
a complete depth sweep.

| Depth | Prefill (pp2048) | Decode (tg512) | Peak |
|---|---|---|---|
| 0   | — (first-run JIT skew) | 21.0 ± 2.2 | 31.0 |
| 8K  | 430.4 ± 0.4 | 19.4 ± 1.7 | 29.0 |
| 32K | 428.2 ± 0.1 | 21.6 ± 2.0 | 29.0 |

Community result on the same stack holds 19.6–25.7 decode to 638k depth (TTFT
at that depth ~24 min — deep-context prefill is the cost of the big window).

### Legacy kernel progression

The no-DCP measurements are retained to isolate kernel changes, not to
recommend the old pruned serving path. With 16 real heads/rank at TP4, the fp8
head-padding progression was:

stock kernel padded queries to 64 heads (75% zeros) → pad-32 (+28–34% prefill)
→ pad-16/none via b12x `mg_n_hg=1` (+6–10% more): 498→666→722 t/s at d0. The
DCP branch carries the equivalent support natively. See `CHANGES.md`.

**Historical fixed-depth result:** k=4 measured optimum on the legacy stack;
fixed k=5 regressed 14% at d0 because prose paid the wider verify cost even when
p4 acceptance was poor. That result motivated adaptation rather than ruling out
k5: the production controller only retains k5 when p4 contributes at least 0.15
accepted tokens per verification batch.

Prefill is bound by sparse-MLA attention + indexer, not MoE GEMM (an NVFP4 MoE
swap moved prefill by nothing); decode is memory-bandwidth-bound.

## Contents

- `bootstrap.sh` / `launch.sh` — legacy no-DCP bring-up and per-node launcher
- `recipes/` — serving specs, including the 290,816-context production vision profile
- `adaptive-mtp/` — exact 2/4/5 controller, build overlay, policy documentation
- `vision/` — zero-copy checkpoint assembler and vLLM GLM-5V overlay
- `patches/draft-quant-packed-mapping.patch` — required DCP-stack fix (quantized-NextN drafts)
- `patches/adaptive-mtp-vllm-hooks.patch` — scheduler/config/CUDA-graph integration
- `kernels/` — legacy-stack Triton sparse-MLA (vLLM/jasl, Apache-2.0, modified — `CHANGES.md`)
- `prune/`, `mtp/`, `model-card/` — historical pre-DCP artifacts
- `docs/retrospective.md`, `docs/licensing.md` — current history, attribution, and downstream packaging rules

## License

The repository's code, patches, recipes, container overlays, and documentation
are Apache-2.0. Derived images embed `LICENSE` and `NOTICE` and carry the OCI
license label.

Model weights are separate: Z.ai GLM-5.2, QuantTrio, cyankiwi, and Baseten
artifacts retain their MIT terms; MoonViT/Kimi components retain their
applicable upstream terms. Apache-2.0 on the tooling does not relicense model
inputs or generated weight composites. See `NOTICE` and `ATTRIBUTION.md`.

See `docs/licensing.md` for downstream packaging rules.
