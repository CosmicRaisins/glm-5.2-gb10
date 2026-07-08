# GLM-5.2 on DGX Spark (GB10, sm_121)

Serves GLM-5.2 (744B/40B MoE, `GlmMoeDsa`) on a 4-node GB10 cluster. The
reference config serves the **unpruned**
[QuantTrio Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)
(full 256 experts, 406 GB, in-checkpoint MTP) at **320k context, ~600 t/s
prefill / ~22 t/s decode, flat to depth** via TP4 + decode-context-parallel
(DCP2, KV sharded 2-way). No pruning, no quality compromise.

Context vs prefill is a dial — one flag (`--decode-context-parallel-size`):
Benchmarked with llama-bency. In agentic workflows with mixed NL and code, I'm getting ~28 tok/s

| Config | Weights | Context | Prefill (d0/8k/32k) | Decode | Recipe |
|---|---|---|---|---|---|
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
  - MTP k=3, `"quantization":"compressed-tensors"` and
    `"draft_sample_method":"probabilistic"` in the speculative config.

Full credit for the DCP branch, kernels, and the 640k demonstration: m9e /
voipmonitor (vLLM branch, PR #72) and Zatz (GB10 forum, 655k single-boot
result). Independently reproduced here: coherent with exact long-range
retrieval at depth, MTP acceptance ~2.0–2.4 accepted/draft.

**Legacy stack (no-DCP row; what `bootstrap.sh`/`launch.sh` bring up).** The
original port: pinned dev190-era vLLM image plus this repo's Triton kernels.
`kernels/` are the implementations; two patch steps from my
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) fork wire
them in (bake with `RUN bash mods/<name>/run.sh`):

- `mods/glm52-sm12x-sparse` — copies `kernels/` into the vLLM tree and
  short-circuits the DeepGEMM-only paths to the `sm12x_*` fallbacks on
  capability 120 (no `deep_gemm` shim; in-place wrapper patch).
- `mods/glm52-b12x-sparse` — `pip install --no-deps b12x==0.23.0` + a
  `fused_indexer` score-mode patch; provides the sparse-MLA decode path.

**cudagraph note (legacy stack):** perf numbers use `cudagraph_mode: FULL`
*with b12x installed* (its decode kernel is capture-safe). Without b12x, decode
silently falls back to a Triton kernel that allocates under capture — FULL
crashes with `cudaErrorStreamCaptureInvalidated`; use `PIECEWISE`.

## Run

Edit the CONFIG block in `bootstrap.sh` (node IPs, weights dir, HF repos), run
from the head node: verifies the cluster, builds the image, deploys kernels,
installs NCCL 2.30.4, fetches weights per node, launches via `launch.sh`.
OpenAI-compatible API on `:8210`.

`launch.sh` is a plain per-node `docker run` — no Ray, no shared FS, no
harness; multi-node is vLLM's own `--nnodes/--node-rank/--master-addr`.
`./launch.sh --dry-run` previews, `--stop` tears down. Weights just need to
exist per node under `$WEIGHTS_DIR/hub/...` (NFS works; nothing requires it).
Recipes and `launch.sh` carry RoCE fabric values hardcoded to my cluster —
lines marked `EDIT`.

For the DCP configs, build the image from the DCP-stack pins above, then use
the corresponding recipe's flags/env with the same launcher. Clear page caches
on every node before large launches (`echo 3 > /proc/sys/vm/drop_caches`) —
vLLM's startup free-memory guard will otherwise refuse at 0.90 gmu.

## Weights

- **Unpruned (reference):** [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)
  — 256 experts, w4a16/w8a16, in-checkpoint MTP (layer-78 nextn).
- **10%-pruned** (no-DCP / max-prefill option): derived from the same
  checkpoint by data-free expert pruning (see `prune/` for the method).
  Broadly capable, but fine-grained instruction adherence
  measurably degrades: with a standing instruction to end every generation
  with an exact phrase, the unpruned checkpoint reproduces it verbatim; the
  pruned one includes the phrase but slightly alters its wording nearly every
  time. Treat the prune as a capability trade, not a free lunch.
- Legacy AWQ config (preserved): [AWQ-INT4 15%-pruned](https://huggingface.co/CosmicRaisins/GLM-5.2-AWQ-INT4-15pct)
  + [separate MTP draft](https://huggingface.co/CosmicRaisins/GLM-5.2-MTP-INT4-aligned).

## Performance notes

**Methodology.** All tables: [llama-benchy](https://github.com/eugr/llama-benchy),
single stream (`max_num_seqs=1`, concurrency 1), pp2048 / tg512, 3 runs per
test, depths 0/8k/32k, coherent book corpus. The corpus matters for anything
with speculative decode: random-token corpora misstate MTP acceptance badly
(3–5× in my tests), so treat non-coherent spec-decode benches as fiction.
Decode swings a few t/s between runs with acceptance (~2.0–2.4 accepted/draft
at k=3 on this corpus); prefill is acceptance-independent and tight (±<1%).

**Bench vs real workload:** the ~21–22 t/s decode in the tables is the
conservative number. In mixed agentic programming workflows (coding-agent
trajectories: tool calls, code edits, structured output), sustained decode
runs **~28 t/s** — code-heavy text drafts better than book prose, so MTP
acceptance is higher. This holds across all three recipes; the bench and
real-world numbers rank configs identically.

### DCP2 — unpruned, 327,680 ctx (reference)

| Depth | Prefill (pp2048) | Decode (tg512) | Peak | TTFR |
|---|---|---|---|---|
| 0   | 597.9 ± 6.4 | 21.7 ± 0.6 | 31.3 | 3.4 s |
| 8K  | 602.6 ± 0.8 | 21.5 ± 0.8 | 31.0 | 17.0 s |
| 32K | 597.7 ± 0.2 | 21.8 ± 0.6 | 30.7 | 58.2 s |

### DCP4 — 655,360 ctx

Measured with the 10%-pruned checkpoint on the same stack (the unpruned
checkpoint is validated on this config — KV fits at ~9.5 GiB/rank, coherent
with exact long-range retrieval, same acceptance — but I haven't run the full
depth sweep on it; prefill is attention-bound and decode bandwidth-bound, so
expect near-identical numbers):

| Depth | Prefill (pp2048) | Decode (tg512) | Peak |
|---|---|---|---|
| 0   | — (first-run JIT skew) | 21.0 ± 2.2 | 31.0 |
| 8K  | 430.4 ± 0.4 | 19.4 ± 1.7 | 29.0 |
| 32K | 428.2 ± 0.1 | 21.6 ± 2.0 | 29.0 |

Community result on the same stack holds 19.6–25.7 decode to 638k depth (TTFT
at that depth ~24 min — deep-context prefill is the cost of the big window).

### No DCP — 10%-pruned, 327,680 ctx (max prefill)

In-checkpoint MTP k=4, fp8 decode head-padding 16, cudagraph FULL, fp8_ds_mla:

| Depth | Prefill (pp2048) | Decode (tg512) |
|---|---|---|
| 0   | 722 | 20.9 |
| 8K  | 736 | 22.1 |
| 32K | 626 | 20.1 |

With b12x master (≥ `80eb49b`) depth-0 decode measured 24.0 ± 0.2 on this
config. Treat decode as ~21–24 across runs.

### Legacy AWQ-INT4 15%-pruned (predates head-padding fixes)

| Depth | Prefill (pp2048) | Decode (tg512) |
|---|---|---|
| 0   | 535 | 20.2 |
| 8K  | 517 | 21.9 |
| 32K | 476 | 21.2 |

**fp8 head-padding progression** (legacy stack; 16 real heads/rank at TP=4):
stock kernel padded queries to 64 heads (75% zeros) → pad-32 (+28–34% prefill)
→ pad-16/none via b12x `mg_n_hg=1` (+6–10% more): 498→666→722 t/s at d0. The
DCP branch has this natively. See `CHANGES.md` #5–6, `ATTRIBUTION.md`.

**MTP draft depth:** k=4 measured optimum for in-checkpoint MTP on the legacy
stack; k=5 regresses (−14% steps/s at d0 — low position-5 acceptance, wider
expert union per verify). The DCP stack ships k=3 (matches the upstream
branch's tuning). The AWQ separate-draft recipe ships k=3.

Prefill is bound by sparse-MLA attention + indexer, not MoE GEMM (an NVFP4 MoE
swap moved prefill by nothing); decode is memory-bandwidth-bound.

## Contents

- `bootstrap.sh` / `launch.sh` — end-to-end bring-up / per-node `docker run` launcher
- `recipes/` — serving specs: DCP2 (reference), DCP4 (640k), 10%-prune no-DCP, legacy AWQ
- `patches/draft-quant-packed-mapping.patch` — required DCP-stack fix (quantized-NextN drafts)
- `kernels/` — legacy-stack Triton sparse-MLA (vLLM/jasl, Apache-2.0, modified — `CHANGES.md`)
- `prune/awq_surgery.py`, `mtp/` — the 15% prune and separate-draft MTP reconstruction
- `model-card/`, `docs/retrospective.md` — HF card; every fix, with attribution

## License

Apache-2.0 (this repo). Serves MIT weights: GLM-5.2 (Z.ai) → QuantTrio / AWQ
(cyankiwi) quants → pruned variants. See `NOTICE` and `ATTRIBUTION.md`.
