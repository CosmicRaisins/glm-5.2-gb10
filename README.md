# GLM-5.2 on DGX Spark (GB10, sm_121)

Serves GLM-5.2 (744B/40B MoE, `GlmMoeDsa`) on a 4-node GB10 cluster with MTP
speculative decode. The reference config serves
[QuantTrio Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)
(10%-pruned, in-checkpoint MTP k=4) at **327k context, ~720 t/s prefill / ~21–24 t/s
decode** — see Performance. The original AWQ-INT4 15%-pruned setup is retained and
still what `bootstrap.sh`/`launch.sh` bring up by default. Getting any of it running
on sm_121 meant porting the sparse-MLA attention off the Hopper-only `_flashmla_C`
path and fixing several sm_121-specific bugs (see `docs/retrospective.md`).

Both expert prunes (10% QuantTrio, 15% AWQ) are data-free and coherence-checked,
**not** benchmarked for quality. Treat quality as unverified.

## Requirements

4× GB10 / DGX Spark (sm_121, aarch64), a node-to-node RoCE fabric, and ~400 GB of
weights reachable from every node. Not portable to single-GPU, x86, or datacenter
Blackwell (sm_100).

## Run

Edit the CONFIG block in `bootstrap.sh` (node IPs, weights location, HF repo ids),
then run it from the head node. It verifies the cluster, builds the pinned vLLM
image, deploys the Triton kernels, installs NCCL 2.30.4, fetches the weights to
every node, and launches via `launch.sh`. Serves an OpenAI-compatible API on
`:8210` as `glm-5.2-15pct`.

**Launch is a plain `docker run`.** `launch.sh` starts the container on each node
directly — no Ray, no shared filesystem, no external harness. Multi-node is vLLM's
own mechanism (`--nnodes/--node-rank/--master-addr/--master-port`): workers come up
headless, then the head serves the API. You can run it on its own once the image,
kernels, and weights are in place; edit its CONFIG block (it mirrors `bootstrap.sh`)
and preview the exact commands with `./launch.sh --dry-run`. Stop with `./launch.sh --stop`.

Weights just need to be present on each node under the configured directory
(`$WEIGHTS_DIR/hub/...`). `bootstrap.sh` fetches a per-node copy; if you have a
shared mount, point `WEIGHTS_DIR` at it instead. Nothing requires NFS.

The image build is not self-contained — it requires two `spark-vllm-docker` mods
that are not vendored here. See **Image build** below before running `bootstrap.sh`.

`launch.sh` (and the reference `recipes/glm52-awq-15pct-prod.yaml`) carry RoCE
fabric values (HCA + interface names) hardcoded to my cluster — set those for
yours. The lines are marked `EDIT`.

## Image build — required vLLM mods (not vendored)

The `kernels/` here are the *implementations*. They do not wire themselves into
vLLM: two patch steps that live in my
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) fork are
required and are **not** vendored in this repo. Build the image at the pinned
ref, then apply both mods (bake them with `RUN bash mods/<name>/run.sh`, as
`Dockerfile.glm52-consolidated` does, or pass `--apply-mod mods/<name>` to
`launch-cluster.sh`):

- **`mods/glm52-sm12x-sparse`** — copies the `kernels/` files into the vLLM tree
  and patches `vllm/utils/deep_gemm.py` + `sparse_attn_indexer.py` in place. On
  capability family 120 it short-circuits `fp8_fp4_mqa_logits` /
  `fp8_fp4_paged_mqa_logits` / `tf32_hc_prenorm_gemm` to the `sm12x_*` fallbacks
  **before** the DeepGEMM `_missing()` gate, and rewrites the `SparseAttnIndexer`
  constructor gate so sm_121 never requires `has_deep_gemm()`. There is **no
  `deep_gemm` package shim** — the activation is this in-place wrapper patch.
- **`mods/glm52-b12x-sparse`** — `pip install --no-deps b12x==0.23.0` plus a
  `fused_indexer` score-mode patch. This provides the `b12x` package that the
  sparse-MLA **decode** path (`b12x_sparse_helpers.py`) calls first.

**cudagraph note (important):** my perf numbers use `cudagraph_mode: FULL` *with
b12x installed*. The b12x decode kernel is cudagraph-capture-safe. If b12x is
absent, `_fp8_flash_mla_kernel` silently falls back to the Triton
`flash_mla_with_kvcache` decode kernel, which does unconditional
`torch.full(..., device=...)` allocations that are illegal under graph capture —
so FULL capture crashes with `cudaErrorStreamCaptureInvalidated`. Without b12x,
run `cudagraph_mode: PIECEWISE`. (The `b12x` import warning is `warning_once`, so
it is emitted once during prefill warmup and then suppressed — decode falls back
silently.)

`build-glm52-awq.sh` builds the base image at the pinned ref;
`Dockerfile.glm52-consolidated` layers `glm52-sm12x-sparse` on top. Apply
`glm52-b12x-sparse` the same way for the `-b12x` image the recipe expects.

## Weights

**Primary — [QuantTrio Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix), 10%-pruned.**
A lighter 10% expert prune in QuantTrio's mixed Int4/Int8 format fits **~340k of KV
(327k served context)** on the same 4× GB10 — a *lighter* prune than the AWQ-INT4
below (15%) yet *more* context than its ~256k, with INT8 attention. MTP is
in-checkpoint (layer-78 nextn weights, k=4) — no separate draft model. The kernels
and launch path are weight-agnostic: point `WEIGHTS_DIR` at the checkpoint and set
`--max-model-len` accordingly, or use the flags in
`recipes/glm52-quanttrio-10pct-prod.yaml`.

Original AWQ config (preserved; `bootstrap.sh` pulls both of these):

- AWQ-INT4, 15%-pruned: https://huggingface.co/CosmicRaisins/GLM-5.2-AWQ-INT4-15pct
- MTP draft: https://huggingface.co/CosmicRaisins/GLM-5.2-MTP-INT4-aligned

## Contents

- `bootstrap.sh` — end-to-end bring-up (build → kernels → NCCL → weights → launch)
- `launch.sh` — self-contained per-node `docker run` launcher (no Ray, no shared FS)
- `kernels/` — portable Triton sparse-MLA (vLLM/jasl, Apache-2.0, modified — `CHANGES.md`)
- `prune/awq_surgery.py` — the data-free 15% expert prune
- `mtp/` — separate-draft MTP reconstruction
- `recipes/` — the serving spec (flags + env; mirrored by `launch.sh`)
- `model-card/` — HuggingFace card for the pruned weights
- `docs/retrospective.md` — every fix, with attribution

## Performance

**Reference config — QuantTrio Int4-Int8Mix (10%-pruned), in-checkpoint MTP k=4,
fp8 decode head-padding 16, cudagraph FULL, kv fp8_ds_mla** (llama-benchy, tg=512,
ctx 327680):

| Depth | Decode (tg) | Prefill (pp) |
|---|---|---|
| 0   | 20.9 t/s | 722 t/s |
| 8K  | 22.1 t/s | 736 t/s |
| 32K | 20.1 t/s | 626 t/s |

Decode on this model is dominated by MTP acceptance, which swings a few t/s
between bench runs on the sampled corpus — treat decode as ~21–24 t/s; the
padding change is decode-neutral once normalized for acceptance (prefill is
acceptance-independent). With b12x master (≥ `80eb49b`, GLM sparse-decode
optimizations; the mods here pin `b12x==0.23.0`) depth-0 decode measured
24.0 ± 0.2 t/s on this config.

**fp8 head-padding progression** (same config throughout; 16 real heads/rank at
TP=4): the stock kernel padded queries to 64 heads — 75% zeros. Padding to 32
(back199640, GB10 forum) lifted prefill +28–34%; padding to 16 — i.e. not at all,
via the b12x `mg_n_hg=1` path — adds another +6–10%. Prefill t/s by pad width:

| Depth | pad 64 | pad 32 | pad 16 |
|---|---|---|---|
| 0   | 498 | 666 | **722** |
| 8K  | 509 | 671 | **736** |
| 32K | 461 | 592 | **626** |

See `CHANGES.md` #5–6 and `ATTRIBUTION.md`.

**Original AWQ-INT4 15%-pruned config** (TP=4, separate MTP draft k=3,
llama-benchy generic corpus; predates the head-padding fixes):

| Depth | Decode (tg) | Prefill (pp) |
|---|---|---|
| 0   | 20.2 t/s | 535 t/s |
| 8K  | 21.9 t/s | 517 t/s |
| 32K | 21.2 t/s | 476 t/s |

The upstream #46862 fused indexer (vendored) left prefill flat and made decode
marginally faster (~0–1 t/s) vs the non-fused path — single config,
cross-harness, unverified.

Numbers will vary with hardware and workload. In my tests decode is
memory-bandwidth-bound and prefill is bound by the sparse-MLA / indexer kernels
rather than the MoE GEMM (an NVFP4 MoE swap changed prefill by nothing) — but I
haven't stress-tested that conclusion broadly.

**MTP draft depth (k):** on the QuantTrio in-checkpoint MTP config, k=4 is the
measured optimum. k=5 (Z.ai's recommendation) benchmarked as a net regression
here: position-5 acceptance is low and each extra position costs a sequential
drafter pass plus a wider MoE expert union per verify step on a
bandwidth-bound decode (−14% engine steps/s at depth 0). The AWQ config's
separate-draft recipe ships k=3, which benchmarked best for that setup.

## License

Apache-2.0 (this repo). Serves MIT weights: GLM-5.2 (Z.ai) → AWQ (cyankiwi) →
pruned here. See `NOTICE` and `ATTRIBUTION.md`.
