# GLM-5.2 on DGX Spark (GB10, sm_121)

Serves GLM-5.2 (744B/40B MoE, `GlmMoeDsa`) on a 4-node GB10 cluster at 256k
context with MTP speculative decode. Getting it running on sm_121 meant porting
the sparse-MLA attention off the Hopper-only `_flashmla_C` path and fixing several
sm_121-specific bugs (see `docs/retrospective.md`).

The 15% expert prune is data-free and coherence-checked, **not** benchmarked.
Treat quality as unverified.

## Requirements

4× GB10 / DGX Spark (sm_121, aarch64), a node-to-node RoCE fabric, and ~400 GB of
weights on an NFS host shared read-only to all nodes. Not portable to single-GPU,
x86, or datacenter Blackwell (sm_100).

## Run

Edit the CONFIG block in `bootstrap.sh` (node IPs, NFS host, HF repo ids), then run
it from the head node. It verifies the cluster, builds the pinned vLLM image,
mounts the Triton kernels, installs NCCL 2.30.4, downloads the weights, and
launches. Serves an OpenAI-compatible API on `:8210` as `glm-5.2-15pct`.

## Contents

- `kernels/` — portable Triton sparse-MLA (vLLM/jasl, Apache-2.0, modified — `CHANGES.md`)
- `prune/awq_surgery.py` — the data-free 15% expert prune
- `mtp/` — separate-draft MTP reconstruction
- `recipes/` — the serving recipe
- `model-card/` — HuggingFace card for the pruned weights
- `docs/retrospective.md` — every fix, with attribution

## Performance

TP=4, MTP k=3, generic corpus: ~22 tok/s decode (flat to 32k), ~515 tok/s prefill
at 8k. Decode is bandwidth-bound; prefill is attention/indexer-bound, not MoE — an
NVFP4 MoE swap changed nothing.

## License

Apache-2.0 (this repo). Serves MIT weights: GLM-5.2 (Z.ai) → AWQ (cyankiwi) →
pruned here. See `NOTICE` and `ATTRIBUTION.md`.
