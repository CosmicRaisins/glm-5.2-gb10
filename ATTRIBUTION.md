# Attribution and contribution boundaries

This repository combines Apache-2.0 inference code with separately licensed
model weights. `NOTICE` is the licensing record; `CHANGES.md` lists prominent
modifications to third-party Apache source.

## Current production stack

| Area | Foundation | Work in this repository |
|---|---|---|
| Adaptive MTP | Aiden Le / aidendle94 and local-inference-lab's acceptance-length adaptive speculative decoding | Forward-port to the pinned DCP scheduler; 2/4/5 ladder; k2 floor; shortened probes; marginal p2/p3 and p4 policy; multi-depth graph coverage; JSON instrumentation |
| DCP + GLM draft path | vLLM; m9e / voipmonitor branch and PR #72 | Reproducible patch set, quantized-draft mapping fix, production recipes, validation, and GB10 packaging |
| Sparse-MLA kernels | vLLM and jasl portable sm12x kernel lineage; B12X work by lukealonso/local-inference-lab | GLM V3.2 adapter, overflow and bounds fixes, fused prefill path, head-padding follow-up, integration and benchmarking |
| Vision | vLLM Kimi multimodal implementation; Baseten GLM-5.2-Vision package; Moonshot MoonViT/Kimi lineage | QuantTrio-preserving GLM-5V wrapper, zero-copy assembler, MTP compatibility fixes, registration overlay, and production profile |
| Model backbone | Z.ai GLM-5.2; QuantTrio Int4-Int8Mix | No weight training or requantization in the current production composite |

## Adaptive MTP lineage

Aiden Le / aidendle94 and local-inference-lab supplied the adaptive
acceptance-length foundation: configuration, scheduler-side depth changes, and
support for multiple speculative query lengths. The implementation here is a
forward-port and modification, not an independent invention of adaptive depth.

The repository-specific contribution is the production policy around that
foundation:

- a discrete `2,4,5` depth ladder with k2 as the safe floor;
- a 32-step k2 baseline and 16-step exploratory k4/k5 windows;
- promotion and retreat based on unconditional marginal accepted tokens at
  p2+p3 and p4;
- CUDA-graph capture for each allowed depth;
- structured `MTP_WINDOW_JSON` telemetry using existing CPU-side accept counts.

See `adaptive-mtp/README.md` for the policy and
`patches/adaptive-mtp-vllm-hooks.patch` for the integration.

## Kernel and vLLM lineage

- **vLLM project** — `GlmMoeDsa`, sparse attention/indexer infrastructure,
  speculative decoding, Marlin, parsers, and the Apache-2.0 source base.
- **jasl** — portable sm12x sparse-MLA and DeepGEMM-fallback kernel lineage used
  by the legacy `kernels/` port.
- **m9e / voipmonitor** — DCP implementation and the GLM DCP draft path carried
  in the pinned branch and PR #72 patches.
- **Aiden Le / local-inference-lab** — B12X kernel and raw-entrypoint lineage in
  addition to the adaptive-depth foundation.
- **lukealonso / B12X** — sparse-MLA backend used by the production stack.
- **yewentao256 / vLLM PR #46862** — fused indexer Q/RoPE quantization code
  vendored in the legacy kernel tree; not authored here.
- **back199640** — the original fp8 decode head-padding reduction from 64 to 32.
  This repository's follow-up reduces 32 to 16 on the B12X path at TP4.
- **Zatz** — public 655k-context GB10 DCP demonstration reproduced here.
- **hazyumps** — NCCL 2.30.4, RDMA/`IPC_LOCK`, and bf16-indexer GB10 runbook.

## Vision and weight lineage

- **Z.ai / Zhipu AI** — GLM-5.2 architecture, weights, native MTP head, and chat
  formats. MIT.
- **QuantTrio** — the current Int4-Int8Mix text and in-checkpoint MTP tensors.
  MIT.
- **Baseten** — trained PatchMerger projector, frozen MoonViT packaging,
  configuration/processing reference, and chat template. MIT for its published
  package; redistributed parent weights retain their upstream terms.
- **Moonshot AI** — MoonViT-3d and Kimi multimodal lineage. Applicable upstream
  Modified MIT terms remain in force.
- **cyankiwi** — AWQ-INT4 parent of the historical pruned checkpoint. MIT.

The vision assembler does not train or requantize weights. It joins QuantTrio's
unchanged text shards with pinned Baseten vision/projector files and records
source revisions and hashes.

## Work authored in this repository

Subject to the third-party boundaries above, the repository-specific work is:

- the adaptive 2/4/5 policy, forward-port integration, instrumentation, and
  production validation;
- the quantized-NextN packed-module mapping fix;
- the QuantTrio GLM-5V wrapper, assembler, registration overlay, and MTP fixes;
- the legacy GLM V3.2 kernel adapter, int64 and bounds fixes, fused prefill
  kernel, and 32-to-16 head-padding follow-up;
- cluster recipes, launch integration, benchmark methodology, and documentation.

Implementation and integration were performed by CosmicRaisins with AI-assisted
development. No affiliation with the projects or authors above is implied.

## Historical artifacts

The 15% data-free expert prune and separate INT4 MTP reconstruction remain for
reproducibility but are no longer the production path. The old MTP draft used
bytes sourced through a 0xSero checkpoint whose repository declared no explicit
license; this repository does not apply Apache-2.0 to those weights. REAP was
evaluated and not used.
