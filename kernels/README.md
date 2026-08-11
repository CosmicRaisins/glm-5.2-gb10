# sm12x sparse-MLA kernel port

This directory is the legacy no-DCP GLM-5.2 path for NVIDIA GB10 (`sm_121`).
The current DCP production image uses its branch-native B12X backend, but this
tree retains the portable implementation and the GB10 fixes developed during
the original port.

## Lineage

The source derives from Apache-2.0 vLLM and jasl's portable sm12x sparse-MLA and
DeepGEMM-fallback work. Original vLLM SPDX and copyright headers are retained.
B12X integration is credited to lukealonso/local-inference-lab. The 64-to-32
fp8 head-padding fix is credited to back199640.

`deepseek_v2.py` and `sparse_attn_indexer.py` also carry vLLM PR #46862 by
yewentao256. Those files are vendored upstream work, not original contributions
of this repository.

## Repository modifications

- Adapt the portable wrappers to GLM's V3.2 `FlashMLASparseImpl` interfaces and
  route sm12x away from unavailable native `_flashmla_C` kernels.
- Promote gathered-KV offset arithmetic to int64 to prevent overflow beyond
  2,048-token prefill chunks at TP4.
- Add upper and lower bounds checks to sparse KV gathers.
- Fuse gather, fp8 dequantization, and sparse attention so prefill does not
  materialize the full `[T, K, 576]` tensor.
- Reduce B12X decode-head padding from 32 to 16 at TP4, following back199640's
  original 64-to-32 finding.

The measured legacy progression and exact implementation changes are recorded
in `../CHANGES.md`. The current architecture and authorship boundaries are in
`../ATTRIBUTION.md`.

## License

Apache-2.0. When this directory is copied into another source tree or image,
include the repository `LICENSE`, `NOTICE`, and `CHANGES.md`, and retain the
per-file SPDX/copyright notices.
