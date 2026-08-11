# Downstream licensing

The repository contains code and recipes under Apache-2.0, but it also operates
on model weights with separate licenses. Keep those scopes distinct.

| Downstream | License handling |
|---|---|
| Source fork, patch bundle, recipe bundle | Include `LICENSE`, `NOTICE`, and `CHANGES.md`; retain file-level SPDX and copyright notices. |
| Adaptive-MTP or vision container image | Apache-2.0 for repository code. Both Dockerfiles embed `LICENSE` and `NOTICE` and set `org.opencontainers.image.licenses=Apache-2.0`. |
| QuantTrio vision composite | The assembler is Apache-2.0; QuantTrio, Baseten, and Moonshot weights retain their upstream terms. Copy their notices with redistributed weights. |
| Legacy pruned AWQ model | The prune tooling is Apache-2.0; the derived weights retain Z.ai/cyankiwi MIT notices. |
| Legacy standalone MTP draft | Do not label the weights Apache-2.0. Its historical intermediate 0xSero source declared no explicit repository license. |

Apache-2.0 on tooling does not relicense model inputs or outputs. `NOTICE`
contains the authoritative upstream list; `ATTRIBUTION.md` states contribution
boundaries.
