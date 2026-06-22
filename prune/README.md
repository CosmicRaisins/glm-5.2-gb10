# prune/ — data-free expert prune

`awq_surgery.py` turns `cyankiwi/GLM-5.2-AWQ-INT4` (256 experts/layer) into the
218-expert build this stack serves, **without any calibration data**.

## Method

For each MoE layer it reads the learned `mlp.gate.e_score_correction_bias` vector
and **drops the highest-bias experts** (the ones the router had to artificially
boost = least intrinsically favored), keeping the lowest-bias survivors. Survivors
are re-indexed `0..N-1`; the router `gate.weight` and `e_score_correction_bias`
are row-sliced; shared experts, dense layers, attention, and norms are untouched.
`num_experts` and `n_routed_experts` both become `N` (uniform). Weight-only — it
operates directly on the safetensors shards, no model load, no forward passes.

## Usage

```bash
# Prune 15% (256 -> 218), reading from a downloaded cyankiwi AWQ snapshot:
python3 awq_surgery.py build <src_snapshot_dir> <out_dir> 0.15

# Or drop a fixed explicit set of expert indices instead of a ratio (see source).
```

The build is deterministic and reproducible from the same source snapshot.

> ⚠️ Quality is coherence-checked, not benchmarked — evaluate before production
> use. This is **not** REAP (which needs calibration data and was found infeasible
> on this hardware); see `../docs/retrospective.md`.
