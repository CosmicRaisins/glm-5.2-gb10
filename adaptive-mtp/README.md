# Adaptive MTP 2/4/5

This overlay forward-ports the acceptance-length adaptive speculative-decoding
controller used by [Aiden Le / aidendle94](https://huggingface.co/aidendle94)
onto the pinned `e232d26` DCP vLLM stack, then applies the production policy we
tuned for the QuantTrio GLM-5.2 checkpoint.

The Aiden-derived foundation adds the vLLM config knob, scheduler-side depth
updates, and CUDA-graph coverage for multiple speculative query lengths. Our
changes replace its general `floor(mean accepted + 1.5)` ratchet with a cheap,
workload-sensitive `2 -> 4 -> 5` ladder.

## What we changed

- **Safe floor:** start at k=2 and never descend below it. Prose normally stays
  here, so low acceptance at p2+ cannot tax every verification step.
- **Discrete depths:** `VLLM_ADAPTIVE_SPEC_DEPTHS=2,4,5` avoids k=3. The
  configured `num_speculative_tokens=5` remains the hard upper bound.
- **Greedy k4 probe:** after 32 k2 scheduler steps, probe k4 when accepted draft
  tokens / attempted draft tokens is at least `0.85`. This is the only head
  acceptance gate; p0/p1 are not gated again once k=2 has justified a probe.
- **Marginal tail decisions:** k4 and k5 are judged by tokens actually earned
  from the extra positions, not by a second whole-prefix acceptance ratio.
- **Fast retreat:** exploratory k4/k5 windows are 16 steps, half the 32-step
  baseline window. A prose/code phase change therefore stops paying for an
  inherited high k quickly.
- **No added model work:** the controller uses CPU-side draft/accept counts the
  scheduler already has after verification. It adds counters and arithmetic,
  not another forward pass or GPU synchronization.
- **Production telemetry:** optional `MTP_WINDOW_JSON` records active/next k,
  decision reason, attempted and accepted drafts, context span, conditional
  per-position acceptance, `tail_gain_23`, and `position_4_gain`.

Positions are zero-indexed below (`p0` is the first speculative token):

| Current k | Observation | Decision |
|---|---|---|
| 2 | head ratio `>= 0.85` over 32 steps | probe k4 |
| 2 | head ratio `< 0.85` | stay k2 |
| 4 | `(accepted p2 + accepted p3) / batches >= 0.70` over 16 steps | probe k5 |
| 4 | tail gain `>= 0.35` but `< 0.70` | stay k4 |
| 4 | tail gain `< 0.35` | fall to k2 |
| 5 | `accepted p4 / batches >= 0.15` over 16 steps | stay k5 |
| 5 | p4 gain `< 0.15`, tail gain `>= 0.35` | fall to k4 |
| 5 | p4 gain `< 0.15`, tail gain `< 0.35` | fall to k2 |

`tail_gain_23` and `position_4_gain` are unconditional marginal tokens per
verification batch. That is deliberate: conditional p2/p3 percentages can look
healthy on the small subset that reaches them while still failing to repay the
cost of drafting them.

The depth is currently batch-wide, not per request. The production recipe uses
`max_num_seqs=1`, so this is equivalent to per-session adaptation for the
single-stream workload it was tuned on.

## Files and build

- `overlay/.../acceptance_length.py` is the exact production controller.
- `../patches/adaptive-mtp-vllm-hooks.patch` adds the config, scheduler,
  instrumentation, and multi-depth CUDA-graph hooks to the pinned vLLM tree.
- `Dockerfile` applies both over the DCP base that already contains the three
  prerequisite patches listed in the root README.

Build from the repository root:

```bash
docker build -f adaptive-mtp/Dockerfile \
  --build-arg BASE_IMAGE=vllm-node-tf5-eldritch-dcp:20260705-e232d26-mtpfix-pr72 \
  -t glm52-adaptive-mtp:k245 .
```

Enable it with both pieces present in the recipe:

```yaml
env:
  VLLM_ADAPTIVE_SPEC_DEPTHS: "2,4,5"
  VLLM_MTP_INSTRUMENT: "1"
  VLLM_MTP_INSTRUMENT_WINDOW: "32"
```

```text
"num_speculative_tokens": 5,
"adaptive_speculative_tokens_window": 32
```

The unified patch is pinned to the `e232d26` package layout and is expected to
fail closed if its anchors drift. Rebase it rather than forcing it onto a newer
vLLM scheduler.
