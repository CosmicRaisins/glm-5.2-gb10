#!/usr/bin/env python3
"""
Quantize the GLM-5.2 MTP layer (layer 78) routed-expert weights from BF16 to
compressed-tensors INT4 pack-quantized (W4A16, group_size=32, asymmetric),
matching the serve-target /srv/hf/hub/glm52-awq-15pct exactly.

Uses the compressed_tensors library's OWN reference functions so the bit packing
order is guaranteed identical to what vLLM unpacks at load:
  - compressed_tensors/quantization/utils/helpers.py :: calculate_qparams  (scale+zp)
  - compressed_tensors/quantization/lifecycle/forward.py :: quantize        (int8 quant)
  - compressed_tensors/compressors/pack_quantized/base.py :: PackedQuantizationCompressor.compress
        which internally calls helpers.py :: pack_to_int32 (weight, packed_dim=1) and
        pack_to_int32(zero_point, packed_dim=0)

Quantization = RTN (round-to-nearest), asymmetric, per-group-32, data-free.
Non-expert tensors (attn, indexer, norms, eh_proj, router gate, shared_experts,
shared_head) are copied BF16 unchanged (target ignores these module types).
"""
import json
import os
import shutil
import math
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.compressors.pack_quantized.base import (
    PackedQuantizationCompressor,
)
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization.lifecycle.forward import dequantize

SRC = "/srv/hf/hub/glm52-mtp-bf16"
TGT = "/srv/hf/hub/glm52-awq-15pct"   # serve target whose scheme we match
OUT = "/srv/hf/hub/glm52-mtp-int4"

NUM_BITS = 4
GROUP_SIZE = 32
LAYER = 78
N_EXPERTS = 256
PROJS = ("gate_proj", "up_proj", "down_proj")

os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Quantization scheme matching the target's config_groups.group_0.weights
# ---------------------------------------------------------------------------
qargs = QuantizationArgs(
    num_bits=NUM_BITS,
    type="int",
    symmetric=False,        # asymmetric -> zero_point present
    strategy="group",
    group_size=GROUP_SIZE,
    observer="mse",
    zp_dtype="torch.int8",  # matches target weights.zp_dtype
)
scheme = QuantizationScheme(
    targets=["Linear"], weights=qargs, input_activations=None
)


def compute_scale_zp(weight_bf16: torch.Tensor):
    """Per-group(32) asymmetric RTN scale+zp via the lib's calculate_qparams.

    weight: [out, in], BF16. Groups along the in (last) dim.
    Returns scale (bf16, [out, in/GS]) and zp (int8, [out, in/GS]).
    Mirrors compressed_tensors.quantization.utils.compute_dynamic_scales_and_zp
    for the GROUP strategy.
    """
    out_f, in_f = weight_bf16.shape
    assert in_f % GROUP_SIZE == 0, (out_f, in_f)
    reshaped = weight_bf16.unflatten(-1, (in_f // GROUP_SIZE, GROUP_SIZE))  # [out,g,GS]
    min_val = torch.amin(reshaped, dim=-1)   # [out, g], keep bf16 dtype
    max_val = torch.amax(reshaped, dim=-1)
    scale, zp = calculate_qparams(min_val, max_val, qargs)
    return scale, zp


def quantize_expert_proj(weight_bf16: torch.Tensor):
    """Returns dict of the 4 compressed tensors using the lib's compressor.compress."""
    scale, zp = compute_scale_zp(weight_bf16)
    sd = {
        "weight": weight_bf16,
        "weight_scale": scale,
        "weight_zero_point": zp,
    }
    comp = PackedQuantizationCompressor.compress(sd, scheme)
    # comp keys: weight_packed (i32), weight_scale (bf16), weight_shape (i64),
    #            weight_zero_point (i32 packed along dim0)
    return comp, scale, zp


def roundtrip_err(weight_bf16, comp, scale):
    """Unpack with lib unpack_from_int32 + dequantize, compare to original BF16."""
    packed = comp["weight_packed"]
    zp_packed = comp["weight_zero_point"]
    orig_shape = comp["weight_shape"]
    # unpack zero_point (packed along dim 0)
    original_zp_shape = (*tuple(orig_shape[:-1].tolist()), scale.shape[-1])
    zp = unpack_from_int32(zp_packed, NUM_BITS, original_zp_shape, packed_dim=0)
    unpacked = unpack_from_int32(packed, NUM_BITS, tuple(orig_shape.tolist()))
    w_deq = dequantize(x_q=unpacked, scale=scale, zero_point=zp).to(torch.float32)
    w32 = weight_bf16.to(torch.float32)
    denom = w32.abs().mean().clamp_min(1e-12)
    err = (w_deq - w32).abs()
    return {
        "mean_abs": err.mean().item(),
        "mean_rel": (err.mean() / denom).item(),
        "max_rel": (err.max() / denom).item(),
        "nan": bool(torch.isnan(w_deq).any()),
        "inf": bool(torch.isinf(w_deq).any()),
    }


# ---------------------------------------------------------------------------
# Main pass: stream the single source safetensors, build the output state dict.
# ---------------------------------------------------------------------------
print(f"[*] Opening source {SRC}/model.safetensors ...", flush=True)
new_state = {}
rt_samples = {}              # round-trip errors for a couple experts
n_quant = 0
n_copy = 0
any_nan = False

with safe_open(os.path.join(SRC, "model.safetensors"), framework="pt") as f:
    all_keys = list(f.keys())
    expert_keys = set()
    for e in range(N_EXPERTS):
        for proj in PROJS:
            expert_keys.add(f"model.layers.{LAYER}.mlp.experts.{e}.{proj}.weight")

    # 1) copy all non-expert tensors unchanged (BF16)
    for k in all_keys:
        if k in expert_keys:
            continue
        t = f.get_tensor(k)
        new_state[k] = t.contiguous()
        if torch.isnan(t).any() or torch.isinf(t).any():
            any_nan = True
            print(f"    !! NaN/Inf in copied tensor {k}", flush=True)
        n_copy += 1
    print(f"[*] Copied {n_copy} non-expert BF16 tensors.", flush=True)

    # 2) quantize each routed expert proj
    for e in range(N_EXPERTS):
        for proj in PROJS:
            base = f"model.layers.{LAYER}.mlp.experts.{e}.{proj}"
            w = f.get_tensor(base + ".weight")
            assert w.dtype == torch.bfloat16, (base, w.dtype)
            comp, scale, zp = quantize_expert_proj(w)
            new_state[base + ".weight_packed"] = comp["weight_packed"].contiguous()
            new_state[base + ".weight_scale"] = comp["weight_scale"].to(
                torch.bfloat16).contiguous()
            new_state[base + ".weight_shape"] = comp["weight_shape"].contiguous()
            new_state[base + ".weight_zero_point"] = comp[
                "weight_zero_point"].contiguous()
            for tt in (comp["weight_packed"], comp["weight_scale"]):
                if torch.isnan(tt.float()).any() or torch.isinf(tt.float()).any():
                    any_nan = True
            n_quant += 1
            # sample round-trip for experts 0 and 128 (all 3 projs)
            if e in (0, 128):
                rt_samples[f"experts.{e}.{proj}"] = roundtrip_err(w, comp, scale)
        if e % 64 == 0:
            print(f"    quantized expert {e}/{N_EXPERTS}", flush=True)

print(f"[*] Quantized {n_quant} expert projections "
      f"(expect {N_EXPERTS*len(PROJS)}).", flush=True)

# ---------------------------------------------------------------------------
# Write model.safetensors (single shard) + index.json
# ---------------------------------------------------------------------------
out_st = os.path.join(OUT, "model.safetensors")
print(f"[*] Saving {len(new_state)} tensors -> {out_st}", flush=True)
save_file(new_state, out_st, metadata={"format": "pt"})

total_bytes = sum(t.numel() * t.element_size() for t in new_state.values())
weight_map = {k: "model.safetensors" for k in new_state.keys()}
index = {
    "metadata": {"total_size": total_bytes},
    "weight_map": weight_map,
}
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as fh:
    json.dump(index, fh, indent=2)

# ---------------------------------------------------------------------------
# config.json: input arch config + target's quantization_config (ignore adapted)
# ---------------------------------------------------------------------------
with open(os.path.join(SRC, "config.json")) as fh:
    cfg = json.load(fh)
with open(os.path.join(TGT, "config.json")) as fh:
    tgt_cfg = json.load(fh)

qc = json.loads(json.dumps(tgt_cfg["quantization_config"]))  # deep copy verbatim

# Adapt the ignore list for layer 78 (MTP). The target ignores, per non-expert
# module type, the same modules that exist in the MTP layer. We replicate the
# SAME patterns referencing layer 78 explicitly (these modules ARE present in
# layer 78 and must NOT be quantized; only mlp.experts.* are quantized).
ignore_78 = [
    f"model.layers.{LAYER}.self_attn.q_a_proj",
    f"model.layers.{LAYER}.self_attn.q_b_proj",
    f"model.layers.{LAYER}.self_attn.kv_a_proj_with_mqa",
    f"model.layers.{LAYER}.self_attn.kv_b_proj",
    f"model.layers.{LAYER}.self_attn.o_proj",
    f"model.layers.{LAYER}.self_attn.indexer.wq_b",
    f"model.layers.{LAYER}.self_attn.indexer.wk",
    f"model.layers.{LAYER}.self_attn.indexer.weights_proj",
    # NOTE: router mlp.gate is intentionally NOT listed — like the target, it is a
    # Gate module (not nn.Linear), so it never matches the "Linear" target and is
    # kept BF16 without needing an ignore entry.
    f"model.layers.{LAYER}.mlp.shared_experts.gate_proj",
    f"model.layers.{LAYER}.mlp.shared_experts.up_proj",
    f"model.layers.{LAYER}.mlp.shared_experts.down_proj",
    f"model.layers.{LAYER}.eh_proj",             # Linear projecting concat -> hidden
    "lm_head",
]
qc["ignore"] = ignore_78
cfg["quantization_config"] = qc

# ensure required arch fields are present/correct (carry from source; assert)
cfg["architectures"] = ["GlmMoeDsaForCausalLM"]
cfg["model_type"] = "glm_moe_dsa"
cfg["num_experts"] = 256
cfg["n_routed_experts"] = 256
cfg["num_nextn_predict_layers"] = 1
cfg["num_hidden_layers"] = 78
cfg["index_topk"] = 2048

with open(os.path.join(OUT, "config.json"), "w") as fh:
    json.dump(cfg, fh, indent=2)

# ---------------------------------------------------------------------------
# tokenizer + generation_config + chat_template (from target serve dir)
# ---------------------------------------------------------------------------
for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
           "chat_template.jinja"):
    src_fn = os.path.join(TGT, fn)
    if os.path.exists(src_fn):
        shutil.copy2(src_fn, os.path.join(OUT, fn))

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n================ REPORT ================")
print(f"output dir: {OUT}")
print(f"total tensors written: {len(new_state)}")
print(f"  non-expert (BF16 copied): {n_copy}")
print(f"  expert projections quantized: {n_quant}  -> {n_quant*4} expert tensors")
print(f"total_size bytes: {total_bytes}  ({total_bytes/1e9:.2f} GB)")
print(f"any NaN/Inf encountered: {any_nan}")
print("\nRound-trip errors (lib unpack_from_int32 + dequantize vs original BF16):")
for k, v in rt_samples.items():
    print(f"  {k:24s} mean_rel={v['mean_rel']*100:6.3f}%  "
          f"max_rel={v['max_rel']*100:7.3f}%  mean_abs={v['mean_abs']:.6e}  "
          f"nan={v['nan']} inf={v['inf']}")
print("========================================\n")
