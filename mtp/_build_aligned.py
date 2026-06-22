#!/usr/bin/env python3
"""
Transform /srv/hf/hub/glm52-mtp-int4-218 -> /srv/hf/hub/glm52-mtp-int4-aligned

Quantize (compressed-tensors INT4 pack, group32, asymmetric) exactly the layer-78
modules the MTP block (built with the TARGET's quant_config) expects quantized:
    self_attn.q_a_proj, kv_a_proj_with_mqa   (stored separate; loader fuses ->fused_qkv_a_proj)
    self_attn.q_b_proj, kv_b_proj, o_proj
    self_attn.indexer.wq_b
    mlp.shared_experts.gate_proj, up_proj     (stored separate; loader fuses ->gate_up_proj)
    mlp.shared_experts.down_proj
Keep BF16 (already correct in source; passthrough):
    all norms, eh_proj, enorm, hnorm, mlp.gate(+e_score_correction_bias),
    indexer.k_norm, indexer.wk, indexer.weights_proj (loader fuses ->wk_weights_proj bf16),
    q_a_layernorm, kv_a_layernorm, shared_head.norm
Routed experts (mlp.experts.*.{gate,up,down}_proj) already INT4 -> passthrough.
Embedding / shared_head.head -> passthrough (runtime-shared, harmless to keep).

Uses the SAME compressed_tensors lib primitives as /srv/hf/glm52_quant_mtp_int4.py.
"""
import json, os, shutil, torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_scheme import QuantizationScheme
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization.lifecycle.forward import dequantize

SRC = "/srv/hf/hub/glm52-mtp-int4-218"
TGT = "/srv/hf/hub/glm52-awq-15pct"
OUT = "/srv/hf/hub/glm52-mtp-int4-aligned"
LAYER = 78
NUM_BITS, GROUP_SIZE = 4, 32
os.makedirs(OUT, exist_ok=True)

qargs = QuantizationArgs(num_bits=NUM_BITS, type="int", symmetric=False,
    strategy="group", group_size=GROUP_SIZE, observer="mse", zp_dtype="torch.int8")
scheme = QuantizationScheme(targets=["Linear"], weights=qargs, input_activations=None)

def compute_scale_zp(w):
    out_f, in_f = w.shape
    assert in_f % GROUP_SIZE == 0, (out_f, in_f)
    r = w.unflatten(-1, (in_f // GROUP_SIZE, GROUP_SIZE))
    return calculate_qparams(torch.amin(r, dim=-1), torch.amax(r, dim=-1), qargs)

def quantize(w):
    scale, zp = compute_scale_zp(w)
    comp = PackedQuantizationCompressor.compress(
        {"weight": w, "weight_scale": scale, "weight_zero_point": zp}, scheme)
    return comp, scale, zp

def roundtrip_err(w, comp, scale):
    packed = comp["weight_packed"]; zp_packed = comp["weight_zero_point"]; orig_shape = comp["weight_shape"]
    zp_shape = (*tuple(orig_shape[:-1].tolist()), scale.shape[-1])
    zp = unpack_from_int32(zp_packed, NUM_BITS, zp_shape, packed_dim=0)
    unpacked = unpack_from_int32(packed, NUM_BITS, tuple(orig_shape.tolist()))
    w_deq = dequantize(x_q=unpacked, scale=scale, zero_point=zp).to(torch.float32)
    w32 = w.to(torch.float32); denom = w32.abs().mean().clamp_min(1e-12)
    err = (w_deq - w32).abs()
    return {"mean_rel": (err.mean()/denom).item(), "max_rel": (err.max()/denom).item(),
            "nan": bool(torch.isnan(w_deq).any()), "inf": bool(torch.isinf(w_deq).any())}

# Modules in layer 78 to quantize (stored as separate shards; suffix .weight in src)
QUANT_MODULES = [
    f"model.layers.{LAYER}.self_attn.q_a_proj",
    f"model.layers.{LAYER}.self_attn.kv_a_proj_with_mqa",
    f"model.layers.{LAYER}.self_attn.q_b_proj",
    f"model.layers.{LAYER}.self_attn.kv_b_proj",
    f"model.layers.{LAYER}.self_attn.o_proj",
    f"model.layers.{LAYER}.self_attn.indexer.wq_b",
    f"model.layers.{LAYER}.mlp.shared_experts.gate_proj",
    f"model.layers.{LAYER}.mlp.shared_experts.up_proj",
    f"model.layers.{LAYER}.mlp.shared_experts.down_proj",
]
quant_weight_keys = {m + ".weight" for m in QUANT_MODULES}

print(f"[*] Opening {SRC}/model.safetensors", flush=True)
new_state = {}; n_quant=0; n_copy=0; any_nan=False; rt={}
with safe_open(os.path.join(SRC, "model.safetensors"), framework="pt") as f:
    keys = list(f.keys())
    # sanity: every quant module must be present as BF16 .weight in source
    for k in quant_weight_keys:
        assert k in keys, f"missing source tensor {k}"
    for k in keys:
        if k in quant_weight_keys:
            w = f.get_tensor(k)
            assert w.dtype == torch.bfloat16, (k, w.dtype)
            base = k[:-len(".weight")]
            comp, scale, zp = quantize(w)
            new_state[base + ".weight_packed"]      = comp["weight_packed"].contiguous()
            new_state[base + ".weight_scale"]       = comp["weight_scale"].to(torch.bfloat16).contiguous()
            new_state[base + ".weight_shape"]       = comp["weight_shape"].contiguous()
            new_state[base + ".weight_zero_point"]  = comp["weight_zero_point"].contiguous()
            n_quant += 1
            rt[base.replace(f"model.layers.{LAYER}.","")] = roundtrip_err(w, comp, scale)
            for tt in (comp["weight_packed"], comp["weight_scale"]):
                if torch.isnan(tt.float()).any() or torch.isinf(tt.float()).any(): any_nan=True
        else:
            t = f.get_tensor(k).contiguous()
            new_state[k] = t
            if t.is_floating_point() and (torch.isnan(t).any() or torch.isinf(t).any()):
                any_nan=True; print("   !! NaN/Inf in", k)
            n_copy += 1

print(f"[*] quantized {n_quant} modules (expect {len(QUANT_MODULES)}); copied {n_copy} tensors", flush=True)

out_st = os.path.join(OUT, "model.safetensors")
print(f"[*] saving {len(new_state)} tensors -> {out_st}", flush=True)
save_file(new_state, out_st, metadata={"format": "pt"})
total = sum(t.numel()*t.element_size() for t in new_state.values())
json.dump({"metadata":{"total_size":total},"weight_map":{k:"model.safetensors" for k in new_state}},
          open(os.path.join(OUT,"model.safetensors.index.json"),"w"), indent=2)

# config.json: start from SRC config, fix quantization_config ignore so the
# DRAFT's own arch config is self-consistent (informational; the runtime MTP block
# is driven by the TARGET quant_config, but keep this faithful).
cfg = json.load(open(os.path.join(SRC,"config.json")))
tgt_cfg = json.load(open(os.path.join(TGT,"config.json")))
qc = json.loads(json.dumps(tgt_cfg["quantization_config"]))
# layer-78 ignore = modules we kept BF16 (norms are not Linear; list the BF16 Linears
# + indexer fused parts + eh_proj + shared_head.head/lm_head). q_a/kv_a are NOW quant
# (fused into fused_qkv_a_proj) so they are NOT ignored.
qc["ignore"] = [
    f"model.layers.{LAYER}.self_attn.indexer.wk",
    f"model.layers.{LAYER}.self_attn.indexer.weights_proj",
    f"model.layers.{LAYER}.eh_proj",
    f"model.layers.{LAYER}.mlp.gate",
    "lm_head",
    "model.layers.78.shared_head.head",
]
cfg["quantization_config"] = qc
cfg["architectures"] = ["GlmMoeDsaForCausalLM"]
cfg["model_type"] = "glm_moe_dsa"
json.dump(cfg, open(os.path.join(OUT,"config.json"),"w"), indent=2)

for fn in ("tokenizer.json","tokenizer_config.json","generation_config.json","chat_template.jinja"):
    s=os.path.join(SRC,fn)
    if not os.path.exists(s): s=os.path.join(TGT,fn)
    if os.path.exists(s): shutil.copy2(s, os.path.join(OUT,fn))

print("\n========== BUILD REPORT ==========")
print(f"out: {OUT}")
print(f"tensors written: {len(new_state)}  total_size: {total/1e9:.3f} GB")
print(f"quantized modules: {n_quant}   copied: {n_copy}   any_nan/inf: {any_nan}")
print("round-trip rel-err for newly-quantized attention/shared-expert modules:")
for k,v in rt.items():
    print(f"   {k:42s} mean_rel={v['mean_rel']*100:6.3f}%  max_rel={v['max_rel']*100:8.3f}%  nan={v['nan']} inf={v['inf']}")
print("==================================")
