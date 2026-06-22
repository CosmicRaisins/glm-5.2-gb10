#!/usr/bin/env python3
"""
Dequantize the GLM-5.2 MTP draft layer (layer 78) from modelopt NVFP4 -> BF16.

Input : /srv/hf/hub/glm52-mtp-nvfp4   (modelopt NVFP4, single MTP decoder layer)
Output: /srv/hf/hub/glm52-mtp-bf16    (same layer, expert weights dequantized to BF16,
        quantization_config removed from config.json)

Dequant math is taken verbatim from vLLM's own reference:
  vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils.dequantize_to_dtype
The on-disk modelopt block-scales are stored LINEAR (un-swizzled): weight_scale has shape
exactly [out, in/16] with no 128-row / 4-col padding, so we call swizzle=False.
On CPU (no GPU in this container) swizzle=False takes vLLM's pure-PyTorch break_fp4_bytes
path, which we have verified is elementwise-identical to the manual formula
(break_fp4 * weight_scale.float() * weight_scale_2).
"""

import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    break_fp4_bytes,
)

SRC = "/srv/hf/hub/glm52-mtp-nvfp4"
DST = "/srv/hf/hub/glm52-mtp-bf16"
WEIGHTS = os.path.join(SRC, "model-mtp.safetensors")
INPUTSCALES = os.path.join(SRC, "model-mtp-inputscales.safetensors")
OUT_WEIGHTS_NAME = "model.safetensors"
BLOCK_SIZE = 16

EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")


def is_expert_weight(key: str) -> bool:
    return ".mlp.experts." in key and key.endswith(".weight")


def is_expert_aux(key: str) -> bool:
    # auxiliary quant tensors we DROP (after using weight_scale / weight_scale_2)
    return ".mlp.experts." in key and (
        key.endswith(".weight_scale")
        or key.endswith(".weight_scale_2")
        or key.endswith(".input_scale")
    )


def main():
    os.makedirs(DST, exist_ok=True)

    # ---- discover keys in the source weights file -------------------------------
    with safe_open(WEIGHTS, framework="pt") as f:
        all_keys = list(f.keys())

    expert_weight_keys = sorted(k for k in all_keys if is_expert_weight(k))
    nonexpert_keys = sorted(
        k for k in all_keys if (not is_expert_weight(k)) and (not is_expert_aux(k))
    )
    dropped_aux = sorted(k for k in all_keys if is_expert_aux(k))

    print(f"[scan] total keys in model-mtp.safetensors : {len(all_keys)}")
    print(f"[scan] expert .weight keys to dequantize    : {len(expert_weight_keys)}")
    print(f"[scan] non-expert keys to copy as-is        : {len(nonexpert_keys)}")
    print(f"[scan] aux quant keys to DROP (scale/inputscale) in weights file: {len(dropped_aux)}")

    out_tensors = {}

    # ---- copy non-expert tensors unchanged --------------------------------------
    with safe_open(WEIGHTS, framework="pt") as f:
        for k in nonexpert_keys:
            t = f.get_tensor(k)
            # they are already bf16 (norms/attn/indexer/gate/shared_experts/eh_proj)
            out_tensors[k] = t.contiguous()

    # report dtypes seen among non-expert tensors
    nonexp_dtypes = sorted({str(out_tensors[k].dtype) for k in nonexpert_keys})
    print(f"[copy] non-expert tensor dtypes: {nonexp_dtypes}")

    # ---- dequantize expert weights ----------------------------------------------
    # verification accumulators
    sample_check = None
    n_done = 0
    with safe_open(WEIGHTS, framework="pt") as f:
        for wk in expert_weight_keys:
            base = wk[: -len(".weight")]  # ...gate_proj
            w = f.get_tensor(wk)                       # uint8 [out, in/2]
            ws = f.get_tensor(base + ".weight_scale")  # f8e4m3 [out, in/16]
            ws2 = f.get_tensor(base + ".weight_scale_2")  # f32 scalar

            assert w.dtype == torch.uint8, f"{wk} not uint8: {w.dtype}"
            assert ws.dtype == torch.float8_e4m3fn, f"{base}.weight_scale dtype {ws.dtype}"

            dq = dequantize_to_dtype(
                w, ws, ws2, torch.bfloat16, block_size=BLOCK_SIZE, swizzle=False
            ).contiguous()

            # sanity
            assert dq.dtype == torch.bfloat16
            assert dq.shape[0] == w.shape[0] and dq.shape[1] == w.shape[1] * 2
            if torch.isnan(dq).any() or torch.isinf(dq).any():
                raise RuntimeError(f"NaN/Inf in dequantized {wk}")

            out_tensors[wk] = dq

            # ---- reference-match gate on a representative sample ----------------
            if wk == "model.layers.78.mlp.experts.0.gate_proj.weight":
                nib = break_fp4_bytes(w, torch.float32)          # [out, in]
                sf = ws.to(torch.float32) * ws2.to(torch.float32)
                sf = sf.repeat_interleave(BLOCK_SIZE, dim=1)     # [out, in]
                manual = (nib * sf).to(torch.bfloat16)
                maxdiff = float((dq.float() - manual.float()).abs().max())
                sample_check = {
                    "key": wk,
                    "max_abs_diff_vs_manual_ref": maxdiff,
                    "shape": tuple(dq.shape),
                    "abs_max": float(dq.abs().max()),
                    "mean_abs": float(dq.abs().float().mean()),
                    "sample_row0_first8": dq[0, :8].float().tolist(),
                }

            n_done += 1
            if n_done % 192 == 0:
                print(f"[dequant] {n_done}/{len(expert_weight_keys)} expert weights done")

    print(f"[dequant] finished {n_done} expert weights")

    # ---- global stats over all dequantized experts ------------------------------
    g_absmax = 0.0
    any_nan = False
    for wk in expert_weight_keys:
        t = out_tensors[wk]
        g_absmax = max(g_absmax, float(t.abs().max()))
        if torch.isnan(t).any() or torch.isinf(t).any():
            any_nan = True
    print(f"[stats] global expert abs-max: {g_absmax}  any_nan_or_inf: {any_nan}")

    # ---- write the single output safetensors file -------------------------------
    out_path = os.path.join(DST, OUT_WEIGHTS_NAME)
    metadata = {"format": "pt"}
    print(f"[write] saving {len(out_tensors)} tensors -> {out_path}")
    save_file(out_tensors, out_path, metadata=metadata)

    # ---- fresh index.json -------------------------------------------------------
    total_bytes = 0
    weight_map = {}
    for k, t in out_tensors.items():
        weight_map[k] = OUT_WEIGHTS_NAME
        total_bytes += t.numel() * t.element_size()
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with open(os.path.join(DST, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    print(f"[write] index.json with {len(weight_map)} tensors, total_size={total_bytes}")

    # ---- config.json (quantization_config REMOVED) ------------------------------
    with open(os.path.join(SRC, "config.json")) as fh:
        cfg = json.load(fh)
    had_qc = "quantization_config" in cfg
    cfg.pop("quantization_config", None)
    # ensure required keys are present/correct
    cfg["architectures"] = ["GlmMoeDsaForCausalLM"]
    cfg["model_type"] = "glm_moe_dsa"
    cfg["num_experts"] = 256
    cfg["num_nextn_predict_layers"] = 1
    cfg["num_hidden_layers"] = 78
    cfg["index_topk"] = 2048
    cfg["dtype"] = "bfloat16"
    with open(os.path.join(DST, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[write] config.json  (removed quantization_config: {had_qc})")

    # ---- copy tokenizer + generation_config + notes -----------------------------
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        src = os.path.join(SRC, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DST, name))
            print(f"[copy] {name}")

    # ---- final verification block ----------------------------------------------
    print("\n========== VERIFICATION ==========")
    # 1. reference-match sample
    print("[1] reference-match (expert0 gate_proj):")
    print(json.dumps(sample_check, indent=2))

    # 2. expert weight properties
    n_expert_out = sum(1 for k in out_tensors if is_expert_weight(k))
    bf16_ok = all(out_tensors[k].dtype == torch.bfloat16 for k in out_tensors if is_expert_weight(k))
    print(f"[2] expert .weight tensors: {n_expert_out}, all BF16: {bf16_ok}, global abs-max: {g_absmax}")

    # 3. tensor counts; no scale tensors remain
    leftover_scales = [
        k for k in out_tensors
        if k.endswith(".weight_scale") or k.endswith(".weight_scale_2") or k.endswith(".input_scale")
    ]
    print(f"[3] total output tensors: {len(out_tensors)}  "
          f"(expert weights: {n_expert_out}, non-expert: {len(nonexpert_keys)})")
    print(f"    leftover scale/inputscale tensors: {len(leftover_scales)} (must be 0)")
    assert len(leftover_scales) == 0

    # 4. config has no quantization_config; index maps all tensors to existing file
    with open(os.path.join(DST, "config.json")) as fh:
        cfg_check = json.load(fh)
    assert "quantization_config" not in cfg_check
    with open(os.path.join(DST, "model.safetensors.index.json")) as fh:
        idx_check = json.load(fh)
    files_referenced = set(idx_check["weight_map"].values())
    for fn in files_referenced:
        assert os.path.exists(os.path.join(DST, fn)), f"index references missing {fn}"
    # every index entry exists in the written file
    with safe_open(out_path, framework="pt") as f:
        written_keys = set(f.keys())
    missing = [k for k in idx_check["weight_map"] if k not in written_keys]
    extra = [k for k in written_keys if k not in idx_check["weight_map"]]
    print(f"[4] config.json has quantization_config: {'quantization_config' in cfg_check}")
    print(f"    index files exist: {files_referenced}")
    print(f"    index entries not in file: {len(missing)}, file tensors not in index: {len(extra)}")
    assert not missing and not extra

    # 5. output size
    out_size = os.path.getsize(out_path)
    print(f"[5] output model.safetensors size: {out_size} bytes ({out_size/1e9:.2f} GB)")
    print("========== VERIFICATION PASSED ==========")


if __name__ == "__main__":
    main()
