#!/usr/bin/env python3
import json, os, sys, re
import torch

DRAFT = sys.argv[1] if len(sys.argv) > 1 else "/srv/hf/hub/glm52-mtp-int4-218"
TARGET = "/srv/hf/hub/glm52-awq-15pct"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
torch.set_default_dtype(torch.bfloat16)  # so MLA backend selector sees bf16

import vllm
# --- monkeypatch attention backend selection to a stub MLA backend ---
import vllm.v1.attention.selector as _sel
class _StubMLABackend:
    @staticmethod
    def is_mla(): return True
    @staticmethod
    def get_name(): return "FLASHMLA_SPARSE"
    @staticmethod
    def get_supported_kernel_block_size(): return [64]
    @staticmethod
    def get_builder_cls(): return None
    @staticmethod
    def get_impl_cls(): 
        class _Impl: 
            def __init__(self,*a,**k): pass
        return _Impl
    @staticmethod
    def get_metadata_cls():
        class _M: pass
        return _M
def _stub_get_attn_backend(*a, **k): return _StubMLABackend
_sel.get_attn_backend = _stub_get_attn_backend
import vllm.model_executor.layers.attention.mla_attention as _mla
_mla.get_attn_backend = _stub_get_attn_backend
# also patch where DeepseekV2 imports it if direct
try:
    import vllm.attention as _va
    _va.get_attn_backend = _stub_get_attn_backend
except Exception: pass

from vllm.engine.arg_utils import EngineArgs
from vllm.config import set_current_vllm_config

print("[*] Building VllmConfig (target + mtp draft) ...", flush=True)
ea = EngineArgs(model=TARGET, tokenizer=TARGET, trust_remote_code=True, dtype="bfloat16",
                max_model_len=4096, enforce_eager=True, skip_tokenizer_init=True,
                speculative_config={"model": DRAFT, "method": "mtp", "num_speculative_tokens": 5})
vconfig = ea.create_engine_config()
print("[*] VllmConfig built. spec method:", vconfig.speculative_config.method, flush=True)

from vllm.distributed import init_distributed_environment, initialize_model_parallel
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29562")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1"); os.environ.setdefault("LOCAL_RANK","0")
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP

with set_current_vllm_config(vconfig):
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="gloo")
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    print("[*] Instantiating DeepSeekMTP on meta device ...", flush=True)
    with torch.device("meta"):
        model = DeepSeekMTP(vllm_config=vconfig, prefix="")

model_params = set(dict(model.named_parameters()).keys())
print(f"[*] model registered params: {len(model_params)}", flush=True)

def cat(name): return name.startswith("model.layers.78.")
l78 = sorted(p for p in model_params if cat(p))
def show(pred, label, limit=80):
    sel = sorted(p for p in l78 if pred(p))
    print(f"--- {label} ({len(sel)}) ---")
    for p in sel[:limit]: print("    ", p.replace("model.layers.78.", ""))
show(lambda p: ".self_attn." in p and ".experts." not in p, "self_attn params")
show(lambda p: ".shared_experts." in p, "shared_experts params")
show(lambda p: ".experts." in p and ".shared_experts." not in p, "FusedMoE expert params", limit=40)
show(lambda p: any(s in p for s in ("eh_proj","enorm","hnorm","shared_head")), "mtp-specific params")
show(lambda p: p.endswith("input_layernorm.weight") or p.endswith("post_attention_layernorm.weight") or "mlp.gate" in p, "norms/gate params")
nonl78 = sorted(p for p in model_params if not cat(p))
print(f"--- non-layer-78 params ({len(nonl78)}) ---")
for p in nonl78[:40]: print("    ", p)

with open("/srv/hf/_mtp_expected_params.json","w") as fh:
    json.dump(sorted(model_params), fh, indent=1)
print("[*] wrote /srv/hf/_mtp_expected_params.json", flush=True)
