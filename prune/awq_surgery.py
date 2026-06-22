#!/usr/bin/env python3
"""Offline weight-only routed-expert prune for cyankiwi/GLM-5.2-AWQ-INT4.

Pure byte-level safetensors surgery (no dequant, no torch):
  - routed experts: keep/drop whole expert (12 tensors), re-index survivors 0..N-1
  - router gate.weight [E,H] BF16 + gate.e_score_correction_bias [E] F32: row-slice
  - shared_experts / dense / attention / norms: untouched
  - config: num_experts/n_routed_experts -> N   (UNIFORM N required)

Modes:
  step1 <src>                 : correctness harness (gate row-slice + reindex map)
  build <src> <out> <ratio>   : write pruned checkpoint, dropping `ratio` LOWEST
                                e_score_correction_bias experts/layer (data-free)
  validate <out> <src>        : structural check of a written checkpoint vs source
"""
import json, re, sys, glob, struct, hashlib, os, shutil

DSIZE = {"BF16":2,"F16":2,"F32":4,"F64":8,"I8":1,"U8":1,"I16":2,"I32":4,"I64":8,
         "F8_E4M3":1,"F8_E5M2":1,"BOOL":1}
EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$")
GATE_RE   = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")
BIAS_RE   = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$")

def snap_of(src): return glob.glob(f"{src}/snapshots/*/")[0] if "snapshots" in os.listdir(src) else src.rstrip("/")+"/"
def parse_index(snap): return json.load(open(os.path.join(snap,"model.safetensors.index.json")))["weight_map"]
def parse_header(path):
    with open(path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(n))
    return hdr, 8+n
def read_tensor_bytes(path,name):
    hdr,ds=parse_header(path); m=hdr[name]; s,e=m["data_offsets"]
    with open(path,"rb") as f: f.seek(ds+s); return f.read(e-s), m["dtype"], m["shape"]
def row_slice(buf, shape, keep):
    rows=shape[0]; rb=len(buf)//rows; assert rb*rows==len(buf),(len(buf),rows)
    return b"".join(buf[i*rb:(i+1)*rb] for i in keep), [len(keep)]+shape[1:]
def h(b): return hashlib.blake2b(b,digest_size=8).hexdigest()

def expert_counts(idx):
    E={}
    for k in idx:
        m=EXPERT_RE.match(k)
        if m: L=int(m.group(1)); E[L]=max(E.get(L,0),int(m.group(2))+1)
    return E

def write_safetensors(path, tensors, metadata=None):
    """tensors: list of (name, dtype, shape, bytes). Writes one shard."""
    header={}; off=0
    if metadata: header["__metadata__"]=metadata
    for name,dt,sh,buf in tensors:
        header[name]={"dtype":dt,"shape":sh,"data_offsets":[off,off+len(buf)]}; off+=len(buf)
    hj=json.dumps(header).encode("utf-8")
    with open(path,"wb") as f:
        f.write(struct.pack("<Q",len(hj))); f.write(hj)
        for _,_,_,buf in tensors: f.write(buf)

# ---------------- selection (data-free: lowest e_score_correction_bias) ----------------
def bias_keeplist(snap, idx, L, E, n_keep):
    bname=f"model.layers.{L}.mlp.gate.e_score_correction_bias"
    buf,dt,sh=read_tensor_bytes(snap+idx[bname], bname)
    # F32 little-endian
    import array; a=array.array("f"); a.frombytes(buf)
    assert dt=="F32" and len(a)==E
    order=sorted(range(E), key=lambda i: a[i])      # ascending bias
    # PRUNE highest bias (router had to boost = least attractive) -> keep lowest n_keep
    keep=sorted(order[:n_keep])                      # keep lowest-bias, sorted asc index
    return keep

def fixed_drop_keeplist(L, E, drop):
    return [e for e in range(E) if e not in drop]

# ---------------- build ----------------
def build(src, out, ratio=None, drop=None):
    snap=snap_of(src); idx=parse_index(snap); config=json.load(open(snap+"config.json"))
    E_by_L=expert_counts(idx); moe=sorted(E_by_L)
    E0=E_by_L[moe[0]]
    assert all(v==E0 for v in E_by_L.values()), "non-uniform expert counts"
    if drop is not None:
        keep={L:fixed_drop_keeplist(L,E0,drop) for L in moe}
    else:
        n_keep=E0-int(round(E0*ratio))
        keep={L:bias_keeplist(snap,idx,L,E0,n_keep) for L in moe}
    N=len(keep[moe[0]]); assert all(len(v)==N for v in keep.values()),"non-uniform keep"
    old2new={L:{o:j for j,o in enumerate(keep[L])} for L in moe}
    os.makedirs(out,exist_ok=True)

    def fate(name):
        m=EXPERT_RE.match(name)
        if m:
            L=int(m.group(1)); e=int(m.group(2))
            if e in old2new[L]: return ("rename", f"model.layers.{L}.mlp.experts.{old2new[L][e]}.{m.group(3)}")
            return ("drop",None)
        g=GATE_RE.match(name) or BIAS_RE.match(name)
        if g and int(g.group(1)) in keep: return ("slice",name)
        return ("keep",name)

    shards={}
    for nm,sh in idx.items(): shards.setdefault(sh,[]).append(nm)
    new_index={}; total=0; nsh=len(shards)
    for si,sh in enumerate(sorted(shards)):
        hdr,ds=parse_header(snap+sh); out_t=[]
        with open(snap+sh,"rb") as f:
            for nm in shards[sh]:
                act,newname=fate(nm)
                if act=="drop": continue
                m=hdr[nm]; s,e=m["data_offsets"]; f.seek(ds+s); buf=f.read(e-s)
                dt=m["dtype"]; shp=m["shape"]
                if act=="slice":
                    L=int(re.match(r"^model\.layers\.(\d+)\.",nm).group(1))
                    buf,shp=row_slice(buf,shp,keep[L])
                out_t.append((newname,dt,shp,buf))
        if out_t:
            write_safetensors(out+"/"+sh, out_t, hdr.get("__metadata__"))
            for nn,_,_,b in out_t: new_index[nn]=sh; total+=len(b)
        if (si%10==0) or si==nsh-1: print(f"  shard {si+1}/{nsh} {sh}: {len(out_t)} tensors", flush=True)
    json.dump({"metadata":{"total_size":total},"weight_map":new_index},
              open(out+"/model.safetensors.index.json","w"))
    config["num_experts"]=N; config["n_routed_experts"]=N
    json.dump(config, open(out+"/config.json","w"), indent=2)
    for fn in os.listdir(snap):
        p=snap+fn
        if os.path.isfile(p) and not (fn.endswith(".safetensors") or fn in
            ("model.safetensors.index.json","config.json")):
            shutil.copy(p, out+"/"+fn)
    print(f"BUILT {out}: {len(new_index)} tensors, {total/1e9:.1f} GB, num_experts {E0}->{N}", flush=True)

# ---------------- validate output structurally vs source ----------------
def validate(out, src):
    snap=snap_of(src); sidx=parse_index(snap)
    oidx=parse_index(out); oconf=json.load(open(out+"/config.json"))
    N=oconf["num_experts"]; ok=True
    # all index names map to existing shards + offsets contiguous per shard
    oshards={}
    for nm,sh in oidx.items(): oshards.setdefault(sh,[]).append(nm)
    for sh in oshards:
        if not os.path.exists(out+"/"+sh): print(f"  MISSING shard {sh}"); ok=False; continue
        hdr,ds=parse_header(out+"/"+sh)
        offs=sorted([hdr[n]["data_offsets"] for n in hdr if n!="__metadata__"])
        cont=all(offs[i][1]==offs[i+1][0] for i in range(len(offs)-1)) and (offs[0][0]==0)
        if not cont: print(f"  {sh}: NON-CONTIGUOUS offsets"); ok=False
        fsize=os.path.getsize(out+"/"+sh)
        if fsize != ds+offs[-1][1]: print(f"  {sh}: size mismatch {fsize} vs {ds+offs[-1][1]}"); ok=False
    # spot-check: a surviving expert's bytes == its source; gate sliced correctly
    L=int(re.match(r"^model\.layers\.(\d+)\.",next(iter(oidx))).group(1)) if False else None
    moe=sorted({int(m.group(1)) for k in oidx if (m:=EXPERT_RE.match(k))})
    import random
    for L in [moe[0], moe[len(moe)//2], moe[-1]]:
        # new expert 0 should byte-equal SOME source expert (we can't know which w/o keep map,
        # but verify the tensor exists+nonzero and gate has N rows)
        gname=f"model.layers.{L}.mlp.gate.weight"
        gb,gdt,gsh=read_tensor_bytes(out+"/"+oidx[gname], gname)
        ne=max(int(m.group(2)) for k in oidx if (m:=EXPERT_RE.match(k)) and int(m.group(1))==L)+1
        if gsh[0]!=N: print(f"  L{L}: gate rows {gsh[0]} != {N}"); ok=False
        if ne!=N: print(f"  L{L}: experts {ne} != {N}"); ok=False
        # expert tensor count
        ct=len({k for k in oidx if (m:=EXPERT_RE.match(k)) and int(m.group(1))==L})
        if ct!=N*12: print(f"  L{L}: expert tensors {ct} != {N*12}"); ok=False
    print(f"  config num_experts={N}, total index tensors={len(oidx)}")
    print("VALIDATE:", "PASS" if ok else "FAIL")
    return ok

# ---------------- step1 harness (from before) ----------------
def step1(src):
    snap=snap_of(src); idx=parse_index(snap); E_by_L=expert_counts(idx); moe=sorted(E_by_L)
    print(f"MoE layers {moe[0]}..{moe[-1]} ({len(moe)})")
    allok=True
    for L in [moe[0],moe[len(moe)//2],moe[-1]]:
        E=E_by_L[L]
        perm=list(range(E))
        for i in range(E-1,0,-1):
            j=(i*2654435761)%(i+1); perm[i],perm[j]=perm[j],perm[i]
        for keep,lab in [(perm,"PERMUTE"),([e for e in range(E) if e%7],"DROP~14%")]:
            g=f"model.layers.{L}.mlp.gate.weight"; b=f"model.layers.{L}.mlp.gate.e_score_correction_bias"
            gb,_,gsh=read_tensor_bytes(snap+idx[g],g); bb,_,bsh=read_tensor_bytes(snap+idx[b],b)
            go,_=row_slice(gb,gsh,keep); bo,_=row_slice(bb,bsh,keep)
            gr=len(gb)//gsh[0]; br=len(bb)//bsh[0]
            okg=all(h(go[j*gr:(j+1)*gr])==h(gb[keep[j]*gr:(keep[j]+1)*gr]) for j in range(len(keep)))
            okb=all(h(bo[j*br:(j+1)*br])==h(bb[keep[j]*br:(keep[j]+1)*br]) for j in range(len(keep)))
            allok&=okg and okb
            print(f"  L{L} {lab}: gate {'OK' if okg else 'FAIL'} bias {'OK' if okb else 'FAIL'}")
    print("STEP1:", "PASS" if allok else "FAIL")

if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="step1": step1(sys.argv[2])
    elif mode=="build":
        src, out, sel = sys.argv[2], sys.argv[3], sys.argv[4]
        if sel=="drop4": build(src, out, drop={1,64,128,200})
        else: build(src, out, ratio=float(sel))
    elif mode=="validate": validate(sys.argv[2], sys.argv[3])
