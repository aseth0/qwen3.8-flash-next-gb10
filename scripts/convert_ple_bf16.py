#!/usr/bin/env python3
"""Flash-Next checkpoint for vLLM v0.30: the NVFP4 PLE (jstarkg/starkweather format)
rewritten in BF16, which v0.30 can read (Unquantized method + blazux's mmap).

The conversion is EXACT with respect to the NVFP4 dequantization: bf16(e2m1 * fp8_scale * global),
the same FP32 operations and the same rounding as the NVFP4 mmap path (LUT and GPU route).

Output: <dst>/ with (relative) symlinks to everything that holds no n-gram tables, and the shards
that do hold them rewritten: `ngram_embedding.shard_N.weight` [rows, head_dim] BF16, without
`weight_packed` / `weight_scale` / `weight_global_scale`. Index regenerated.

Usage: convert_ple_bf16.py <src> <dst>
"""
import json, os, re, sys, time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, DST = sys.argv[1], sys.argv[2]
LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
NG = re.compile(r"(.*\.ple\.ple_embedding\.ngram_embedding)\.(shard_\d+)\.(weight_packed|weight_scale)$")
GS = re.compile(r"(.*\.ple\.ple_embedding\.ngram_embedding)\.weight_global_scale$")
torch.set_num_threads(16)

idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wmap = idx["weight_map"]
ple_files = sorted({f for k, f in wmap.items() if NG.match(k) or GS.match(k)})
print(f"{len(ple_files)} files with n-gram tables", flush=True)

# global scale per layer (may live in a different file than its shards)
gscale = {}
for k, f in wmap.items():
    m = GS.match(k)
    if m:
        with safe_open(os.path.join(SRC, f), "pt") as sf:
            gscale[m[1]] = sf.get_tensor(k).float().reshape(())

os.makedirs(DST, exist_ok=True)
for fn in os.listdir(SRC):
    if fn in ple_files or fn == "model.safetensors.index.json" or fn.startswith("."):
        continue
    dst = os.path.join(DST, fn)
    if not os.path.lexists(dst):
        # relative: works with any mount point and when copying both directories
        os.symlink(os.path.relpath(os.path.join(SRC, fn), DST), dst)

new_map = {k: f for k, f in wmap.items() if f not in ple_files}


def dequant(packed: torch.Tensor, scale: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    rows = packed.shape[0]
    out = torch.empty((rows, packed.shape[1] * 2), dtype=torch.bfloat16)
    step = 1 << 16
    for a in range(0, rows, step):
        p = packed[a:a + step]
        codes = torch.stack((p & 0xF, p >> 4), dim=-1).reshape(p.shape[0], -1)
        v = LEVELS[(codes & 7).long()]
        v = torch.where(codes & 8 > 0, -v, v)
        eff = scale[a:a + step].view(torch.float8_e4m3fn).float() * g
        n = v.shape[0]
        out[a:a + step] = (v.view(n, -1, 16) * eff.unsqueeze(-1)).view(n, -1).to(torch.bfloat16)
    return out


t0 = time.time()
for i, fn in enumerate(ple_files):
    out_fn = fn.replace(".safetensors", "-plebf16.safetensors")
    out_path = os.path.join(DST, out_fn)
    with safe_open(os.path.join(SRC, fn), "pt") as sf:
        keys = list(sf.keys())
        tensors, pend = {}, {}
        for k in keys:
            m = NG.match(k)
            if m:
                pend.setdefault((m[1], m[2]), {})[m[3]] = k
            elif GS.match(k):
                continue
            else:
                tensors[k] = sf.get_tensor(k)
        for (base, shard), parts in pend.items():
            w = dequant(sf.get_tensor(parts["weight_packed"]), sf.get_tensor(parts["weight_scale"]), gscale[base])
            tensors[f"{base}.{shard}.weight"] = w
    if not os.path.exists(out_path):
        save_file(tensors, out_path + ".tmp", metadata={"format": "pt"})
        os.replace(out_path + ".tmp", out_path)
    for k in tensors:
        new_map[k] = out_fn
    print(f"[{i + 1}/{len(ple_files)}] {fn} -> {out_fn} ({len(pend)} tables) {time.time() - t0:.0f}s", flush=True)

idx["weight_map"] = dict(sorted(new_map.items()))
json.dump(idx, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=1)
print("OK", flush=True)
