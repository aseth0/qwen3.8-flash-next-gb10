#!/usr/bin/env python3
"""Builds the checkpoint this config serves, from two public checkpoints at pinned revisions:

  nvidia/Qwen3.8-Flash-Next-NVFP4            -> routed experts (NVFP4, ModelOpt), every BF16 tensor, tokenizer, configs
  starkweatherdigital/qwen3.8-flash-next-nvfp4 -> the MTP drafter's experts in NVFP4 and the n-gram table (PLE) in NVFP4

Why a hybrid: NVIDIA's checkpoint ships the MTP experts as block-FP8 (on vLLM v0.30 they load through DeepGEMM and
cost +3.2 GiB of GPU memory and +1.3 ms per decode step) and the PLE as FP8 (47.7 GiB). The second checkpoint has both in
NVFP4: same source tensors, half the size, and the PLE (26.9 GiB) fits in the page cache of a GB10. vLLM loads every tensor
of every file named in the index, so the pieces are rewritten into files of their own (streaming byte copies, no RAM).

Output <dst>/: relative symlinks to NVIDIA's shards, `model-mtp-experts-nvfp4.safetensors` (6,144 tensors),
`model-ple-nvfp4.safetensors` (257 tensors), a regenerated index, config.json / hf_quant_config.json with the MTP experts
as NVFP4 and the PLE unquantized (the mmap loader builds the table itself), and generation_config.json with this config's
default sampling (temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05: at temperature 0 the model can loop on
tool calls, and clients that send temperature 0 never send a repetition penalty, so the server default is what stops it).

Usage: build-hybrid.py <nvidia_dir> <stark_dir> <dst>      (idempotent: existing output files are kept)"""
import json, os, struct, sys, time

NV, ST, DST = sys.argv[1], sys.argv[2], sys.argv[3]
MTP_EXP = "mtp.layers.0.mlp.experts."
PLE_KEY = "ple_embedding.ngram_embedding"
NV_BIG = "model-fp8-mtp-ple.safetensors"           # NVIDIA's FP8 MTP experts + FP8 PLE: not used at all
MTP_F, PLE_F = "model-mtp-experts-nvfp4.safetensors", "model-ple-nvfp4.safetensors"
SKIP = {"config.json", "hf_quant_config.json", "generation_config.json", "model.safetensors.index.json", NV_BIG}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return h, 8 + n, meta


def write_streaming(out_path, entries, meta):
    """entries: (name, dtype, shape, src_path, src_abs_offset, nbytes). Byte copies, one tensor at a time."""
    if os.path.exists(out_path):
        print(f"  {os.path.basename(out_path)} exists, kept"); return
    hdr, off = {}, 0
    for name, dtype, shape, _, _, nb in entries:
        hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nb]}; off += nb
    hdr["__metadata__"] = meta
    hj = json.dumps(hdr, separators=(",", ":")).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
    tmp, t0, cur = out_path + ".tmp", time.time(), None
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hj))); out.write(hj)
        for i, (name, _, _, src, so, nb) in enumerate(entries):
            if cur is None or cur.name != src:
                if cur: cur.close()
                cur = open(src, "rb")
            cur.seek(so); left = nb
            while left:
                chunk = cur.read(min(left, 64 << 20))
                if not chunk: raise IOError(f"short read in {name}")
                out.write(chunk); left -= len(chunk)
            if (i + 1) % 1024 == 0 or i + 1 == len(entries):
                print(f"  {os.path.basename(out_path)}: {i + 1}/{len(entries)} tensors, {out.tell() / 2**30:.1f} GiB, {time.time() - t0:.0f} s", flush=True)
        if cur: cur.close()
    os.replace(tmp, out_path)


def entries_from(src_dir, index, select):
    by_file = {}
    for k, f in index.items():
        if select(k): by_file.setdefault(f, []).append(k)
    out = []
    for f, names in sorted(by_file.items()):
        h, base, _ = header(os.path.join(src_dir, f))
        for k in sorted(names):
            a, b = h[k]["data_offsets"]
            out.append((k, h[k]["dtype"], h[k]["shape"], os.path.join(src_dir, f), base + a, b - a))
    return out


os.makedirs(DST, exist_ok=True)
nv_idx = json.load(open(os.path.join(NV, "model.safetensors.index.json")))
st_idx = json.load(open(os.path.join(ST, "model.safetensors.index.json")))["weight_map"]
src_tag = "starkweatherdigital/qwen3.8-flash-next-nvfp4@1b304e5f"

mtp = entries_from(ST, st_idx, lambda k: k.startswith(MTP_EXP)); assert len(mtp) == 6144, len(mtp)
write_streaming(os.path.join(DST, MTP_F), mtp, {"format": "pt", "source": src_tag + " (MTP routed experts, NVFP4)"})
ple = entries_from(ST, st_idx, lambda k: PLE_KEY in k); assert len(ple) == 257, len(ple)
write_streaming(os.path.join(DST, PLE_F), ple, {"format": "pt", "source": src_tag + " (n-gram table, NVFP4)"})

for fn in os.listdir(NV):
    if fn in SKIP or fn.startswith("."): continue
    dst = os.path.join(DST, fn)
    if not os.path.lexists(dst): os.symlink(os.path.relpath(os.path.join(NV, fn), DST), dst)

wm = {k: f for k, f in nv_idx["weight_map"].items() if f != NV_BIG}
dropped = len(nv_idx["weight_map"]) - len(wm); assert dropped == 3201, dropped   # 3,072 FP8 MTP experts + 129 FP8 PLE
for k, *_ in mtp: wm[k] = MTP_F
for k, *_ in ple: wm[k] = PLE_F
nv_idx["weight_map"] = dict(sorted(wm.items()))
json.dump(nv_idx, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=1)

cfg = json.load(open(os.path.join(NV, "config.json"))); q = cfg["quantization_config"]
q["quantized_layers"]["mtp.layers.0.mlp.experts"] = {"quant_algo": "NVFP4", "group_size": 16}
q["quantized_layers"].pop("model.language_model.layers.1.ple.ple_embedding.ngram_embedding", None)
g0 = q["config_groups"]["group_0"]
q["config_groups"]["group_1"] = {"input_activations": g0["input_activations"], "weights": g0["weights"], "targets": ["mtp.layers.0.mlp.experts"]}
q["config_groups"].pop("group_2", None)
cfg["_hybrid"] = {"experts": "nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54", "mtp_experts_and_ple": src_tag, "built_by": "scripts/build-hybrid.py"}
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
hq = json.load(open(os.path.join(NV, "hf_quant_config.json")))
hq["quantization"]["quantized_layers"]["mtp.layers.0.mlp.experts"] = {"quant_algo": "NVFP4", "group_size": 16}
hq["quantization"]["quantized_layers"].pop("model.language_model.layers.1.ple.ple_embedding.ngram_embedding", None)
json.dump(hq, open(os.path.join(DST, "hf_quant_config.json"), "w"), indent=2)
gen = json.load(open(os.path.join(NV, "generation_config.json")))
gen.update({"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.05})
json.dump(gen, open(os.path.join(DST, "generation_config.json"), "w"), indent=4)

seen = {}
for f in sorted(x for x in os.listdir(DST) if x.endswith(".safetensors")):
    hh, _, _ = header(os.path.join(DST, f))
    for k in hh:
        assert k not in seen, f"duplicate tensor {k} in {f} and {seen[k]}"; seen[k] = f
assert set(seen) == set(wm) and all(seen[k] == wm[k] for k in wm), "index and files disagree"
open(os.path.join(DST, ".build-complete"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
print(f"OK: {len(seen)} tensors in {len(set(seen.values()))} files -> {DST}")
