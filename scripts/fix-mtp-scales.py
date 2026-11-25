#!/usr/bin/env python3
"""Builds a copy of the hybrid checkpoint whose MTP drafter experts have ONE global NVFP4 scale for gate_proj and up_proj.

Why: vLLM fuses gate/up into w13 and keeps a single global scale per expert (`w13_weight_scale_2[:, 0]`, the gate's).
In the MTP experts of starkweatherdigital/qwen3.8-flash-next-nvfp4 the two scales differ in 506 of 512 experts, so the
runtime dequantizes every up_proj with the gate's scale: its output is off by g/u (median 24 %, range 0.37x..3.6x).
vLLM warns at startup: "w1_weight_scale_2 must match w3_weight_scale_2. Accuracy may be affected."
Only the drafter is affected (speed through acceptance, never output quality): the 48 main layers already match.

Fix, per expert: the shared scale is the larger of the two (the block scales already reach the FP8 maximum, so only
shrinking is safe). The projection with the smaller scale gets the ratio folded into its per-block FP8 scales, its FP4
codes re-rounded against the weights the checkpoint meant (code * block_scale * own scale), and the shared scale stored
as its weight_scale_2. Everything else is copied byte for byte; header, tensor names, shapes and offsets do not change.

Measured on a DGX Spark (BENCHMARKS.md): acceptance and speed unchanged within noise (+0.2 %), quality unchanged; it removes
the startup warning and serves the drafter the weights the checkpoint meant.

Usage: fix-mtp-scales.py <src_hybrid_dir> <dst_dir>          (never writes into <src_hybrid_dir>; idempotent)
       fix-mtp-scales.py --verify <src_hybrid_dir> <dst_dir>"""
import json, os, shutil, struct, sys, time
import numpy as np

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
E2M1_MID = ((E2M1[1:] + E2M1[:-1]) / 2).astype(np.float32)


def _e4m3_table():
    t = np.zeros(256, dtype=np.float64)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
        if e == 0xF and m == 7:
            v = np.nan
        elif e == 0:
            v = (m / 8.0) * 2.0 ** -6
        else:
            v = (1 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


E4M3 = _e4m3_table()                       # byte -> value
E4M3_POS = E4M3[:127].astype(np.float32)   # 0x00..0x7E, increasing, finite, >= 0


class ST:
    """Minimal safetensors reader (pread, no mmap of the whole file)."""

    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        n = struct.unpack("<Q", os.pread(self.fd, 8, 0))[0]
        self.hraw = os.pread(self.fd, n, 8)
        self.h = json.loads(self.hraw)
        self.base = 8 + n

    def raw(self, k):
        a, b = self.h[k]["data_offsets"]
        return os.pread(self.fd, b - a, self.base + a)

    def abs_off(self, k):
        a, b = self.h[k]["data_offsets"]
        return self.base + a, b - a

    def f32(self, k):
        return float(np.frombuffer(self.raw(k), np.float32)[0])

    def u8(self, k):
        return np.frombuffer(self.raw(k), np.uint8).reshape(self.h[k]["shape"])

    def bf16(self, k):
        u = np.frombuffer(self.raw(k), np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(self.h[k]["shape"])


def unpack_fp4(packed):
    """[N, K/2] bytes -> [N, K] float codes (low nibble = even index)."""
    lo, hi = packed & 0x0F, packed >> 4
    c = np.stack([lo, hi], -1).reshape(packed.shape[0], -1)
    return E2M1[c & 7] * np.where(c & 8, np.float32(-1), np.float32(1))


def pack_fp4(val):
    """[N, K] float values already on the E2M1 grid -> [N, K/2] bytes."""
    idx = np.searchsorted(E2M1, np.abs(val)).astype(np.uint8)
    idx |= (np.signbit(val) & (val != 0)).astype(np.uint8) << 3
    idx = idx.reshape(val.shape[0], -1, 2)
    return (idx[..., 0] | (idx[..., 1] << 4)).astype(np.uint8)


def round_e2m1(x):
    mag = np.minimum(np.abs(x), 6.0)
    return np.copysign(E2M1[np.digitize(mag, E2M1_MID)], x)


def dequant_nvfp4(packed, sc_bytes, sc2):
    """ModelOpt NVFP4: W = code * block_scale(FP8, per 16 inputs) * weight_scale_2."""
    bs = E4M3[sc_bytes].astype(np.float32)
    return unpack_fp4(packed) * np.repeat(bs, 16, axis=1) * np.float32(sc2)


MTP_F, PLE_F = "model-mtp-experts-nvfp4.safetensors", "model-ple-nvfp4.safetensors"
DONE = ".mtpfix-complete"
P = "mtp.layers.0.mlp.experts."
NCAND = (-2, -1, 0, 1)          # FP8 neighbours of the ideal block scale that are tried (MSE picks one)


def fix_expert(src, e):
    """Returns (projection to rewrite or None, packed codes, block-scale bytes, shared scale, relative error).
    The shared scale is the LARGER of the two: the block scales already reach the FP8 maximum (448), so the
    projection with the smaller global scale is the one re-expressed (its block scales shrink, nothing saturates)."""
    g = np.float32(src.f32(f"{P}{e}.gate_proj.weight_scale_2"))
    u = np.float32(src.f32(f"{P}{e}.up_proj.weight_scale_2"))
    if g == u:
        return None, None, None, g, 0.0
    proj, own, shared = ("up_proj", u, g) if u < g else ("gate_proj", g, u)
    pk, sc = src.u8(f"{P}{e}.{proj}.weight"), src.u8(f"{P}{e}.{proj}.weight_scale")
    T = dequant_nvfp4(pk, sc, own)                                 # what the checkpoint means
    n, k = T.shape
    Tb = T.reshape(n, k // 16, 16)
    ideal = E4M3[sc].astype(np.float32) * (own / shared)           # exact block scale under the shared scale (< original)
    hi = np.clip(np.searchsorted(E4M3_POS, ideal), 1, 126)
    best_err = np.full(ideal.shape, np.inf, np.float32)
    best_idx = np.zeros(ideal.shape, np.uint8)
    best_q = np.zeros_like(Tb)
    for d in NCAND:
        idx = np.clip(hi + d, 1, 126).astype(np.uint8)
        s = (E4M3_POS[idx] * shared)[..., None]
        q = round_e2m1(Tb / s)
        err = ((q * s - Tb) ** 2).sum(-1)
        m = err < best_err
        best_err[m], best_idx[m] = err[m], idx[m]
        best_q[m] = q[m]
    zero = sc == 0                                                 # empty blocks stay exactly as they were
    best_idx[zero] = 0
    new_pk = pack_fp4(best_q.reshape(n, k))
    new_pk = np.where(np.repeat(zero, 8, axis=1), pk, new_pk)
    rel = float(np.sqrt(best_err[~zero].sum() / max(float((T ** 2).sum()), 1e-30)))
    return proj, new_pk, best_idx, shared, rel


def build(src_dir, dst_dir):
    if os.path.realpath(src_dir) == os.path.realpath(dst_dir):
        sys.exit("dst must differ from src")
    os.makedirs(dst_dir, exist_ok=True)
    for fn in sorted(os.listdir(src_dir)):
        s, d = os.path.join(src_dir, fn), os.path.join(dst_dir, fn)
        if fn in (MTP_F, DONE) or ".bak" in fn or os.path.lexists(d):
            continue
        if os.path.islink(s):
            os.symlink(os.readlink(s), d)                          # same relative target: dst is a sibling of src
        elif fn == PLE_F:
            os.symlink(os.path.relpath(s, dst_dir), d)             # 28.8 GB, untouched: link, do not copy
        else:
            shutil.copy2(s, d)
    if os.path.exists(os.path.join(dst_dir, DONE)):
        print(f"{dst_dir} already fixed, kept"); return
    out, tmp = os.path.join(dst_dir, MTP_F), os.path.join(dst_dir, MTP_F + ".tmp")
    src = ST(os.path.join(src_dir, MTP_F))
    shutil.copyfile(src.path, tmp)
    fd = os.open(tmp, os.O_RDWR)
    t0, rels, which = time.time(), [], {"up_proj": 0, "gate_proj": 0}
    for e in range(512):
        proj, pk, sc, shared, rel = fix_expert(src, e)
        if proj:
            for key, data in ((".weight", pk.tobytes()), (".weight_scale", sc.tobytes()), (".weight_scale_2", np.float32(shared).tobytes())):
                off, nb = src.abs_off(f"{P}{e}.{proj}{key}")
                assert len(data) == nb, (e, key, len(data), nb)
                os.pwrite(fd, data, off)
            which[proj] += 1; rels.append(rel)
        if (e + 1) % 64 == 0:
            print(f"  {e + 1}/512 experts, {sum(which.values())} rewritten, {time.time() - t0:.0f} s", flush=True)
    os.fsync(fd); os.close(fd); os.replace(tmp, out)
    cfg_p = os.path.join(dst_dir, "config.json")
    cfg = json.load(open(cfg_p))
    cfg.setdefault("_hybrid", {})["mtp_scale_fix"] = (
        f"MTP experts: gate_proj/up_proj re-expressed under one global scale, the larger of the two "
        f"({which['up_proj']} up_proj + {which['gate_proj']} gate_proj of 512; scripts/fix-mtp-scales.py)")
    json.dump(cfg, open(cfg_p, "w"), indent=2)
    open(os.path.join(dst_dir, DONE), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    r = np.array(rels)
    print(f"rewritten {which} in {time.time() - t0:.0f} s; relative error of the rewritten projection against the "
          f"intended weights: median {np.median(r):.4f}, max {r.max():.4f}")


def verify(src_dir, dst_dir):
    a, b = ST(os.path.join(src_dir, MTP_F)), ST(os.path.join(dst_dir, MTP_F))
    assert a.hraw == b.hraw, "header differs"
    assert os.path.getsize(a.path) == os.path.getsize(b.path), "size differs"
    same = diff = 0
    for k in a.h:
        if k == "__metadata__":
            continue
        if a.raw(k) == b.raw(k):
            same += 1
        else:
            diff += 1
            assert ".up_proj.weight" in k or ".gate_proj.weight" in k, f"unexpected change in {k}"
    err = {"gate_proj": ([], []), "up_proj": ([], [])}
    mism = 0
    for e in range(512):
        g2, u2 = b.f32(f"{P}{e}.gate_proj.weight_scale_2"), b.f32(f"{P}{e}.up_proj.weight_scale_2")
        mism += int(g2 != u2)
        g0 = a.f32(f"{P}{e}.gate_proj.weight_scale_2")
        for proj in ("gate_proj", "up_proj"):
            own = a.f32(f"{P}{e}.{proj}.weight_scale_2")
            T = dequant_nvfp4(a.u8(f"{P}{e}.{proj}.weight"), a.u8(f"{P}{e}.{proj}.weight_scale"), own)
            # the runtime uses the GATE's global scale for both halves
            N = dequant_nvfp4(b.u8(f"{P}{e}.{proj}.weight"), b.u8(f"{P}{e}.{proj}.weight_scale"), g2)
            err[proj][0].append(abs(g0 / own - 1)); err[proj][1].append(float(np.linalg.norm(N - T) / np.linalg.norm(T)))
    print(f"tensors identical {same}, changed {diff} (gate_proj/up_proj only); experts with gate != up global scale: {mism}")
    print("projection as the runtime computes it vs intended weights, relative error over 512 experts:")
    for proj, (o, n) in err.items():
        o, n = np.array(o), np.array(n)
        print(f"  {proj:9s} before: median {np.median(o):.4f} mean {o.mean():.4f} max {o.max():.4f} | "
              f"after: median {np.median(n):.4f} mean {n.mean():.4f} max {n.max():.4f}")
    for fn in sorted(os.listdir(dst_dir)):
        if not os.path.exists(os.path.join(dst_dir, fn)):
            sys.exit(f"broken link: {fn}")
    print("all files of the destination resolve")


if __name__ == "__main__":
    if sys.argv[1] == "--verify":
        verify(sys.argv[2], sys.argv[3])
    else:
        build(sys.argv[1], sys.argv[2])
