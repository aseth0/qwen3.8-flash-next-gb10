#!/bin/bash
# Fill the page cache with the n-gram table (PLE) AFTER the container has loaded the weights. Run it once the service answers
# /health. Needs no root: it works from inside the container.
#
# Why not VLLM_PLE_MMAP_PREWARM=1: the loader reads the table before the weights are loaded, and the ~75 GiB weight load then
# evicts it (measured: 0.1 % resident at startup). Worse, the weight shards, already copied to the GPU, stay in the page cache
# (25 GiB of useless pages). This script drops them with posix_fadvise(DONTNEED) and reads the PLE file once: with ~26 GiB free
# the NVFP4 table (27 GiB) ends ~85-90 % resident and stays there.
#
# Note: PLE residency does not change decode speed (measured at 0 %, 46 % and 90 %: same 57.5 ms/step). It takes the NVMe out
# of the inference path and protects TTFT from other disk activity.
# Usage: scripts/ple-fill.sh [container=vllm-fn] [model dir inside the container=/models/<model in config/model.yaml>]
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
C=${1:-vllm-fn}
D=${2:-$(grep -oP '^model:\s*\K\S+' "$HERE/config/model.yaml")}
docker exec -i -e D="$D" "$C" python3 - <<'PY'
import ctypes, glob, json, os, time
D = os.environ["D"]; PAGE = 4096
libc = ctypes.CDLL("libc.so.6", use_errno=True); libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
def resident(p):
    size = os.path.getsize(p); fd = os.open(p, os.O_RDONLY); addr = libc.mmap(None, size, 1, 2, fd, 0); n = (size + PAGE - 1) // PAGE
    vec = (ctypes.c_ubyte * n)(); libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec); r = sum(v & 1 for v in vec)
    libc.munmap(ctypes.c_void_p(addr), size); os.close(fd); return r * PAGE / 2**30, size / 2**30
idx = json.load(open(os.path.join(D, "model.safetensors.index.json")))["weight_map"]
ple_files = sorted({f for k, f in idx.items() if "ple_embedding.ngram_embedding.shard_" in k})
weight_files = sorted(set(idx.values()) - set(ple_files))
for f in weight_files:  # already on the GPU: drop their pages so the PLE can use the cache
    fd = os.open(os.path.realpath(os.path.join(D, f)), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
t0 = time.time(); n = 0
for f in ple_files:
    with open(os.path.join(D, f), "rb", buffering=0) as fh:
        while True:
            b = fh.read(64 << 20)
            if not b: break
            n += len(b)
r = sum(resident(os.path.join(D, f))[0] for f in ple_files); s = sum(resident(os.path.join(D, f))[1] for f in ple_files)
print(json.dumps({"ple_files": len(ple_files), "read_GiB": round(n / 2**30, 1), "seconds": round(time.time() - t0), "resident_GiB": round(r, 1), "percent": round(100 * r / s, 1)}))
PY
