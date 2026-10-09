#!/bin/bash
# Memory plan for a GB10 BEFORE starting the container: how much KV cache this host can give the model, and what is
# left as page cache for the n-gram table (PLE). Reads the host (RAM, GPU memory already in use) and the checkpoint
# directory (which shards go to the GPU, which hold the PLE). Changes nothing.
#
# Why: on a GB10 the GPU and the host share one memory pool. vLLM sizes the KV cache dynamically (gpu-memory-utilization):
# it measures what is free *after* loading the weights, and on unified memory that includes the page cache the load itself
# just filled, so the KV cache varies by about +/-1 GiB (+/-30k tokens) between starts. This config keeps it dynamic on
# every host (vLLM's pinned mode, kv-cache-memory, skips the profiling and can end in an out-of-memory at runtime). The
# script tells you what to expect for a utilization and which utilization aims at a given KV cache size.
#
# Usage: scripts/plan-memory.sh [model_dir] [--kv GiB]   (--kv: target KV cache; model_dir defaults to $MODELS_DIR/<model in config/model.yaml>)
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$HERE/.env" ] && set -a && . "$HERE/.env" && set +a
KV_WANT=""
while [ $# -gt 0 ]; do case "$1" in --kv) KV_WANT=$2; shift 2;; *) MODEL_DIR=$1; shift;; esac; done
if [ -z "${MODEL_DIR:-}" ]; then
  rel=$(grep -oP '^model:\s*/models/\K\S+' "$HERE/config/model.yaml" 2>/dev/null)
  MODEL_DIR="${MODELS_DIR:-/opt/models-vllm}/${rel:-}"
fi
[ -d "$MODEL_DIR" ] || { echo "model dir not found: $MODEL_DIR"; exit 1; }
MAXLEN=$(grep -oP '^max-model-len:\s*\K[0-9]+' "$HERE/config/model.yaml" 2>/dev/null || echo 131072)
UTIL=$(grep -oP '^gpu-memory-utilization:\s*\K[0-9.]+' "$HERE/config/model.yaml" 2>/dev/null || echo 0.715)
python3 - "$MODEL_DIR" "$MAXLEN" "$UTIL" "$KV_WANT" <<'PY'
import json, os, re, subprocess, sys
D, MAXLEN, UTIL, KV_WANT = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
G = 2**30
# --- measured on a DGX Spark (2026-10-09), see BENCHMARKS.md ---
KV_BYTES_PER_TOKEN = 31.0e3       # 9.0 GiB -> 311,860 tokens; 8.71 GiB -> 301,314 (max-model-len 131072, MTP-3, RecoverSSM)
DENSE_FP8_SAVING = 3.11 * G       # linear_attn + self_attn + lm_head BF16 -> FP8 at load (config/fn_dense_fp8.conf)
RUNTIME_OVERHEAD = 5.1 * G        # measured "weights + non-torch" minus the shards: CUDA context, FlashInfer workspaces, Mamba
                                  # states, drafter head, PLE loader buffers (4.1 GiB on the BF16 PLE, 5.3 on the NVFP4 PLE)
ACTIVATION_PEAK = 2.5 * G         # vLLM's measured peak activation during profiling
GRAPHS = 0.5 * G                  # CUDA graphs
OS_RESERVE = 6.0 * G              # kernel, docker, python processes outside the pool (measured: ~6 GiB on a DGX Spark)
# --- host ---
mem = {l.split(':')[0]: int(l.split()[1]) * 1024 for l in open('/proc/meminfo') if ':' in l}
total = mem['MemTotal']
other_gpu, running_model = 0, 0
try:
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
    sizes = sorted((int(l.split(',')[1]) * 2**20 for l in out.splitlines() if l.strip()), reverse=True)
    if sizes and sizes[0] > 50 * G:      # a >50 GiB process can only be this model, already running: plan as if it were stopped
        running_model, sizes = sizes[0], sizes[1:]
    other_gpu = sum(sizes)
except Exception:
    pass
# --- checkpoint: which files go to the GPU, which are the PLE table (served from the page cache) ---
idx = json.load(open(os.path.join(D, 'model.safetensors.index.json')))['weight_map']
ple_files = {f for k, f in idx.items() if 'ple_embedding.ngram_embedding.shard_' in k}
gpu_files = set(idx.values()) - ple_files
size = lambda fs: sum(os.path.getsize(os.path.realpath(os.path.join(D, f))) for f in fs)
weights_disk, ple_disk = size(gpu_files), size(ple_files)
dense_fp8 = True  # config/fn_dense_fp8.conf ships with the repo (linear_attn,self_attn,lm_head)
weights_gpu = weights_disk - (DENSE_FP8_SAVING if dense_fp8 else 0) + RUNTIME_OVERHEAD
fixed = weights_gpu + ACTIVATION_PEAK + GRAPHS
budget = total - other_gpu - OS_RESERVE          # what the model + its KV + the PLE page cache must share
pool = UTIL * total
kv_auto = pool - fixed                           # what gpu-memory-utilization would give (before the +/-1 GiB noise)
def tokens(b): return int(b / KV_BYTES_PER_TOKEN)
def ple_pct(kv): return max(0.0, min(100.0, 100 * (budget - fixed - kv) / ple_disk)) if ple_disk else 0.0
print(f"host: {total/G:.1f} GiB RAM, {other_gpu/G:.1f} GiB used on the GPU by other services, {OS_RESERVE/G:.0f} GiB kept for the OS"
      + (f" (a {running_model/G:.1f} GiB process is this model already running: ignored)" if running_model else ""))
print(f"model: {weights_disk/G:.1f} GiB of shards to the GPU (-{DENSE_FP8_SAVING/G:.1f} dense FP8, +{RUNTIME_OVERHEAD/G:.1f} runtime) = {weights_gpu/G:.1f} GiB; PLE table {ple_disk/G:.1f} GiB served from the page cache")
print(f"fixed cost (weights + activation {ACTIVATION_PEAK/G:.1f} + graphs {GRAPHS/G:.1f}): {fixed/G:.1f} GiB; left for KV cache + PLE page cache: {(budget-fixed)/G:.1f} GiB")
print()
print(f"{'utilization':>11} {'KV cache':>9} {'tokens':>9} {'full ctx reqs':>13} {'PLE in RAM':>11}   note")
rows = [(UTIL, "current config/model.yaml")]
for u in (0.70, 0.72, 0.74, 0.76):
    rows.append((u, ""))
if KV_WANT:
    rows.append(((fixed + float(KV_WANT) * G) / total, f"aims at {float(KV_WANT):.1f} GiB of KV cache"))
for u, note in sorted(rows):
    kv = u * total - fixed
    if kv <= 0: print(f"{u:11.3f} {'-':>9} {'-':>9} {'-':>13} {'-':>11}   {note}  <- does not fit"); continue
    print(f"{u:11.3f} {kv/G:7.1f} G {tokens(kv):9,d} {kv/KV_BYTES_PER_TOKEN/MAXLEN:13.2f} {ple_pct(kv):10.0f} %   {note}")
print()
print("Expected values: the real KV cache will land within about +/-1 GiB (+/-30k tokens) of these, never above what fits.")
print("The KV cache stays dynamic on purpose: vLLM measures free memory after loading and cannot run out of memory at runtime.")
print("PLE residency does not change decode speed (measured 0 %, 46 %, 90 %: same 57.5 ms/step); it only removes disk I/O from the path.")
PY
