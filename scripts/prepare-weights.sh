#!/usr/bin/env bash
# Runs INSIDE the image (docker compose run --rm prepare): uses its hf and its torch. Idempotent: safe to re-run.
# 1) downloads nvidia/Qwen3.8-Flash-Next-NVFP4 at the validated revision, without its FP8 MTP+PLE file (50 GB, not used);
# 2) downloads the 36 files of starkweatherdigital/qwen3.8-flash-next-nvfp4 that hold the NVFP4 MTP experts and PLE;
# 3) builds the checkpoint this config serves (scripts/build-hybrid.py: streaming byte copies, ~2 min).
set -euo pipefail
NV_REPO=nvidia/Qwen3.8-Flash-Next-NVFP4;              NV_REV=fc694b54fb0174e0913e6adf86691ef85a4ead47
ST_REPO=starkweatherdigital/qwen3.8-flash-next-nvfp4; ST_REV=1b304e5f99de0faaf43c3a959f2b4000294bf65c
NV=/models/qwen3.8-flash-next-nvidia-nvfp4
ST=/models/qwen3.8-flash-next-nvfp4-parts
DST=/models/qwen3.8-flash-next-nvfp4-hybrid
ST_FILES=(model.safetensors.index.json config.json model-00026.safetensors model-00028.safetensors)
for i in $(seq 98 131); do ST_FILES+=("model-$(printf %05d "$i").safetensors"); done

# complete = every file we need exists and is not empty (so a copy made by hand is recognised without any marker).
have(){ local d=$1; shift; for f in "$@"; do [ -s "$d/$f" ] || { echo "   missing $f"; return 1; }; done; }
nv_needed(){ python3 - "$NV" <<'P'
import json, os, sys
d = sys.argv[1]
try: wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
except Exception: sys.exit(1)
files = {f for f in set(wm.values()) if f != "model-fp8-mtp-ple.safetensors"} | {"config.json", "hf_quant_config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "merges.txt", "vocab.json", "preprocessor_config.json", "video_preprocessor_config.json"}
missing = sorted(f for f in files if not os.path.isfile(os.path.join(d, f)) or os.path.getsize(os.path.join(d, f)) == 0)
for f in missing[:5]: print(f"   missing {f}")
sys.exit(1 if missing else 0)
P
}
if nv_needed; then echo "== NVIDIA checkpoint complete in $NV"
else
  echo "== downloading $NV_REPO@${NV_REV:0:8} (~74 GB; the FP8 MTP+PLE file is skipped)"
  hf download "$NV_REPO" --revision "$NV_REV" --local-dir "$NV" --exclude "model-fp8-mtp-ple.safetensors" || {
    echo "   ✗ download failed (network or HF_TOKEN?). Re-running resumes the download."; exit 1; }
fi
if have "$ST" "${ST_FILES[@]}"; then echo "== NVFP4 MTP + PLE parts complete in $ST"
else
  echo "== downloading 36 files of $ST_REPO@${ST_REV:0:8} (~30 GB)"
  hf download "$ST_REPO" --revision "$ST_REV" --local-dir "$ST" --include "${ST_FILES[@]}" || {
    echo "   ✗ download failed. Re-running resumes the download."; exit 1; }
fi
if [ -f "$DST/.build-complete" ]; then echo "== hybrid already built in $DST"
else
  echo "== building the hybrid checkpoint (~28 GB of output, ~2 min)"
  python3 /scripts/build-hybrid.py "$NV" "$ST" "$DST"
fi
echo "== done: $(du -shL "$DST" | cut -f1) in $DST"
