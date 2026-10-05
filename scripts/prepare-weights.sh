#!/usr/bin/env bash
# Runs INSIDE the image (docker compose run --rm prepare): uses its hf and its torch.
# 1) downloads the NVFP4 checkpoint at the exact revision that was validated;
# 2) rewrites the PLE in BF16 (v0.30 can't read the NVFP4 PLE). Idempotent: safe to re-run.
set -euo pipefail
REPO=starkweatherdigital/qwen3.8-flash-next-nvfp4
REV=1b304e5f99de0faaf43c3a959f2b4000294bf65c
SRC=/models/qwen3.8-flash-next-nvfp4
DST=/models/qwen3.8-flash-next-nvfp4-plebf16

# Complete = the index exists and so does every shard it names. That way a checkpoint copied from
# another machine is recognised without the marker, and a partial one isn't.
complete(){ python3 - "$SRC" <<'P'
import json, os, sys
d = sys.argv[1]
try: wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
except Exception: sys.exit(1)
missing = sorted(f for f in set(wm.values()) if not os.path.isfile(os.path.join(d, f)) or os.path.getsize(os.path.join(d, f)) == 0)
for f in missing[:5]: print(f"   missing {f}")
sys.exit(1 if missing else 0)
P
}
if [ -f "$SRC/.download-complete" ] || complete; then echo "== weights complete in $SRC"; touch "$SRC/.download-complete"
else
  echo "== downloading $REPO@${REV:0:8} (~102 GB)"
  hf download "$REPO" --revision "$REV" --local-dir "$SRC" || {
    echo "   ✗ download failed (network or HF_TOKEN?). Re-running resumes the download."; exit 1; }
  touch "$SRC/.download-complete"
fi

if [ -f "$DST/.conversion-complete" ]; then echo "== PLE already converted in $DST"
else
  echo "== converting the PLE to BF16 (~96 GB of output)"
  python3 /scripts/convert_ple_bf16.py "$SRC" "$DST"
  touch "$DST/.conversion-complete"
fi
echo "== done: $(du -shL "$DST" | cut -f1) in $DST"
