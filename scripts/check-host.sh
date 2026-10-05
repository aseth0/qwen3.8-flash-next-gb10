#!/usr/bin/env bash
# Pre-flight host check. Changes nothing.
# The config is measured on a GB10 (DGX Spark / ASUS Ascent GX10; 121 GiB of unified memory).
# This decides whether another host can serve it and, if not, why: what usually fails is memory,
# not the architecture.
set -uo pipefail
ok(){ echo -e "  \033[32m✓\033[0m $*"; }; bad(){ echo -e "  \033[31m✗\033[0m $*"; F=1; }; warn(){ echo -e "  \033[33m!\033[0m $*"; W=1; }
info(){ echo "      $*"; }
F=0; W=0
ENV_FILE="$(dirname "$0")/../.env"
MODELS_DIR=${MODELS_DIR:-$(grep -oP '^MODELS_DIR=\K.*' "$ENV_FILE" 2>/dev/null || echo /opt/models-vllm)}

# Requirements measured on a DGX Spark
NEED_GPU=86     # GiB: weights ~75 + KV + graphs (85.5 actual with util 0.715)
NEED_PLE=100    # GiB of RAM for the n-gram table via mmap (96 GB; reaches 76 % resident)
NEED_DISK=210   # GiB: checkpoint 102 + BF16 PLE 96 (+ ~22 of image in /var/lib/docker)

echo "== System"
case "$(uname -m)" in
  aarch64) ok "aarch64 (the reference)" ;;
  x86_64)  ok "x86_64 (the base image is also published for amd64)" ;;
  *)       bad "architecture $(uname -m): the base image only exists for arm64 and amd64" ;;
esac
# Daemon first: if docker info fails (permissions, daemon down) nothing can be said about the
# runtime, and it used to be mistaken for "toolkit missing".
if ! dinfo=$(docker info 2>&1); then
  bad "cannot talk to the docker daemon"
  info "$(grep -m1 -iE 'permission denied|cannot connect|error' <<<"$dinfo")"
  info "is your user in the docker group? (sudo usermod -aG docker \$USER and log in again)"
elif grep -qi 'runtimes.*nvidia' <<<"$dinfo"; then ok "nvidia runtime in docker"
elif command -v nvidia-ctk >/dev/null; then
  bad "nvidia-container-toolkit installed but not registered with docker"
  info "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
else bad "nvidia-container-toolkit missing (see README, Requirements)"; fi
docker compose version >/dev/null 2>&1 && ok "docker compose" || bad "docker compose v2 missing"

echo "== GPU"
if ! command -v nvidia-smi >/dev/null; then
  bad "no nvidia-smi"; echo "  fix the above first"; exit 1
fi
# One line per GPU: name, VRAM (MiB, or [N/A] with unified memory) and compute capability.
mapfile -t GPUS < <(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null)
[ "${#GPUS[@]}" -gt 0 ] || { bad "nvidia-smi sees no GPU"; exit 1; }
name=$(awk -F', *' '{print $1}' <<<"${GPUS[0]}")
vram_mib=$(awk -F', *' '{print $2}' <<<"${GPUS[0]}")
cc=$(awk -F', *' '{print $3}' <<<"${GPUS[0]}")
[ "${#GPUS[@]}" -gt 1 ] && warn "${#GPUS[@]} GPUs: the config uses ONE (no tensor parallelism); checking the first"

case "$cc" in
  12.1) ok "$name, sm121 (GB10: DGX Spark, Ascent GX10…, the reference)" ;;
  12.0) warn "$name, sm120 (desktop Blackwell)"
        info "NVFP4 and the FLA tweak (99 KB of shared memory) should work, but it's untested" ;;
  10.*) warn "$name, sm${cc/./} (datacenter Blackwell): NVFP4 yes, FLA tweaks untested" ;;
  *)    bad "$name, compute capability $cc: the weights are NVFP4 and need Blackwell (10.x/12.x)" ;;
esac

ram=$(free -g | awk '/^Mem:/{print $2}')
echo "== Memory"
if [[ "$vram_mib" =~ ^[0-9]+$ ]]; then
  # Discrete GPU: VRAM and RAM are separate and both must be enough.
  vram=$(( vram_mib / 1024 ))
  warn "discrete GPU: the config is designed for unified memory"
  info "on a GB10 the GPU reads the PLE table (96 GB) straight from RAM; here it would cross PCIe"
  if [ "$vram" -ge "$NEED_GPU" ]; then ok "VRAM ${vram} GiB (~${NEED_GPU} needed)"
  else bad "VRAM ${vram} GiB: weights and KV need ~${NEED_GPU} GiB"
       info "not even lowering max-model-len: the weights alone take ~75 GiB"; fi
  if [ "$ram" -ge "$NEED_PLE" ]; then ok "RAM ${ram} GiB for the PLE table (~${NEED_PLE})"
  else bad "RAM ${ram} GiB: the PLE table (96 GB) doesn't fit and would be served from disk"; fi
else
  # Unified memory: what free reports is what there is for everything.
  if   [ "$ram" -ge 121 ]; then ok "unified memory ${ram} GiB (reference 121)"
  elif [ "$ram" -ge 110 ]; then warn "unified memory ${ram} GiB < 121: if it doesn't start, lower max-model-len in config/model.yaml"
  else bad "unified memory ${ram} GiB: ~121 needed"; fi
fi

echo "== Disk"
# If MODELS_DIR doesn't exist yet, check the closest existing ancestor, which is where it will be created.
d=$MODELS_DIR; while [ ! -d "$d" ]; do d=$(dirname "$d"); done
[ "$d" = "$MODELS_DIR" ] || info "$MODELS_DIR doesn't exist yet (will be created); checking $d"
free_gb=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc 0-9)
if [ "${free_gb:-0}" -ge "$NEED_DISK" ]; then ok "${free_gb} GiB free (~${NEED_DISK} needed + ~22 for the image)"
else bad "${free_gb:-?} GiB free in $d: ~${NEED_DISK} needed for weights + ~22 for the image"
     info "point MODELS_DIR in .env to a disk with more space"; fi

echo
if [ "$F" = 0 ] && [ "$W" = 0 ]; then ok "host is suitable"; exit 0; fi
if [ "$F" = 0 ]; then
  warn "it fits, but this config is not validated on this hardware: measure with scripts/verify.sh"
  info "GB10 reference: ~55 t/s on code; if it's far below, the PLE is the bottleneck"; exit 0
fi
echo "  This host cannot serve the config as is."
if [[ "$vram_mib" =~ ^[0-9]+$ ]] && [ "$(( vram_mib / 1024 ))" -lt "$NEED_GPU" ]; then
  info "Options: use it as a client of a GB10's Flash-Next (http://<host>:8010/v1),"
  info "or serve a model here that fits in ${vram:-?} GiB of VRAM (a different config, not this one)."
fi
exit 1
