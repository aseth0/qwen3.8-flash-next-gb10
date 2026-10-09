# Qwen3.8-Flash-Next on a GB10 (vLLM v0.30 + RecoverSSM)

Qwen3.8-Flash-Next (NVFP4) served with vLLM v0.30 on NVIDIA GB10 machines: a patched image,
the vLLM config and helper scripts. Current release: **`v030-nvidia`**: the routed experts come from NVIDIA's
official NVFP4 checkpoint, the n-gram table (PLE) is served in NVFP4 (26.8 GiB, it fits in RAM) and there is no
conversion step any more; see [CHANGELOG.md](CHANGELOG.md) and [BENCHMARKS.md](BENCHMARKS.md). Weights and base
image are not included; both are public and are downloaded on the target machine.

**Machines:** any GB10 with 128 GB of unified memory.

| Machine | Status |
|---|---|
| **NVIDIA DGX Spark** | reference: every number in this README was measured on one |
| **ASUS Ascent GX10** | same SoC, memory and software (DGX OS); same config |
| Other GB10 (Dell Pro Max GB10, Lenovo ThinkStation PGX, MSI EdgeXpert…) | should behave the same; not tested |

**Requirements:** aarch64, ~121 GiB of unified memory as reported by `free -g`, Docker with
`nvidia-container-toolkit` and Compose v2 (DGX OS ships all three), and **~160 GB of disk**
(74 GB NVIDIA checkpoint + 30 GB of NVFP4 parts + 28 GB built + 22 GB image). It fits on the 1 TB
variants, but check first: `scripts/check-host.sh` does.

On a **discrete GPU** (RTX 50xx, RTX PRO…) you need ~89 GiB of VRAM **and** ~30 GiB of RAM
for the PLE table; below that no tuning will help. `scripts/check-host.sh` checks and explains
this. This config is only validated on GB10.

**nvidia-container-toolkit** (if `check-host.sh` reports it missing; Ubuntu/Debian):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi    # must show the GPU
```

The `nvidia` runtime does not need to be registered with Docker (`Default Runtime: runc` is fine):
`--gpus` / `gpus: all` uses the toolkit's prestart hook, which mounts the GPU, the driver libraries
and `nvidia-smi` into the container. What matters is that the command above shows the GPU.

## Install

```bash
git clone https://github.com/aseth0/qwen3.8-flash-next-gb10.git && cd qwen3.8-flash-next-gb10
cp .env.example .env               # fill in API_KEY and MODELS_DIR
scripts/check-host.sh              # checks the host, changes nothing
docker compose build               # ~22 GB, almost all of it is the vLLM base
docker compose run --rm prepare    # once: downloads ~104 GB and builds the checkpoint (idempotent, ~2 min of build)
docker compose up -d flashnext
scripts/verify.sh                  # waits, checks the patches and benchmarks
scripts/ple-fill.sh                # optional: puts the n-gram table in the page cache (see Memory plan)
```

Endpoint: `http://<host>:8010/v1` (OpenAI-compatible), model `qwen3.8-flash-next`,
`Authorization: Bearer <API_KEY>`.

### Upgrading from `v030-rssm` or `v030`

The checkpoint changes, so `prepare` runs again: it downloads ~104 GB (NVIDIA's checkpoint without its unused 50 GB
file, plus 36 files of the other) and builds the served directory in ~2 minutes. The previous
`qwen3.8-flash-next-nvfp4-plebf16/` (96 GB) and the original `qwen3.8-flash-next-nvfp4/` can be deleted afterwards
if you do not plan to roll back. The base image is the same (same digest): the build adds a few small layers.

```bash
git pull
docker compose build               # new tag flashnext-gb10:v030-nvidia; the previous image stays
docker compose run --rm prepare    # downloads and builds; idempotent
docker compose up -d flashnext     # recreates vllm-fn: ~7 min of downtime (a normal startup)
scripts/verify.sh
```

`.env`, clients, port, model name and API key are unchanged. The vLLM cache now lives in `$CACHE_DIR/v030-nvidia`.
**Rollback:** `git checkout v030-rssm && docker compose up -d flashnext` (its image and weights are still there).

### What `prepare` does

vLLM v0.30 reads NVIDIA's checkpoint as published, but two of its pieces cost memory on a GB10: the MTP drafter's
experts are block-FP8 (loaded through DeepGEMM: +3.2 GiB of GPU memory and +1.3 ms per decode step) and the n-gram
table (PLE) is FP8 (47.7 GiB). `prepare` downloads `nvidia/Qwen3.8-Flash-Next-NVFP4` (pinned revision, skipping that
file) and the 36 files of `starkweatherdigital/qwen3.8-flash-next-nvfp4` (pinned revision) that hold the same two
pieces in NVFP4, and `scripts/build-hybrid.py` assembles `qwen3.8-flash-next-nvfp4-hybrid/`: symlinks to NVIDIA's shards,
two new files with the NVFP4 MTP experts (1.3 GB) and the NVFP4 PLE (26.8 GB), a regenerated index and configs. Streaming
byte copies: no RAM, ~2 minutes. The PLE loader in the image reads the NVFP4 table straight from those files via mmap and
dequantizes rows on the GPU, bit-for-bit equal to the BF16 conversion the previous releases used.

## What's inside

| | |
|---|---|
| `docker/Dockerfile` | `vllm/vllm-openai:v0.30.0` **pinned by digest** + patches |
| `docker/myllmbox/overlays-rssm/` | RecoverSSM (vLLM PR #58863, Apache-2.0) ported to v0.30.0 by myllmbox; whole files, base hashes checked |
| `docker/myllmbox/patches/`, `overlays-v030/` | fused multi-step draft metadata for the QSA cache and a QSA RoPE clamp (myllmbox, MIT) |
| `docker/vllm_ple_mmap.py` | PLE served via mmap from disk (blazux, Apache-2.0) + zeros during memory profiling + NVFP4 tables dequantized on the GPU |
| `docker/patch_mamba_block_size.py`, `patch_moe_load_clone.py` | prefix-cache block_size and clone-on-load for the MoE (blazux) |
| `docker/fn_dense_fp8.py`, `patch_flashnext.py` | row-wise FP8 for the dense Linears (**+16 % on code**) and NVFP4 draft head (**+13-25 %**) |
| `config/model.yaml` | the vLLM config (no api-key: it comes from `VLLM_API_KEY`) |
| `config/fn_dense_fp8.conf` | which layers go to FP8. **If missing, the patch silently turns off** |
| `scripts/build-hybrid.py`, `prepare-weights.sh` | download the two checkpoints at pinned revisions and build the served directory |
| `scripts/plan-memory.sh`, `ple-fill.sh` | memory arithmetic before starting; put the PLE in the page cache after starting |
| `compose.yaml` | `flashnext` service (container `vllm-fn`) and `prepare` (separate profile) |
| `CHANGELOG.md`, `BENCHMARKS.md` | what changed per release, and every measurement behind it |

## Performance (DGX Spark)

Measured on a DGX Spark running this config on 2026-10-09, with an embeddings model loaded but idle on the same
machine. Full tables, the previous release side by side and what was tried and rejected: [BENCHMARKS.md](BENCHMARKS.md).

**Single request, official sampling** (512 output tokens): **55 t/s** on mixed code and prose prompts. At temperature 0:
SQL 57.0, refactor 54.0, prose 45.9 t/s (MTP tokens per step 3.28 / 3.14 / 2.63); the decode step is 57.4-58.2 ms, the same
as the previous release.

**Concurrency** (512 output tokens per request, official sampling, mixed prompts, median of 3 runs; measured on the same
experts with the FP8 PLE, the NVFP4 PLE does not change it):

| Concurrent requests | Aggregate t/s | Per-request t/s |
|---|---|---|
| 1 | 53-55 | 53-55 |
| 2 | 77 | 38 |
| 4 | 103 | 26 |
| 8 | 140-142 | 18 |
| 16 | **205** | 13 |

The server-side `repetition_penalty 1.05` default costs about 5 % of aggregate throughput at 8 concurrent requests and
nothing at 1; see BENCHMARKS.md for why it is there.

**Prefix cache** (fixed prefix + new tail, without priming): a 7.4k prefix answers its 2nd request in 0.94 s (first: 2.64 s),
an 18k prefix in 1.04 s (first: 6.13 s). Same as the previous release.

**Quality:**

| Benchmark | Result |
|---|---|
| HumanEval+ (164 problems) | 95.1 % base / 89.6 % plus (156 / 147) |
| τ²-bench retail (74 tasks with deterministic grading, self-play) | 60-63/74 (0.81-0.85; previous release 57-59/74) |
| Needle in a haystack | 16/16 (4/4 at 30k, 60k, 90k and 115k tokens) |
| Long agentic scenarios (multi-step tool use, 3 scenarios) | 3/3 |
| Tool calls (6 cases) | 6/6 |

**Memory:** `gpu-memory-utilization 0.73`, KV cache of ~340-360k tokens (2.6-2.7 full-length 131k requests), PLE table 76-90 %
resident in the page cache after `scripts/ple-fill.sh`.

`verify.sh` reports a code-generation figure that includes prefill: if it is well below ~50 t/s, check that the logs show
`fn_dense_fp8: 145 layers`, the draft head, RecoverSSM and the NVFP4 PLE line.

## Measured rules (don't change without re-measuring)

- **Official sampling:** `temperature 0.7, top_p 0.8, presence_penalty 1.5`. At temperature 0 it loops; the built checkpoint's
  `generation_config.json` therefore gives every client that sends no sampling parameters temperature 0.7 / top_p 0.8 / top_k 20
  and a `repetition_penalty` of 1.05, which also stops the loop for clients that insist on temperature 0 (see BENCHMARKS.md).
- **Never fewer than 10 experts.** Below that quality drops and there is no speed gain.
- **MTP-3** is optimal; MTP-4 ties and uses more memory.
- **Keep the KV cache in BF16.** Stock v0.30 rejects fp8 (`NotImplementedError`). Patched to allow it, the KV grows
  58 % but decode drops 11 % (lower MTP acceptance), concurrency does not improve (the per-request Mamba state
  dominates) and prefix-cache hits get twice as slow (3,200-token blocks).
- **If memory is short, lower `max-model-len`**, not `gpu-memory-utilization`: below ~0.70 it hangs while loading weights.
  Never pin the KV cache with `kv-cache-memory` (see Memory plan).
- **Put the stable part first in the prompt**: any early change breaks the prefix cache.
- **`prefix-cache-retention-interval: 1696`** (a multiple of the 1696-token block that RecoverSSM brings) makes a
  fixed prefix (system prompt, documents) with a new tail hit from its first reuse. Without it, v0.30 only hits
  from the second reuse. If the block size changes, this must change with it or vLLM refuses to start.
- No reasoning: `chat_template_kwargs: {"enable_thinking": false}`.

## Memory plan: KV cache vs. PLE page cache

The KV cache is sized **dynamically** (`gpu-memory-utilization` in `config/model.yaml`): vLLM measures what is free after
loading the weights and gives the rest to the KV cache, so it cannot overshoot. On a GB10 the GPU and the host share one memory
pool and that measurement includes the page cache the load itself just filled, which is why the KV cache varies by about
±1 GiB (±30k tokens) from one start to the next with the same config. That variation is the price of never running out of
memory, and this config pays it on purpose: vLLM also has a pinned mode (`kv-cache-memory`) that skips the profiling, and
with it a host that ends up with less memory than planned fails with an out-of-memory at runtime. **Not used here, on any host.**

`scripts/plan-memory.sh` does the arithmetic before the first start (or before changing `max-model-len`, the services on the
host or the utilization): expected KV cache for the current utilization, the utilization that would give a target KV cache,
and how much page cache is left for the n-gram table (PLE). It changes nothing.

```bash
scripts/plan-memory.sh                 # uses MODELS_DIR from .env and the model in config/model.yaml
scripts/plan-memory.sh --kv 9          # which gpu-memory-utilization gives ~9 GiB of KV cache on this host
```

Measured on a DGX Spark: ≈31 KB of KV cache per token at `max-model-len` 131072 (MTP-3, RecoverSSM), so 1 GiB ≈ 34k tokens.
Whatever is left after weights, KV cache and ~6 GiB for the OS is page cache for the PLE table. `scripts/ple-fill.sh` fills it
right after startup (the loader's own `VLLM_PLE_MMAP_PREWARM` runs before the weights are loaded and is undone by them). PLE
residency does **not** change decode speed (measured at 0 %, 46 % and 90 %: the same 57.5 ms per step); it only takes the NVMe
out of the inference path and protects TTFT from other disk activity.

## Operations

- **Startup order:** if other GPU services (embeddings, ASR…) run on the same host, start them
  **after** `flashnext` is serving. In parallel, vLLM's memory profiling takes everything.
- **A failed startup does not stop the container:** with `restart: unless-stopped` it loops and
  `docker ps` still says `Up`. Check `docker inspect -f '{{.RestartCount}}' vllm-fn`.
- **First startup takes 10 to 30 min** (5 for weight loading + graphs). The PLE table starts almost empty in RAM;
  `scripts/ple-fill.sh` puts it there in ~30 s (it does not change decode speed, it takes the disk out of the path).
- **`docker stats` is misleading** with unified memory. The real figure:
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
- If you change `max-model-len` or `gpu-memory-utilization` and startup never finishes, delete `$CACHE_DIR/v030-nvidia`.

## License

Apache-2.0, see [LICENSE](LICENSE). The served checkpoint combines weights under two licenses: the routed experts and
all BF16 tensors are NVIDIA's `Qwen3.8-Flash-Next-NVFP4` (NVIDIA Open Model License; its card allows commercial use), the
NVFP4 MTP experts and n-gram table come from `starkweatherdigital/qwen3.8-flash-next-nvfp4` (Qwen Community License 1.0,
inherited from Qwen/Qwen3.8-Flash-Next). The patches in `docker/vllm_ple_mmap.py`,
`patch_mamba_block_size.py` and `patch_moe_load_clone.py` come from
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX), also Apache-2.0
([docker/LICENSE.blazux](docker/LICENSE.blazux)). `docker/myllmbox/` comes from
[myllmbox/qwen38-flash-next-recipe](https://github.com/myllmbox/qwen38-flash-next-recipe): its patches are MIT and
the RecoverSSM overlay is vLLM code (Apache-2.0, vllm-project/vllm PR #58863); see
[docker/myllmbox/LICENSE.myllmbox](docker/myllmbox/LICENSE.myllmbox). The model weights have their own license.
