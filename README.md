# Qwen3.8-Flash-Next on a GB10 (vLLM v0.30 + RecoverSSM)

Qwen3.8-Flash-Next (NVFP4) served with vLLM v0.30 on NVIDIA GB10 machines: a patched image,
the vLLM config and helper scripts. Current release: **`v030-rssm`** (adds RecoverSSM: +45 %
aggregate throughput at 16 concurrent requests, faster prefix-cache hits, same quality);
see [CHANGELOG.md](CHANGELOG.md) and [BENCHMARKS.md](BENCHMARKS.md). Weights and base image are not included; both are public
and are downloaded on the target machine.

**Machines:** any GB10 with 128 GB of unified memory.

| Machine | Status |
|---|---|
| **NVIDIA DGX Spark** | reference: every number in this README was measured on one |
| **ASUS Ascent GX10** | same SoC, memory and software (DGX OS); same config |
| Other GB10 (Dell Pro Max GB10, Lenovo ThinkStation PGX, MSI EdgeXpert…) | should behave the same; not tested |

**Requirements:** aarch64, ~121 GiB of unified memory as reported by `free -g`, Docker with
`nvidia-container-toolkit` and Compose v2 (DGX OS ships all three), and **~235 GB of disk**
(102 GB checkpoint + 96 GB BF16 PLE + 22 GB image). It fits on the 1 TB variants, but check
first: `scripts/check-host.sh` does.

On a **discrete GPU** (RTX 50xx, RTX PRO…) you need ~86 GiB of VRAM **and** ~100 GiB of RAM
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
docker compose run --rm prepare    # once: downloads 102 GB and converts the PLE (idempotent)
docker compose up -d flashnext
scripts/verify.sh                  # waits, checks the patches and benchmarks
```

Endpoint: `http://<host>:8010/v1` (OpenAI-compatible), model `qwen3.8-flash-next`,
`Authorization: Bearer <API_KEY>`.

### Upgrading from `v030`

The weights do not change and the base image is the same (same digest), so there is no `prepare`
and no large download: only a few small layers are built on top of the base you already have.

```bash
git pull
docker compose build               # new tag flashnext-gb10:v030-rssm; the v030 image stays
docker compose up -d flashnext     # recreates vllm-fn: ~7 min of downtime (a normal startup)
scripts/verify.sh                  # now also checks RecoverSSM and the 1696-token block
```

`.env`, clients, port, model name and API key are unchanged. The vLLM cache now lives in
`$CACHE_DIR/v030-rssm`, so the old one is left untouched. **Rollback:**
`git checkout v030 && docker compose up -d flashnext` (its image is still there).

### Why there is a `prepare` step

vLLM v0.30 cannot read the n-gram table (PLE) in the NVFP4 format of the published checkpoint.
`prepare` downloads `starkweatherdigital/qwen3.8-flash-next-nvfp4` at a **pinned revision**
(`1b304e5f`, the validated one) and rewrites those shards in BF16 (exact conversion,
`scripts/convert_ple_bf16.py`). This creates `…-nvfp4-plebf16/`: 32 new shards plus
**relative** symlinks to the rest of the original. **Do not delete the original directory**:
the converted one depends on it. It runs inside the image, so no Python is needed on the host.

**If you already have the checkpoint on another GB10**, copy it instead of downloading:
`prepare` sees it is complete (every shard in the index) and only converts.

```bash
rsync -a --info=progress2 <other-gb10>:/opt/models-vllm/qwen3.8-flash-next-nvfp4/ /opt/models-vllm/qwen3.8-flash-next-nvfp4/
```

## What's inside

| | |
|---|---|
| `docker/Dockerfile` | `vllm/vllm-openai:v0.30.0` **pinned by digest** + patches |
| `docker/myllmbox/overlays-rssm/` | RecoverSSM (vLLM PR #58863, Apache-2.0) ported to v0.30.0 by myllmbox; whole files, base hashes checked |
| `docker/myllmbox/patches/`, `overlays-v030/` | fused multi-step draft metadata for the QSA cache and a QSA RoPE clamp (myllmbox, MIT) |
| `docker/vllm_ple_mmap.py` | PLE served via mmap from disk (blazux, Apache-2.0) + zeros during memory profiling |
| `docker/patch_mamba_block_size.py`, `patch_moe_load_clone.py` | prefix-cache block_size and clone-on-load for the MoE (blazux) |
| `docker/fn_dense_fp8.py`, `patch_flashnext.py` | row-wise FP8 for the dense Linears (**+16 % on code**) and NVFP4 draft head (**+13-25 %**) |
| `config/model.yaml` | the vLLM config (no api-key: it comes from `VLLM_API_KEY`) |
| `config/fn_dense_fp8.conf` | which layers go to FP8. **If missing, the patch silently turns off** |
| `compose.yaml` | `flashnext` service (container `vllm-fn`) and `prepare` (separate profile) |
| `CHANGELOG.md`, `BENCHMARKS.md` | what changed per release, and every measurement behind it |

## Performance (DGX Spark)

Measured on a DGX Spark running exactly this config on 2026-10-06, with other idle services
(embeddings, ASR) loaded on the same machine. Full tables, the previous release side by side and
what was tried and rejected: [BENCHMARKS.md](BENCHMARKS.md).

**Single request, official sampling** (512 output tokens): **54 t/s** on mixed code and prose prompts.
Prose alone (4 prose prompts × 4 runs): 41 t/s. At temperature 0: SQL 55.7, refactor 58.5, prose 46.8 t/s
(MTP tokens per step 3.22 / 3.39 / 2.70).

**Concurrency** (512 output tokens per request, official sampling, mixed prompts, median of 3 runs):

| Concurrent requests | Aggregate t/s | Per-request t/s | Median TTFT |
|---|---|---|---|
| 1 | 48-54 | 49-54 | 0.2 s |
| 2 | 82 | 42 | 0.26 s |
| 4 | 100 | 29 | 0.37 s |
| 8 | 144 | 21 | 0.34 s |
| 16 | **206** | 15 | 0.53 s |

All 16 now run at once (the previous release queued ~6 of them: 146 t/s).

**Prefill and prefix cache** (a new prompt, then the same prompt again):

| Prompt tokens | TTFT, new prompt | Prefill t/s | TTFT, repeated |
|---|---|---|---|
| 7.8k | 2.8 s | 2,725 | 1.0 s |
| 32k | 10.8 s | 2,955 | 1.2 s |
| 65k | 21.8 s | 2,975 | 0.9 s |

Fixed prefix + new tail, without priming (the agent / RAG case): an 18k prefix answers its 2nd
request in **1.0 s** (previous release: 3.6 s). A conversation growing turn by turn (2.8k → 18k
tokens) keeps a flat ~1.5 s TTFT.

**Quality:**

| Benchmark | Result |
|---|---|
| HumanEval+ (164 problems) | 94.5 % base / 91.5 % plus (155 / 150) |
| τ²-bench retail (74 tasks with deterministic grading, self-play) | 0.797 (59/74) |
| Needle in a haystack | 16/16 (4/4 at 30k, 60k, 90k and 115k tokens) |
| Long agentic scenarios (multi-step tool use, 3 scenarios) | 3/3 |
| Thinking mode (24 coding tasks, 8k tokens) | 24/24, no truncation |
| 2-hour soak (472 mixed requests, 3 concurrent) | 0 errors, 0 restarts, no speed drift |

**Memory:** same `gpu-memory-utilization`; KV cache of ~294-309k tokens (2.2-2.4 full-length 131k requests).

`verify.sh` reports a code-generation figure that includes prefill: if it is well below
~50 t/s, check that the logs show `fn_dense_fp8: 145 layers`, the draft head and RecoverSSM.

## Measured rules (don't change without re-measuring)

- **Official sampling:** `temperature 0.7, top_p 0.8, presence_penalty 1.5`. At temperature 0 it loops.
- **Never fewer than 10 experts.** Below that quality drops and there is no speed gain.
- **MTP-3** is optimal; MTP-4 ties and uses more memory.
- **Keep the KV cache in BF16.** Stock v0.30 rejects fp8 (`NotImplementedError`). Patched to allow it, the KV grows
  58 % but decode drops 11 % (lower MTP acceptance), concurrency does not improve (the per-request Mamba state
  dominates) and prefix-cache hits get twice as slow (3,200-token blocks).
- **If memory is short, lower `max-model-len`**, not `gpu-memory-utilization`: below ~0.70 it hangs while loading weights.
- **Put the stable part first in the prompt**: any early change breaks the prefix cache.
- **`prefix-cache-retention-interval: 1696`** (a multiple of the 1696-token block that RecoverSSM brings) makes a
  fixed prefix (system prompt, documents) with a new tail hit from its first reuse. Without it, v0.30 only hits
  from the second reuse. If the block size changes, this must change with it or vLLM refuses to start.
- No reasoning: `chat_template_kwargs: {"enable_thinking": false}`.

## Operations

- **Startup order:** if other GPU services (embeddings, ASR…) run on the same host, start them
  **after** `flashnext` is serving. In parallel, vLLM's memory profiling takes everything.
- **A failed startup does not stop the container:** with `restart: unless-stopped` it loops and
  `docker ps` still says `Up`. Check `docker inspect -f '{{.RestartCount}}' vllm-fn`.
- **First startup takes 10 to 30 min** (5 for weight loading + graphs). The PLE starts almost
  empty in RAM and fills with use: the first long requests are slow, that is expected.
- **`docker stats` is misleading** with unified memory. The real figure:
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.
- If you change `max-model-len` or `gpu-memory-utilization` and startup never finishes, delete `$CACHE_DIR/v030-rssm`.

## License

Apache-2.0, see [LICENSE](LICENSE). The patches in `docker/vllm_ple_mmap.py`,
`patch_mamba_block_size.py` and `patch_moe_load_clone.py` come from
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX), also Apache-2.0
([docker/LICENSE.blazux](docker/LICENSE.blazux)). `docker/myllmbox/` comes from
[myllmbox/qwen38-flash-next-recipe](https://github.com/myllmbox/qwen38-flash-next-recipe): its patches are MIT and
the RecoverSSM overlay is vLLM code (Apache-2.0, vllm-project/vllm PR #58863); see
[docker/myllmbox/LICENSE.myllmbox](docker/myllmbox/LICENSE.myllmbox). The model weights have their own license.
