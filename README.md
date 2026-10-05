# Qwen3.8-Flash-Next on a GB10 (vLLM v0.30)

Qwen3.8-Flash-Next (NVFP4) served with vLLM v0.30 on NVIDIA GB10 machines: a patched image,
the vLLM config and helper scripts. Weights and base image are not included; both are public
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
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all ubuntu nvidia-smi    # must show the GPU
```

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
| `docker/vllm_ple_mmap.py` | PLE served via mmap from disk (blazux, Apache-2.0) + zeros during memory profiling |
| `docker/patch_mamba_block_size.py`, `patch_moe_load_clone.py` | prefix-cache block_size and clone-on-load for the MoE (blazux) |
| `docker/fn_dense_fp8.py`, `patch_flashnext.py` | row-wise FP8 for the dense Linears (**+16 % on code**) and NVFP4 draft head (**+13-25 %**) |
| `config/model.yaml` | the vLLM config (no api-key: it comes from `VLLM_API_KEY`) |
| `config/fn_dense_fp8.conf` | which layers go to FP8. **If missing, the patch silently turns off** |
| `compose.yaml` | `flashnext` service (container `vllm-fn`) and `prepare` (separate profile) |

## Performance (DGX Spark)

Measured on a DGX Spark running exactly this config (speed: 2026-10-05; quality: 2026-09-26).
Other idle services (embeddings, ASR) were loaded on the same machine.

**Single request** (512 output tokens, median of 4 runs):

| Workload | t/s | MTP tokens per step |
|---|---|---|
| SQL | 54.5 | 3.17 |
| Code refactor | 55.9 | 3.25 |
| Prose | 50.6 | 2.95 |

**Concurrency** (512 output tokens per request, official sampling, mixed code and prose prompts,
median of 3 runs):

| Concurrent requests | Aggregate t/s | Per-request t/s | Median TTFT |
|---|---|---|---|
| 1 | 53 | 54 | 0.19 s |
| 2 | 75 | 39 | 0.32 s |
| 4 | 94 | 28 | 0.30 s |
| 8 | 139 | 20 | 0.48 s |
| 16 | 146 | 18.5 | 0.37 s |

With 16 requests only ~10 run at once and the rest queue: the KV cache reaches 96 % because each
request takes whole 1,600-token blocks plus its Mamba state. In practice the ceiling is
**~145 t/s aggregate with ~10 concurrent requests**.

**Prefill and prefix cache** (a new prompt, then the same prompt again):

| Prompt tokens | TTFT, new prompt | Prefill t/s | TTFT, repeated |
|---|---|---|---|
| 1.2k | 0.6 s | 2,000 | 0.5 s |
| 9.7k | 3.5 s | 2,740 | 0.7 s |
| 40k | 13.8 s | 2,900 | 1.2 s |
| 81k | 27.9 s | 2,910 | 1.2 s |
| 122k | 42.7 s | 2,860 | 1.3 s |

**Quality:**

| Benchmark | Result |
|---|---|
| HumanEval+ (164 problems) | 95.7 % base / 91.5 % plus |
| τ²-bench retail (74 tasks with deterministic grading, self-play) | 0.784 (58/74) |
| Needle in a haystack | 16/16 (4/4 at 33k, 67k, 100k and 128k tokens) |
| Long-context retrieval (12 facts at 17k and 57k tokens) | 12/12, both cold and with a prefix-cache hit |

**Memory:** ~88 GiB used, KV cache of 239,238 tokens (1.8 full-length 131k requests).

`verify.sh` reports a code-generation figure that includes prefill: if it is well below
~50 t/s, check that the logs show `fn_dense_fp8: 145 layers` and the draft head.

## Measured rules (don't change without re-measuring)

- **Official sampling:** `temperature 0.7, top_p 0.8, presence_penalty 1.5`. At temperature 0 it loops.
- **Never fewer than 10 experts.** Below that quality drops and there is no speed gain.
- **MTP-3** is optimal; MTP-4 ties and uses more memory.
- **The KV cache cannot be fp8**: the QSA indexer requires BF16 (`NotImplementedError`).
- **If memory is short, lower `max-model-len`**, not `gpu-memory-utilization`: below ~0.70 it hangs while loading weights.
- **Put the stable part first in the prompt**: any early change breaks the prefix cache.
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
- If you change `max-model-len` or `gpu-memory-utilization` and startup never finishes, delete `CACHE_DIR`.

## License

Apache-2.0, see [LICENSE](LICENSE). The patches in `docker/vllm_ple_mmap.py`,
`patch_mamba_block_size.py` and `patch_moe_load_clone.py` come from
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX), also Apache-2.0
([docker/LICENSE.blazux](docker/LICENSE.blazux)). The model weights have their own license.
