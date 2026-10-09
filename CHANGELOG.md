# Changelog

Releases are git tags; each builds its own image tag (`flashnext-gb10:<tag>`) and uses its own cache
subdirectory, so going back is `git checkout <tag> && docker compose up -d flashnext`.

## Unreleased

## v030-nvidia — 2026-10-09

**Official NVIDIA experts, NVFP4 n-gram table in RAM, no conversion step.** Same image base and the same
patches as v030-rssm; the checkpoint changes. Measurements in [BENCHMARKS.md](BENCHMARKS.md).

Added
- `scripts/build-hybrid.py`: builds the served checkpoint from two public checkpoints at pinned revisions —
  `nvidia/Qwen3.8-Flash-Next-NVFP4` (routed experts, all BF16 tensors, tokenizer) and the NVFP4 MTP experts and
  NVFP4 n-gram table (PLE) of `starkweatherdigital/qwen3.8-flash-next-nvfp4`. Streaming byte copies, ~2 min, 28 GB.
- PLE loader (`docker/vllm_ple_mmap.py`): NVFP4 tables (`weight_packed` + per-row FP8 scales + global scale) served
  via mmap and dequantized on the GPU, bit-for-bit equal to the BF16 conversion the previous releases used.
  The table shrinks from 95.6 GiB (BF16) to 26.8 GiB and fits in the page cache of a GB10.
- `scripts/ple-fill.sh`: fills the page cache with the PLE after startup (drops the weight shards' pages first).
- `scripts/plan-memory.sh`: expected KV cache per `gpu-memory-utilization` and page cache left for the PLE, before starting.
- Server-side default sampling in the built checkpoint's `generation_config.json`: temperature 0.7, top_p 0.8, top_k 20,
  `repetition_penalty 1.05`. At temperature 0 the model can loop on tool calls (reproduced 6/6 in one agentic scenario,
  up to the harness cap); the penalty stops it (9 calls, 6/6) with HumanEval+ unchanged (155/149).
- `verify.sh` checks the NVFP4 PLE line and the sampling defaults; `CONTAINER` / `API_KEY` / `PORT` can be overridden.

Changed
- Checkpoint: `/models/qwen3.8-flash-next-nvfp4-hybrid` (built by `prepare`). The old `…-nvfp4-plebf16` directory and the
  BF16 PLE conversion are no longer used; `scripts/convert_ple_bf16.py` removed.
- `prepare` downloads ~74 GB (NVIDIA, without its unused FP8 MTP+PLE file) + ~30 GB (36 files) instead of 102 GB, and
  builds in 2 minutes instead of converting 96 GB for an hour. Disk: ~140 GB instead of ~210.
- `gpu-memory-utilization` 0.715 → **0.73** (covers the ~0.5 GiB the NVFP4 loader keeps in buffers). The KV cache stays
  dynamic on purpose; see README, Memory plan.
- Image tag `flashnext-gb10:v030-rssm` → `flashnext-gb10:v030-nvidia`; vLLM cache in `$CACHE_DIR/v030-nvidia`.
- `check-host.sh`: ~30 GiB of RAM for the PLE (was 100), ~140 GiB of disk (was 210).

Results (DGX Spark, against v030-rssm started the same way)
- Decode step identical (57.4-58.2 ms vs 57.4-57.7); single request 55.4 t/s (54.6); 16 concurrent 205 (212) t/s.
- KV cache 301k → **343k** tokens (util 0.73, dynamic); PLE 0 % → **76-90 %** resident after `ple-fill.sh`.
- HumanEval+ 156/147 (155/147); τ²-bench 63/74 (57/74, +13/−8, p = 0.38); needle 16/16; tools 6/6; long agentic 3/3.
- Measured and documented: PLE residency does not change decode speed (0 %, 46 %, 90 %: same step time).

Fixed
- `check-host.sh` no longer fails when the toolkit is installed but the `nvidia` runtime is not registered
  with Docker: `--gpus` works through the toolkit's prestart hook (or CDI) with the default `runc`.
- README: `nvidia-ctk runtime configure` removed from the toolkit install; it is not needed for `--gpus`.

## v030-rssm — 2026-10-06

**RecoverSSM on top of v030.** Same weights, same base image (same digest), no `prepare` needed.
Measurements in [BENCHMARKS.md](BENCHMARKS.md).

Added
- RecoverSSM (vllm-project/vllm PR #58863, ported to v0.30.0 by myllmbox): speculative verify replays only
  the accepted tokens from one read-only SSM checkpoint instead of keeping a GDN/PLE state copy per draft
  position. `use-replayssm: true`, `mamba-backend: triton`, `VLLM_USE_V2_MODEL_RUNNER=1`.
- Fused multi-step draft metadata for the QSA side cache (code of an upstream vLLM PR, via myllmbox;
  `MBX_FUSED_DRAFT=1`). Removes the "Fused multi-step draft decode is not supported" fallback.
- QSA pre-indexer RoPE-position clamp (myllmbox): avoids an illegal memory access during CUDA-graph warmup.
- `verify.sh` checks that RecoverSSM is active and that the attention block is 1696 tokens.
- `CHANGELOG.md`, `BENCHMARKS.md`.

Changed
- Attention block 1600 → **1696** tokens (vLLM sizes it to the Mamba page, which RecoverSSM enlarges);
  `prefix-cache-retention-interval` 1600 → **1696** (it must be a multiple of the block).
- Image tag `flashnext-gb10:v030` → `flashnext-gb10:v030-rssm`; vLLM cache in `$CACHE_DIR/v030-rssm`.
- `patch_flashnext.py`: the NVFP4 draft head asks for the Marlin workspace per call when the layer has none
  (needed on vLLM ≥ 0.31; no effect on v0.30).

Results (DGX Spark, against v030 started the same way)
- KV cache ~250-274k → **~294-309k** tokens; 16 concurrent requests 146 → **206-212 t/s** aggregate (all 16 fit).
- Prefix cache, fixed prefix + new tail, 18k: 2nd request 3.6 s → **1.0 s** TTFT. Prefill +3-18 %.
- Single-request decode, prose and quality unchanged: HumanEval+ 155/150 (v030 157/150), τ² 59/74 (58/74),
  needle 16/16, long agentic 3/3, thinking 24/24, tool-overuse suite identical.

Not included on purpose
- myllmbox's dynamic MTP depth: at MTP-3 it measured identical (its hooks change nothing), and its
  K=7 configuration was 22-41 % slower with the official sampling.

## v030 — 2026-10-05

- vLLM v0.30.0 pinned by digest; PLE via mmap from a BF16 conversion (blazux); row-wise FP8 for the dense
  Linears; NVFP4 draft head for MTP-3; 10 experts.
- `prefix-cache-retention-interval: 1600`: a fixed prefix with a new tail hits from its first reuse.
