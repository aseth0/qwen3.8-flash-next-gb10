# Changelog

Releases are git tags; each builds its own image tag (`flashnext-gb10:<tag>`) and uses its own cache
subdirectory, so going back is `git checkout <tag> && docker compose up -d flashnext`.

## Unreleased

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
