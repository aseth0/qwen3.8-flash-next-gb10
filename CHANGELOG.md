# Changelog

Releases are git tags; each builds its own image tag (`flashnext-gb10:<tag>`) and uses its own cache
subdirectory, so going back is `git checkout <tag> && docker compose up -d flashnext`.

## v034-rssm-mtpfix — 2026-11-25

**Same image, the MTP drafter's NVFP4 scales made consistent.** `prepare` gains a fourth step, `scripts/fix-mtp-scales.py`,
and `config/model.yaml` serves `qwen3.8-flash-next-nvfp4-hybrid-mtpfix/`. The hybrid is not modified.

Changed
- vLLM keeps one global scale for each fused `gate_proj`/`up_proj` pair, but the NVFP4 MTP experts of
  `starkweatherdigital/qwen3.8-flash-next-nvfp4` carry two different ones in 506 of 512 experts, so the drafter computed
  `up_proj` with the gate's scale (median error 24 %, up to 3.6x). The fix puts both under the larger scale and re-rounds
  the other projection: 2.6 % median relative error against the intended weights, startup warning gone.
- `verify.sh` checks that vLLM no longer warns `w1_weight_scale_2 must match w3_weight_scale_2`.

Measured (on v0.33, same harness as below): MTP tokens per step 2.500 → 2.506 (+0.2 %, 95 % interval −0.5..+1.0), HumanEval+
157 / 150 against 158 / 152 (p = 1.00 / 0.69), τ² 62/74 against 59/74 (+12 / −9, p = 0.66). No speed or quality change; the
drafter can only change speed. It is in the release because it serves the weights the checkpoint meant.

## v034-rssm — 2026-11-25

**vLLM v0.34.0 with RecoverSSM, same checkpoint and config.** Measurements in [BENCHMARKS.md](BENCHMARKS.md). Decode,
concurrency and the prefix cache tie with `v033-rssm`; prefill in cold is 3-16 % faster and startup about 2 minutes shorter.
Quality ties: HumanEval+ 156 / 147, τ² 60/74.

Changed
- Base image `vllm/vllm-openai:v0.34.0`, pinned by digest (PyTorch 2.14, CUDA 13.0). New image tag and cache subdirectory.
- RecoverSSM diff (PR #58863 at 3694757f) rebased onto v0.34: two hunks, because v0.34 adds `KimiK31ForConditionalGeneration`
  to the recoverable architectures and turns the last `else:` of the GDN forward into `elif not prefill_in_place:`. v0.34
  already ships the Kimi half of RecoverSSM (`use_kda_recoverssm`); the Qwen half this model needs is still only in the PR.
- `VLLM_USE_V2_MODEL_RUNNER` removed from `compose.yaml`: v0.34 dropped Model Runner V1 and warns that the variable is redundant.

Removed
- `docker/patch_moe_load_clone.py`: upstream now does the same with `copy_weight_` (`vllm/model_executor/utils.py`), which touches every page
  of a mmap-backed weight on the CPU before the copy on integrated GPUs.

## v033-rssm — 2026-11-04

**vLLM v0.33.0 with RecoverSSM, same checkpoint.** Measurements in [BENCHMARKS.md](BENCHMARKS.md). Throughput ties with
`v030-nvidia`; repeated prefixes answer about twice as fast. Quality ties too: HumanEval+ 158 / 152, τ² 59/74.

Changed
- Base image `vllm/vllm-openai:v0.33.0`, pinned by digest (PyTorch 2.14, CUDA 13.0). New image tag and cache subdirectory.
- RecoverSSM now comes from vLLM PR #58863 at 3694757f (still open upstream), applied as a diff without its Kimi-K3 files.
- Attention block 1696 → 1728 tokens, so `prefix-cache-retention-interval` is 1728. KV cache ~340-360k → ~385-408k tokens at util 0.73.
- `mamba-cache-mode` removed from the config: deprecated in v0.33, it follows from `enable-prefix-caching`.
- `verify.sh` expects 133 dense FP8 layers (145 on v0.30, same 6.22 GiB) and the 1728-token block.

Removed
- `docker/myllmbox/`. Fused draft metadata for the QSA cache (vLLM PR #58449) no longer applies to v0.33 and only gave +1-3 %.
  The QSA RoPE clamp patched a file that no longer exists; the startup captures run without it.
- The worker half of `patch_mamba_block_size.py`: upstream fixed it.

Kept on purpose
- This repo's NVFP4 draft head: vLLM's `draft_lm_head_quantization` needs an unquantized target `lm_head`, and dense FP8 converts it.
- PLE via mmap: vLLM's own NVFP4 table support keeps the 26.8 GiB resident in GPU memory.

Known
- Prefill in cold is 14 % slower at 8k tokens and 5 % at 64k than on `v030-nvidia`.
- PR #58863 reports −15-18 % at 32 concurrent requests with draft-less rows; `max-num-seqs` here is 16.

## v030-nvidia-no-rp — 2026-10-25

Changed
- No server-side `repetition_penalty` any more (2026-10-25). The 1.05 default stopped the tool-call loop at temperature 0,
  but vLLM applies it to every token already seen in prompt and output, and over long texts the model then drops accents
  and letters (3-4 misspellings per 1,600-word essay; 25 in a 2,380-token rewrite; none at 1.0 — see BENCHMARKS.md).
  HumanEval+ with the official sampling and no penalty: 156 / 149 (with it: 156 / 147 and 155 / 149).
  `scripts/build-hybrid.py` leaves it out, `verify.sh` checks that it is absent. Checkpoints built earlier: delete the
  `repetition_penalty` line from `/models/qwen3.8-flash-next-nvfp4-hybrid/generation_config.json` and restart.

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
