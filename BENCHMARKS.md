# Benchmarks

All on one NVIDIA DGX Spark (GB10, 128 GB), 2026-10-05/06, with an embeddings model and an ASR model
loaded but idle on the same machine. "v030" is the previous release, started the same way as each
candidate (fresh container, same harness) — between two startups of the *same* config, single-request
numbers move by ±5 %, so only consistent or structural differences (KV size, acceptance, concurrency)
are read as real.

**Sampling.** "Official" = `temperature 0.7, top_p 0.8, presence_penalty 1.5`, thinking off; it is what
clients use. Temperature 0 is only used to compare MTP acceptance; **it can mislead**: several candidates
looked faster at temperature 0 and were slower with the official sampling.

## v030-rssm vs v030

| | v030 | **v030-rssm** |
|---|---|---|
| KV cache | ~260-274k tokens | **~294-309k** |
| Attention block / prefix retention | 1600 / 1600 | 1696 / 1696 |
| Single request, official sampling, mixed prompts | 54.0 t/s | 54.4 t/s |
| Single request, official sampling, prose only | 40.4 t/s | 41.1 t/s |
| temp 0: SQL / refactor / prose (MTP tokens per step) | 54.4 / 55.7 / 50.4 (3.17 / 3.25 / 2.95) | 55.7 / 58.5 / 46.8 (3.22 / 3.39 / 2.70) |
| Aggregate, 1 / 2 / 4 / 8 / 16 concurrent | 53 / 75 / 94 / 139 / 146 | 48-54 / 82 / 100 / 144 / **206-212** |
| Prefill 8k / 32k / 64k | 2,314 / 2,800 / 2,876 t/s | **2,725 / 2,955 / 2,975** t/s |
| Prefix cache, fixed 18k prefix + new tail (TTFT of requests A → B → C) | 6.3 → 3.56 → 1.53 s | 6.1 → **1.04** → 1.07 s |
| Prefix cache, conversation growing 2.8k → 18k over 14 turns | flat, median 1.56 s | flat, median 1.53 s |
| HumanEval+ base / plus (164, official sampling) | 157 / 150 | 155 / 150 (McNemar p = 0.73 / 1.00) |
| τ²-bench retail, 74 tasks, self-play | 58/74 | 59/74 (+8 / −7, p = 1.00) |
| Needle in a haystack, 30k / 60k / 90k / 115k | 16/16 | 16/16 |
| Long agentic scenarios (incl. an arithmetic stop condition) | 3/3 | 3/3 (×3 runs) |
| Tool calls (6 cases) | 6/6 | 6/6 |
| Thinking mode, 24 coding tasks, t = 1.0, 8k tokens | 24/24, 0 truncated | 24/24, 0 truncated |
| Vision (5 questions) | 3-4/5 | 4/5 (×3) |
| Typed decisions, 92 (one-letter / choice / JSON) | 85 / 85 / 83 | 86 / 85 / 91 |
| Tool overuse (unneeded calls: default / with "when not to use" descriptions + system policy) | 6.7 % / 0 % | 6.7 % / 0 % |

The temperature-0 prose drop (MTP acceptance 2.95 → 2.70) does **not** show with the official sampling.

**2-hour soak** (3 threads, mixed code / prose / tool calls / thinking, official sampling): 472 requests,
0 errors, 0 empty answers, tool calls 84/84, 0 container restarts, per-request speed stable over the
2 hours. One thinking request out of 146 reached its 6,000-token limit with repetition (possible loop,
not confirmed); the repetition flags on code were false positives (legitimately repetitive tests and
tokenizer branches, checked on saved samples).

## Variants of v030-rssm

| | KV | official c=1 / c=8 | c=16 | long agentic |
|---|---|---|---|---|
| with myllmbox's MTP-depth hooks active (depth pinned at 3) | 309k | 54.1 / 145.5 | 210.5 | 3/3 |
| same image, hooks off | 309k | 54.5 / 144.4 | 210.6 | 3/3 |
| **without the hooks (this repo)** | 294k | 54.1 / 144.3 | 208.9 | 3/3 |

MTP acceptance identical to the hundredth in the three: the hooks do nothing at MTP-3, so they are left out.

## Tried and rejected

Against v030 started the same way (official sampling c=1 54.0 / c=8 137.9 t/s):

| Idea | Result | Why not |
|---|---|---|
| Other NVFP4 MoE backends | marlin ≈ (not deterministic at temp 0), humming ≈ but half the KV, VLLM_CUTLASS slower; flashinfer_cutedsl / trtllm / b12x do not start on sm121 | `auto` (FLASHINFER_CUTLASS) stays |
| Dynamic MTP depth 3-7 (myllmbox) | temp 0 code +16 %, but official sampling −22 % (c=1) / −41 % (c=8) | configuration for K=7 (buffers, block, graphs) |
| Fused draft metadata or block / probabilistic rejection alone | within noise | kept the fused draft (part of this release), no gain alone |
| vLLM v0.31.0 / nightly with sm121 skinny-GEMM plans | within noise; without our row-wise FP8 the step goes 57 → 71 ms | no decode gain over v0.30 |
| v0.31 official FP8 KV for QSA | KV +52 %, same step time, lower MTP acceptance: −7 % c=1, −4 % c=8 | only pays off beyond ~10 long concurrent requests |
| local-inference-lab QAD checkpoint (NVFP4 + MXFP8) | decode +3-5 %, HumanEval+ 152-154, τ² 61/74 | **prefill −25 %** (MXFP8 through Marlin), and with RecoverSSM it fails an arithmetic agentic test every time |
