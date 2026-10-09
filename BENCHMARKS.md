# Benchmarks

All on one NVIDIA DGX Spark (GB10, 128 GB), 2026-10-05/06, with an embeddings model and an ASR model
loaded but idle on the same machine. "v030" is the previous release, started the same way as each
candidate (fresh container, same harness) — between two startups of the *same* config, single-request
numbers move by ±5 %, so only consistent or structural differences (KV size, acceptance, concurrency)
are read as real.

**Sampling.** "Official" = `temperature 0.7, top_p 0.8, presence_penalty 1.5`, thinking off; it is what
clients use. Temperature 0 is only used to compare MTP acceptance; **it can mislead**: several candidates
looked faster at temperature 0 and were slower with the official sampling.

## v030-nvidia vs v030-rssm

All on one DGX Spark, 2026-10-09, same harness as below. "v030-rssm" is the previous release measured on 2026-10-05/06
(it serves the third-party NVFP4 checkpoint with its PLE converted to BF16). The candidates were started fresh, with an
embeddings model loaded but idle; the ASR model and a small auxiliary model that were running on 10-05/06 were stopped
during the day, which is why the page cache figures differ. Between two startups of the same config single-request
numbers move by ±5 % and the KV cache by ±1 GiB.

### The checkpoint, step by step

| | v030-rssm | NVIDIA checkpoint as published | + NVFP4 MTP experts | **+ NVFP4 PLE (v030-nvidia)** |
|---|---|---|---|---|
| Routed experts | third-party NVFP4 (ModelOpt) | NVIDIA NVFP4 (ModelOpt, MSE-calibrated) | NVIDIA | NVIDIA |
| MTP drafter experts | NVFP4 | block-FP8 → DeepGEMM | NVFP4 | NVFP4 |
| PLE (n-gram table) | 95.6 GiB BF16, mmap | 47.7 GiB FP8, mmap | 47.7 GiB FP8 | **26.8 GiB NVFP4**, mmap + GPU dequant |
| Weights + non-torch on the GPU | 75.8 GiB | 79.0 | 76.7 | 76.5-77.1 |
| KV cache (util 0.715, dynamic) | 301k | 211k | 277-285k | 259k; **343k at util 0.73** |
| Decode step, temp 0 | 57.4-57.7 ms | 58.2-59.2 | 57.4-57.8 | 57.4-58.2 |
| temp 0: SQL / refactor / prose | 56.1 / 58.8 / 47.0 | 58.7 / 55.4 / 49.0 | 56.0 / 56.1 / 46.9 | 57.0 / 54.0 / 45.9 |
| Single request, official sampling | 54.6 t/s | 55.3 | 53.8 | 55.4 |
| 8 concurrent, official sampling | 144 t/s | 143 | 148 | 140 (with the repetition penalty, see below) |
| HumanEval+ base / plus | 155 / 147 | 155 / 148 | 154 / 145 | **156 / 147** |
| Tools 6 / needle 16 | 6/6, 16/16 | 6/6, 16/16 | 6/6, 8/8 (60k, 115k) | 6/6, 8/8 (60k, 115k) |

The published NVIDIA checkpoint costs 3.2 GiB and 1.3 ms per step for its block-FP8 drafter on vLLM v0.30; grafting
the NVFP4 MTP experts removes both. The NVFP4 PLE is dequantized on the GPU bit-for-bit to the values the BF16
conversion held, so its quality is the previous release's; its size is what lets it sit in RAM.

### Full battery on the NVIDIA experts (same experts as v030-nvidia, FP8 PLE)

| | v030-rssm | NVIDIA experts |
|---|---|---|
| Aggregate, 1 / 2 / 4 / 8 / 16 concurrent | 53 / 74 / 96 / 147 / 212 | 53 / 77 / 103 / 142 / 205 |
| Prefix cache, 7.4k and 18k prefix + new tail (TTFT A → B → C) | 2.75 → 0.96 → 0.92 s · 6.14 → 1.14 → 1.03 s | 2.64 → 0.94 → 0.92 s · 6.13 → 1.04 → 1.04 s |
| HumanEval+ base / plus | 155 / 147 | 155 / 148 (p = 0.69 vs v030) |
| τ²-bench retail, 74 tasks, self-play | 57/74 | **63/74** (+13 / −8 vs v030's 58, p = 0.38) |
| Needle 30k / 60k / 90k / 115k | 16/16 | 16/16 |
| Long agentic scenarios | 3/3, 44 tool calls | 3/3, 74-75 tool calls (see "Tool-call loop") |
| Vision (5 questions, 4 runs) | 4/5 | 3, 4, 4, 3 / 5 |

### The release image, end to end

The `v030-nvidia` image and the directory built by `prepare` (not the hand-assembled candidates above), started as
`compose.yaml` starts them, util 0.73: KV cache 357k tokens; temp 0 SQL / refactor / prose 57.0 / 54.0 / 45.9 t/s;
official sampling 53.4 t/s at 1 request, 139 at 8; 1 / 2 / 4 / 8 / 16 concurrent 53 / 77 / 103 / 140 / 201 t/s;
HumanEval+ 155 / 147; tools 6/6; needle 16/16; long agentic 3/3 in 43 tool calls (the loop is gone with the server
default); prefix cache 2.63 → 0.95 s and 6.05 → 1.04 s; vision 3/5; **τ²-bench 60/74** (+9 / −7 vs v030's 58, p = 0.80).
τ² with the repetition penalty (60) and without it (63, previous section) differ by less than the run-to-run spread of
this benchmark (the previous release scored 57, 58 and 59 in three runs).

### Tool-call loop at temperature 0, and the repetition penalty

With the NVIDIA experts, one long agentic scenario (edit a set of config files, verify, stop) enters a verify-rewrite loop
at temperature 0 until the harness cap of 40 calls — 6 runs out of 6 when it runs right after another scenario on the same
server, 0 out of 3 in isolation; the previous release never did it in 5 runs. The result is still correct; the calls are wasted.
Greedy decoding is the trigger (the model's known rule: at temperature 0 it loops), and any repetition penalty breaks it:

| sampling | calls in the scenario | HumanEval+ |
|---|---|---|
| temperature 0 | 40, 40, 40, 41, 40, 41 | — |
| official (0.7 / 0.8 / presence 1.5) | 11, 9, 9 | — |
| temperature 0 + `repetition_penalty 1.05` | 9, 9, 9 (full test: 43 calls, previous release 44) | **155 / 149** |
| temperature 0 + `repetition_penalty 1.02` | 9, 9 | — |

Of the official parameters only `repetition_penalty` can be a server default (`generation_config.json`; vLLM does not read
`presence_penalty` from it), and OpenAI-style clients never send it, so it applies even to clients that ask for temperature 0.
Cost, A/B on the same server: none at 1 request; **−5 % aggregate at 8 concurrent** with 1.05 (158 → 150 t/s), −3 % with 1.02.
The built checkpoint ships 1.05, the value validated with HumanEval+.

### PLE residency does not change decode speed

| PLE | resident | decode step (temp 0) | 1 request | 8 concurrent |
|---|---|---|---|---|
| 47.7 GiB FP8 | 0-11 % | 57.4-57.8 ms | 53.8 | 147.5 |
| 47.7 GiB FP8 | 46 % (after `ple-fill`) | 57.2-57.8 ms | 53.2 | 134.0 (repetition penalty on) |
| 26.8 GiB NVFP4 | 90 % (after `ple-fill`) | 57.4-58.2 ms | 55.4 | 140.0 (repetition penalty on) |

The per-step gather of PLE rows takes 10-14 ms of CPU whether the table is on disk or in RAM (it is NumPy indexing, not
I/O) and runs overlapped with the GPU; the NVFP4 dequantization adds 0.3-0.6 ms. What residency buys is taking the NVMe
out of the inference path (and protecting TTFT from other disk activity), not speed. `VLLM_PLE_MMAP_PREWARM=1` does not
achieve it: it reads the table before the weights, and loading 75 GiB of weights evicts it (0.1 % resident at startup) while
the weight shards' own pages stay in the cache (25 GiB, useless). `scripts/ple-fill.sh` drops those and reads the table.

### KV cache sizing

| | KV cache |
|---|---|
| dynamic, util 0.715, three starts of the same config | 259k, 277k, 285k |
| pinned 9.0 GiB (`kv-cache-memory`) | 312k, deterministic — **not used**: skips profiling, can end in a runtime OOM |
| dynamic, util 0.73 | 343k |

## v030-rssm vs v030

| | v030 | **v030-rssm** |
|---|---|---|
| KV cache | ~250-274k tokens | **~294-309k** |
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
