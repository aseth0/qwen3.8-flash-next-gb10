#!/usr/bin/env python3
"""Prefix caching (align mode) on hybrid models with a small-block KV group, vLLM v0.33-v0.34. Upstream already fixed the worker's
state-slot seed (it uses mamba_block_size). This keeps the scheduler's prefill split on Mamba block boundaries even if
cache_config.block_size is pulled down by another group: self.block_size is the scheduler block size, and when both match
this changes nothing. Originally diagnosed on GB10 with state checksums (blazux/qwen3.8-Flash-DGX)."""
import ast
import sys

SP = sys.argv[1]
MH = f"{SP}/vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
SC = f"{SP}/vllm/v1/core/sched/scheduler.py"
assert "(new_req_data.num_computed_tokens - 1) // mamba_block_size" in open(MH).read(), "upstream worker fix not found"
t = open(SC).read()
b = "        block_size = self.cache_config.block_size\n        # The last block-aligned position whose state can be cached."
assert t.count(b) == 1, "scheduler split line not found"
t = t.replace(b, "        block_size = self.block_size  # scheduler block size == mamba block size\n"
                 "        # The last block-aligned position whose state can be cached.")
ast.parse(t)
open(SC, "w").write(t)
print("mamba block_size fix (scheduler) applied OK")
