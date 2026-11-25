# Qwen3.8-Flash-Next on vLLM v0.34.0 for GB10 (sm121, aarch64): NVIDIA DGX Spark,
# ASUS Ascent GX10 and other GB10 machines. The base is pinned by digest so the build is
# reproducible.
#
# Layers (in order):
#  1. RecoverSSM: vllm-project/vllm PR #58863 (Apache-2.0, still open) at its head 3694757f, without its Kimi-K3 files (v0.34
#     ships that half upstream; this model does not use it), with two hunks rebased onto v0.34. Speculative verify replays
#     only the accepted tokens from one read-only SSM checkpoint instead of keeping a state copy per draft position.
#  2. blazux/qwen3.8-Flash-DGX (Apache-2.0, LICENSE.blazux): PLE via mmap (extended here to NVFP4 tables, dequantized on
#     the GPU), 2 FLA fixes for GB10, prefix-cache block_size in the scheduler. Their clone-on-load for the MoE is upstream
#     since v0.34 (copy_weight_ in model_executor/utils.py).
#  3. This repo: the PLE returns zeros only during memory profiling (vllm_ple_mmap.py), row-wise
#     FP8 for the dense Linears (fn_dense_fp8.py, +16 % on code) and NVFP4 draft head (+13-25 %).
ARG BASE=vllm/vllm-openai@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1
FROM ${BASE}
LABEL qwen38.base=v0.34 qwen38.extra="recoverssm-58863 ple-mmap-nvfp4"
ARG SP=/usr/local/lib/python3.12/dist-packages

# --- 1. RecoverSSM. The PR renames use_kda_recoverssm; its Kimi-K3 side is left out, so the rename is applied here ---
COPY recoverssm-pr58863-3694757f.diff /tmp/recoverssm.diff
RUN set -e; cd ${SP} && patch -p1 --batch --forward --no-backup-if-mismatch < /tmp/recoverssm.diff >/dev/null && rm /tmp/recoverssm.diff \
 && grep -rl use_kda_recoverssm ${SP}/vllm --include='*.py' | xargs -r sed -i 's/use_kda_recoverssm/use_recoverssm/g'

# --- 2-3. blazux + this repo ---
ARG PLE=${SP}/vllm/models/qwen4_exp/nvidia/ngram_embedding.py
COPY vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py
RUN printf '\n\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen4ExpNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read())"
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem|' ${FLA_UTILS} && grep -q spark-fla-shmem ${FLA_UTILS} \
 && sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps|' ${FLA_CDH} && grep -q spark-fla-warps ${FLA_CDH}
COPY patch_mamba_block_size.py /tmp/p1.py
RUN python3 /tmp/p1.py ${SP} && rm /tmp/p1.py
COPY fn_dense_fp8.py ${SP}/vllm/models/qwen4_exp/nvidia/fn_dense_fp8.py
COPY patch_flashnext.py /tmp/patch_flashnext.py
RUN python3 /tmp/patch_flashnext.py ${SP} && rm /tmp/patch_flashnext.py

COPY check_image.py /tmp/check_image.py
RUN python3 /tmp/check_image.py && rm /tmp/check_image.py
