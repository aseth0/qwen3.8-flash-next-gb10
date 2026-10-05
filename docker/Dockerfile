# Qwen3.8-Flash-Next on vLLM v0.30.0 for GB10 (sm121, aarch64): NVIDIA DGX Spark,
# ASUS Ascent GX10 and other GB10 machines. The base is pinned by digest so the build is
# reproducible.
#
# Patches (in order):
#  - blazux/qwen3.8-Flash-DGX (Apache-2.0, LICENSE.blazux): PLE via mmap, 2 FLA fixes for GB10,
#    prefix-cache block_size, clone-on-load for the MoE.
#  - This repo: the PLE returns zeros only during memory profiling (vllm_ple_mmap.py), row-wise
#    FP8 for the dense Linears (fn_dense_fp8.py, +16 % on code) and NVFP4 draft head (+13-25 %).
ARG BASE=vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
FROM ${BASE}
LABEL qwen38.base=v0.30
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG PLE=${SP}/vllm/models/qwen4_exp/nvidia/ngram_embedding.py
COPY vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py
RUN printf '\n\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen4ExpNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read())"
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem|' ${FLA_UTILS} && grep -q spark-fla-shmem ${FLA_UTILS} \
 && sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps|' ${FLA_CDH} && grep -q spark-fla-warps ${FLA_CDH}
COPY patch_mamba_block_size.py /tmp/p1.py
COPY patch_moe_load_clone.py /tmp/p2.py
RUN python3 /tmp/p1.py ${SP} && python3 /tmp/p2.py ${SP} && rm /tmp/p1.py /tmp/p2.py
COPY fn_dense_fp8.py ${SP}/vllm/models/qwen4_exp/nvidia/fn_dense_fp8.py
COPY patch_flashnext.py /tmp/patch_flashnext.py
RUN python3 /tmp/patch_flashnext.py ${SP} && rm /tmp/patch_flashnext.py
