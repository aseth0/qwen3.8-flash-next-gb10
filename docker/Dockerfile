# Qwen3.8-Flash-Next on vLLM v0.30.0 for GB10 (sm121, aarch64): NVIDIA DGX Spark,
# ASUS Ascent GX10 and other GB10 machines. The base is pinned by digest so the build is
# reproducible.
#
# Layers (in order):
#  1. RecoverSSM (vllm-project/vllm PR #58863, Apache-2.0), as ported to v0.30.0 by myllmbox
#     (github.com/myllmbox/qwen38-flash-next-recipe, MIT; see myllmbox/LICENSE.myllmbox). It replaces
#     whole vLLM files, so it goes FIRST, onto the pristine base; their hashes are checked.
#     Speculative verify replays only the accepted tokens from one read-only SSM checkpoint instead of
#     keeping a state copy per draft position: more KV in the same memory (~309k vs ~250-274k tokens).
#  2. blazux/qwen3.8-Flash-DGX (Apache-2.0, LICENSE.blazux): PLE via mmap (extended here to NVFP4 tables, dequantized on
#     the GPU), 2 FLA fixes for GB10, prefix-cache block_size, clone-on-load for the MoE.
#  3. This repo: the PLE returns zeros only during memory profiling (vllm_ple_mmap.py), row-wise
#     FP8 for the dense Linears (fn_dense_fp8.py, +16 % on code) and NVFP4 draft head (+13-25 %).
#  4. myllmbox (MIT): fused multi-step draft metadata for the QSA side cache (the code of the upstream
#     PR, gated by MBX_FUSED_DRAFT) and a RoPE-position clamp for the QSA pre-indexer (CUDA-graph warmup).
#     Their dynamic-MTP-depth patch is deliberately NOT included (measured: no effect at MTP-3).
ARG BASE=vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
FROM ${BASE}
LABEL qwen38.base=v0.30 qwen38.extra="recoverssm-58863 qsa-fused-draft qsa-rope-clamp ple-mmap-nvfp4"
ARG SP=/usr/local/lib/python3.12/dist-packages

# --- 1. RecoverSSM: whole-file overlay onto the verified v0.30.0 base ---
COPY myllmbox/overlays-rssm/ /opt/mbx/overlays-rssm/
RUN set -e; cd ${SP}/vllm && sha256sum -c --quiet /opt/mbx/overlays-rssm/base-0.30.0.sha256 \
 && cp -r /opt/mbx/overlays-rssm/vllm/. ${SP}/vllm/ && rm -rf /opt/mbx/overlays-rssm

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
COPY patch_moe_load_clone.py /tmp/p2.py
RUN python3 /tmp/p1.py ${SP} && python3 /tmp/p2.py ${SP} && rm /tmp/p1.py /tmp/p2.py
COPY fn_dense_fp8.py ${SP}/vllm/models/qwen4_exp/nvidia/fn_dense_fp8.py
COPY patch_flashnext.py /tmp/patch_flashnext.py
RUN python3 /tmp/patch_flashnext.py ${SP} && rm /tmp/patch_flashnext.py

# --- 4. myllmbox QSA patches (anchors/hashes asserted by the scripts) ---
COPY myllmbox/patches/ /opt/mbx/patches/
COPY myllmbox/overlays-v030/ /opt/mbx/overlays-v030/
RUN set -e; for p in 03-qsa-pre-indexer-rope-clamp 05-qsa-fused-draft-v2; do python3 /opt/mbx/patches/$p.py; done \
 && python3 -c "import inspect; \
import vllm.model_executor.layers.mamba.gdn.recoverssm_gdn; from vllm.config.cache import CacheConfig; \
assert 'use_recoverssm' in CacheConfig.__dataclass_fields__; \
import vllm.models.qwen4_exp.nvidia.mtp as m; assert '_fn_build_quantized_heads' in inspect.getsource(m); \
import vllm.models.qwen4_exp.nvidia.model as mm; assert 'enable_fn_dense_fp8' in inspect.getsource(mm) and 'SupportsReplaySSM' in inspect.getsource(mm); \
print('flashnext-gb10 v030-nvidia OK')"
