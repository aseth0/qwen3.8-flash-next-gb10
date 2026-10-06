# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp decode GEMM selection on Hopper and Blackwell.

Dispatch follows Kimi-K3 and uses the local ``(N, K)`` shape and token count.
Plans contain measured CUDA graph capture sizes; other token counts use the
standard linear implementation.
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

QWEN4_EXP_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=4),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=4),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=2, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=2, vector_width=4),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
        16: SkinnyGemmConfig(16, 128, 1, k_unroll=4, vector_width=4),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 64, 2, k_unroll=2),
        4: SkinnyGemmConfig(4, 64, 2, k_unroll=2),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 2, k_unroll=4, vector_width=4),
        4: SkinnyGemmConfig(4, 128, 2, k_unroll=4, vector_width=4),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=4),
        16: SkinnyGemmConfig(16, 128, 2, k_unroll=4, vector_width=4),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 4, k_unroll=2),
        2: SkinnyGemmConfig(2, 32, 4, k_unroll=2),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, static_k=10240),
        4: SkinnyGemmConfig(4, 128, 2, static_k=10240),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4),
    },
}

# H200 plans selected by exhaustive CUDA graph replay measurements over
# M={1, 2, 4, 8, 16}. Only points that beat the standard linear implementation
# in both hot-cache and L2-flush measurements are retained; other token counts
# keep the standard implementation and its GEMM heuristics.
QWEN4_EXP_SM90_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # GDN fused QKVZ projection, TP=4.
    (4096, 2560): {
        1: SkinnyGemmConfig(1, 128, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, vector_width=4, static_k=2560),
    },
    # GDN and QSA output projections, TP=4.
    (2560, 1536): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=2, static_k=1536),
        2: SkinnyGemmConfig(2, 128, 4, vector_width=4, static_k=1536),
        4: SkinnyGemmConfig(4, 64, 4, k_unroll=6, vector_width=4),
    },
    # GDN fused B/A projection, TP=4.
    (24, 2560): {
        1: SkinnyGemmConfig(1, 128, 3, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 1, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # QSA fused QKV/gate projection, TP=4.
    (3584, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 4, k_unroll=5),
    },
    # QSA indexer Q/K projection, replicated in a TP=4 deployment.
    (640, 2560): {
        1: SkinnyGemmConfig(1, 256, 2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 2, vector_width=4, static_k=2560),
    },
    # Shared-expert fused gate/up projection, TP=4.
    (320, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=5, vector_width=4),
        4: SkinnyGemmConfig(4, 160, 1, k_unroll=2),
        8: SkinnyGemmConfig(8, 128, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # LM head, TP=4.
    (62080, 2560): {
        1: SkinnyGemmConfig(1, 64, 2, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 2, vector_width=2, static_k=2560),
    },
    # HC merged down/injection projection, replicated in a TP=4 deployment.
    (336, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=5),
        2: SkinnyGemmConfig(2, 256, 3, static_k=10240),
        4: SkinnyGemmConfig(4, 256, 3, static_k=10240),
        8: SkinnyGemmConfig(8, 256, 3, static_k=10240),
    },
    # Final HC down projection, replicated in a TP=4 deployment.
    (320, 10240): {
        1: SkinnyGemmConfig(1, 256, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=10),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=10),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=10),
    },
}


# GB10 (SM12x) plans at TP=1 (myllmbox solo), timed 2026-10-02 on one GB10: CUDA-graph replay, weights rotated past L2,
# each entry beat cuBLAS F.linear by >= 3 % (cuBLAS picks SM80 WMMA kernels for these on GB10). Method after the TP=2
# table in myllmbox/qwen38-flash-next-cluster-recipe#4 (sethforprivacy). A missing M keeps the standard linear path.
QWEN4_EXP_SM12X_TP1_GEMM_PLANS: dict[tuple[int, int], dict[int, SkinnyGemmConfig]] = {
    # QSA fused Q/gate/K/V
    (13312, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 32, 1, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 32, 1, vector_width=4, static_k=2560),
        5: SkinnyGemmConfig(5, 32, 1, vector_width=4, static_k=2560),
        6: SkinnyGemmConfig(6, 32, 1, vector_width=4, static_k=2560),
        7: SkinnyGemmConfig(7, 32, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 32, 1, vector_width=4, static_k=2560),
        10: SkinnyGemmConfig(10, 32, 1, vector_width=4, static_k=2560),
        12: SkinnyGemmConfig(12, 32, 1, vector_width=4, static_k=2560),
        14: SkinnyGemmConfig(14, 32, 1, vector_width=4, static_k=2560),
        15: SkinnyGemmConfig(15, 32, 1, vector_width=4, static_k=2560),
        16: SkinnyGemmConfig(16, 32, 1, vector_width=4, static_k=2560),
    },
    # QSA output
    (2560, 6144): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=4, static_k=6144),
        2: SkinnyGemmConfig(2, 256, 1, vector_width=4, static_k=6144),
        3: SkinnyGemmConfig(3, 128, 1, k_unroll=4, static_k=6144),
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=4, vector_width=4, static_k=6144),
        5: SkinnyGemmConfig(5, 256, 1, k_unroll=4, vector_width=4, static_k=6144),
        6: SkinnyGemmConfig(6, 128, 1, k_unroll=4, static_k=6144),
        7: SkinnyGemmConfig(7, 128, 1, k_unroll=4, static_k=6144),
        8: SkinnyGemmConfig(8, 256, 1, k_unroll=4, vector_width=4, static_k=6144),
        10: SkinnyGemmConfig(10, 128, 1, k_unroll=4, static_k=6144),
        12: SkinnyGemmConfig(12, 256, 1),
        14: SkinnyGemmConfig(14, 128, 2, k_unroll=4),
        15: SkinnyGemmConfig(15, 128, 2, k_unroll=4),
        16: SkinnyGemmConfig(16, 128, 2, k_unroll=4),
    },
    # HC up (replicated)
    (10240, 320): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, vector_width=2, static_k=320),
        2: SkinnyGemmConfig(2, 32, 1, vector_width=2, static_k=320),
        3: SkinnyGemmConfig(3, 32, 1, k_unroll=4, vector_width=2, static_k=320),
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=4, vector_width=2, static_k=320),
        5: SkinnyGemmConfig(5, 32, 1, vector_width=2, static_k=320),
        6: SkinnyGemmConfig(6, 32, 1, k_unroll=2, vector_width=2, static_k=320),
        7: SkinnyGemmConfig(7, 32, 1, k_unroll=4, vector_width=2, static_k=320),
        8: SkinnyGemmConfig(8, 32, 1, k_unroll=4, vector_width=2, static_k=320),
        10: SkinnyGemmConfig(10, 32, 2, vector_width=2, static_k=320),
        12: SkinnyGemmConfig(12, 32, 4, vector_width=2, static_k=320),
        14: SkinnyGemmConfig(14, 32, 4, vector_width=2, static_k=320),
    },
    # router gate (replicated)
    (512, 2560): {
        1: SkinnyGemmConfig(1, 64, 1, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 1, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 128, 1, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, vector_width=4, static_k=2560),
        5: SkinnyGemmConfig(5, 64, 1, static_k=2560),
        6: SkinnyGemmConfig(6, 64, 1, static_k=2560),
        7: SkinnyGemmConfig(7, 128, 1, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, static_k=2560),
        10: SkinnyGemmConfig(10, 128, 1, vector_width=4, static_k=2560),
        12: SkinnyGemmConfig(12, 64, 1, static_k=2560),
        14: SkinnyGemmConfig(14, 64, 1, static_k=2560),
        15: SkinnyGemmConfig(15, 64, 1, static_k=2560),
        16: SkinnyGemmConfig(16, 64, 1, static_k=2560),
    },
    # GDN fused B/A
    (96, 2560): {
        1: SkinnyGemmConfig(1, 128, 4, k_unroll=4, vector_width=4),
        2: SkinnyGemmConfig(2, 128, 4, k_unroll=4, vector_width=4),
        3: SkinnyGemmConfig(3, 64, 1, k_unroll=4, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 1, k_unroll=4, static_k=2560),
        5: SkinnyGemmConfig(5, 64, 1, k_unroll=4, static_k=2560),
        6: SkinnyGemmConfig(6, 128, 4, k_unroll=4, vector_width=4),
        7: SkinnyGemmConfig(7, 64, 1, k_unroll=4, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1),
        10: SkinnyGemmConfig(10, 64, 1, k_unroll=4, static_k=2560),
        12: SkinnyGemmConfig(12, 64, 1, k_unroll=4, static_k=2560),
        14: SkinnyGemmConfig(14, 64, 1, k_unroll=4),
        15: SkinnyGemmConfig(15, 64, 1, k_unroll=4, static_k=2560),
        16: SkinnyGemmConfig(16, 64, 1),
    },
    # MTP fc_embedding / fc_hidden
    (2560, 2560): {
        1: SkinnyGemmConfig(1, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        2: SkinnyGemmConfig(2, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        3: SkinnyGemmConfig(3, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        4: SkinnyGemmConfig(4, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        5: SkinnyGemmConfig(5, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        6: SkinnyGemmConfig(6, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        7: SkinnyGemmConfig(7, 256, 1, k_unroll=4, vector_width=2, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, k_unroll=4, static_k=2560),
        10: SkinnyGemmConfig(10, 64, 1, static_k=2560),
        12: SkinnyGemmConfig(12, 64, 1, k_unroll=4, static_k=2560),
        14: SkinnyGemmConfig(14, 64, 1, k_unroll=4, static_k=2560),
        15: SkinnyGemmConfig(15, 64, 1, k_unroll=4, static_k=2560),
        16: SkinnyGemmConfig(16, 64, 1, k_unroll=4, static_k=2560),
    },
    # HC merged down/injection (replicated)
    (336, 10240): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, static_k=10240),
        3: SkinnyGemmConfig(3, 128, 2, static_k=10240),
        4: SkinnyGemmConfig(4, 128, 1, static_k=10240),
        5: SkinnyGemmConfig(5, 128, 1, static_k=10240),
        6: SkinnyGemmConfig(6, 128, 1, static_k=10240),
        7: SkinnyGemmConfig(7, 128, 2, static_k=10240),
        8: SkinnyGemmConfig(8, 128, 2, static_k=10240),
        10: SkinnyGemmConfig(10, 128, 2, static_k=10240),
        12: SkinnyGemmConfig(12, 64, 2, static_k=10240),
        14: SkinnyGemmConfig(14, 64, 2, k_unroll=4, static_k=10240),
        15: SkinnyGemmConfig(15, 64, 2, static_k=10240),
        16: SkinnyGemmConfig(16, 64, 2, k_unroll=4, static_k=10240),
    },
    # shared-expert gate/up
    (1280, 2560): {
        1: SkinnyGemmConfig(1, 64, 1, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 1, k_unroll=2, static_k=2560),
        3: SkinnyGemmConfig(3, 64, 1, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 1, static_k=2560),
        5: SkinnyGemmConfig(5, 64, 1, k_unroll=2, static_k=2560),
        6: SkinnyGemmConfig(6, 64, 1, static_k=2560),
        7: SkinnyGemmConfig(7, 64, 1, k_unroll=2, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, static_k=2560),
        10: SkinnyGemmConfig(10, 64, 1, k_unroll=2, static_k=2560),
        12: SkinnyGemmConfig(12, 64, 1, k_unroll=2, static_k=2560),
        14: SkinnyGemmConfig(14, 64, 1, k_unroll=2, static_k=2560),
        15: SkinnyGemmConfig(15, 64, 1, static_k=2560),
        16: SkinnyGemmConfig(16, 64, 1, k_unroll=2, static_k=2560),
    },
    # QSA indexer Q/K (replicated)
    (640, 2560): {
        1: SkinnyGemmConfig(1, 32, 1, static_k=2560),
        2: SkinnyGemmConfig(2, 64, 1, static_k=2560),
        3: SkinnyGemmConfig(3, 64, 1, static_k=2560),
        4: SkinnyGemmConfig(4, 64, 1, static_k=2560),
        5: SkinnyGemmConfig(5, 64, 1, static_k=2560),
        6: SkinnyGemmConfig(6, 64, 1, k_unroll=2, static_k=2560),
        7: SkinnyGemmConfig(7, 64, 1, k_unroll=2, static_k=2560),
        8: SkinnyGemmConfig(8, 64, 1, static_k=2560),
        10: SkinnyGemmConfig(10, 64, 1, static_k=2560),
        12: SkinnyGemmConfig(12, 64, 1, k_unroll=2, static_k=2560),
        14: SkinnyGemmConfig(14, 64, 1, k_unroll=2, static_k=2560),
        15: SkinnyGemmConfig(15, 64, 1, k_unroll=2, static_k=2560),
        16: SkinnyGemmConfig(16, 32, 1, k_unroll=2, static_k=2560),
    },
    # shared-expert down
    (2560, 640): {
        1: SkinnyGemmConfig(1, 32, 1, k_unroll=4, vector_width=2, static_k=640),
        2: SkinnyGemmConfig(2, 32, 1, k_unroll=4, vector_width=2, static_k=640),
        3: SkinnyGemmConfig(3, 32, 1, k_unroll=4, vector_width=2, static_k=640),
        4: SkinnyGemmConfig(4, 32, 1, k_unroll=4, vector_width=2, static_k=640),
        5: SkinnyGemmConfig(5, 32, 1, k_unroll=4, vector_width=2, static_k=640),
        6: SkinnyGemmConfig(6, 32, 1, k_unroll=4, vector_width=2, static_k=640),
        7: SkinnyGemmConfig(7, 32, 1, k_unroll=4, vector_width=2, static_k=640),
        8: SkinnyGemmConfig(8, 32, 1, k_unroll=4, vector_width=2, static_k=640),
        10: SkinnyGemmConfig(10, 32, 1, k_unroll=4, vector_width=2, static_k=640),
        12: SkinnyGemmConfig(12, 32, 1, k_unroll=4, vector_width=2, static_k=640),
        14: SkinnyGemmConfig(14, 32, 1, k_unroll=4, vector_width=2, static_k=640),
    },
    # shared-expert sigmoid gate (replicated)
    (1, 2560): {
        1: SkinnyGemmConfig(1, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        2: SkinnyGemmConfig(2, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        3: SkinnyGemmConfig(3, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        4: SkinnyGemmConfig(4, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        5: SkinnyGemmConfig(5, 128, 1, vector_width=4, static_k=2560),
        6: SkinnyGemmConfig(6, 128, 1, vector_width=4, static_k=2560),
        7: SkinnyGemmConfig(7, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        8: SkinnyGemmConfig(8, 128, 1, k_unroll=4, vector_width=4, static_k=2560),
        10: SkinnyGemmConfig(10, 128, 1, vector_width=4, static_k=2560),
        12: SkinnyGemmConfig(12, 128, 1, k_unroll=2, vector_width=4, static_k=2560),
        14: SkinnyGemmConfig(14, 128, 1, vector_width=2, static_k=2560),
        15: SkinnyGemmConfig(15, 128, 1, vector_width=2, static_k=2560),
        16: SkinnyGemmConfig(16, 128, 1, vector_width=4, static_k=2560),
    },
    # final HC down (replicated)
    (320, 10240): {
        1: SkinnyGemmConfig(1, 128, 1, static_k=10240),
        2: SkinnyGemmConfig(2, 128, 1, static_k=10240),
        3: SkinnyGemmConfig(3, 128, 2, static_k=10240),
        4: SkinnyGemmConfig(4, 128, 1, static_k=10240),
        5: SkinnyGemmConfig(5, 128, 1, static_k=10240),
        6: SkinnyGemmConfig(6, 128, 1, static_k=10240),
        7: SkinnyGemmConfig(7, 128, 2, static_k=10240),
        8: SkinnyGemmConfig(8, 128, 2, static_k=10240),
        10: SkinnyGemmConfig(10, 128, 2, static_k=10240),
        12: SkinnyGemmConfig(12, 64, 2, k_unroll=4, static_k=10240),
        14: SkinnyGemmConfig(14, 64, 2, k_unroll=2, static_k=10240),
        15: SkinnyGemmConfig(15, 32, 1, static_k=10240),
        16: SkinnyGemmConfig(16, 64, 2, static_k=10240),
    },
}


def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))


def _is_sm90() -> bool:
    return current_platform.is_device_capability((9, 0))


def _gemm_plans() -> dict[tuple[int, int], dict[int, SkinnyGemmConfig]]:
    if _is_sm103():
        return QWEN4_EXP_GEMM_PLANS
    if _is_sm90():
        return QWEN4_EXP_SM90_GEMM_PLANS
    # GB10 (SM12x): the measured TP=1 table; off by default (end-to-end neutral on solo, A/B 2026-10-02); MBX_SKINNY_GEMM_SM12X=1 enables.
    if current_platform.is_device_capability_family(120) and __import__("os").environ.get("MBX_SKINNY_GEMM_SM12X", "0") == "1":
        return QWEN4_EXP_SM12X_TP1_GEMM_PLANS
    return {}


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        not envs.VLLM_BATCH_INVARIANT
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


class _Qwen4ExpLowLatencyApply:
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.qwen4_exp_low_latency_gemm(x, layer.weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Qwen4ExpLowLatencyLinearMethod(_Qwen4ExpLowLatencyApply, UnquantizedLinearMethod):
    pass


class Qwen4ExpLowLatencyEmbeddingMethod(
    _Qwen4ExpLowLatencyApply, UnquantizedEmbeddingMethod
):
    pass


def _qwen4_exp_low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    plan = _gemm_plans().get((weight.shape[0], weight.shape[1]))
    config = None if plan is None else plan.get(x.shape[0])
    if (
        config is not None
        and _runtime_ok(x, weight)
        and shape_dynamic_skinny_gemm.is_available()
    ):
        return shape_dynamic_skinny_gemm(x, weight, config)
    return torch.nn.functional.linear(x, weight)


def _qwen4_exp_low_latency_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen4_exp_low_latency_gemm",
    op_func=_qwen4_exp_low_latency_gemm,
    fake_impl=_qwen4_exp_low_latency_gemm_fake,
)


def enable_qwen4_exp_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    plans = _gemm_plans()
    if dtype != torch.bfloat16 or not plans:
        return
    if not shape_dynamic_skinny_gemm.is_available():
        return

    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():
        is_linear = (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        )
        is_head = (
            isinstance(child, ParallelLMHead)
            and type(child.quant_method) is UnquantizedEmbeddingMethod
        )
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        plan = plans.get((weight.shape[0], weight.shape[1]))
        if plan is None:
            continue
        if is_linear:
            child.quant_method = Qwen4ExpLowLatencyLinearMethod()
        else:
            child.quant_method = Qwen4ExpLowLatencyEmbeddingMethod()
        warmup_configs.update(plan.values())

    if warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)
    # which bf16 shapes took the skinny path, and which bf16 shapes had no plan (once per model: target and MTP)
    from collections import Counter as _Counter
    from vllm.logger import init_logger as _il
    _hit, _miss = _Counter(), _Counter()
    for c in module.modules():
        w = getattr(c, "weight", None)
        if w is None or getattr(w, "dim", lambda: 0)() != 2 or w.dtype != torch.bfloat16:
            continue
        if isinstance(getattr(c, "quant_method", None), _Qwen4ExpLowLatencyApply):
            _hit[tuple(w.shape)] += 1
        elif isinstance(c, LinearBase) and type(c.quant_method) is UnquantizedLinearMethod:
            _miss[tuple(w.shape)] += 1
    _il(__name__).info("Qwen4Exp low-latency GEMM: %d modules on skinny plans %s · bf16 linears without a plan %s",
                       sum(_hit.values()), dict(_hit), dict(_miss))
