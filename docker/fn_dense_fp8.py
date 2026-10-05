"""fn_dense_fp8 — FP8 (e4m3, per-row scale) for the dense BF16 Linears of
Qwen3.8-Flash-Next that NVIDIA's NVFP4 checkpoint leaves unquantized.

Why: profiling splits the decode step (~71 ms) into dense BF16 GEMMs ~45 ms, experts 16,
draft head 5. Those GEMMs read 9.3 GiB of BF16 per step at ~207 GB/s: pure memory
bandwidth. The quantization_config ``ignore`` list excludes ``*.self_attn.*``,
``*.linear_attn.*``, ``*hyper_connection*`` and ``lm_head``. Quantizing half of those
bytes to FP8 is the biggest lever left in the model.

How: when the model is built (right after ``enable_qwen38next_low_latency_gemm``) the
``quant_method`` of the selected Linears is replaced with ``FnDenseFp8LinearMethod``.
The Linears still create and load their BF16 weight as usual; in
``process_weights_after_loading`` (called by vLLM after loading) the weight is quantized
to row-wise FP8, ``weight`` (fp8, [N,K]) + ``weight_scale`` ([N] fp32) are registered
and the BF16 is freed. ``apply`` quantizes the activation per token (dynamic,
``ops.scaled_fp8_quant``) and calls ``torch._scaled_mm``.

Selection (at load time, without rebuilding the image): the file
``/root/.cache/vllm/fn_dense_fp8.conf`` (mounted from ``config/fn_dense_fp8.conf``)
holds a comma-separated list of substrings that must appear in the layer ``prefix``,
e.g. ``linear_attn`` or ``linear_attn,self_attn,hyper_connection,lm_head``.
Empty or missing = disabled. Fixed exclusions: ``input_mix_weight_up`` (hyperconnection
reads ``.weight`` directly with ``F.linear``), ``mlp.gate`` (router), ``conv1d``
(3-D weight), ``ple.``, ``visual``, ``mtp.`` (the drafter has its own head patch).

A/B: by restart, because keeping BF16 and FP8 at the same time would cost 9 GiB of KV.
NOTE: torch.compile's AOT cache doesn't see quant_method changes; clear the vLLM cache
(CACHE_DIR) when switching modes.
"""
from __future__ import annotations

import os

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

CONF_PATH = "/root/.cache/vllm/fn_dense_fp8.conf"
FP8_MAX = 448.0
# Layers that are never touched even if they match the selection.
ALWAYS_EXCLUDE = (
    "input_mix_weight_up",  # common/hyperconnection.py:216 uses F.linear(gate, .weight)
    "input_mix_weight_down",  # v0.30: common/hyperconnection.py:213, same F.linear(.weight)
    "block_inject_weight",    # v0.30: common/hyperconnection.py:238
    "mlp.gate",             # MoE router: 0.12 GiB and picks experts, not worth the risk
    "conv1d",               # 3-D weight, not a GEMM
    ".ple.",
    "visual",
    "mtp.",
)


def read_targets(path: str = CONF_PATH) -> list[str]:
    try:
        with open(path) as fh:
            raw = fh.read()
    except FileNotFoundError:
        return []
    return [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]


# ----------------------------------------------------------------------------
# Kernel: per-token FP8 activation + _scaled_mm. Registered as a custom op so
# torch.compile / the CUDA graph treat it as opaque (like the model's
# low_latency_gemm).
# ----------------------------------------------------------------------------
def _fn_dense_fp8_gemm(
    x: torch.Tensor, weight_t: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    # x: [M, K] bf16; weight_t: [K, N] fp8 (transposed view, col-major);
    # weight_scale: [1, N] fp32.
    x2 = x.reshape(-1, x.shape[-1])
    xq, xs = ops.scaled_fp8_quant(x2, None, use_per_token_if_dynamic=True)
    out = torch._scaled_mm(
        xq, weight_t, scale_a=xs, scale_b=weight_scale, out_dtype=x.dtype
    )
    return out.reshape(*x.shape[:-1], weight_t.shape[1])


def _fn_dense_fp8_gemm_fake(
    x: torch.Tensor, weight_t: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight_t.shape[1]))


direct_register_custom_op(
    op_name="fn_dense_fp8_gemm",
    op_func=_fn_dense_fp8_gemm,
    fake_impl=_fn_dense_fp8_gemm_fake,
)


def quantize_rowwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [N,K] -> (fp8 [N,K], fp32 scale [N]). Scale = row_amax / 448."""
    with torch.no_grad():
        row_amax = w.abs().amax(dim=1).float().clamp_min(1e-12)
        scale = row_amax / FP8_MAX
        w8 = (w.float() / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w8, scale


class _FnDenseFp8Apply:
    """Mixin shared by the Linear and the LMHead versions."""

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        w = layer.weight
        if w.dtype == torch.float8_e4m3fn:
            return  # already processed (weight reload)
        assert w.dim() == 2 and w.dtype == torch.bfloat16, (w.shape, w.dtype)
        n, k = w.shape
        assert n % 16 == 0 and k % 16 == 0, (n, k)
        w8, scale = quantize_rowwise(w.data)
        # weight_t is a transposed [K,N] col-major VIEW of the same buffer: it's what
        # _scaled_mm requires for mat2 and it doesn't duplicate memory.
        layer.register_parameter("weight", nn.Parameter(w8, requires_grad=False))
        layer.register_parameter(
            "weight_scale", nn.Parameter(scale.reshape(1, n).contiguous(), requires_grad=False)
        )
        layer._fn_fp8_weight_t = layer.weight.t()
        del w
        self._bytes_saved += n * k  # BF16 (2B) -> FP8 (1B): 1 byte per element

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.ops.vllm.fn_dense_fp8_gemm(x, layer._fn_fp8_weight_t, layer.weight_scale)
        if bias is not None:
            out = out + bias
        return out


class FnDenseFp8LinearMethod(_FnDenseFp8Apply, UnquantizedLinearMethod):
    def __init__(self):
        super().__init__()
        self._bytes_saved = 0


class FnDenseFp8EmbeddingMethod(_FnDenseFp8Apply, UnquantizedEmbeddingMethod):
    """Only for ParallelLMHead (logits projection); embed_tokens is not touched."""

    def __init__(self):
        super().__init__()
        self._bytes_saved = 0

    def embedding(self, layer, input_):  # pragma: no cover
        raise RuntimeError("FnDenseFp8EmbeddingMethod is only for lm_head, not for lookups")


def _selected(prefix: str, targets: list[str]) -> bool:
    if any(x in prefix for x in ALWAYS_EXCLUDE):
        return False
    return any(t in prefix for t in targets)


def enable_fn_dense_fp8(model: nn.Module, targets: list[str] | None = None) -> int:
    """Replaces the quant_method of the selected BF16 Linears. Returns how many."""
    if targets is None:
        targets = read_targets()
    if not targets:
        logger.info("fn_dense_fp8: disabled (no %s)", CONF_PATH)
        return 0
    count = 0
    est = 0
    for name, child in model.named_modules():
        prefix = getattr(child, "prefix", None) or name
        if isinstance(child, LinearBase):
            qm = child.quant_method
            # UnquantizedLinearMethod or its LowLatency subclass (already set by the model).
            if not isinstance(qm, UnquantizedLinearMethod) or isinstance(qm, _FnDenseFp8Apply):
                continue
            new_qm = FnDenseFp8LinearMethod
        elif isinstance(child, ParallelLMHead) and "lm_head" in targets:
            qm = child.quant_method
            if not isinstance(qm, UnquantizedEmbeddingMethod):
                continue
            new_qm = FnDenseFp8EmbeddingMethod
        else:
            continue
        if not _selected(prefix, targets):
            continue
        w = getattr(child, "weight", None)
        if w is None or w.dim() != 2 or w.dtype != torch.bfloat16:
            continue
        if w.shape[0] % 16 or w.shape[1] % 16:
            logger.warning("fn_dense_fp8: skipping %s, shape %s", prefix, tuple(w.shape))
            continue
        child.quant_method = new_qm()
        count += 1
        est += w.numel()
    logger.info(
        "fn_dense_fp8: %d layers -> row-wise FP8 (targets=%s); ~%.2f GiB of BF16 become %.2f GiB",
        count, targets, est * 2 / 2**30, est / 2**30,
    )
    return count
