#!/usr/bin/env python3
"""Applies this repo's two patches to vLLM v0.30 (qwen4_exp):

  (1) Row-wise FP8 for the dense BF16 Linears (fn_dense_fp8.py): hook at the end of the
      Qwen4ExpForCausalLM constructor.
  (2) NVFP4 lm_head ONLY for the MTP drafter: the MTP's compute_logits uses an
      NVFP4-marlin copy of the lm_head; verification still uses the base model's lm_head.
      Marker files /tmp/vllm-draft-head-{bf16,fp8} switch it for A/B without restarting.

Usage: patch_flashnext.py <site-packages>
"""
import ast, sys

PKG = f"{sys.argv[1]}/vllm/models/qwen4_exp/nvidia"

# --- (1) model.py -----------------------------------------------------------------
p = f"{PKG}/model.py"
s = open(p).read()
hook = "        enable_qwen4_exp_low_latency_gemm(self, self.model_config.dtype)\n"
assert s.count(hook) == 1, "model.py hook not found"
s = s.replace(hook, hook +
    "        # flashnext-gb10: row-wise FP8 for the dense BF16 Linears (selection in\n"
    "        # /root/.cache/vllm/fn_dense_fp8.conf).\n"
    "        from .fn_dense_fp8 import enable_fn_dense_fp8\n"
    "        enable_fn_dense_fp8(self)\n")
ast.parse(s); open(p, "w").write(s)

# --- (2) mtp.py -------------------------------------------------------------------
p = f"{PKG}/mtp.py"
s = open(p).read()
old_logits = (
    "    def compute_logits(\n"
    "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
    "    ) -> torch.Tensor | None:\n"
    "        return self.logits_processor(self.lm_head, hidden_states)\n")
assert s.count(old_logits) == 1, "MTP compute_logits not found"
s = s.replace(old_logits,
    "    def compute_logits(\n"
    "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
    "    ) -> torch.Tensor | None:\n"
    "        heads = getattr(self, \"_fn_draft_heads\", None)\n"
    "        mode = _fn_draft_head_mode()\n"
    "        if heads is None or mode == \"bf16\":\n"
    "            return self.logits_processor(self.lm_head, hidden_states)\n"
    "        return heads[mode](hidden_states)\n")
old_load = "        return loader.load_weights(remap_weight_names(), mapper=mapper)\n"
assert s.count(old_load) == 1, "MTP load_weights not found"
s = s.replace(old_load,
    "        loaded = loader.load_weights(remap_weight_names(), mapper=mapper)\n"
    "        # flashnext-gb10: build the drafter's quantized head here (outside graph captures).\n"
    "        w = getattr(self.lm_head, \"weight\", None)\n"
    "        if w is not None and w.dtype == torch.bfloat16 and w.dim() == 2:\n"
    "            self._fn_draft_heads = _fn_build_quantized_heads(\n"
    "                w, self.logits_processor.org_vocab_size)\n"
    "        else:\n"
    "            from vllm.logger import init_logger\n"
    "            init_logger(__name__).warning(\n"
    "                \"Flash-Next MTP: lm_head is not 2-D BF16 (%s); draft head left unquantized\",\n"
    "                None if w is None else (w.dtype, tuple(w.shape)))\n"
    "        return loaded\n")
anchor = '\n\n__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]\n'
assert s.count(anchor) == 1
s = s.replace(anchor, '''


# --- flashnext-gb10: quantized lm_head for the drafter only ----------------------------
def _fn_draft_head_mode() -> str:
    import os
    if os.path.exists("/tmp/vllm-draft-head-bf16"):
        return "bf16"
    if os.path.exists("/tmp/vllm-draft-head-fp8"):
        return "fp8"
    return "nvfp4"


def _fn_build_quantized_heads(weight: torch.Tensor, org_vocab_size: int) -> dict:
    """{'nvfp4': fn, 'fp8': fn}; each maps hidden [M, H] -> logits [M, V]."""
    import types
    from vllm import _custom_ops as ops
    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        prepare_fp4_layer_for_marlin,
    )

    logger = init_logger(__name__)
    w = weight.detach()
    vocab, hidden = w.shape
    with torch.no_grad():
        amax = w.abs().amax().float()
        gs_quant = (448.0 * 6.0) / amax
        packed, scales = ops.scaled_fp4_quant(w, gs_quant, is_sf_swizzled_layout=False)
        layer = types.SimpleNamespace()
        layer.weight = nn.Parameter(packed, requires_grad=False)
        layer.weight_scale = nn.Parameter(scales, requires_grad=False)
        layer.weight_global_scale = nn.Parameter((1.0 / gs_quant).reshape(()), requires_grad=False)
        layer.output_size_per_partition = vocab
        layer.input_size_per_partition = hidden
        layer.params_dtype = w.dtype
        prepare_fp4_layer_for_marlin(layer)

    fp8_state: dict = {}

    def _fp8_weights():
        if not fp8_state:
            with torch.no_grad():
                row_amax = w.abs().amax(dim=1).float().clamp_min(1e-12)
                w_scale = row_amax / 448.0
                w8 = (w.float() / w_scale[:, None]).to(torch.float8_e4m3fn)
                fp8_state["w8_t"] = w8.t()
                fp8_state["w_scale_row"] = w_scale.reshape(1, vocab).contiguous()
        return fp8_state["w8_t"], fp8_state["w_scale_row"]

    def nvfp4_head(hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.reshape(-1, hidden)
        if x.dtype != layer.params_dtype:
            x = x.to(layer.params_dtype)
        out = apply_fp4_marlin_linear(
            x, layer.weight, layer.weight_scale, layer.weight_global_scale,
            layer.workspace, vocab, hidden,
        )
        return out[..., :org_vocab_size]

    def fp8_head(hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.reshape(-1, hidden)
        x_scale = (x.abs().amax(dim=1, keepdim=True).float() / 448.0).clamp_min(1e-12)
        x8 = (x.float() / x_scale).to(torch.float8_e4m3fn)
        w8_t, w_scale_row = _fp8_weights()
        out = torch._scaled_mm(x8, w8_t, scale_a=x_scale.contiguous(), scale_b=w_scale_row,
                               out_dtype=w.dtype)
        return out[..., :org_vocab_size]

    logger.info("Flash-Next MTP: built NVFP4 draft head (%.0f MB; fp8 lazy); mode=%s",
                layer.weight.numel() * 4 / 1e6, _fn_draft_head_mode())
    return {"nvfp4": nvfp4_head, "fp8": fp8_head}
''' + anchor)
ast.parse(s); open(p, "w").write(s)
print("flashnext-gb10 patches (dense-fp8 + draft head) applied OK")
