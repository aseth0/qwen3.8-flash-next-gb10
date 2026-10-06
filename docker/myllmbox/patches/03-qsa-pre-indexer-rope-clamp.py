#!/usr/bin/env python3
"""MBX: clamp QSA pre-indexer RoPE positions (graph-warmup IMA on SM121).

Crash: `_qsa_pre_indexer_kernel` Triton IMA during CUDA-graph warmup
(ops/qsa_pre_indexer.py). `_norm_rope` loads `cos_sin[pos]` with no bounds
check. Capture dummy positions are out of range of the cos/sin table
(max_model_len). Clamp pos_t/h/w into [0, n_rope)` inside `_norm_rope` so
both the Q tile and the K-pool call sites are covered. `--check` = anchors.
"""
import pathlib
import sys

import vllm

CHECK = "--check" in sys.argv
P = str(pathlib.Path(vllm.__file__).parent / "models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py")
s = open(P).read()
assert "MBX SM121 rope clamp" not in s, "already patched"

edits = []
edits.append(("norm_rope sig",
    """    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    \"\"\"Apply Gemma RMSNorm and selected-axis NeoX RoPE to register rows.\"\"\"""",
    """    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    \"\"\"Apply Gemma RMSNorm and selected-axis NeoX RoPE to register rows.\"\"\"
    pos_t = tl.minimum(tl.maximum(pos_t, 0), MAX_POS)  # MBX SM121 rope clamp
    pos_h = tl.minimum(tl.maximum(pos_h, 0), MAX_POS)
    pos_w = tl.minimum(tl.maximum(pos_w, 0), MAX_POS)"""))

edits.append(("q call",
    """            IS_2D_POSITIONS,
            MROPE_H,
            MROPE_W,
        )""",
    """            IS_2D_POSITIONS,
            MROPE_H,
            MROPE_W,
            MAX_POS,
        )"""))

edits.append(("k call",
    """                IS_K_MROPE,
                MROPE_H,
                MROPE_W,
            )""",
    """                IS_K_MROPE,
                MROPE_H,
                MROPE_W,
                MAX_POS,
            )"""))

edits.append(("kernel sig",
    """    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    pid = tl.program_id(0)""",
    """    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    pid = tl.program_id(0)"""))

edits.append(("launch",
    """        MROPE_H=section[1],
        MROPE_W=section[2],
        num_warps=1,""",
    """        MROPE_H=section[1],
        MROPE_W=section[2],
        MAX_POS=max(int(cos_sin_cache.shape[0]) - 1, 0),
        num_warps=1,"""))

for label, a, b in edits:
    assert s.count(a) == 1, f"anchor '{label}' not found/unique ({s.count(a)})"
    s = s.replace(a, b, 1)

if not CHECK:
    open(P, "w").write(s)
print("QSA pre-indexer rope clamp:", "anchors OK" if CHECK else f"applied to {P}")
