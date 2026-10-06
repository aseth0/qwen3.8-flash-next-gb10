#!/usr/bin/env python3
"""MBX: QSA fused multi-step draft metadata, v2 = the upstream PR code (vLLM 0.30). Replaces patch 05 v1.

v1 refreshed logical_positions / slot_mapping / k_work_metadata by hand in an update hook and never refreshed
visible_blocks (which decode selection reads). v2 is the change prepared for vllm-project/vllm (branch
qwen4exp-qsa-fused-draft-metadata in /var/www/vllm): `update_draft_decode_metadata` re-launches the builder's own
`_build_qsa_metadata_kernel` in place (the DeepseekSparseSWA pattern), via `_launch_qsa_metadata_kernel` shared with
build(); QSAForwardMetadata gains `common_slot_mapping` + `num_mapped_tokens`. Exact by construction.

Shipped as a whole-file overlay (docker/overlays-v030/qsa_cache.py) = 0.30's qsa_cache.py + the PR diff (applies
cleanly) + ONE local line: the MBX_FUSED_DRAFT A/B gate on the opt-in flag (unset/"1" = fused, "0" = stock).
PROVENANCE IS CHECKED: BASE_SHA256 = vllm/vllm-openai:v0.30.0's original file (no other patch touches it);
OVERLAY_SHA256 = the shipped overlay. Both asserted. `--check` = hashes only.
"""
import hashlib, pathlib, shutil, sys
import vllm

CHECK = "--check" in sys.argv
BASE_SHA256 = "c6651a0cf27e2ffb23b90eafe21e044df5609ce976f52ebffa7a4fbea688fc04"
OVERLAY_SHA256 = "b0eb2fb7def9c8d63b1d063e9854d517ecd911c37c95ad51a6db1348c3bb45fc"
DST = pathlib.Path(vllm.__file__).parent / "models/qwen4_exp/common/qsa_cache.py"
SRC = pathlib.Path(__file__).resolve().parent.parent / "overlays-v030/qsa_cache.py"
if not SRC.exists():
    SRC = pathlib.Path("/opt/mbx/overlays-v030/qsa_cache.py")

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
got_base = sha(DST)
assert got_base != OVERLAY_SHA256, "already patched"
assert got_base == BASE_SHA256, f"qsa_cache.py is not the v0.30.0 original (sha {got_base}) — re-port the overlay"
assert sha(SRC) == OVERLAY_SHA256, f"overlay {SRC} changed (sha {sha(SRC)}) — update OVERLAY_SHA256 deliberately"
if not CHECK:
    shutil.copyfile(SRC, DST)
    assert sha(DST) == OVERLAY_SHA256
print("QSA fused draft v2 (upstream PR code):", "hashes OK" if CHECK else f"applied to {DST}")
