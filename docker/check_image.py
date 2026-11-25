"""Build-time check, no GPU: imports the patched modules and builds the placeholder n-gram table."""
import inspect
import os
import types

os.environ["VLLM_PLE_MMAP"] = "1"
import torch
import vllm
from vllm.model_executor.utils import copy_weight_
from vllm.config.cache import CacheConfig
import vllm.model_executor.layers.mamba.gdn.recoverssm_gdn
import vllm.models.qwen4_exp.nvidia.fn_dense_fp8
import vllm.models.qwen4_exp.nvidia.model as model
import vllm.models.qwen4_exp.nvidia.mtp as mtp
import vllm.models.qwen4_exp.nvidia.ngram_embedding as ng
import vllm.v1.attention.backends.ple_recoverssm

assert vllm.__version__ == "0.34.0", vllm.__version__
assert "use_recoverssm" in CacheConfig.__dataclass_fields__
assert "_fn_build_quantized_heads" in inspect.getsource(mtp)
source = inspect.getsource(model)
assert "enable_fn_dense_fp8" in source and "SupportsReplaySSM" in source
assert "_is_integrated_gpu" in inspect.getsource(copy_weight_)  # upstream replacement for the MoE clone-on-load
assert getattr(ng.Qwen4ExpNGramEmbedding, "_ple_mmap_patched", False), "PLE mmap not applied"

ng.get_current_vllm_config = lambda: types.SimpleNamespace(engram_config=None)  # no active config outside a server
config = types.SimpleNamespace(ngram_size=3, heads_per_ngram=2, eos_token_id=2, vocab_size=1000, split_ngram_parts=4,
                               seed=1234, ngram_vocab_size_base=101, make_ngram_vocab_size_divisible_by=16)
table = ng.Qwen4ExpNGramEmbedding(config, 160, 0, 64, data_parallel_rank=0, prefix="t.ple_embedding", params_dtype=torch.bfloat16)
assert type(table.ngram_embedding).__name__ == "_MmapNgramEmbeddingV030" and not table.ngram_embedding.supports_prefetch
ids = table.compute_ngram_ids(torch.tensor([5, 6, 7]), torch.tensor([0, 3]), torch.full((1, 2), 2))
assert ids.shape == (3, table.ngram_heads), ids.shape
print("flashnext-gb10 v034-rssm OK")
