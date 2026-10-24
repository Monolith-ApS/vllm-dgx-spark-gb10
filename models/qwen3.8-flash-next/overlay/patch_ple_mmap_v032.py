#!/usr/bin/env python3
"""Step 1 on vLLM v0.32: vllm_ple_mmap.py's v0.30-layout hook replaces Qwen4ExpNGramEmbedding.__init__ with a fixed
signature, and v0.32 added the keyword use_sequence_parallel (nvidia/model.py passes it). Accept it and pass it on.
The hook's forward returns the full-token lookup and never shards it, so sequence parallel is refused (one Spark,
TP 1: the model passes False)."""
import ast
import sys

p = sys.argv[1]
s = open(p).read()
pairs = [(
    """    def __init__(self, config, embedding_dim, ple_dense_layer_id, max_total_tokens, *,
                 data_parallel_rank, prefix, quant_config=None, params_dtype=None):
        real = {a: getattr(mod, a) for a in embed_attrs}
""",
    """    def __init__(self, config, embedding_dim, ple_dense_layer_id, max_total_tokens, *,
                 data_parallel_rank, prefix, quant_config=None, params_dtype=None,
                 use_sequence_parallel=False):
        if use_sequence_parallel:
            raise RuntimeError("PLE mmap: sequence parallel is not supported (the lookup is never sharded)")
        real = {a: getattr(mod, a) for a in embed_attrs}
"""), (
    """                      data_parallel_rank=data_parallel_rank, prefix=prefix,
                      quant_config=None, params_dtype=params_dtype)
""",
    """                      data_parallel_rank=data_parallel_rank, prefix=prefix,
                      quant_config=None, params_dtype=params_dtype,
                      use_sequence_parallel=use_sequence_parallel)
""")]
for old, new in pairs:
    assert s.count(old) == 1, f"anchor found {s.count(old)} times:\n{old}"
    s = s.replace(old, new)
ast.parse(s)
open(p, "w").write(s)
print("vllm_ple_mmap.py: use_sequence_parallel accepted (v0.32)")
