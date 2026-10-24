#!/usr/bin/env python3
"""Step 4 on vLLM v0.32: only the scheduler half of patch_mamba_block_size.py is still needed.

v0.32 seeds the align-mode state slot with cache_config.mamba_block_size itself (upstream #55601,
v1/worker/gpu/model_states/mamba_hybrid.py), so the worker half is gone. The block-aligned prefill split in
Scheduler._mamba_block_aligned_split still reads cache_config.block_size, which EngineCore overwrites with the
MIN group block size (the QSA raw-key ring, 8/16): chunk ends then never land on mamba boundaries (1680) and
states are almost never cached. Same replacement as on v0.31: the scheduler's own block size (LCM of the groups).
Fails loudly when either side drifted."""
import ast
import sys

SP = sys.argv[1]
MH = f"{SP}/vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
SC = f"{SP}/vllm/v1/core/sched/scheduler.py"
s = open(MH).read()
assert "(new_req_data.num_computed_tokens - 1) // mamba_block_size\n" in s, \
    "mamba_hybrid no longer seeds with mamba_block_size: use the full patch_mamba_block_size.py"
t = open(SC).read()
b = "        block_size = self.cache_config.block_size\n        # The last block-aligned position whose state can be cached."
assert t.count(b) == 1, "scheduler split line not found"
t = t.replace(b, "        block_size = self.block_size  # scheduler block size (LCM of groups) == mamba block size\n"
                 "        # The last block-aligned position whose state can be cached.")
ast.parse(t)
open(SC, "w").write(t)
print("mamba block_size fix (scheduler half, v0.32) applied OK")
