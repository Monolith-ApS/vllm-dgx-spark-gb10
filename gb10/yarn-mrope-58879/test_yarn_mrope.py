"""Does vLLM's rotary cache match the HF/vendor YaRN formula for Qwen3.8 Flash Next? CPU only, no GPU, ~2-3 GB RAM.
Run inside an image:  docker run --rm --gpus all --memory 8g --entrypoint python3 <image> /opt/vllm-dgx-spark/test_yarn_mrope.py
Exit code 0 when every factor matches transformers' ROPE_INIT_FUNCTIONS["yarn"]; the unpatched image is expected to FAIL."""
import sys

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
try:
    from transformers import PreTrainedConfig
except ImportError:  # older transformers
    from transformers import PretrainedConfig as PreTrainedConfig
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope

HEAD, PARTIAL, THETA, SECTION, ORIG = 256, 0.25, 1e7, [11, 11, 10], 262144
bad = 0
with set_current_vllm_config(VllmConfig()):
    for ctx, factor in ((524288, 2.0), (786432, 3.0), (1048576, 4.0)):
        rp = {"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": ORIG, "rope_theta": THETA,
              "partial_rotary_factor": PARTIAL, "mrope_section": SECTION, "mrope_interleaved": True}
        hf = PreTrainedConfig(hidden_size=HEAD, num_attention_heads=1, head_dim=HEAD, max_position_embeddings=ORIG, rope_parameters=dict(rp))
        inv_freq, attn = ROPE_INIT_FUNCTIONS["yarn"](hf, "cpu")
        rope = get_rope(head_size=HEAD, max_position=ORIG, rope_parameters=dict(rp), dtype=torch.float32)
        n = 4096
        freqs = torch.outer(torch.arange(n, dtype=torch.float32), inv_freq)
        want = torch.cat((freqs.cos(), freqs.sin()), dim=-1) * attn
        got = rope.cos_sin_cache[:n].float().cpu()
        err = (got - want).abs().max().item()
        ok = torch.allclose(got, want)
        bad += not ok
        print("ctx %7d factor %.0f: max |vllm - hf| = %.3e  %s" % (ctx, factor, err, "MATCH" if ok else "DIFFERENT"), flush=True)
sys.exit(1 if bad else 0)
